import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_lab.storage import (
    IncompatibleSchemaError,
    MigrationRequiredError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_connection import open_sqlite_connection
from inference_lab.storage._sqlite_initialisation import initialise_empty_database
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    open_current_database,
    validate_database_schema,
)
from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    MigrationDefinition,
)

_APPLIED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _initialise(
    path: Path,
    *,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
) -> None:
    connection = open_sqlite_connection(path)
    try:
        initialise_empty_database(
            connection,
            migrations=migrations,
            applied_at_factory=_fixed_applied_at,
        )
    finally:
        connection.close()


def _second_migration() -> MigrationDefinition:
    return MigrationDefinition(
        revision=2,
        migration_id="test_add_marker",
        statements=("CREATE TABLE migration_test_marker (value TEXT) STRICT",),
    )


def test_current_database_inspection_is_read_only(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path)
    connection = open_sqlite_connection(database_path)
    try:
        changes_before = connection.total_changes

        inspection = validate_database_schema(connection)

        assert inspection == SchemaInspection(
            current_revision=3,
            latest_supported_revision=3,
            applied_migration_count=3,
        )
        assert connection.total_changes == changes_before
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()


def test_validated_open_enables_wal_only_after_validation(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path)

    connection = open_current_database(database_path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_empty_database_requires_explicit_initialisation_without_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"

    with pytest.raises(MigrationRequiredError):
        open_current_database(database_path)

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_unrelated_database_is_rejected_without_changing_journal_mode(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "unrelated.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.commit()
    connection.close()

    with pytest.raises(IncompatibleSchemaError):
        open_current_database(database_path)

    reopened = sqlite3.connect(database_path)
    try:
        assert reopened.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert reopened.execute(
            "SELECT name FROM sqlite_schema WHERE name = 'unrelated'"
        ).fetchone() == ("unrelated",)
    finally:
        reopened.close()


def test_supported_pending_migration_is_reported_without_application(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path, migrations=(MIGRATIONS[0],))
    migrations = (MIGRATIONS[0], _second_migration())
    connection = open_sqlite_connection(database_path)
    try:
        changes_before = connection.total_changes

        with pytest.raises(MigrationRequiredError):
            validate_database_schema(connection, migrations=migrations)

        assert connection.total_changes == changes_before
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT current_revision FROM metadata_schema"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT count(*) FROM migration_history").fetchone()[0]
            == 1
        )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "PRAGMA user_version = 4",
        "UPDATE metadata_schema SET current_revision = 0",
        "DELETE FROM migration_history",
        "UPDATE migration_history SET checksum = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'",
        "UPDATE migration_history SET migration_id = 'changed_migration' WHERE revision = 1",
        "UPDATE migration_history SET applied_at_utc = 'not-a-timestamp'",
    ],
)
def test_inconsistent_schema_or_history_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path)
    connection = open_sqlite_connection(database_path)
    try:
        connection.execute(mutation)

        with pytest.raises(IncompatibleSchemaError):
            validate_database_schema(connection)
    finally:
        connection.close()


def test_migration_history_must_be_contiguous_and_complete(tmp_path: Path) -> None:
    second = _second_migration()
    database_path = tmp_path / "metadata.sqlite3"
    connection = open_sqlite_connection(database_path)
    try:
        initialise_empty_database(
            connection,
            migrations=(MIGRATIONS[0], second),
            applied_at_factory=_fixed_applied_at,
        )
        connection.execute("DELETE FROM migration_history WHERE revision = 1")

        with pytest.raises(IncompatibleSchemaError):
            validate_database_schema(
                connection,
                migrations=(MIGRATIONS[0], second),
            )
    finally:
        connection.close()


def test_inspection_rejects_an_existing_transaction(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path)
    connection = open_sqlite_connection(database_path)
    try:
        connection.execute("BEGIN")

        with pytest.raises(StorageTransactionError):
            validate_database_schema(connection)

        assert connection.in_transaction is True
        connection.rollback()
    finally:
        connection.close()
