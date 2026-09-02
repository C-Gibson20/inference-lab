import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from inference_lab.storage import (
    IncompatibleSchemaError,
    StorageConcurrencyError,
    StorageMigrationError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_connection import open_sqlite_connection
from inference_lab.storage._sqlite_initialisation import initialise_empty_database
from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    SQLITE_APPLICATION_ID,
    MigrationDefinition,
)

_APPLIED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _schema_objects(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = connection.execute(
        """SELECT type, name
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name"""
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def test_initialisation_atomically_publishes_current_revision(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    connection = open_sqlite_connection(database_path)
    try:
        initialise_empty_database(
            connection,
            applied_at_factory=_fixed_applied_at,
        )

        assert connection.in_transaction is False
        assert connection.execute("PRAGMA application_id").fetchone()[0] == (
            SQLITE_APPLICATION_ID
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        schema_row = connection.execute(
            "SELECT schema_name, current_revision FROM metadata_schema"
        ).fetchone()
        assert tuple(schema_row) == ("inference_lab_metadata", 4)
        history_rows = connection.execute(
            """SELECT revision, migration_id, checksum, applied_at_utc
            FROM migration_history ORDER BY revision"""
        ).fetchall()
        assert [tuple(row) for row in history_rows] == [
            (
                1,
                "initial_metadata",
                MIGRATIONS[0].checksum,
                "2026-09-01T12:30:00.000000+00:00",
            ),
            (
                2,
                "source_observation_identity",
                MIGRATIONS[1].checksum,
                "2026-09-01T12:30:00.000000+00:00",
            ),
            (
                3,
                "duplicate_hash_integrity",
                MIGRATIONS[2].checksum,
                "2026-09-01T12:30:00.000000+00:00",
            ),
            (
                4,
                "corpus_catalogue_membership",
                MIGRATIONS[3].checksum,
                "2026-09-01T12:30:00.000000+00:00",
            ),
        ]
    finally:
        connection.close()

    reopened = open_sqlite_connection(database_path)
    try:
        assert reopened.execute("PRAGMA application_id").fetchone()[0] == (
            SQLITE_APPLICATION_ID
        )
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 4
        assert (
            reopened.execute("SELECT current_revision FROM metadata_schema").fetchone()[
                0
            ]
            == 4
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("identity_pragma", ["application_id", "user_version"])
def test_initialisation_rejects_nonzero_identity_metadata_without_changes(
    tmp_path: Path,
    identity_pragma: str,
) -> None:
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(f"PRAGMA {identity_pragma} = 99")

        with pytest.raises(IncompatibleSchemaError):
            initialise_empty_database(connection)

        assert connection.execute(f"PRAGMA {identity_pragma}").fetchone()[0] == 99
        assert _schema_objects(connection) == []
    finally:
        connection.close()


def test_initialisation_rejects_unrelated_schema_without_modifying_it(
    tmp_path: Path,
) -> None:
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("CREATE TABLE unrelated (value TEXT NOT NULL)")
        before = _schema_objects(connection)

        with pytest.raises(IncompatibleSchemaError):
            initialise_empty_database(connection)

        assert _schema_objects(connection) == before
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        connection.close()


def test_failed_later_migration_rolls_back_revision_one(tmp_path: Path) -> None:
    failing_migration = MigrationDefinition(
        revision=2,
        migration_id="failing_test_migration",
        statements=(
            "CREATE TABLE should_be_rolled_back (value TEXT NOT NULL)",
            "CREATE TABLE documents (conflicting_value TEXT)",
        ),
    )
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageMigrationError):
            initialise_empty_database(
                connection,
                migrations=(MIGRATIONS[0], failing_migration),
                applied_at_factory=_fixed_applied_at,
            )

        assert connection.in_transaction is False
        assert _schema_objects(connection) == []
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.parametrize(
    "invalid_value",
    [
        "2026-09-01T12:30:00Z",
        datetime(2026, 9, 1, 12, 30),
        datetime(2026, 9, 1, 13, 30, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_invalid_applied_time_rolls_back_initialisation(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageMigrationError):
            initialise_empty_database(
                connection,
                applied_at_factory=lambda: invalid_value,  # type: ignore[return-value]
            )

        assert _schema_objects(connection) == []
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
    finally:
        connection.close()


def test_initialisation_rejects_an_existing_transaction(tmp_path: Path) -> None:
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("BEGIN")

        with pytest.raises(StorageTransactionError):
            initialise_empty_database(connection)

        assert connection.in_transaction is True
        assert _schema_objects(connection) == []
        connection.rollback()
    finally:
        connection.close()


def test_initialisation_lock_timeout_is_typed_and_retryable(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    writer = open_sqlite_connection(database_path, busy_timeout_ms=1)
    contender = open_sqlite_connection(database_path, busy_timeout_ms=1)
    try:
        writer.execute("BEGIN IMMEDIATE")

        with pytest.raises(StorageConcurrencyError) as captured:
            initialise_empty_database(contender)

        assert captured.value.retryable is True
        assert contender.in_transaction is False
        assert _schema_objects(contender) == []
    finally:
        writer.rollback()
        writer.close()
        contender.close()
