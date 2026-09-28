"""Tests for SQLite schema inspection."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.storage import (
    IncompatibleSchemaError,
    MigrationRequiredError,
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
    SQLITE_APPLICATION_ID,
    MigrationDefinition,
)

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252


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
            current_revision=7,
            latest_supported_revision=7,
            applied_migration_count=7,
        )
        assert connection.total_changes == changes_before
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()


def test_superseded_pre_release_repository_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "pre-release.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute(f"PRAGMA application_id = {SQLITE_APPLICATION_ID}")
    connection.execute("PRAGMA user_version = 5")
    connection.execute("CREATE TABLE superseded_schema_marker (value TEXT)")
    connection.commit()
    connection.close()

    with pytest.raises(IncompatibleSchemaError):
        open_current_database(database_path)

    reopened = sqlite3.connect(database_path)
    try:
        assert reopened.execute("PRAGMA application_id").fetchone()[0] == (
            SQLITE_APPLICATION_ID
        )
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 5
        assert reopened.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert reopened.execute(
            "SELECT name FROM sqlite_schema WHERE name = 'superseded_schema_marker'"
        ).fetchone() == ("superseded_schema_marker",)
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


def test_inconsistent_schema_or_history_is_rejected(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path)
    connection = open_sqlite_connection(database_path)
    try:
        connection.execute(
            "UPDATE migration_history SET checksum = "
            "'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'"
        )

        with pytest.raises(IncompatibleSchemaError):
            validate_database_schema(connection)
    finally:
        connection.close()


def test_schema_definition_drift_is_rejected_on_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise(database_path)
    connection = open_sqlite_connection(database_path)
    try:
        connection.execute("DROP TABLE document_family_memberships")
        schema_before = tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT type, name, tbl_name, sql
                FROM sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name, tbl_name"""
            ).fetchall()
        )
    finally:
        connection.close()

    with pytest.raises(IncompatibleSchemaError):
        open_current_database(database_path)

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        schema_after = tuple(
            connection.execute(
                """SELECT type, name, tbl_name, sql
                FROM sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name, tbl_name"""
            ).fetchall()
        )
        assert schema_after == schema_before
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
