"""Tests for explicit SQLite schema upgrades."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    IncompatibleSchemaError,
    MigrationRequiredError,
    StorageMigrationError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_connection import open_sqlite_connection
from inference_lab.storage._sqlite_initialisation import initialise_empty_database
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    validate_database_schema,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_upgrade import apply_pending_migrations

_APPLIED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
_CORPUS_ID = "00000000-0000-0000-0000-000000000001"


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _revision_two() -> MigrationDefinition:
    return MigrationDefinition(
        revision=2,
        migration_id="test_add_corpus_label",
        statements=("ALTER TABLE corpora ADD COLUMN corpus_label TEXT",),
    )


def _revision_three(*, failing: bool = False) -> MigrationDefinition:
    statements = ["CREATE TABLE revision_three_marker (value TEXT) STRICT"]
    if failing:
        statements.append("CREATE TABLE documents (conflicting_value TEXT)")
    return MigrationDefinition(
        revision=3,
        migration_id="test_revision_three",
        statements=tuple(statements),
    )


def _initialise_baseline(path: Path) -> sqlite3.Connection:
    connection = open_sqlite_connection(path)
    initialise_empty_database(
        connection,
        migrations=(MIGRATIONS[0],),
        applied_at_factory=_fixed_applied_at,
    )
    connection.execute(
        "INSERT INTO corpora (corpus_id, display_name) VALUES (?, 'Test corpus')",
        (_CORPUS_ID,),
    )
    return connection


def test_supported_forward_migration_preserves_data_and_reopens(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    connection = _initialise_baseline(database_path)
    revision_two = _revision_two()
    migrations = (MIGRATIONS[0], revision_two)
    try:
        inspection = apply_pending_migrations(
            connection,
            migrations=migrations,
            applied_at_factory=_fixed_applied_at,
        )

        assert inspection == SchemaInspection(2, 2, 2)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT current_revision FROM metadata_schema"
            ).fetchone()[0]
            == 2
        )
        corpus_row = connection.execute(
            "SELECT corpus_id, corpus_label FROM corpora"
        ).fetchone()
        assert tuple(corpus_row) == (_CORPUS_ID, None)
    finally:
        connection.close()

    reopened = open_sqlite_connection(database_path)
    try:
        assert validate_database_schema(reopened, migrations=migrations) == (
            SchemaInspection(2, 2, 2)
        )
    finally:
        reopened.close()


def test_repeated_upgrade_is_a_no_op(tmp_path: Path) -> None:
    connection = _initialise_baseline(tmp_path / "metadata.sqlite3")
    migrations = (MIGRATIONS[0], _revision_two())
    try:
        apply_pending_migrations(
            connection,
            migrations=migrations,
            applied_at_factory=_fixed_applied_at,
        )
        changes_before = connection.total_changes

        inspection = apply_pending_migrations(
            connection,
            migrations=migrations,
            applied_at_factory=_fixed_applied_at,
        )

        assert inspection == SchemaInspection(2, 2, 2)
        assert connection.total_changes == changes_before
        assert (
            connection.execute("SELECT count(*) FROM migration_history").fetchone()[0]
            == 2
        )
    finally:
        connection.close()


def test_failed_later_revision_leaves_last_successful_revision_reopenable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    connection = _initialise_baseline(database_path)
    revision_two = _revision_two()
    revision_three = _revision_three(failing=True)
    migrations = (MIGRATIONS[0], revision_two, revision_three)
    try:
        with pytest.raises(StorageMigrationError):
            apply_pending_migrations(
                connection,
                migrations=migrations,
                applied_at_factory=_fixed_applied_at,
            )

        assert connection.in_transaction is False
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT current_revision FROM metadata_schema"
            ).fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT count(*) FROM migration_history").fetchone()[0]
            == 2
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE name = 'revision_three_marker'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute("SELECT corpus_id FROM corpora").fetchone()[0]
            == _CORPUS_ID
        )
    finally:
        connection.close()

    reopened = open_sqlite_connection(database_path)
    try:
        assert validate_database_schema(
            reopened,
            migrations=(MIGRATIONS[0], revision_two),
        ) == SchemaInspection(2, 2, 2)
        with pytest.raises(MigrationRequiredError):
            validate_database_schema(reopened, migrations=migrations)
    finally:
        reopened.close()


def test_upgrade_rejects_modified_history_before_changing_schema(
    tmp_path: Path,
) -> None:
    connection = _initialise_baseline(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "UPDATE migration_history SET checksum = ? WHERE revision = 1",
            ("b" * 64,),
        )

        with pytest.raises(IncompatibleSchemaError):
            apply_pending_migrations(
                connection,
                migrations=(MIGRATIONS[0], _revision_two()),
            )

        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT count(*) FROM pragma_table_info('corpora') WHERE name = 'corpus_label'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_invalid_application_time_rolls_back_only_pending_revision(
    tmp_path: Path,
) -> None:
    connection = _initialise_baseline(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageMigrationError):
            apply_pending_migrations(
                connection,
                migrations=(MIGRATIONS[0], _revision_two()),
                applied_at_factory=lambda: "not-a-time",  # type: ignore[return-value]
            )

        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT count(*) FROM pragma_table_info('corpora') WHERE name = 'corpus_label'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT corpus_id FROM corpora").fetchone()[0]
            == _CORPUS_ID
        )
    finally:
        connection.close()


def test_upgrade_rejects_an_existing_transaction(tmp_path: Path) -> None:
    connection = _initialise_baseline(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("BEGIN")

        with pytest.raises(StorageTransactionError):
            apply_pending_migrations(
                connection,
                migrations=(MIGRATIONS[0], _revision_two()),
            )

        assert connection.in_transaction is True
        connection.rollback()
    finally:
        connection.close()


def test_migration_logs_are_structured_and_payload_free(tmp_path: Path) -> None:
    connection = _initialise_baseline(tmp_path / "metadata.sqlite3")
    migrations = (MIGRATIONS[0], _revision_two())
    try:
        with capture_logs() as logs:
            apply_pending_migrations(
                connection,
                migrations=migrations,
                applied_at_factory=_fixed_applied_at,
            )

        assert [(log["event"], log["log_level"]) for log in logs] == [
            ("migration_started", "info"),
            ("migration_completed", "info"),
        ]
        assert all(log["pipeline_stage"] == "storage" for log in logs)
        assert all(log["migration_revision"] == 2 for log in logs)
        assert all(log["migration_id"] == "test_add_corpus_label" for log in logs)
        rendered = repr(logs)
        assert _CORPUS_ID not in rendered
        assert "ALTER TABLE" not in rendered
        assert str(tmp_path) not in rendered
    finally:
        connection.close()


def test_failed_migration_log_contains_only_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    connection = _initialise_baseline(tmp_path / "metadata.sqlite3")
    failing = MigrationDefinition(
        revision=2,
        migration_id="test_failing_migration",
        statements=("CREATE TABLE documents (conflicting_value TEXT)",),
    )
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageMigrationError):
                apply_pending_migrations(
                    connection,
                    migrations=(MIGRATIONS[0], failing),
                    applied_at_factory=_fixed_applied_at,
                )

        assert [(log["event"], log["log_level"]) for log in logs] == [
            ("migration_started", "info"),
            ("migration_failed", "error"),
        ]
        assert logs[-1]["error_category"] == "migration"
        assert logs[-1]["retryable"] is False
        rendered = repr(logs)
        assert "CREATE TABLE" not in rendered
        assert _CORPUS_ID not in rendered
        assert str(tmp_path) not in rendered
    finally:
        connection.close()
