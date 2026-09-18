"""Tests for SQLite database opening and schema lifecycle."""

import sqlite3
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import MigrationRequiredError, StorageMigrationError
from inference_lab.storage._sqlite_connection import open_sqlite_connection
from inference_lab.storage._sqlite_initialisation import initialise_empty_database
from inference_lab.storage._sqlite_lifecycle import open_sqlite_database
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252


def _revision_two(*, failing: bool = False) -> MigrationDefinition:
    statements = ["ALTER TABLE corpora ADD COLUMN corpus_label TEXT"]
    if failing:
        statements.append("CREATE TABLE documents (conflicting_value TEXT)")
    return MigrationDefinition(
        revision=2,
        migration_id="test_add_corpus_label",
        statements=tuple(statements),
    )


def _initialise_baseline(path: Path) -> None:
    connection = open_sqlite_connection(path)
    try:
        initialise_empty_database(
            connection,
            migrations=(MIGRATIONS[0],),
            applied_at_factory=_fixed_applied_at,
        )
    finally:
        connection.close()


def test_default_open_is_read_only_and_requires_initialisation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"

    with pytest.raises(MigrationRequiredError):
        open_sqlite_database(database_path)

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


def test_explicit_schema_policy_initialises_and_reopens_idempotently(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"

    connection = open_sqlite_database(
        database_path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert (
            connection.execute("SELECT count(*) FROM migration_history").fetchone()[0]
            == 7
        )
    finally:
        connection.close()

    reopened = open_sqlite_database(database_path, apply_migrations=True)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 7
        assert (
            reopened.execute("SELECT count(*) FROM migration_history").fetchone()[0]
            == 7
        )
    finally:
        reopened.close()


def test_older_schema_is_not_migrated_without_explicit_permission(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise_baseline(database_path)
    migrations = (MIGRATIONS[0], _revision_two())

    with pytest.raises(MigrationRequiredError):
        open_sqlite_database(database_path, migrations=migrations)

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert (
            connection.execute(
                "SELECT count(*) FROM pragma_table_info('corpora') WHERE name = 'corpus_label'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_explicit_schema_policy_applies_supported_forward_migrations(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _initialise_baseline(database_path)
    migrations = (MIGRATIONS[0], _revision_two())

    connection = open_sqlite_database(
        database_path,
        apply_migrations=True,
        migrations=migrations,
        applied_at_factory=_fixed_applied_at,
    )
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert (
            connection.execute("SELECT count(*) FROM migration_history").fetchone()[0]
            == 2
        )
    finally:
        connection.close()


def test_initialisation_failure_is_atomic_and_logged_without_payloads(
    tmp_path: Path,
) -> None:
    private_marker = "private-storage-path-marker"
    database_path = tmp_path / private_marker / "metadata.sqlite3"
    migrations = (MIGRATIONS[0], _revision_two(failing=True))

    with capture_logs() as logs:
        with pytest.raises(StorageMigrationError):
            open_sqlite_database(
                database_path,
                apply_migrations=True,
                migrations=migrations,
                applied_at_factory=_fixed_applied_at,
            )

    assert [(log["event"], log["log_level"]) for log in logs] == [
        ("repository_open_started", "info"),
        ("repository_initialisation_started", "info"),
        ("repository_initialisation_failed", "error"),
        ("repository_open_failed", "error"),
    ]
    assert logs[-1]["error_category"] == "migration"
    assert all(log["pipeline_stage"] == "storage" for log in logs)
    rendered = repr(logs)
    assert private_marker not in rendered
    assert "ALTER TABLE" not in rendered
    assert "CREATE TABLE" not in rendered

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_successful_open_logs_bounded_lifecycle_metadata(tmp_path: Path) -> None:
    database_path = tmp_path / "private-marker" / "metadata.sqlite3"

    with capture_logs() as logs:
        connection = open_sqlite_database(
            database_path,
            apply_migrations=True,
            applied_at_factory=_fixed_applied_at,
        )
    connection.close()

    assert [log["event"] for log in logs] == [
        "repository_open_started",
        "repository_initialisation_started",
        "repository_initialisation_completed",
        "repository_open_completed",
    ]
    assert logs[-1]["schema_revision"] == 7
    assert logs[-1]["applied_migration_count"] == 7
    assert all(log["pipeline_stage"] == "storage" for log in logs)
    assert str(database_path) not in repr(logs)


def test_schema_update_policy_requires_an_explicit_boolean(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        open_sqlite_database(
            tmp_path / "metadata.sqlite3",
            apply_migrations=1,  # type: ignore[arg-type]
        )


def test_lifecycle_rejects_an_invalid_time_factory_before_open(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"

    with pytest.raises(TypeError):
        open_sqlite_database(
            database_path,
            applied_at_factory=None,  # type: ignore[arg-type]
        )

    assert not database_path.exists()
