"""Shared execution helpers for atomic SQLite metadata migrations."""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from inference_lab.storage._sqlite_migrations import MigrationDefinition
from inference_lab.storage.errors import StorageMigrationError

AppliedAtFactory = Callable[[], datetime]


def _format_applied_at(factory: AppliedAtFactory) -> str:
    try:
        value = factory()
    except Exception:
        raise StorageMigrationError() from None
    if not isinstance(value, datetime):
        raise StorageMigrationError()
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError):
        raise StorageMigrationError() from None
    if value.tzinfo is None or offset != timedelta(0):
        raise StorageMigrationError()
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def execute_migration(
    connection: sqlite3.Connection,
    migration: MigrationDefinition,
    applied_at_factory: AppliedAtFactory,
) -> None:
    """Execute and record one migration inside its caller-owned transaction."""
    for statement in migration.statements:
        connection.execute(statement)
    connection.execute(
        """INSERT INTO migration_history (
            revision, migration_id, checksum, applied_at_utc
        ) VALUES (?, ?, ?, ?)""",
        (
            migration.revision,
            migration.migration_id,
            migration.checksum,
            _format_applied_at(applied_at_factory),
        ),
    )
    connection.execute(
        """UPDATE metadata_schema
        SET current_revision = ?
        WHERE singleton = 1 AND current_revision = ?""",
        (migration.revision, migration.revision - 1),
    )
    if connection.execute("SELECT changes()").fetchone()[0] != 1:
        raise StorageMigrationError()
    connection.execute(f"PRAGMA user_version = {migration.revision}")
