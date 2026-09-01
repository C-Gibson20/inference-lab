"""Atomic initialisation for an empty SQLite metadata repository."""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    SQLITE_APPLICATION_ID,
    MigrationDefinition,
    validate_migration_registry,
)
from inference_lab.storage.errors import (
    IncompatibleSchemaError,
    StorageError,
    StorageMigrationError,
    StorageTransactionError,
)

AppliedAtFactory = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


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


def _read_pragma_integer(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None or not isinstance(row[0], int) or isinstance(row[0], bool):
        raise StorageMigrationError()
    return row[0]


def _ensure_empty_target(connection: sqlite3.Connection) -> None:
    if _read_pragma_integer(connection, "application_id") != 0:
        raise IncompatibleSchemaError()
    if _read_pragma_integer(connection, "user_version") != 0:
        raise IncompatibleSchemaError()

    existing_object = connection.execute(
        """SELECT 1
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        LIMIT 1"""
    ).fetchone()
    if existing_object is not None:
        raise IncompatibleSchemaError()


def _record_migration(
    connection: sqlite3.Connection,
    migration: MigrationDefinition,
    applied_at_factory: AppliedAtFactory,
) -> None:
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


def _rollback_preserving_primary(connection: sqlite3.Connection) -> None:
    if not connection.in_transaction:
        return
    try:
        connection.rollback()
    except sqlite3.Error:
        pass


def initialise_empty_database(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
    applied_at_factory: AppliedAtFactory = _utc_now,
) -> None:
    """Atomically create and publish all known migrations on an empty target."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection.")
    if not callable(applied_at_factory):
        raise TypeError("applied_at_factory must be callable.")
    try:
        validate_migration_registry(migrations)
    except (TypeError, ValueError):
        raise StorageMigrationError() from None
    if connection.in_transaction:
        raise StorageTransactionError()

    try:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_empty_target(connection)

        for migration in migrations:
            for statement in migration.statements:
                connection.execute(statement)
            _record_migration(connection, migration, applied_at_factory)

        connection.execute(f"PRAGMA application_id = {SQLITE_APPLICATION_ID}")
        if _read_pragma_integer(connection, "application_id") != SQLITE_APPLICATION_ID:
            raise StorageMigrationError()
        connection.commit()
    except StorageError:
        _rollback_preserving_primary(connection)
        raise
    except sqlite3.Error as error:
        _rollback_preserving_primary(connection)
        raise translate_sqlite_error(
            error,
            default=StorageMigrationError,
            constraint=StorageMigrationError,
        ) from None
    except Exception:
        _rollback_preserving_primary(connection)
        raise StorageMigrationError() from None
