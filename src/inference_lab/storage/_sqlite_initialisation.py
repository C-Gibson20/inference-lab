"""Atomic initialisation for an empty SQLite metadata repository."""

import sqlite3

from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_migration_execution import (
    AppliedAtFactory,
    execute_migration,
    rollback_preserving_primary,
    utc_now,
)
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


def initialise_empty_database(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
    applied_at_factory: AppliedAtFactory = utc_now,
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
            execute_migration(connection, migration, applied_at_factory)

        connection.execute(f"PRAGMA application_id = {SQLITE_APPLICATION_ID}")
        if _read_pragma_integer(connection, "application_id") != SQLITE_APPLICATION_ID:
            raise StorageMigrationError()
        connection.commit()
    except StorageError:
        rollback_preserving_primary(connection)
        raise
    except sqlite3.Error as error:
        rollback_preserving_primary(connection)
        raise translate_sqlite_error(
            error,
            default=StorageMigrationError,
            constraint=StorageMigrationError,
        ) from None
    except Exception:
        rollback_preserving_primary(connection)
        raise StorageMigrationError() from None
