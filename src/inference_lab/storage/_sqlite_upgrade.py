"""Supported forward migration execution for SQLite metadata repositories."""

import sqlite3

from inference_lab._clock import utc_now
from inference_lab.storage._sqlite_cleanup import rollback_preserving_primary
from inference_lab.storage._sqlite_connection import (
    enable_durable_wal,
    translate_sqlite_error,
)
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    _inspect_database_schema,
    inspect_database_schema,
    validate_database_schema,
)
from inference_lab.storage._sqlite_logging import safe_log, safe_log_failure
from inference_lab.storage._sqlite_migration_execution import (
    AppliedAtFactory,
    execute_migration,
)
from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    MigrationDefinition,
    validate_migration_registry,
)
from inference_lab.storage.errors import (
    StorageError,
    StorageMigrationError,
    StorageTransactionError,
)


def _log_migration_failure(
    migration: MigrationDefinition,
    error: StorageError,
) -> None:
    """Log one typed migration failure with its bounded identity."""
    safe_log_failure(
        "migration_failed",
        error,
        migration_revision=migration.revision,
        migration_id=migration.migration_id,
    )


def _apply_one_migration(
    connection: sqlite3.Connection,
    migration: MigrationDefinition,
    *,
    migrations: tuple[MigrationDefinition, ...],
    applied_at_factory: AppliedAtFactory,
) -> SchemaInspection:
    safe_log(
        "info",
        "migration_started",
        migration_revision=migration.revision,
        migration_id=migration.migration_id,
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        locked_state = _inspect_database_schema(connection, migrations=migrations)
        if locked_state.current_revision >= migration.revision:
            connection.rollback()
            safe_log(
                "info",
                "migration_skipped",
                migration_revision=migration.revision,
                migration_id=migration.migration_id,
            )
            return locked_state
        if locked_state.current_revision != migration.revision - 1:
            raise StorageMigrationError()
        execute_migration(connection, migration, applied_at_factory)
        connection.commit()
    except StorageError as error:
        rollback_preserving_primary(connection)
        _log_migration_failure(migration, error)
        raise
    except sqlite3.Error as error:
        rollback_preserving_primary(connection)
        translated = translate_sqlite_error(
            error,
            default=StorageMigrationError,
            constraint=StorageMigrationError,
        )
        _log_migration_failure(migration, translated)
        raise translated from None
    except Exception:
        rollback_preserving_primary(connection)
        translated = StorageMigrationError()
        _log_migration_failure(migration, translated)
        raise translated from None

    safe_log(
        "info",
        "migration_completed",
        migration_revision=migration.revision,
        migration_id=migration.migration_id,
    )
    return SchemaInspection(
        current_revision=migration.revision,
        latest_supported_revision=migrations[-1].revision,
        applied_migration_count=migration.revision,
    )


def apply_pending_migrations(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
    applied_at_factory: AppliedAtFactory = utc_now,
) -> SchemaInspection:
    """Apply supported pending revisions as durable, ordered checkpoints."""
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

    inspection = inspect_database_schema(connection, migrations=migrations)
    if inspection.current_revision == inspection.latest_supported_revision:
        return inspection

    enable_durable_wal(connection)
    while inspection.current_revision < inspection.latest_supported_revision:
        migration = migrations[inspection.current_revision]
        inspection = _apply_one_migration(
            connection,
            migration,
            migrations=migrations,
            applied_at_factory=applied_at_factory,
        )

    return validate_database_schema(connection, migrations=migrations)
