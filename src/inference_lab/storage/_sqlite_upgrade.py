"""Supported forward migration execution for SQLite metadata repositories."""

import sqlite3

import structlog

from inference_lab._clock import utc_now
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
from inference_lab.storage._sqlite_migration_execution import (
    AppliedAtFactory,
    execute_migration,
    rollback_preserving_primary,
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

_LOGGER = structlog.get_logger(__name__)


def _safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit operational migration metadata without affecting migration state."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="storage", **fields)
    except Exception:
        pass


def _log_failure(migration: MigrationDefinition, error: StorageError) -> None:
    _safe_log(
        "error",
        "migration_failed",
        migration_revision=migration.revision,
        migration_id=migration.migration_id,
        error_category=error.category.value,
        retryable=error.retryable,
    )


def _apply_one_migration(
    connection: sqlite3.Connection,
    migration: MigrationDefinition,
    *,
    migrations: tuple[MigrationDefinition, ...],
    applied_at_factory: AppliedAtFactory,
) -> SchemaInspection:
    _safe_log(
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
            _safe_log(
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
        _log_failure(migration, error)
        raise
    except sqlite3.Error as error:
        rollback_preserving_primary(connection)
        translated = translate_sqlite_error(
            error,
            default=StorageMigrationError,
            constraint=StorageMigrationError,
        )
        _log_failure(migration, translated)
        raise translated from None
    except Exception:
        rollback_preserving_primary(connection)
        translated = StorageMigrationError()
        _log_failure(migration, translated)
        raise translated from None

    _safe_log(
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
