"""Explicit open and schema-update policy for SQLite metadata storage."""

import sqlite3
from pathlib import Path

import structlog

from inference_lab._clock import utc_now
from inference_lab.storage._sqlite_connection import (
    DEFAULT_BUSY_TIMEOUT_MS,
    enable_durable_wal,
    open_sqlite_connection,
)
from inference_lab.storage._sqlite_initialisation import initialise_empty_database
from inference_lab.storage._sqlite_inspection import (
    inspect_database_schema,
    validate_database_schema,
)
from inference_lab.storage._sqlite_migration_execution import AppliedAtFactory
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage._sqlite_upgrade import apply_pending_migrations
from inference_lab.storage.errors import (
    MigrationRequiredError,
    StorageError,
)

_LOGGER = structlog.get_logger(__name__)


def _safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded lifecycle metadata without changing storage outcomes."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="storage", **fields)
    except Exception:
        pass


def _log_failure(event: str, error: StorageError) -> None:
    _safe_log(
        "error",
        event,
        error_category=error.category.value,
        retryable=error.retryable,
    )


def _close_preserving_primary(connection: sqlite3.Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except sqlite3.Error:
        pass


def _initialise_with_logs(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...],
    applied_at_factory: AppliedAtFactory,
) -> None:
    _safe_log("info", "repository_initialisation_started")
    try:
        initialise_empty_database(
            connection,
            migrations=migrations,
            applied_at_factory=applied_at_factory,
        )
    except StorageError as error:
        _log_failure("repository_initialisation_failed", error)
        raise
    _safe_log(
        "info",
        "repository_initialisation_completed",
        schema_revision=migrations[-1].revision,
        applied_migration_count=len(migrations),
    )


def open_sqlite_database(
    path: Path,
    *,
    apply_migrations: bool = False,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
    applied_at_factory: AppliedAtFactory = utc_now,
) -> sqlite3.Connection:
    """Open current metadata, mutating schema only when explicitly allowed."""
    if not isinstance(apply_migrations, bool):
        raise TypeError("apply_migrations must be a boolean.")
    if not callable(applied_at_factory):
        raise TypeError("applied_at_factory must be callable.")

    connection: sqlite3.Connection | None = None
    _safe_log(
        "info",
        "repository_open_started",
        schema_updates_allowed=apply_migrations,
    )
    try:
        connection = open_sqlite_connection(
            path,
            busy_timeout_ms=busy_timeout_ms,
        )
        try:
            inspection = inspect_database_schema(
                connection,
                migrations=migrations,
            )
        except MigrationRequiredError:
            if not apply_migrations:
                raise
            _initialise_with_logs(
                connection,
                migrations=migrations,
                applied_at_factory=applied_at_factory,
            )
        else:
            if inspection.current_revision < inspection.latest_supported_revision:
                if not apply_migrations:
                    raise MigrationRequiredError()
                apply_pending_migrations(
                    connection,
                    migrations=migrations,
                    applied_at_factory=applied_at_factory,
                )

        current = validate_database_schema(connection, migrations=migrations)
        enable_durable_wal(connection)
    except StorageError as error:
        _close_preserving_primary(connection)
        _log_failure("repository_open_failed", error)
        raise

    _safe_log(
        "info",
        "repository_open_completed",
        schema_updates_allowed=apply_migrations,
        schema_revision=current.current_revision,
        applied_migration_count=current.applied_migration_count,
    )
    return connection
