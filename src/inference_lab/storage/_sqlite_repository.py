"""Internal SQLite metadata repository lifecycle owner."""

import sqlite3
from pathlib import Path
from typing import Self

import structlog

from inference_lab.storage._sqlite_connection import (
    DEFAULT_BUSY_TIMEOUT_MS,
    translate_sqlite_error,
)
from inference_lab.storage._sqlite_inspection import (
    SchemaInspection,
    validate_database_schema,
)
from inference_lab.storage._sqlite_lifecycle import open_sqlite_database
from inference_lab.storage._sqlite_migration_execution import (
    AppliedAtFactory,
    utc_now,
)
from inference_lab.storage._sqlite_migrations import MIGRATIONS, MigrationDefinition
from inference_lab.storage.errors import (
    RepositoryClosedError,
    RepositoryUnavailableError,
)

_LOGGER = structlog.get_logger(__name__)


def _safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded repository lifecycle metadata without changing outcomes."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="storage", **fields)
    except Exception:
        pass


class SQLiteMetadataRepository:
    """Own one validated SQLite connection and its explicit lifecycle."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        migrations: tuple[MigrationDefinition, ...],
    ) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be a sqlite3.Connection.")
        if not isinstance(migrations, tuple):
            raise TypeError("migrations must be a tuple.")
        self._connection: sqlite3.Connection | None = connection
        self._migrations = migrations

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        apply_migrations: bool = False,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
        applied_at_factory: AppliedAtFactory = utc_now,
    ) -> Self:
        """Open a validated repository under the explicit schema policy."""
        connection = open_sqlite_database(
            path,
            apply_migrations=apply_migrations,
            busy_timeout_ms=busy_timeout_ms,
            migrations=migrations,
            applied_at_factory=applied_at_factory,
        )
        return cls(connection, migrations)

    @property
    def is_closed(self) -> bool:
        """Return whether this repository has been permanently closed."""
        return self._connection is None

    @property
    def schema_inspection(self) -> SchemaInspection:
        """Return the validated current schema state."""
        return validate_database_schema(
            self._connection_or_raise(),
            migrations=self._migrations,
        )

    def _connection_or_raise(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise RepositoryClosedError()
        return connection

    def close(self) -> None:
        """Close the owned connection idempotently and reject future use."""
        connection = self._connection
        if connection is None:
            return
        self._connection = None

        try:
            connection.close()
        except sqlite3.Error as error:
            translated = translate_sqlite_error(
                error,
                default=RepositoryUnavailableError,
            )
            _safe_log(
                "error",
                "repository_close_failed",
                error_category=translated.category.value,
                retryable=translated.retryable,
            )
            raise translated from None

        _safe_log("info", "repository_close_completed")
