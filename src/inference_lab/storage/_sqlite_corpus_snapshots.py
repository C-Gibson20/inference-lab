"""Dedicated SQLite snapshot lifecycle for version-pinned corpus readers."""

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage._sqlite_cleanup import (
    close_preserving_primary,
    rollback_preserving_primary,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_publication import _published_manifest
from inference_lab.storage._sqlite_corpus_updates import get_corpus_update
from inference_lab.storage._sqlite_corpus_versions import (
    get_active_corpus_version,
    get_corpus_version,
)
from inference_lab.storage._sqlite_inspection import open_current_database
from inference_lab.storage._sqlite_logging import safe_log, safe_log_failure
from inference_lab.storage._sqlite_migrations import MigrationDefinition
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
    StorageTransactionError,
)
from inference_lab.storage.versioning import (
    CorpusUpdateState,
    CorpusVersion,
    CorpusVersionManifest,
)


def _database_details(connection: sqlite3.Connection) -> tuple[Path, int]:
    try:
        database_rows = connection.execute("PRAGMA database_list").fetchall()
        main_rows = tuple(row for row in database_rows if row["name"] == "main")
        timeout_row = connection.execute("PRAGMA busy_timeout").fetchone()
        if len(main_rows) != 1 or timeout_row is None:
            raise RepositoryUnavailableError()
        database_value = main_rows[0]["file"]
        timeout_value = timeout_row[0]
        if not isinstance(database_value, str) or not database_value:
            raise RepositoryUnavailableError()
        if (
            not isinstance(timeout_value, int)
            or isinstance(timeout_value, bool)
            or timeout_value <= 0
        ):
            raise RepositoryUnavailableError()
        return Path(database_value), timeout_value
    except RepositoryUnavailableError:
        raise
    except (KeyError, IndexError, TypeError):
        raise RepositoryUnavailableError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


class SQLiteCorpusSnapshot:
    """Own one dedicated read transaction pinned to a committed version."""

    def __init__(
        self,
        database_path: Path,
        busy_timeout_ms: int,
        migrations: tuple[MigrationDefinition, ...],
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId | None,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a Path.")
        if not isinstance(busy_timeout_ms, int) or isinstance(busy_timeout_ms, bool):
            raise TypeError("busy_timeout_ms must be an integer.")
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be greater than zero.")
        if not isinstance(migrations, tuple):
            raise TypeError("migrations must be a tuple.")
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if corpus_version is not None and not isinstance(
            corpus_version, CorpusVersionId
        ):
            raise TypeError("corpus_version must be a CorpusVersionId or None.")
        self._database_path = database_path
        self._busy_timeout_ms = busy_timeout_ms
        self._migrations = migrations
        self._requested_corpus_id = corpus_id
        self._requested_corpus_version = corpus_version
        self._connection: sqlite3.Connection | None = None
        self._version: CorpusVersion | None = None
        self._manifest: CorpusVersionManifest | None = None
        self._closed = False

    @property
    def corpus_id(self) -> CorpusId:
        """Return the selected corpus identity while the snapshot is open."""
        return self._open_version().corpus_id

    @property
    def version(self) -> CorpusVersion:
        """Return the committed version selected on entry."""
        return self._open_version()

    @property
    def manifest(self) -> CorpusVersionManifest:
        """Return the complete committed manifest selected on entry."""
        self._open_version()
        manifest = self._manifest
        if manifest is None:
            raise StorageTransactionError()
        return manifest

    @property
    def is_closed(self) -> bool:
        """Return whether this snapshot has permanently released its reader."""
        return self._closed

    def __enter__(self) -> Self:
        """Open and pin the dedicated SQLite read transaction."""
        if self._closed:
            raise RepositoryClosedError()
        if self._connection is not None:
            raise StorageTransactionError()

        connection: sqlite3.Connection | None = None
        try:
            connection = open_current_database(
                self._database_path,
                busy_timeout_ms=self._busy_timeout_ms,
                migrations=self._migrations,
            )
            connection.execute("PRAGMA query_only = ON")
            query_only = connection.execute("PRAGMA query_only").fetchone()
            if query_only is None or query_only[0] != 1:
                raise RepositoryUnavailableError()
            connection.execute("BEGIN")
            version = self._select_version(connection)
            update = get_corpus_update(
                connection,
                version.corpus_id,
                version.corpus_version,
            )
            if update is None or update.state is not CorpusUpdateState.COMMITTED:
                raise CorruptStorageError()
            manifest = _published_manifest(connection, update)
        except StorageError as error:
            close_preserving_primary(connection)
            self._closed = True
            safe_log_failure(
                "corpus_snapshot_open_failed",
                error,
                corpus_id=self._requested_corpus_id.as_str(),
                corpus_version=(
                    None
                    if self._requested_corpus_version is None
                    else self._requested_corpus_version.as_str()
                ),
            )
            raise
        except sqlite3.Error as sqlite_error:
            close_preserving_primary(connection)
            self._closed = True
            error = translate_sqlite_error(
                sqlite_error,
                default=StorageTransactionError,
            )
            safe_log_failure(
                "corpus_snapshot_open_failed",
                error,
                corpus_id=self._requested_corpus_id.as_str(),
                corpus_version=(
                    None
                    if self._requested_corpus_version is None
                    else self._requested_corpus_version.as_str()
                ),
            )
            raise error from None

        self._connection = connection
        self._version = version
        self._manifest = manifest
        safe_log(
            "info",
            "corpus_snapshot_opened",
            corpus_id=version.corpus_id.as_str(),
            corpus_version=version.corpus_version.as_str(),
            sequence=version.sequence,
            document_count=len(manifest.document_memberships),
            component_count=len(manifest.component_generations),
        )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Close the reader without replacing a primary exception."""
        del exception_type, traceback
        try:
            self.close()
        except StorageError:
            if exception is None:
                raise
        return False

    def close(self) -> None:
        """Release the reader idempotently and reject subsequent use."""
        if self._closed:
            return
        connection = self._connection
        version = self._version
        self._connection = None
        self._closed = True
        if connection is None:
            return

        rolled_back = rollback_preserving_primary(connection)
        closed = close_preserving_primary(connection)
        if not rolled_back or not closed:
            error: StorageError = (
                StorageTransactionError()
                if not rolled_back
                else RepositoryUnavailableError()
            )
            safe_log_failure(
                "corpus_snapshot_close_failed",
                error,
                corpus_id=(
                    self._requested_corpus_id.as_str()
                    if version is None
                    else version.corpus_id.as_str()
                ),
                corpus_version=(
                    None if version is None else version.corpus_version.as_str()
                ),
            )
            raise error

        if version is not None:
            safe_log(
                "info",
                "corpus_snapshot_closed",
                corpus_id=version.corpus_id.as_str(),
                corpus_version=version.corpus_version.as_str(),
                sequence=version.sequence,
            )

    def _select_version(self, connection: sqlite3.Connection) -> CorpusVersion:
        requested = self._requested_corpus_version
        version = (
            get_active_corpus_version(connection, self._requested_corpus_id)
            if requested is None
            else get_corpus_version(connection, self._requested_corpus_id, requested)
        )
        if version is None:
            raise StorageConflictError()
        return version

    def _open_version(self) -> CorpusVersion:
        if self._closed:
            raise RepositoryClosedError()
        version = self._version
        if self._connection is None or version is None:
            raise StorageTransactionError()
        return version


def create_active_corpus_snapshot(
    connection: sqlite3.Connection,
    migrations: tuple[MigrationDefinition, ...],
    corpus_id: CorpusId,
) -> SQLiteCorpusSnapshot:
    """Create an unopened owner that will select the active version on entry."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    database_path, busy_timeout_ms = _database_details(connection)
    return SQLiteCorpusSnapshot(
        database_path,
        busy_timeout_ms,
        migrations,
        corpus_id,
        None,
    )


def create_corpus_snapshot(
    connection: sqlite3.Connection,
    migrations: tuple[MigrationDefinition, ...],
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> SQLiteCorpusSnapshot:
    """Create an unopened owner for one explicit historical version."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    database_path, busy_timeout_ms = _database_details(connection)
    return SQLiteCorpusSnapshot(
        database_path,
        busy_timeout_ms,
        migrations,
        corpus_id,
        corpus_version,
    )
