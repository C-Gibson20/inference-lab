"""Internal SQLite connection policy and exception translation."""

import sqlite3
from pathlib import Path
from typing import Final

from inference_lab._validation import is_local_path_shape
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
    StorageError,
)

DEFAULT_BUSY_TIMEOUT_MS: Final = 5_000
_SQLITE_PRIMARY_CODE_MASK: Final = 0xFF


def _primary_sqlite_error_code(error: sqlite3.Error) -> int | None:
    code = getattr(error, "sqlite_errorcode", None)
    if not isinstance(code, int) or isinstance(code, bool):
        return None
    return code & _SQLITE_PRIMARY_CODE_MASK


def translate_sqlite_error(
    error: sqlite3.Error,
    *,
    default: type[StorageError],
    constraint: type[StorageError] = StorageConflictError,
) -> StorageError:
    """Translate a database failure without retaining its message or payload."""
    if not isinstance(error, sqlite3.Error):
        raise TypeError("error must be a sqlite3.Error.")
    if not isinstance(default, type) or not issubclass(default, StorageError):
        raise TypeError("default must be a StorageError type.")
    if not isinstance(constraint, type) or not issubclass(constraint, StorageError):
        raise TypeError("constraint must be a StorageError type.")

    primary_code = _primary_sqlite_error_code(error)
    if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return StorageConcurrencyError()
    if primary_code == sqlite3.SQLITE_CONSTRAINT:
        return constraint()
    if primary_code in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
        return CorruptStorageError()
    return default()


def _validated_database_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path.")

    raw_path = str(path)
    if not is_local_path_shape(path) or raw_path == ":memory:":
        raise RepositoryUnavailableError()

    try:
        resolved_path = path.expanduser().resolve(strict=False)
        if resolved_path.exists() and not resolved_path.is_file():
            raise RepositoryUnavailableError()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    except RepositoryUnavailableError:
        raise
    except (OSError, RuntimeError):
        raise RepositoryUnavailableError() from None

    return resolved_path


def _validate_busy_timeout(busy_timeout_ms: int) -> None:
    if not isinstance(busy_timeout_ms, int) or isinstance(busy_timeout_ms, bool):
        raise TypeError("busy_timeout_ms must be an integer.")
    if busy_timeout_ms <= 0:
        raise ValueError("busy_timeout_ms must be greater than zero.")


def _close_after_failed_open(connection: sqlite3.Connection | None) -> None:
    """Release a partial connection without replacing the primary failure."""
    if connection is None:
        return
    try:
        connection.close()
    except sqlite3.Error:
        pass


def open_sqlite_connection(
    path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a local connection with safe per-connection metadata settings.

    WAL is deliberately enabled only after schema identity is validated, so
    inspecting an unrelated SQLite file does not change its journal mode.
    """
    _validate_busy_timeout(busy_timeout_ms)
    database_path = _validated_database_path(path)
    connection: sqlite3.Connection | None = None

    try:
        connection = sqlite3.connect(
            database_path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA trusted_schema = OFF")

        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise RepositoryUnavailableError()
        if connection.execute("PRAGMA synchronous").fetchone()[0] != 2:
            raise RepositoryUnavailableError()
        if connection.execute("PRAGMA trusted_schema").fetchone()[0] != 0:
            raise RepositoryUnavailableError()
    except StorageError:
        _close_after_failed_open(connection)
        raise
    except sqlite3.Error as error:
        _close_after_failed_open(connection)
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None

    return connection


def enable_durable_wal(connection: sqlite3.Connection) -> None:
    """Enable and verify durable WAL mode after repository identity validation."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection.")

    try:
        journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if journal_mode != "wal":
            raise RepositoryUnavailableError()
    except StorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
