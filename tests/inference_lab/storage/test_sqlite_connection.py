"""Tests for SQLite connection policy and error translation."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.storage import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_connection import (
    open_sqlite_connection,
    translate_sqlite_error,
)


def test_open_rejects_nonlocal_paths(
    rejected_local_path_shape: str | Path,
) -> None:
    with pytest.raises(RepositoryUnavailableError):
        open_sqlite_connection(Path(rejected_local_path_shape))


def test_sqlite_error_translation_uses_codes_not_messages() -> None:
    private_marker = "private-sqlite-message"
    cases = (
        (sqlite3.OperationalError(private_marker), RepositoryUnavailableError),
        (sqlite3.IntegrityError(private_marker), StorageConflictError),
        (sqlite3.DatabaseError(private_marker), CorruptStorageError),
    )
    codes = (
        sqlite3.SQLITE_CANTOPEN,
        sqlite3.SQLITE_CONSTRAINT_UNIQUE,
        sqlite3.SQLITE_NOTADB,
    )

    for (error, expected_type), code in zip(cases, codes, strict=True):
        error.sqlite_errorcode = code
        translated = translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        )
        assert isinstance(translated, expected_type)
        assert private_marker not in str(translated)


def test_busy_and_locked_codes_are_retryable_concurrency_errors() -> None:
    for code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED_SHAREDCACHE):
        error = sqlite3.OperationalError("private-lock-details")
        error.sqlite_errorcode = code

        translated = translate_sqlite_error(
            error,
            default=StorageTransactionError,
        )

        assert isinstance(translated, StorageConcurrencyError)
        assert translated.retryable is True
