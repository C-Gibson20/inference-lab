"""Tests for SQLite connection policy and error translation."""

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from inference_lab.storage import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_connection import (
    DEFAULT_BUSY_TIMEOUT_MS,
    enable_durable_wal,
    open_sqlite_connection,
    translate_sqlite_error,
)


def test_open_creates_parent_and_applies_connection_policy(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata" / "metadata.sqlite3"

    connection = open_sqlite_connection(database_path)
    try:
        assert database_path.is_file()
        assert connection.row_factory is sqlite3.Row
        assert connection.isolation_level is None
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA trusted_schema").fetchone()[0] == 0
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == (
            DEFAULT_BUSY_TIMEOUT_MS
        )
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()


def test_wal_is_enabled_only_when_explicitly_requested(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    connection = open_sqlite_connection(database_path)
    try:
        enable_durable_wal(connection)

        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        connection.close()

    reopened = open_sqlite_connection(database_path)
    try:
        assert reopened.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        reopened.close()


def test_open_rejects_nonlocal_paths(
    rejected_local_path_shape: str | Path,
) -> None:
    with pytest.raises(RepositoryUnavailableError):
        open_sqlite_connection(Path(rejected_local_path_shape))


def test_open_rejects_nonpersistent_memory_database() -> None:
    with pytest.raises(RepositoryUnavailableError):
        open_sqlite_connection(Path(":memory:"))


def test_open_rejects_a_directory_target_without_exposing_it(tmp_path: Path) -> None:
    private_marker = "private-storage-marker"
    database_path = tmp_path / private_marker
    database_path.mkdir()

    with pytest.raises(RepositoryUnavailableError) as captured:
        open_sqlite_connection(database_path)

    assert private_marker not in str(captured.value)


def test_open_rejects_a_non_database_file_as_corrupt(tmp_path: Path) -> None:
    private_marker = "private-database-payload"
    database_path = tmp_path / "metadata.sqlite3"
    database_path.write_text(private_marker, encoding="utf-8")

    with pytest.raises(CorruptStorageError) as captured:
        open_sqlite_connection(database_path)

    assert private_marker not in str(captured.value)
    assert str(database_path) not in str(captured.value)


@pytest.mark.parametrize("busy_timeout_ms", [True, 1.5, "10"])
def test_busy_timeout_rejects_non_integer_values(
    tmp_path: Path,
    busy_timeout_ms: object,
) -> None:
    with pytest.raises(TypeError):
        open_sqlite_connection(
            tmp_path / "metadata.sqlite3",
            busy_timeout_ms=busy_timeout_ms,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("busy_timeout_ms", [0, -1])
def test_busy_timeout_must_be_positive(
    tmp_path: Path,
    busy_timeout_ms: int,
) -> None:
    with pytest.raises(ValueError):
        open_sqlite_connection(
            tmp_path / "metadata.sqlite3",
            busy_timeout_ms=busy_timeout_ms,
        )


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


def test_error_translation_rejects_invalid_contract_types() -> None:
    error = sqlite3.OperationalError("private-details")

    with pytest.raises(TypeError, match="error"):
        translate_sqlite_error(
            cast("sqlite3.Error", RuntimeError("not SQLite")),
            default=RepositoryUnavailableError,
        )
    with pytest.raises(TypeError, match="default"):
        translate_sqlite_error(
            error,
            default=RuntimeError,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="constraint"):
        translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
            constraint=RuntimeError,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("error_code", [True, "5"])
def test_invalid_sqlite_error_codes_fall_back_to_the_default(
    error_code: object,
) -> None:
    error = sqlite3.OperationalError("private-details")
    setattr(error, "sqlite_errorcode", error_code)

    translated = translate_sqlite_error(
        error,
        default=RepositoryUnavailableError,
    )

    assert isinstance(translated, RepositoryUnavailableError)


def test_connection_functions_reject_wrong_runtime_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="path"):
        open_sqlite_connection(str(tmp_path / "metadata.sqlite3"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="connection"):
        enable_durable_wal(object())  # type: ignore[arg-type]


def test_enabling_wal_on_a_closed_connection_is_unavailable(tmp_path: Path) -> None:
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    connection.close()

    with pytest.raises(RepositoryUnavailableError):
        enable_durable_wal(connection)
