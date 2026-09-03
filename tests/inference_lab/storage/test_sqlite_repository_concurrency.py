"""Tests for SQLite repository concurrency."""

import sqlite3
from pathlib import Path
from time import perf_counter

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
)
from inference_lab.storage import StorageConcurrencyError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252

_BASELINE_CORPUS_ID = CorpusId("08b00c09-b9c0-44b7-a3a4-1acb526162ab")
_BASELINE_CORPUS = CorpusMetadata(_BASELINE_CORPUS_ID, "Baseline corpus")
_UNCOMMITTED_DOCUMENT_ID = DocumentId("ce118577-f665-419f-9b4a-c9f875a59718")
_AFTER_LOCK_CORPUS_ID = CorpusId("c2a62ed4-de89-4b4c-a5e2-a8e6340e2d81")
_AFTER_LOCK_CORPUS = CorpusMetadata(_AFTER_LOCK_CORPUS_ID, "After-lock corpus")


def _initialise_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def _journal_mode(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        return row[0]
    finally:
        connection.close()


def test_repository_enables_durable_wal_across_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _initialise_repository(database_path)
    try:
        assert _journal_mode(database_path) == "wal"
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert _journal_mode(database_path) == "wal"
    finally:
        reopened.close()


def test_reader_continues_with_committed_state_during_write(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    writer = _initialise_repository(database_path)
    with writer.transaction() as transaction:
        transaction.add_corpus(_BASELINE_CORPUS)
    reader = SQLiteMetadataRepository.open(database_path)
    try:
        with writer.transaction() as transaction:
            transaction.add_document(DocumentIdentity(_UNCOMMITTED_DOCUMENT_ID))

            started_at = perf_counter()
            assert reader.get_corpus(_BASELINE_CORPUS_ID) == _BASELINE_CORPUS
            assert reader.get_document(_UNCOMMITTED_DOCUMENT_ID) is None
            assert perf_counter() - started_at < 0.5

        assert reader.get_document(_UNCOMMITTED_DOCUMENT_ID) == DocumentIdentity(
            _UNCOMMITTED_DOCUMENT_ID
        )
    finally:
        reader.close()
        writer.close()


def test_second_writer_times_out_safely_and_recovers_after_commit(
    tmp_path: Path,
) -> None:
    private_marker = "private-lock-path"
    database_path = tmp_path / private_marker / "metadata.sqlite3"
    writer = _initialise_repository(database_path)
    contender = SQLiteMetadataRepository.open(database_path, busy_timeout_ms=25)
    try:
        with writer.transaction() as transaction:
            transaction.add_corpus(_BASELINE_CORPUS)

            with capture_logs() as logs:
                started_at = perf_counter()
                with pytest.raises(StorageConcurrencyError) as raised:
                    with contender.transaction():
                        pass
                elapsed = perf_counter() - started_at

            assert elapsed < 1.0
            assert raised.value.retryable is True
            assert private_marker not in str(raised.value)
            assert "locked" not in str(raised.value).lower()
            assert logs == [
                {
                    "event": "transaction_failed",
                    "pipeline_stage": "storage",
                    "error_category": "concurrency",
                    "retryable": True,
                    "log_level": "error",
                }
            ]
            assert private_marker not in repr(logs)

        with contender.transaction() as transaction:
            transaction.add_corpus(_AFTER_LOCK_CORPUS)

        assert contender.get_corpus(_BASELINE_CORPUS_ID) == _BASELINE_CORPUS
        assert contender.get_corpus(_AFTER_LOCK_CORPUS_ID) == _AFTER_LOCK_CORPUS
    finally:
        contender.close()
        writer.close()
