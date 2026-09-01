from datetime import UTC, datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import CorpusId, DocumentId, DocumentIdentity
from inference_lab.storage import (
    RepositoryClosedError,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
_SHARED_UUID = "b4c4bb8a-8d23-4c20-a26f-2ca25f614b35"
_OTHER_UUID = "af93dceb-40fc-414d-b3e9-50eb66f433ed"


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _open_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def test_core_identities_round_trip_after_restart_without_row_identity(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    corpus_id = CorpusId(_SHARED_UUID)
    document = DocumentIdentity(DocumentId(_SHARED_UUID))
    first = _open_repository(database_path)
    with first.transaction() as transaction:
        transaction.add_corpus(corpus_id)
        transaction.add_document(document)
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        stored_corpus = second.get_corpus(corpus_id)
        stored_document = second.get_document(document.document_id)

        assert stored_corpus == corpus_id
        assert stored_document == document
        assert isinstance(stored_corpus, CorpusId)
        assert stored_document is not None
        assert isinstance(stored_document.document_id, DocumentId)
        assert stored_corpus != stored_document.document_id
        assert not hasattr(stored_corpus, "row_id")
        assert not hasattr(stored_document, "row_id")
    finally:
        second.close()


def test_missing_core_identities_return_none(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        assert repository.get_corpus(CorpusId(_SHARED_UUID)) is None
        assert repository.get_document(DocumentId(_SHARED_UUID)) is None
    finally:
        repository.close()


def test_exact_core_identity_replay_is_idempotent(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    corpus_id = CorpusId(_SHARED_UUID)
    document = DocumentIdentity(DocumentId(_OTHER_UUID))
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(corpus_id)
            transaction.add_corpus(corpus_id)
            transaction.add_document(document)
            transaction.add_document(document)

        assert repository.get_corpus(corpus_id) == corpus_id
        assert repository.get_document(document.document_id) == document
    finally:
        repository.close()


def test_primary_failure_rolls_back_all_core_writes(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    corpus_id = CorpusId(_SHARED_UUID)
    document = DocumentIdentity(DocumentId(_OTHER_UUID))
    primary = RuntimeError("private-payload")
    try:
        with pytest.raises(RuntimeError) as raised:
            with repository.transaction() as transaction:
                transaction.add_corpus(corpus_id)
                transaction.add_document(document)
                raise primary

        assert raised.value is primary
        assert repository.get_corpus(corpus_id) is None
        assert repository.get_document(document.document_id) is None
    finally:
        repository.close()


def test_reader_sees_only_committed_core_state(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    writer = _open_repository(database_path)
    reader = SQLiteMetadataRepository.open(database_path)
    corpus_id = CorpusId(_SHARED_UUID)
    try:
        with writer.transaction() as transaction:
            transaction.add_corpus(corpus_id)
            assert reader.get_corpus(corpus_id) is None

        assert reader.get_corpus(corpus_id) == corpus_id
    finally:
        reader.close()
        writer.close()


def test_transaction_events_are_bounded_and_payload_free(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "private-path" / "metadata.sqlite3")
    private_payload = "private-payload"
    try:
        with capture_logs() as logs:
            with repository.transaction() as transaction:
                transaction.add_corpus(CorpusId(_SHARED_UUID))
            with pytest.raises(RuntimeError, match=private_payload):
                with repository.transaction():
                    raise RuntimeError(private_payload)

        assert logs == [
            {
                "event": "transaction_completed",
                "pipeline_stage": "storage",
                "log_level": "info",
            },
            {
                "event": "transaction_rolled_back",
                "pipeline_stage": "storage",
                "log_level": "info",
            },
        ]
        assert private_payload not in repr(logs)
        assert _SHARED_UUID not in repr(logs)
        assert "private-path" not in repr(logs)
    finally:
        repository.close()


def test_transaction_rejects_nesting_reuse_and_close_while_active(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    transaction = repository.transaction()
    try:
        with transaction:
            with pytest.raises(StorageTransactionError):
                with repository.transaction():
                    pass
            with pytest.raises(StorageTransactionError):
                repository.close()

        with pytest.raises(StorageTransactionError):
            with transaction:
                pass
    finally:
        repository.close()


def test_inactive_transaction_operations_and_closed_repository_are_rejected(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    transaction = repository.transaction()

    with pytest.raises(StorageTransactionError):
        transaction.get_corpus(CorpusId(_SHARED_UUID))

    repository.close()

    with pytest.raises(RepositoryClosedError):
        repository.get_corpus(CorpusId(_SHARED_UUID))
    with pytest.raises(RepositoryClosedError):
        repository.get_document(DocumentId(_SHARED_UUID))
    with pytest.raises(RepositoryClosedError):
        repository.transaction()
