import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    ExternalIdentifier,
    HashAlgorithm,
)
from inference_lab.storage import (
    MetadataRepository,
    MetadataTransaction,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage import _sqlite_repository as repository_module
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 11, 0, tzinfo=UTC)
_CORPUS_ID = CorpusId("62738887-ac02-4c50-98d3-637658880de5")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Test corpus")
_FAMILY_ID = DocumentFamilyId("66f40173-12bb-4054-a427-0bb08f79aa34")
_FIRST_DOCUMENT_ID = DocumentId("429569cc-9094-4b22-97a4-a43838b4cd44")
_SECOND_DOCUMENT_ID = DocumentId("59c7ea2d-72d9-4af7-b76f-822dd84c6e35")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_FIRST_VERSION = DocumentVersion(
    _FIRST_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
    (ExternalIdentifier("doi", "10.1000/private-first"),),
)
_SECOND_VERSION = DocumentVersion(
    _SECOND_VERSION_ID,
    _SECOND_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_DUPLICATE = DuplicateRelationship(
    _FIRST_VERSION_ID,
    _SECOND_VERSION_ID,
    DuplicateClass.BYTE_IDENTICAL,
)
_MEMBERSHIP = DocumentFamilyMembership(
    _FAMILY_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _open_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def _add_complete_graph(transaction: MetadataTransaction) -> None:
    transaction.add_corpus(_CORPUS)
    transaction.add_document_family(_FAMILY_ID)
    transaction.add_document(DocumentIdentity(_FIRST_DOCUMENT_ID))
    transaction.add_document(DocumentIdentity(_SECOND_DOCUMENT_ID))
    transaction.add_document_version(_FIRST_VERSION)
    transaction.add_document_version(_SECOND_VERSION)
    for version in (_FIRST_VERSION, _SECOND_VERSION):
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                version.document_id,
                version.document_version_id,
            )
        )
    transaction.add_duplicate_relationship(_DUPLICATE)
    transaction.add_family_membership(_MEMBERSHIP)


def test_concrete_repository_and_transaction_satisfy_public_protocols(
    tmp_path: Path,
) -> None:
    concrete = _open_repository(tmp_path / "metadata.sqlite3")
    repository: MetadataRepository = concrete
    transaction = repository.transaction()
    transaction_contract: MetadataTransaction = transaction
    try:
        assert isinstance(repository, MetadataRepository)
        assert isinstance(transaction_contract, MetadataTransaction)
        assert not hasattr(repository, "add_document")
        assert not hasattr(repository, "add_document_version")

        with transaction_contract:
            transaction_contract.add_corpus(_CORPUS)
    finally:
        repository.close()


def test_complete_metadata_graph_commits_atomically(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with repository.transaction() as transaction:
            _add_complete_graph(transaction)

        assert repository.get_corpus(_CORPUS_ID) == _CORPUS
        assert repository.get_document_family(_FAMILY_ID) == _FAMILY_ID
        assert repository.get_document(_FIRST_DOCUMENT_ID) == DocumentIdentity(
            _FIRST_DOCUMENT_ID
        )
        assert repository.get_document_version(_FIRST_VERSION_ID) == _FIRST_VERSION
        assert repository.list_duplicate_relationships(
            _CORPUS_ID, _FIRST_VERSION_ID
        ) == (_DUPLICATE,)
        assert repository.list_family_memberships_by_family(_CORPUS_ID, _FAMILY_ID) == (
            _MEMBERSHIP,
        )
    finally:
        repository.close()


def test_failure_at_final_relationship_step_rolls_back_complete_graph(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    invalid_membership = DocumentFamilyMembership(
        _FAMILY_ID,
        _SECOND_DOCUMENT_ID,
        _FIRST_VERSION_ID,
    )
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                _add_complete_graph(transaction)
                transaction.add_family_membership(invalid_membership)

        assert repository.get_corpus(_CORPUS_ID) is None
        assert repository.get_document_family(_FAMILY_ID) is None
        assert repository.get_document(_FIRST_DOCUMENT_ID) is None
        assert repository.get_document(_SECOND_DOCUMENT_ID) is None
        assert repository.get_document_version(_FIRST_VERSION_ID) is None
        assert repository.get_document_version(_SECOND_VERSION_ID) is None
        assert (
            repository.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == ()
        )
        assert (
            repository.list_family_memberships_by_family(_CORPUS_ID, _FAMILY_ID) == ()
        )
    finally:
        repository.close()


def test_caught_constraint_failure_marks_transaction_rollback_only(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "private-path" / "metadata.sqlite3")
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageTransactionError):
                with repository.transaction() as transaction:
                    transaction.add_corpus(_CORPUS)
                    try:
                        transaction.add_document_version(_FIRST_VERSION)
                    except StorageConflictError:
                        pass
                    with pytest.raises(StorageTransactionError):
                        transaction.get_corpus(_CORPUS_ID)

        assert repository.get_corpus(_CORPUS_ID) is None
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "transaction",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert "private-path" not in repr(logs)
        assert _FIRST_VERSION.content_hash.digest not in repr(logs)
        assert _FIRST_VERSION.external_identifiers[0].value not in repr(logs)
    finally:
        repository.close()


def test_caught_conversion_failure_marks_transaction_rollback_only(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageTransactionError):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                try:
                    transaction.add_document("private-value")  # type: ignore[arg-type]
                except TypeError:
                    pass

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()


def test_logging_failure_does_not_replace_primary_transaction_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingLogger:
        def info(self, event: str, **fields: object) -> None:
            raise RuntimeError("logging failed")

        def error(self, event: str, **fields: object) -> None:
            raise RuntimeError("logging failed")

    repository = _open_repository(tmp_path / "metadata.sqlite3")
    monkeypatch.setattr(repository_module, "_LOGGER", FailingLogger())
    primary = RuntimeError("private-primary")
    try:
        with pytest.raises(RuntimeError) as raised:
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                raise primary

        assert raised.value is primary
        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()


def test_rollback_failure_does_not_replace_primary_transaction_exception(
    tmp_path: Path,
) -> None:
    class RollbackFailingConnection(sqlite3.Connection):
        def rollback(self) -> None:
            raise sqlite3.OperationalError("private-rollback-payload")

    database_path = tmp_path / "metadata.sqlite3"
    initial = _open_repository(database_path)
    initial.close()
    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=RollbackFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    repository = SQLiteMetadataRepository(connection, MIGRATIONS)
    primary = RuntimeError("private-primary")
    try:
        with capture_logs() as logs:
            with pytest.raises(RuntimeError) as raised:
                with repository.transaction() as transaction:
                    transaction.add_corpus(_CORPUS)
                    raise primary

        assert raised.value is primary
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "transaction",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert "private-rollback-payload" not in repr(logs)
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.get_corpus(_CORPUS_ID) is None
    finally:
        reopened.close()
