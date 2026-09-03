"""Failure atomicity and corpus isolation for SQLite metadata writes."""

import sqlite3
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    HashAlgorithm,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import StorageConflictError, StorageTransactionError
from inference_lab.storage._sqlite_migrations import MIGRATIONS
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import open_repository as _open_repository  # noqa: TID252

_FIRST_CORPUS = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"), "First atomicity corpus"
)
_SECOND_CORPUS = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"), "Second atomicity corpus"
)
_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_FIRST_VERSION = DocumentVersion(
    DocumentVersionId("44444444-4444-4444-8444-444444444444"),
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_SECOND_VERSION = DocumentVersion(
    DocumentVersionId("55555555-5555-4555-8555-555555555555"),
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_SOURCE_ID = SourceObservationId("66666666-6666-4666-8666-666666666666")
_INITIAL_STATE = SourceObservationState(
    _FIRST_CORPUS.corpus_id,
    _SOURCE_ID,
    1,
    _DOCUMENT_ID,
    _FIRST_VERSION.document_version_id,
)
_PRIVATE_TRIGGER_PAYLOAD = "private-trigger-payload"


def _add_catalogue(repository: SQLiteMetadataRepository) -> None:
    """Add two corpora and a two-version document visible only in the first."""
    with repository.transaction() as transaction:
        transaction.add_corpus(_FIRST_CORPUS)
        transaction.add_corpus(_SECOND_CORPUS)
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        for version in (_FIRST_VERSION, _SECOND_VERSION):
            transaction.add_document_version(version)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _FIRST_CORPUS.corpus_id,
                    _DOCUMENT_ID,
                    version.document_version_id,
                )
            )


def _install_trigger(path: Path, statement: str) -> None:
    """Install one deterministic fault-injection trigger."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def test_source_creation_failure_rolls_back_current_and_history_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    _install_trigger(
        database_path,
        f"""CREATE TRIGGER reject_source_history
        BEFORE INSERT ON source_observation_history
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_TRIGGER_PAYLOAD}');
        END""",
    )
    try:
        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                transaction.add_source_observation(_INITIAL_STATE)

        assert _PRIVATE_TRIGGER_PAYLOAD not in str(raised.value)
        assert (
            repository.get_source_observation(_FIRST_CORPUS.corpus_id, _SOURCE_ID)
            is None
        )
        assert (
            repository.list_source_observation_history(
                _FIRST_CORPUS.corpus_id, _SOURCE_ID
            )
            == ()
        )
        assert repository.get_document_version(_FIRST_VERSION.document_version_id) == (
            _FIRST_VERSION
        )
    finally:
        repository.close()


def test_source_advance_failure_rolls_back_appended_history(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
    _install_trigger(
        database_path,
        f"""CREATE TRIGGER reject_source_pointer_update
        BEFORE UPDATE ON source_observations
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_TRIGGER_PAYLOAD}');
        END""",
    )
    try:
        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                transaction.advance_source_observation(
                    _FIRST_CORPUS.corpus_id,
                    _SOURCE_ID,
                    expected_version_id=_FIRST_VERSION.document_version_id,
                    new_version_id=_SECOND_VERSION.document_version_id,
                )

        assert _PRIVATE_TRIGGER_PAYLOAD not in str(raised.value)
        assert repository.get_source_observation(
            _FIRST_CORPUS.corpus_id, _SOURCE_ID
        ) == (_INITIAL_STATE)
        assert repository.list_source_observation_history(
            _FIRST_CORPUS.corpus_id, _SOURCE_ID
        ) == (_INITIAL_STATE,)
    finally:
        repository.close()


def test_assertion_batch_failure_rolls_back_earlier_batch_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
    _install_trigger(
        database_path,
        f"""CREATE TRIGGER reject_second_assertion
        BEFORE INSERT ON external_identifier_assertions
        WHEN NEW.namespace = 'pmid'
        BEGIN
            SELECT RAISE(ABORT, '{_PRIVATE_TRIGGER_PAYLOAD}');
        END""",
    )
    assertions = (
        ExternalIdentifierAssertion(
            _SOURCE_ID,
            _FIRST_VERSION.document_version_id,
            ExternalIdentifier("doi", "10.1000/atomicity"),
        ),
        ExternalIdentifierAssertion(
            _SOURCE_ID,
            _FIRST_VERSION.document_version_id,
            ExternalIdentifier("pmid", "12345678"),
        ),
    )
    try:
        with pytest.raises(StorageConflictError) as raised:
            with repository.transaction() as transaction:
                transaction.add_external_identifier_assertions(
                    _FIRST_CORPUS.corpus_id, assertions
                )

        assert _PRIVATE_TRIGGER_PAYLOAD not in str(raised.value)
        assert (
            repository.list_external_identifier_assertions(
                _FIRST_CORPUS.corpus_id, _FIRST_VERSION.document_version_id
            )
            == ()
        )
        for assertion in assertions:
            assert (
                repository.find_document_versions_by_external_identifiers(
                    _FIRST_CORPUS.corpus_id,
                    (assertion.external_identifier,),
                )
                == ()
            )
    finally:
        repository.close()


def test_cross_corpus_assertion_failure_rolls_back_other_staged_changes(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
    assertion = ExternalIdentifierAssertion(
        _SOURCE_ID,
        _FIRST_VERSION.document_version_id,
        ExternalIdentifier("doi", "10.1000/cross-corpus"),
    )
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.set_corpus_display_name(
                    _SECOND_CORPUS.corpus_id, "Uncommitted rename"
                )
                transaction.add_external_identifier_assertions(
                    _SECOND_CORPUS.corpus_id, (assertion,)
                )

        assert repository.get_corpus(_SECOND_CORPUS.corpus_id) == _SECOND_CORPUS
        for corpus in (_FIRST_CORPUS, _SECOND_CORPUS):
            assert (
                repository.list_external_identifier_assertions(
                    corpus.corpus_id, _FIRST_VERSION.document_version_id
                )
                == ()
            )
    finally:
        repository.close()


def test_source_creation_requires_revision_one_and_rolls_back_prior_writes(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    invalid_state = SourceObservationState(
        _FIRST_CORPUS.corpus_id,
        _SOURCE_ID,
        2,
        _DOCUMENT_ID,
        _FIRST_VERSION.document_version_id,
    )
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.set_corpus_display_name(
                    _SECOND_CORPUS.corpus_id, "Uncommitted rename"
                )
                transaction.add_source_observation(invalid_state)

        assert repository.get_corpus(_SECOND_CORPUS.corpus_id) == _SECOND_CORPUS
        assert (
            repository.get_source_observation(_FIRST_CORPUS.corpus_id, _SOURCE_ID)
            is None
        )
    finally:
        repository.close()


def test_source_identity_cannot_be_replaced_at_revision_one(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    conflicting_state = SourceObservationState(
        _FIRST_CORPUS.corpus_id,
        _SOURCE_ID,
        1,
        _DOCUMENT_ID,
        _SECOND_VERSION.document_version_id,
    )
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_source_observation(conflicting_state)

        assert repository.get_source_observation(
            _FIRST_CORPUS.corpus_id, _SOURCE_ID
        ) == (_INITIAL_STATE)
        assert repository.list_source_observation_history(
            _FIRST_CORPUS.corpus_id, _SOURCE_ID
        ) == (_INITIAL_STATE,)
    finally:
        repository.close()


def test_advancing_an_unknown_source_fails_without_creating_history(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.advance_source_observation(
                    _FIRST_CORPUS.corpus_id,
                    _SOURCE_ID,
                    expected_version_id=_FIRST_VERSION.document_version_id,
                    new_version_id=_SECOND_VERSION.document_version_id,
                )

        assert (
            repository.list_source_observation_history(
                _FIRST_CORPUS.corpus_id, _SOURCE_ID
            )
            == ()
        )
    finally:
        repository.close()


def test_same_version_advance_is_an_exact_no_op(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
    replayed: SourceObservationState | None = None
    try:
        with repository.transaction() as transaction:
            replayed = transaction.advance_source_observation(
                _FIRST_CORPUS.corpus_id,
                _SOURCE_ID,
                expected_version_id=_FIRST_VERSION.document_version_id,
                new_version_id=_FIRST_VERSION.document_version_id,
            )

        assert replayed == _INITIAL_STATE
        assert repository.list_source_observation_history(
            _FIRST_CORPUS.corpus_id, _SOURCE_ID
        ) == (_INITIAL_STATE,)
    finally:
        repository.close()


class _CommitFailingConnection(sqlite3.Connection):
    """Connection that deterministically fails transaction commits."""

    def commit(self) -> None:
        raise sqlite3.OperationalError(_PRIVATE_TRIGGER_PAYLOAD)


def test_commit_failure_rolls_back_and_leaves_repository_readable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    initial = _open_repository(database_path)
    initial.close()
    connection = sqlite3.connect(
        database_path,
        isolation_level=None,
        factory=_CommitFailingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    repository = SQLiteMetadataRepository(connection, MIGRATIONS)
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageTransactionError) as raised:
                with repository.transaction() as transaction:
                    transaction.add_corpus(_FIRST_CORPUS)

        assert _PRIVATE_TRIGGER_PAYLOAD not in str(raised.value)
        assert repository.get_corpus(_FIRST_CORPUS.corpus_id) is None
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "transaction",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert _PRIVATE_TRIGGER_PAYLOAD not in repr(logs)
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.get_corpus(_FIRST_CORPUS.corpus_id) is None
    finally:
        reopened.close()
