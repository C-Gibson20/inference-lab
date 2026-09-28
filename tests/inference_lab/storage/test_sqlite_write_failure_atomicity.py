"""Failure atomicity and corpus isolation for SQLite metadata writes."""

import sqlite3
from pathlib import Path

import pytest

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
from inference_lab.storage import StorageConflictError
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
