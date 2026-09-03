"""Tests for SQLite corpus catalogue membership."""

from datetime import UTC, datetime
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
    HashAlgorithm,
)
from inference_lab.storage import StorageConflictError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 16, 0, tzinfo=UTC)
_FIRST_CORPUS = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"), "First corpus"
)
_SECOND_CORPUS = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"), "Second corpus"
)
_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_OTHER_DOCUMENT_ID = DocumentId("44444444-4444-4444-8444-444444444444")
_VERSION_ID = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_VERSION = DocumentVersion(
    _VERSION_ID,
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_FIRST_MEMBERSHIP = CorpusDocumentVersionMembership(
    _FIRST_CORPUS.corpus_id, _DOCUMENT_ID, _VERSION_ID
)
_SECOND_MEMBERSHIP = CorpusDocumentVersionMembership(
    _SECOND_CORPUS.corpus_id, _DOCUMENT_ID, _VERSION_ID
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _open(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def _add_shared_version(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_FIRST_CORPUS)
        transaction.add_corpus(_SECOND_CORPUS)
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_VERSION)


def test_same_immutable_version_can_belong_to_multiple_corpora_after_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open(database_path)
    _add_shared_version(first)
    with first.transaction() as transaction:
        transaction.add_corpus_document_version_membership(_SECOND_MEMBERSHIP)
        transaction.add_corpus_document_version_membership(_FIRST_MEMBERSHIP)
        transaction.add_corpus_document_version_membership(_FIRST_MEMBERSHIP)
    first.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert (
            reopened.get_corpus_document_version_membership(
                _FIRST_CORPUS.corpus_id, _VERSION_ID
            )
            == _FIRST_MEMBERSHIP
        )
        assert reopened.list_corpus_document_version_memberships(
            _FIRST_CORPUS.corpus_id
        ) == (_FIRST_MEMBERSHIP,)
        assert reopened.list_corpus_document_version_memberships(
            _SECOND_CORPUS.corpus_id
        ) == (_SECOND_MEMBERSHIP,)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "membership",
    [
        CorpusDocumentVersionMembership(
            CorpusId("66666666-6666-4666-8666-666666666666"),
            _DOCUMENT_ID,
            _VERSION_ID,
        ),
        CorpusDocumentVersionMembership(
            _FIRST_CORPUS.corpus_id,
            _OTHER_DOCUMENT_ID,
            _VERSION_ID,
        ),
        CorpusDocumentVersionMembership(
            _FIRST_CORPUS.corpus_id,
            _DOCUMENT_ID,
            DocumentVersionId("77777777-7777-4777-8777-777777777777"),
        ),
    ],
)
def test_membership_requires_existing_consistent_records(
    tmp_path: Path,
    membership: CorpusDocumentVersionMembership,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        _add_shared_version(repository)
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_corpus_document_version_membership(membership)

        assert (
            repository.list_corpus_document_version_memberships(_FIRST_CORPUS.corpus_id)
            == ()
        )
    finally:
        repository.close()


def test_membership_rolls_back_with_its_transaction(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        _add_shared_version(repository)
        with pytest.raises(RuntimeError, match="primary"):
            with repository.transaction() as transaction:
                transaction.add_corpus_document_version_membership(_FIRST_MEMBERSHIP)
                raise RuntimeError("primary")

        assert (
            repository.get_corpus_document_version_membership(
                _FIRST_CORPUS.corpus_id, _VERSION_ID
            )
            is None
        )
    finally:
        repository.close()
