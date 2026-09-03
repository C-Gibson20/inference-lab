"""Tests for SQLite corpus isolation."""

from pathlib import Path

import pytest

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
    ExternalIdentifierAssertion,
    HashAlgorithm,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import StorageConflictError

from .fixtures import open_repository as _open  # noqa: TID252

_FIRST_CORPUS = CorpusMetadata(
    CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"), "First corpus"
)
_SECOND_CORPUS = CorpusMetadata(
    CorpusId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"), "Second corpus"
)
_FAMILY_ID = DocumentFamilyId("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_SHARED_DOCUMENT_ID = DocumentId("11111111-1111-4111-8111-111111111111")
_FIRST_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_SECOND_DOCUMENT_ID = DocumentId("33333333-3333-4333-8333-333333333333")
_SHARED_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_FIRST_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_SECOND_VERSION_ID = DocumentVersionId("33333333-3333-4333-8333-333333333333")
_FIRST_SHARED_NEXT_VERSION_ID = DocumentVersionId(
    "66666666-6666-4666-8666-666666666666"
)
_FIRST_SOURCE_ID = SourceObservationId("44444444-4444-4444-8444-444444444444")
_SECOND_SOURCE_ID = SourceObservationId("55555555-5555-4555-8555-555555555555")
_SHARED_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_SHARED_VERSION = DocumentVersion(
    _SHARED_VERSION_ID,
    _SHARED_DOCUMENT_ID,
    _SHARED_HASH,
    (ExternalIdentifier("doi", "10.1000/shared"),),
)
_FIRST_VERSION = DocumentVersion(
    _FIRST_VERSION_ID,
    _FIRST_DOCUMENT_ID,
    _SHARED_HASH,
    (ExternalIdentifier("scope", "first-only"),),
)
_SECOND_VERSION = DocumentVersion(
    _SECOND_VERSION_ID,
    _SECOND_DOCUMENT_ID,
    _SHARED_HASH,
    (ExternalIdentifier("scope", "second-only"),),
)
_FIRST_SHARED_NEXT_VERSION = DocumentVersion(
    _FIRST_SHARED_NEXT_VERSION_ID,
    _SHARED_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_FIRST_DUPLICATE = DuplicateRelationship(
    _SHARED_VERSION_ID, _FIRST_VERSION_ID, DuplicateClass.BYTE_IDENTICAL
)
_SECOND_DUPLICATE = DuplicateRelationship(
    _SHARED_VERSION_ID, _SECOND_VERSION_ID, DuplicateClass.BYTE_IDENTICAL
)
_SHARED_FAMILY_MEMBERSHIP = DocumentFamilyMembership(_FAMILY_ID, _SHARED_DOCUMENT_ID)
_FIRST_FAMILY_MEMBERSHIP = DocumentFamilyMembership(
    _FAMILY_ID, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID
)
_SECOND_FAMILY_MEMBERSHIP = DocumentFamilyMembership(
    _FAMILY_ID, _SECOND_DOCUMENT_ID, _SECOND_VERSION_ID
)
_FIRST_STATE = SourceObservationState(
    _FIRST_CORPUS.corpus_id,
    _FIRST_SOURCE_ID,
    1,
    _SHARED_DOCUMENT_ID,
    _SHARED_VERSION_ID,
)
_SECOND_STATE = SourceObservationState(
    _SECOND_CORPUS.corpus_id,
    _SECOND_SOURCE_ID,
    1,
    _SHARED_DOCUMENT_ID,
    _SHARED_VERSION_ID,
)
_FIRST_ASSERTION = ExternalIdentifierAssertion(
    _FIRST_SOURCE_ID,
    _SHARED_VERSION_ID,
    ExternalIdentifier("observed", "first-only"),
)
_SECOND_ASSERTION = ExternalIdentifierAssertion(
    _SECOND_SOURCE_ID,
    _SHARED_VERSION_ID,
    ExternalIdentifier("observed", "second-only"),
)


def test_corpus_scope_isolates_candidates_sources_and_relationships(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    versions = (
        _SHARED_VERSION,
        _FIRST_VERSION,
        _SECOND_VERSION,
        _FIRST_SHARED_NEXT_VERSION,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_FIRST_CORPUS)
            transaction.add_corpus(_SECOND_CORPUS)
            transaction.add_document_family(_FAMILY_ID)
            for version in versions:
                transaction.add_document(DocumentIdentity(version.document_id))
                transaction.add_document_version(version)
            for corpus, visible_versions in (
                (
                    _FIRST_CORPUS,
                    (_SHARED_VERSION, _FIRST_VERSION, _FIRST_SHARED_NEXT_VERSION),
                ),
                (_SECOND_CORPUS, (_SHARED_VERSION, _SECOND_VERSION)),
            ):
                for version in visible_versions:
                    transaction.add_corpus_document_version_membership(
                        CorpusDocumentVersionMembership(
                            corpus.corpus_id,
                            version.document_id,
                            version.document_version_id,
                        )
                    )
            transaction.add_duplicate_relationship(_FIRST_DUPLICATE)
            transaction.add_duplicate_relationship(_SECOND_DUPLICATE)
            transaction.add_family_membership(_SHARED_FAMILY_MEMBERSHIP)
            transaction.add_family_membership(_FIRST_FAMILY_MEMBERSHIP)
            transaction.add_family_membership(_SECOND_FAMILY_MEMBERSHIP)
            transaction.add_source_observation(_FIRST_STATE)
            transaction.add_source_observation(_SECOND_STATE)
            transaction.add_external_identifier_assertions(
                _FIRST_CORPUS.corpus_id, (_FIRST_ASSERTION,)
            )
            transaction.add_external_identifier_assertions(
                _SECOND_CORPUS.corpus_id, (_SECOND_ASSERTION,)
            )

        assert repository.find_document_versions_by_content_hash(
            _FIRST_CORPUS.corpus_id, _SHARED_HASH
        ) == (_SHARED_VERSION, _FIRST_VERSION)
        assert repository.find_document_versions_by_content_hash(
            _SECOND_CORPUS.corpus_id, _SHARED_HASH
        ) == (_SHARED_VERSION, _SECOND_VERSION)
        assert (
            repository.find_document_versions_by_external_identifiers(
                _SECOND_CORPUS.corpus_id,
                (ExternalIdentifier("scope", "first-only"),),
            )
            == ()
        )
        assert repository.find_document_versions_by_external_identifiers(
            _FIRST_CORPUS.corpus_id,
            (ExternalIdentifier("observed", "first-only"),),
        ) == (_SHARED_VERSION,)
        assert (
            repository.find_document_versions_by_external_identifiers(
                _SECOND_CORPUS.corpus_id,
                (ExternalIdentifier("observed", "first-only"),),
            )
            == ()
        )

        assert (
            repository.get_source_observation(_FIRST_CORPUS.corpus_id, _FIRST_SOURCE_ID)
            == _FIRST_STATE
        )
        assert (
            repository.get_source_observation(
                _SECOND_CORPUS.corpus_id, _FIRST_SOURCE_ID
            )
            is None
        )
        assert (
            repository.list_source_observation_history(
                _SECOND_CORPUS.corpus_id, _FIRST_SOURCE_ID
            )
            == ()
        )
        assert repository.list_external_identifier_assertions(
            _FIRST_CORPUS.corpus_id, _SHARED_VERSION_ID
        ) == (_FIRST_ASSERTION,)
        assert repository.list_external_identifier_assertions(
            _SECOND_CORPUS.corpus_id, _SHARED_VERSION_ID
        ) == (_SECOND_ASSERTION,)

        assert repository.list_duplicate_relationships(
            _FIRST_CORPUS.corpus_id, _SHARED_VERSION_ID
        ) == (_FIRST_DUPLICATE,)
        assert repository.list_duplicate_relationships(
            _SECOND_CORPUS.corpus_id, _SHARED_VERSION_ID
        ) == (_SECOND_DUPLICATE,)
        assert repository.list_family_memberships_by_family(
            _FIRST_CORPUS.corpus_id, _FAMILY_ID
        ) == (_SHARED_FAMILY_MEMBERSHIP, _FIRST_FAMILY_MEMBERSHIP)
        assert repository.list_family_memberships_by_family(
            _SECOND_CORPUS.corpus_id, _FAMILY_ID
        ) == (_SHARED_FAMILY_MEMBERSHIP, _SECOND_FAMILY_MEMBERSHIP)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_source_observation(
                    SourceObservationState(
                        _SECOND_CORPUS.corpus_id,
                        _FIRST_SOURCE_ID,
                        1,
                        _SHARED_DOCUMENT_ID,
                        _SHARED_VERSION_ID,
                    )
                )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.advance_source_observation(
                    _SECOND_CORPUS.corpus_id,
                    _SECOND_SOURCE_ID,
                    expected_version_id=_SHARED_VERSION_ID,
                    new_version_id=_FIRST_SHARED_NEXT_VERSION_ID,
                )
        assert (
            repository.get_source_observation(
                _SECOND_CORPUS.corpus_id, _SECOND_SOURCE_ID
            )
            == _SECOND_STATE
        )
    finally:
        repository.close()
