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
    DuplicateClass,
    DuplicateRelationship,
    HashAlgorithm,
)
from inference_lab.storage import StorageConflictError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_APPLIED_AT = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)
_CORPUS_ID = CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Duplicate test corpus")
_DOCUMENT_ID = DocumentId("89606249-7390-4ff5-936f-0309f511924d")
_FIRST_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SECOND_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_THIRD_VERSION_ID = DocumentVersionId("33333333-3333-4333-8333-333333333333")
_FIRST_RELATIONSHIP = DuplicateRelationship(
    _FIRST_VERSION_ID,
    _SECOND_VERSION_ID,
    DuplicateClass.BYTE_IDENTICAL,
)
_SECOND_RELATIONSHIP = DuplicateRelationship(
    _FIRST_VERSION_ID,
    _THIRD_VERSION_ID,
    DuplicateClass.BYTE_IDENTICAL,
)


def _fixed_applied_at() -> datetime:
    return _APPLIED_AT


def _open_repository(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        applied_at_factory=_fixed_applied_at,
    )


def _version(
    version_id: DocumentVersionId,
    digest_character: str = "a",
) -> DocumentVersion:
    return DocumentVersion(
        version_id,
        _DOCUMENT_ID,
        ContentHash(HashAlgorithm.SHA256, digest_character * 64),
    )


def _add_versions(
    repository: SQLiteMetadataRepository,
    version_ids: tuple[DocumentVersionId, ...],
) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        for version_id in version_ids:
            transaction.add_document_version(_version(version_id))
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(_CORPUS_ID, _DOCUMENT_ID, version_id)
            )


def test_duplicate_relationships_round_trip_and_list_by_either_endpoint(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open_repository(database_path)
    _add_versions(
        first,
        (_FIRST_VERSION_ID, _SECOND_VERSION_ID, _THIRD_VERSION_ID),
    )
    with first.transaction() as transaction:
        transaction.add_duplicate_relationship(_SECOND_RELATIONSHIP)
        transaction.add_duplicate_relationship(_FIRST_RELATIONSHIP)
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        stored = second.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID)

        assert stored == (_FIRST_RELATIONSHIP, _SECOND_RELATIONSHIP)
        assert second.list_duplicate_relationships(_CORPUS_ID, _SECOND_VERSION_ID) == (
            _FIRST_RELATIONSHIP,
        )
        assert second.list_duplicate_relationships(_CORPUS_ID, _THIRD_VERSION_ID) == (
            _SECOND_RELATIONSHIP,
        )
        assert not hasattr(stored[0], "row_id")
        assert not hasattr(stored[0], "storage_id")
    finally:
        second.close()


def test_duplicate_relationship_exact_replay_is_idempotent(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_versions(repository, (_FIRST_VERSION_ID, _SECOND_VERSION_ID))
    try:
        with repository.transaction() as transaction:
            transaction.add_duplicate_relationship(_FIRST_RELATIONSHIP)
            transaction.add_duplicate_relationship(_FIRST_RELATIONSHIP)
            assert transaction.list_duplicate_relationships(
                _CORPUS_ID, _FIRST_VERSION_ID
            ) == (_FIRST_RELATIONSHIP,)

        assert repository.list_duplicate_relationships(
            _CORPUS_ID, _FIRST_VERSION_ID
        ) == (_FIRST_RELATIONSHIP,)
    finally:
        repository.close()


def test_missing_duplicate_endpoint_rolls_back_the_composite_write(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
                transaction.add_document_version(_version(_FIRST_VERSION_ID, "a"))
                transaction.add_duplicate_relationship(_FIRST_RELATIONSHIP)

        assert repository.get_document(_DOCUMENT_ID) is None
        assert repository.get_document_version(_FIRST_VERSION_ID) is None
        assert (
            repository.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == ()
        )
    finally:
        repository.close()


def test_byte_identical_relationship_rejects_mismatched_content_hashes(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    with repository.transaction() as transaction:
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(_version(_FIRST_VERSION_ID))
        transaction.add_document_version(_version(_SECOND_VERSION_ID, "b"))

    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_duplicate_relationship(_FIRST_RELATIONSHIP)

        assert (
            repository.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == ()
        )
    finally:
        repository.close()


def test_unknown_version_has_no_duplicate_relationships(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        assert (
            repository.list_duplicate_relationships(_CORPUS_ID, _FIRST_VERSION_ID) == ()
        )
    finally:
        repository.close()
