"""Tests for manifest-filtered metadata reads through SQLite snapshots."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, cast

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
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
from inference_lab.storage import (
    CommitCorpusUpdateOutcome,
    CorpusSnapshot,
    RepositoryClosedError,
    SealCorpusUpdateOutcome,
    StorageTransactionError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CREATED_AT = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
_CORPUS_A = CorpusMetadata(
    CorpusId("11111111-1111-4111-8111-111111111111"),
    "Private projection corpus",
)
_CORPUS_B = CorpusMetadata(
    CorpusId("22222222-2222-4222-8222-222222222222"),
    "Other private projection corpus",
)
_VERSION_A1 = CorpusVersionId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_VERSION_A2 = CorpusVersionId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_VERSION_B1 = CorpusVersionId("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_DOCUMENT_ONE = DocumentId("11111111-1111-4111-8111-111111111112")
_DOCUMENT_TWO = DocumentId("22222222-2222-4222-8222-222222222223")
_FOREIGN_DOCUMENT = DocumentId("33333333-3333-4333-8333-333333333334")
_DOCUMENT_ONE_V1_ID = DocumentVersionId("11111111-1111-4111-8111-111111111113")
_DOCUMENT_ONE_V2_ID = DocumentVersionId("22222222-2222-4222-8222-222222222224")
_DOCUMENT_TWO_V1_ID = DocumentVersionId("33333333-3333-4333-8333-333333333335")
_FOREIGN_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444445")
_SHARED_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)
_SHARED_IDENTIFIER = ExternalIdentifier("doi", "10.1000/snapshot")
_ASSERTED_IDENTIFIER = ExternalIdentifier("pmid", "12345678")
_DOCUMENT_ONE_V1 = DocumentVersion(
    _DOCUMENT_ONE_V1_ID,
    _DOCUMENT_ONE,
    _SHARED_HASH,
    (_SHARED_IDENTIFIER,),
)
_DOCUMENT_ONE_V2 = DocumentVersion(
    _DOCUMENT_ONE_V2_ID,
    _DOCUMENT_ONE,
    _SHARED_HASH,
    (_SHARED_IDENTIFIER,),
)
_DOCUMENT_TWO_V1 = DocumentVersion(
    _DOCUMENT_TWO_V1_ID,
    _DOCUMENT_TWO,
    _SHARED_HASH,
)
_FOREIGN_VERSION = DocumentVersion(
    _FOREIGN_VERSION_ID,
    _FOREIGN_DOCUMENT,
    _SHARED_HASH,
    (_SHARED_IDENTIFIER,),
)
_MEMBERSHIP_ONE_V1 = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _DOCUMENT_ONE,
    _DOCUMENT_ONE_V1_ID,
)
_MEMBERSHIP_ONE_V2 = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _DOCUMENT_ONE,
    _DOCUMENT_ONE_V2_ID,
)
_MEMBERSHIP_TWO_V1 = CorpusDocumentVersionMembership(
    _CORPUS_A.corpus_id,
    _DOCUMENT_TWO,
    _DOCUMENT_TWO_V1_ID,
)
_FOREIGN_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_B.corpus_id,
    _FOREIGN_DOCUMENT,
    _FOREIGN_VERSION_ID,
)
_SOURCE_ID = SourceObservationId("55555555-5555-4555-8555-555555555555")
_SOURCE_V1 = SourceObservationState(
    _CORPUS_A.corpus_id,
    _SOURCE_ID,
    1,
    _DOCUMENT_ONE,
    _DOCUMENT_ONE_V1_ID,
)
_SOURCE_V2 = SourceObservationState(
    _CORPUS_A.corpus_id,
    _SOURCE_ID,
    2,
    _DOCUMENT_ONE,
    _DOCUMENT_ONE_V2_ID,
)
_ASSERTION_V1 = ExternalIdentifierAssertion(
    _SOURCE_ID,
    _DOCUMENT_ONE_V1_ID,
    _ASSERTED_IDENTIFIER,
)
_ASSERTION_V2 = ExternalIdentifierAssertion(
    _SOURCE_ID,
    _DOCUMENT_ONE_V2_ID,
    _ASSERTED_IDENTIFIER,
)
_DUPLICATE_V1 = DuplicateRelationship(
    _DOCUMENT_ONE_V1_ID,
    _DOCUMENT_TWO_V1_ID,
    DuplicateClass.BYTE_IDENTICAL,
)
_DUPLICATE_V2 = DuplicateRelationship(
    _DOCUMENT_ONE_V2_ID,
    _DOCUMENT_TWO_V1_ID,
    DuplicateClass.BYTE_IDENTICAL,
)
_VISIBLE_FAMILY = DocumentFamilyId("66666666-6666-4666-8666-666666666666")
_FOREIGN_FAMILY = DocumentFamilyId("77777777-7777-4777-8777-777777777777")
_FAMILY_DOCUMENT_ONE = DocumentFamilyMembership(_VISIBLE_FAMILY, _DOCUMENT_ONE)
_FAMILY_ONE_V1 = DocumentFamilyMembership(
    _VISIBLE_FAMILY,
    _DOCUMENT_ONE,
    _DOCUMENT_ONE_V1_ID,
)
_FAMILY_ONE_V2 = DocumentFamilyMembership(
    _VISIBLE_FAMILY,
    _DOCUMENT_ONE,
    _DOCUMENT_ONE_V2_ID,
)
_FAMILY_TWO_V1 = DocumentFamilyMembership(
    _VISIBLE_FAMILY,
    _DOCUMENT_TWO,
    _DOCUMENT_TWO_V1_ID,
)
_FAMILY_FOREIGN = DocumentFamilyMembership(_FOREIGN_FAMILY, _FOREIGN_DOCUMENT)


def _fixed_created_at() -> datetime:
    return _CREATED_AT


def _open(path: Path) -> SQLiteMetadataRepository:
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        created_at_factory=_fixed_created_at,
    )


def _populate_catalogue(repository: SQLiteMetadataRepository) -> None:
    versions = (
        _DOCUMENT_ONE_V1,
        _DOCUMENT_ONE_V2,
        _DOCUMENT_TWO_V1,
        _FOREIGN_VERSION,
    )
    memberships = (
        _MEMBERSHIP_ONE_V1,
        _MEMBERSHIP_ONE_V2,
        _MEMBERSHIP_TWO_V1,
        _FOREIGN_MEMBERSHIP,
    )
    family_memberships = (
        _FAMILY_DOCUMENT_ONE,
        _FAMILY_ONE_V1,
        _FAMILY_ONE_V2,
        _FAMILY_TWO_V1,
        _FAMILY_FOREIGN,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS_A)
        transaction.add_corpus(_CORPUS_B)
        transaction.add_document_family(_VISIBLE_FAMILY)
        transaction.add_document_family(_FOREIGN_FAMILY)
        for document_id in (_DOCUMENT_ONE, _DOCUMENT_TWO, _FOREIGN_DOCUMENT):
            transaction.add_document(DocumentIdentity(document_id))
        for version in versions:
            transaction.add_document_version(version)
        for membership in memberships:
            transaction.add_corpus_document_version_membership(membership)
        transaction.add_source_observation(_SOURCE_V1)
        transaction.add_external_identifier_assertions(
            _CORPUS_A.corpus_id,
            (_ASSERTION_V1,),
        )
        assert (
            transaction.advance_source_observation(
                _CORPUS_A.corpus_id,
                _SOURCE_ID,
                expected_version_id=_DOCUMENT_ONE_V1_ID,
                new_version_id=_DOCUMENT_ONE_V2_ID,
            )
            == _SOURCE_V2
        )
        transaction.add_external_identifier_assertions(
            _CORPUS_A.corpus_id,
            (_ASSERTION_V2,),
        )
        transaction.add_duplicate_relationship(_DUPLICATE_V1)
        transaction.add_duplicate_relationship(_DUPLICATE_V2)
        for membership in family_memberships:
            transaction.add_family_membership(membership)


def _publish(
    repository: SQLiteMetadataRepository,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    expected_corpus_version: CorpusVersionId | None,
    memberships: tuple[CorpusDocumentVersionMembership, ...],
    replacements: tuple[
        tuple[
            CorpusDocumentVersionMembership,
            CorpusDocumentVersionMembership,
        ],
        ...,
    ] = (),
) -> None:
    sealed: object = None
    committed: object = None
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            corpus_id,
            corpus_version,
            expected_corpus_version=expected_corpus_version,
        )
        for membership in memberships:
            transaction.add_corpus_update_document_membership(
                corpus_id,
                corpus_version,
                membership,
            )
        for expected, replacement in replacements:
            transaction.replace_corpus_update_document_membership(
                corpus_id,
                corpus_version,
                expected,
                replacement,
            )
        sealed = transaction.seal_corpus_update(corpus_id, corpus_version)
        committed = transaction.commit_corpus_update(corpus_id, corpus_version)
    assert isinstance(sealed, SealCorpusUpdateOutcome)
    assert isinstance(committed, CommitCorpusUpdateOutcome)


def _assert_version_one_projection(snapshot: CorpusSnapshot) -> None:
    assert snapshot.get_document(_DOCUMENT_ONE) == DocumentIdentity(_DOCUMENT_ONE)
    assert snapshot.get_document(_DOCUMENT_TWO) == DocumentIdentity(_DOCUMENT_TWO)
    assert snapshot.get_document(_FOREIGN_DOCUMENT) is None
    assert snapshot.get_document_version(_DOCUMENT_ONE_V1_ID) == _DOCUMENT_ONE_V1
    assert snapshot.get_document_version(_DOCUMENT_ONE_V2_ID) is None
    assert snapshot.get_document_version(_FOREIGN_VERSION_ID) is None
    assert snapshot.list_document_versions(_DOCUMENT_ONE) == (_DOCUMENT_ONE_V1,)
    assert snapshot.list_document_versions(_FOREIGN_DOCUMENT) == ()
    assert snapshot.find_document_versions_by_content_hash(_SHARED_HASH) == (
        _DOCUMENT_ONE_V1,
        _DOCUMENT_TWO_V1,
    )
    assert snapshot.find_document_versions_by_external_identifiers(
        (_SHARED_IDENTIFIER,)
    ) == (_DOCUMENT_ONE_V1,)
    assert snapshot.get_source_observation(_SOURCE_ID) == _SOURCE_V1
    assert snapshot.list_source_observation_history(_SOURCE_ID) == (_SOURCE_V1,)
    assert snapshot.list_external_identifier_assertions(_DOCUMENT_ONE_V1_ID) == (
        _ASSERTION_V1,
    )
    assert snapshot.list_external_identifier_assertions(_DOCUMENT_ONE_V2_ID) == ()
    assert snapshot.list_duplicate_relationships(_DOCUMENT_ONE_V1_ID) == (
        _DUPLICATE_V1,
    )
    assert snapshot.list_duplicate_relationships(_DOCUMENT_TWO_V1_ID) == (
        _DUPLICATE_V1,
    )
    assert snapshot.list_duplicate_relationships(_DOCUMENT_ONE_V2_ID) == ()
    assert snapshot.get_document_family(_VISIBLE_FAMILY) == _VISIBLE_FAMILY
    assert snapshot.get_document_family(_FOREIGN_FAMILY) is None
    assert snapshot.list_family_memberships_by_family(_VISIBLE_FAMILY) == (
        _FAMILY_DOCUMENT_ONE,
        _FAMILY_ONE_V1,
        _FAMILY_TWO_V1,
    )
    assert snapshot.list_family_memberships_by_document(_DOCUMENT_ONE) == (
        _FAMILY_DOCUMENT_ONE,
        _FAMILY_ONE_V1,
    )
    assert snapshot.list_family_memberships_by_document_version(
        _DOCUMENT_ONE_V1_ID
    ) == (_FAMILY_ONE_V1,)
    assert (
        snapshot.list_family_memberships_by_document_version(_DOCUMENT_ONE_V2_ID) == ()
    )


def _assert_version_two_projection(snapshot: CorpusSnapshot) -> None:
    assert snapshot.get_document_version(_DOCUMENT_ONE_V1_ID) is None
    assert snapshot.get_document_version(_DOCUMENT_ONE_V2_ID) == _DOCUMENT_ONE_V2
    assert snapshot.list_document_versions(_DOCUMENT_ONE) == (_DOCUMENT_ONE_V2,)
    assert snapshot.find_document_versions_by_content_hash(_SHARED_HASH) == (
        _DOCUMENT_ONE_V2,
        _DOCUMENT_TWO_V1,
    )
    assert snapshot.find_document_versions_by_external_identifiers(
        (_SHARED_IDENTIFIER,)
    ) == (_DOCUMENT_ONE_V2,)
    assert snapshot.get_source_observation(_SOURCE_ID) == _SOURCE_V2
    assert snapshot.list_source_observation_history(_SOURCE_ID) == (_SOURCE_V2,)
    assert snapshot.list_external_identifier_assertions(_DOCUMENT_ONE_V1_ID) == ()
    assert snapshot.list_external_identifier_assertions(_DOCUMENT_ONE_V2_ID) == (
        _ASSERTION_V2,
    )
    assert snapshot.list_duplicate_relationships(_DOCUMENT_ONE_V2_ID) == (
        _DUPLICATE_V2,
    )
    assert snapshot.list_family_memberships_by_family(_VISIBLE_FAMILY) == (
        _FAMILY_DOCUMENT_ONE,
        _FAMILY_ONE_V2,
        _FAMILY_TWO_V1,
    )


def test_snapshot_reads_remain_manifest_filtered_while_active_version_advances(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _populate_catalogue(repository)
    _publish(
        repository,
        _CORPUS_A.corpus_id,
        _VERSION_A1,
        None,
        (_MEMBERSHIP_ONE_V1, _MEMBERSHIP_TWO_V1),
    )
    _publish(
        repository,
        _CORPUS_B.corpus_id,
        _VERSION_B1,
        None,
        (_FOREIGN_MEMBERSHIP,),
    )
    try:
        with repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id) as original:
            _assert_version_one_projection(original)

            _publish(
                repository,
                _CORPUS_A.corpus_id,
                _VERSION_A2,
                _VERSION_A1,
                (),
                ((_MEMBERSHIP_ONE_V1, _MEMBERSHIP_ONE_V2),),
            )

            _assert_version_one_projection(original)
            with repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id) as current:
                _assert_version_two_projection(current)

        with repository.open_corpus_snapshot(
            _CORPUS_A.corpus_id,
            _VERSION_A1,
        ) as historical:
            _assert_version_one_projection(historical)

        with repository.open_active_corpus_snapshot(_CORPUS_B.corpus_id) as foreign:
            assert foreign.get_document(_FOREIGN_DOCUMENT) == DocumentIdentity(
                _FOREIGN_DOCUMENT
            )
            assert foreign.get_document(_DOCUMENT_ONE) is None
            assert foreign.find_document_versions_by_content_hash(_SHARED_HASH) == (
                _FOREIGN_VERSION,
            )
            assert foreign.find_document_versions_by_external_identifiers(
                (_SHARED_IDENTIFIER,)
            ) == (_FOREIGN_VERSION,)
            assert foreign.get_document_family(_VISIBLE_FAMILY) is None
            assert foreign.get_document_family(_FOREIGN_FAMILY) == _FOREIGN_FAMILY
    finally:
        repository.close()


def test_snapshot_read_methods_enforce_owned_lifecycle(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _populate_catalogue(repository)
    _publish(
        repository,
        _CORPUS_A.corpus_id,
        _VERSION_A1,
        None,
        (_MEMBERSHIP_ONE_V1,),
    )
    snapshot = repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id)
    try:
        with pytest.raises(StorageTransactionError):
            snapshot.get_document(_DOCUMENT_ONE)
        with snapshot:
            assert snapshot.get_document(_DOCUMENT_ONE) == DocumentIdentity(
                _DOCUMENT_ONE
            )
    finally:
        snapshot.close()
        repository.close()

    with pytest.raises(RepositoryClosedError):
        snapshot.get_document(_DOCUMENT_ONE)


def test_snapshot_read_methods_validate_public_inputs(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _populate_catalogue(repository)
    _publish(
        repository,
        _CORPUS_A.corpus_id,
        _VERSION_A1,
        None,
        (_MEMBERSHIP_ONE_V1,),
    )
    operations: tuple[Callable[[CorpusSnapshot], object], ...] = (
        lambda snapshot: snapshot.get_document(cast("DocumentId", object())),
        lambda snapshot: snapshot.get_document_version(
            cast("DocumentVersionId", object())
        ),
        lambda snapshot: snapshot.list_document_versions(cast("DocumentId", object())),
        lambda snapshot: snapshot.find_document_versions_by_content_hash(
            cast("ContentHash", object())
        ),
        lambda snapshot: snapshot.find_document_versions_by_external_identifiers(
            cast("tuple[ExternalIdentifier, ...]", object())
        ),
        lambda snapshot: snapshot.get_source_observation(
            cast("SourceObservationId", object())
        ),
        lambda snapshot: snapshot.list_source_observation_history(
            cast("SourceObservationId", object())
        ),
        lambda snapshot: snapshot.list_external_identifier_assertions(
            cast("DocumentVersionId", object())
        ),
        lambda snapshot: snapshot.list_duplicate_relationships(
            cast("DocumentVersionId", object())
        ),
        lambda snapshot: snapshot.get_document_family(
            cast("DocumentFamilyId", object())
        ),
        lambda snapshot: snapshot.list_family_memberships_by_family(
            cast("DocumentFamilyId", object())
        ),
        lambda snapshot: snapshot.list_family_memberships_by_document(
            cast("DocumentId", object())
        ),
        lambda snapshot: snapshot.list_family_memberships_by_document_version(
            cast("DocumentVersionId", object())
        ),
    )
    try:
        with repository.open_active_corpus_snapshot(_CORPUS_A.corpus_id) as snapshot:
            for operation in operations:
                with pytest.raises(TypeError):
                    operation(snapshot)
    finally:
        repository.close()
