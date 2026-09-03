"""Tests for persisted source continuity and identifier assertions."""

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
    IdentityResolutionRequest,
    ResolutionKind,
    SourceObservationId,
    SourceObservationState,
    UuidIdentityIdAllocator,
    resolve_identity,
)
from inference_lab.storage import CorruptStorageError, StorageConflictError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import open_repository as _open_repository  # noqa: TID252

_CORPUS_ID = CorpusId("99999999-9999-4999-8999-999999999999")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Source test corpus")
_OTHER_CORPUS_ID = CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_OTHER_CORPUS = CorpusMetadata(_OTHER_CORPUS_ID, "Other source test corpus")
_DOCUMENT_ID = DocumentId("11111111-1111-4111-8111-111111111111")
_OTHER_DOCUMENT_ID = DocumentId("22222222-2222-4222-8222-222222222222")
_FIRST_VERSION_ID = DocumentVersionId("33333333-3333-4333-8333-333333333333")
_SECOND_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_OTHER_VERSION_ID = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_CONFLICTING_VERSION_ID = DocumentVersionId("88888888-8888-4888-8888-888888888888")
_SOURCE_OBSERVATION_ID = SourceObservationId("66666666-6666-4666-8666-666666666666")
_OTHER_SOURCE_OBSERVATION_ID = SourceObservationId(
    "77777777-7777-4777-8777-777777777777"
)
_DOI = ExternalIdentifier("doi", "10.1000/example")
_PMID = ExternalIdentifier("pmid", "12345678")
_FIRST_VERSION = DocumentVersion(
    _FIRST_VERSION_ID,
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
    (_DOI,),
)
_SECOND_VERSION = DocumentVersion(
    _SECOND_VERSION_ID,
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "b" * 64),
)
_OTHER_VERSION = DocumentVersion(
    _OTHER_VERSION_ID,
    _OTHER_DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "c" * 64),
)
_CONFLICTING_VERSION = DocumentVersion(
    _CONFLICTING_VERSION_ID,
    _DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "d" * 64),
    (ExternalIdentifier("doi", "10.1000/different"),),
)
_INITIAL_STATE = SourceObservationState(
    _CORPUS_ID,
    _SOURCE_OBSERVATION_ID,
    1,
    _DOCUMENT_ID,
    _FIRST_VERSION_ID,
)


def _populate_versions(repository: SQLiteMetadataRepository) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        transaction.add_corpus(_OTHER_CORPUS)
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document(DocumentIdentity(_OTHER_DOCUMENT_ID))
        transaction.add_document_version(_FIRST_VERSION)
        transaction.add_document_version(_SECOND_VERSION)
        transaction.add_document_version(_OTHER_VERSION)
        transaction.add_document_version(_CONFLICTING_VERSION)
        for version in (
            _FIRST_VERSION,
            _SECOND_VERSION,
            _OTHER_VERSION,
            _CONFLICTING_VERSION,
        ):
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS_ID,
                    version.document_id,
                    version.document_version_id,
                )
            )


def test_source_observation_and_history_round_trip_after_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open_repository(database_path)
    _populate_versions(first)
    with first.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
        transaction.add_source_observation(_INITIAL_STATE)
    first.close()

    second = SQLiteMetadataRepository.open(database_path)
    try:
        assert (
            second.get_source_observation(_CORPUS_ID, _SOURCE_OBSERVATION_ID)
            == _INITIAL_STATE
        )
        assert second.list_source_observation_history(
            _CORPUS_ID, _SOURCE_OBSERVATION_ID
        ) == (_INITIAL_STATE,)
        assert (
            second.get_source_observation(_CORPUS_ID, _OTHER_SOURCE_OBSERVATION_ID)
            is None
        )
        assert (
            second.list_source_observation_history(
                _CORPUS_ID, _OTHER_SOURCE_OBSERVATION_ID
            )
            == ()
        )
    finally:
        second.close()


def test_source_observation_read_rejects_current_pointer_history_divergence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _populate_versions(repository)
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)
    repository.close()

    connection = sqlite3.connect(database_path)
    connection.execute(
        """UPDATE source_observations
        SET current_document_version_id = ?, current_revision = 2
        WHERE source_observation_id = ?""",
        (_SECOND_VERSION_ID.as_str(), _SOURCE_OBSERVATION_ID.as_str()),
    )
    connection.commit()
    connection.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            reopened.get_source_observation(_CORPUS_ID, _SOURCE_OBSERVATION_ID)
    finally:
        reopened.close()


def test_source_observation_advance_is_atomic_and_idempotent(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _populate_versions(repository)
    advanced: SourceObservationState | None = None
    replayed: SourceObservationState | None = None
    try:
        with repository.transaction() as transaction:
            transaction.add_source_observation(_INITIAL_STATE)
            advanced = transaction.advance_source_observation(
                _CORPUS_ID,
                _SOURCE_OBSERVATION_ID,
                expected_version_id=_FIRST_VERSION_ID,
                new_version_id=_SECOND_VERSION_ID,
            )
            replayed = transaction.advance_source_observation(
                _CORPUS_ID,
                _SOURCE_OBSERVATION_ID,
                expected_version_id=_FIRST_VERSION_ID,
                new_version_id=_SECOND_VERSION_ID,
            )

        expected = SourceObservationState(
            _CORPUS_ID,
            _SOURCE_OBSERVATION_ID,
            2,
            _DOCUMENT_ID,
            _SECOND_VERSION_ID,
        )
        assert advanced is not None
        assert replayed is not None
        assert advanced == expected
        assert replayed == expected
        assert (
            repository.get_source_observation(_CORPUS_ID, _SOURCE_OBSERVATION_ID)
            == expected
        )
        assert repository.list_source_observation_history(
            _CORPUS_ID, _SOURCE_OBSERVATION_ID
        ) == (
            _INITIAL_STATE,
            expected,
        )
    finally:
        repository.close()


def test_source_observation_advance_rejects_stale_or_cross_document_version(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _populate_versions(repository)
    with repository.transaction() as transaction:
        transaction.add_source_observation(_INITIAL_STATE)

    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.advance_source_observation(
                    _CORPUS_ID,
                    _SOURCE_OBSERVATION_ID,
                    expected_version_id=_SECOND_VERSION_ID,
                    new_version_id=_FIRST_VERSION_ID,
                )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.advance_source_observation(
                    _CORPUS_ID,
                    _SOURCE_OBSERVATION_ID,
                    expected_version_id=_FIRST_VERSION_ID,
                    new_version_id=_OTHER_VERSION_ID,
                )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.advance_source_observation(
                    _CORPUS_ID,
                    _SOURCE_OBSERVATION_ID,
                    expected_version_id=_FIRST_VERSION_ID,
                    new_version_id=_CONFLICTING_VERSION_ID,
                )

        assert repository.get_source_observation(
            _CORPUS_ID, _SOURCE_OBSERVATION_ID
        ) == (_INITIAL_STATE)
        assert repository.list_source_observation_history(
            _CORPUS_ID, _SOURCE_OBSERVATION_ID
        ) == (_INITIAL_STATE,)
    finally:
        repository.close()


def test_identifier_assertion_round_trips_and_participates_in_lookup(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _populate_versions(repository)
    assertion = ExternalIdentifierAssertion(
        _SOURCE_OBSERVATION_ID,
        _FIRST_VERSION_ID,
        _PMID,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_source_observation(_INITIAL_STATE)
            transaction.add_external_identifier_assertions(_CORPUS_ID, (assertion,))
            transaction.add_external_identifier_assertions(_CORPUS_ID, (assertion,))

        assert repository.list_external_identifier_assertions(
            _CORPUS_ID, _FIRST_VERSION_ID
        ) == (assertion,)
        assert repository.find_document_versions_by_external_identifiers(
            _CORPUS_ID, (_PMID,)
        ) == (_FIRST_VERSION,)
    finally:
        repository.close()


def test_identifier_assertion_batch_preserves_complete_namespace_value_set(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _populate_versions(repository)
    assertions = (
        ExternalIdentifierAssertion(
            _SOURCE_OBSERVATION_ID,
            _FIRST_VERSION_ID,
            ExternalIdentifier("publication", "EP1"),
        ),
        ExternalIdentifierAssertion(
            _SOURCE_OBSERVATION_ID,
            _FIRST_VERSION_ID,
            ExternalIdentifier("publication", "EP2"),
        ),
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_source_observation(_INITIAL_STATE)
            transaction.add_external_identifier_assertions(_CORPUS_ID, assertions)
            transaction.add_external_identifier_assertions(_CORPUS_ID, assertions)

        assert (
            repository.list_external_identifier_assertions(
                _CORPUS_ID, _FIRST_VERSION_ID
            )
            == assertions
        )
        assert repository.find_document_versions_by_external_identifiers(
            _CORPUS_ID, (ExternalIdentifier("publication", "EP2"),)
        ) == (_FIRST_VERSION,)
    finally:
        repository.close()


def test_identifier_assertion_rejects_conflict_and_unobserved_version(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _populate_versions(repository)
    assertion = ExternalIdentifierAssertion(
        _SOURCE_OBSERVATION_ID,
        _FIRST_VERSION_ID,
        _PMID,
    )
    try:
        with repository.transaction() as transaction:
            transaction.add_source_observation(_INITIAL_STATE)
            transaction.add_external_identifier_assertions(_CORPUS_ID, (assertion,))

        conflicting = ExternalIdentifierAssertion(
            _SOURCE_OBSERVATION_ID,
            _FIRST_VERSION_ID,
            ExternalIdentifier("pmid", "different"),
        )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_external_identifier_assertions(
                    _CORPUS_ID, (conflicting,)
                )

        unobserved = ExternalIdentifierAssertion(
            _SOURCE_OBSERVATION_ID,
            _SECOND_VERSION_ID,
            ExternalIdentifier("arxiv", "example"),
        )
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_external_identifier_assertions(
                    _CORPUS_ID, (unobserved,)
                )

        assert repository.list_external_identifier_assertions(
            _CORPUS_ID, _FIRST_VERSION_ID
        ) == (assertion,)
    finally:
        repository.close()


def test_repository_values_compose_into_idempotent_resolution_request(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _populate_versions(repository)
    try:
        with repository.transaction() as transaction:
            transaction.add_source_observation(_INITIAL_STATE)

        current = repository.get_source_observation(_CORPUS_ID, _SOURCE_OBSERVATION_ID)
        assert current is not None
        candidate = repository.get_document_version(current.document_version_id)
        assert candidate is not None
        first_outcome = resolve_identity(
            IdentityResolutionRequest(
                content_hash=_FIRST_VERSION.content_hash,
                source_observation_id=_SOURCE_OBSERVATION_ID,
                external_identifiers=(_DOI, _PMID),
                continuity_candidates=(candidate,),
                content_hash_candidates=(candidate,),
                external_identifier_candidates=(candidate,),
            ),
            allocator=UuidIdentityIdAllocator(),
            pipeline_stage="ingest",
        )
        assert first_outcome.kind is ResolutionKind.UNCHANGED
        assert len(first_outcome.identifier_assertions_to_add) == 1

        with repository.transaction() as transaction:
            transaction.add_external_identifier_assertions(
                _CORPUS_ID, first_outcome.identifier_assertions_to_add
            )

        assertions = repository.list_external_identifier_assertions(
            _CORPUS_ID, current.document_version_id
        )
        second_outcome = resolve_identity(
            IdentityResolutionRequest(
                content_hash=_FIRST_VERSION.content_hash,
                source_observation_id=_SOURCE_OBSERVATION_ID,
                external_identifiers=(_DOI, _PMID),
                continuity_candidates=(candidate,),
                content_hash_candidates=(candidate,),
                external_identifier_candidates=(candidate,),
                identifier_assertions=assertions,
            ),
            allocator=UuidIdentityIdAllocator(),
            pipeline_stage="ingest",
        )

        assert second_outcome.kind is ResolutionKind.UNCHANGED
        assert second_outcome.version == _FIRST_VERSION
        assert second_outcome.identifier_assertions_to_add == ()
    finally:
        repository.close()
