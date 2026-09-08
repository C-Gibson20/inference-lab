"""Tests for successful durable identity persistence."""

from pathlib import Path

from inference_lab.ingest import (
    IdentityApplicationResult,
    UuidDiscoveryIdentityIdAllocator,
    persist_discovered_source_identity,
)
from inference_lab.provenance import (
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    ResolutionKind,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.ingest.identity_persistence_fixtures import (
    _CORPUS_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
    _HASH,
    _SOURCE_ID,
    _case,
    _CountingAllocator,
)


def test_real_repository_resolves_and_persists_one_atomic_source(
    tmp_path: Path,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))
    try:
        result = persist_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=UuidDiscoveryIdentityIdAllocator(),
        )

        assert isinstance(result, IdentityApplicationResult)
        assert result.created_document_version
        outcome = result.source.identity_outcome
        assert outcome.document is not None
        assert outcome.version is not None
        assert repository.get_document(outcome.document.document_id) == outcome.document
        assert (
            repository.get_document_version(outcome.version.document_version_id)
            == outcome.version
        )
        assert (
            repository.get_source_observation(
                _CORPUS_ID,
                result.source.source_observation_id,
            )
            is not None
        )
        assert (
            repository.get_source_locator(
                _CORPUS_ID,
                result.source.source_observation_id,
            )
            is not None
        )
    finally:
        repository.close()


def test_repeated_discovery_is_a_complete_durable_identity_noop(
    tmp_path: Path,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    allocator = _CountingAllocator()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))
    try:
        first = persist_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=allocator,
        )
        assert isinstance(first, IdentityApplicationResult)
        assert first.created_document_version
        assert first.source.source_observation_id == _SOURCE_ID
        assert first.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
        assert first.source.identity_outcome.document == DocumentIdentity(
            _FIRST_DOCUMENT_ID
        )
        assert first.source.identity_outcome.version == DocumentVersion(
            _FIRST_VERSION_ID,
            _FIRST_DOCUMENT_ID,
            _HASH,
        )
        assert allocator.counts() == (1, 1, 1)

        durable_before = (
            repository.list_document_versions(_FIRST_DOCUMENT_ID),
            repository.list_corpus_document_version_memberships(_CORPUS_ID),
            repository.list_source_observation_history(_CORPUS_ID, _SOURCE_ID),
            repository.list_source_locator_history(_CORPUS_ID, _SOURCE_ID),
            repository.list_external_identifier_assertions(
                _CORPUS_ID,
                _FIRST_VERSION_ID,
            ),
            repository.list_duplicate_relationships(
                _CORPUS_ID,
                _FIRST_VERSION_ID,
            ),
        )

        repeated = persist_discovered_source_identity(
            source,
            snapshot,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=allocator,
        )

        assert isinstance(repeated, IdentityApplicationResult)
        assert not repeated.created_document_version
        assert repeated.source.source_observation_id == _SOURCE_ID
        assert repeated.source.identity_outcome.kind is ResolutionKind.UNCHANGED
        assert (
            repeated.source.identity_outcome.document
            == first.source.identity_outcome.document
        )
        assert (
            repeated.source.identity_outcome.version
            == first.source.identity_outcome.version
        )
        assert allocator.counts() == (1, 1, 1)
        assert (
            repository.list_document_versions(_FIRST_DOCUMENT_ID),
            repository.list_corpus_document_version_memberships(_CORPUS_ID),
            repository.list_source_observation_history(_CORPUS_ID, _SOURCE_ID),
            repository.list_source_locator_history(_CORPUS_ID, _SOURCE_ID),
            repository.list_external_identifier_assertions(
                _CORPUS_ID,
                _FIRST_VERSION_ID,
            ),
            repository.list_duplicate_relationships(
                _CORPUS_ID,
                _FIRST_VERSION_ID,
            ),
        ) == durable_before
    finally:
        repository.close()
