"""Tests for concurrent identity persistence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import inference_lab.ingest.identity_persistence as persistence_module
from inference_lab.ingest import (
    ClassifiedSourceCandidate,
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryIdentityIdAllocator,
    DiscoveryRequest,
    IdentityApplicationResult,
    TraversalInventory,
    VerifiedSourceSnapshot,
    persist_discovered_source_identity,
)
from inference_lab.provenance import (
    CorpusMetadata,
    IdentityIdCollisionError,
    ResolutionKind,
)
from inference_lab.storage import (
    MetadataTransaction,
    StorageConcurrencyError,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.ingest.identity_persistence_fixtures import (
    _CORPUS_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
    _SOURCE_ID,
    _AlternateCountingAllocator,
    _case,
    _CountingAllocator,
)


def test_concurrent_discovery_rebuilds_to_one_unchanged_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    database_path = tmp_path / "metadata.sqlite3"
    initial = SQLiteMetadataRepository.open(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))
    initial.close()

    holder_allocator = _CountingAllocator()
    contender_allocator = _AlternateCountingAllocator()
    holder_entered = Event()
    release_holder = Event()
    holder_done = Event()
    retry_calls = 0
    original_resolver = persistence_module.resolve_discovered_source_identity
    original_log_retry = persistence_module._log_retry

    def gated_resolver(
        candidate: ClassifiedSourceCandidate,
        verified: VerifiedSourceSnapshot,
        *,
        request: DiscoveryRequest,
        inventory: TraversalInventory,
        transaction: MetadataTransaction,
        allocator: DiscoveryIdentityIdAllocator,
    ) -> DiscoveredSource | DiscoveryFailure:
        if allocator is holder_allocator:
            holder_entered.set()
            if not release_holder.wait(timeout=5):
                raise AssertionError("concurrent discovery holder was not released")
        return original_resolver(
            candidate,
            verified,
            request=request,
            inventory=inventory,
            transaction=transaction,
            allocator=allocator,
        )

    def release_after_retry(
        error: IdentityIdCollisionError
        | StorageConcurrencyError
        | StorageConflictError,
    ) -> None:
        nonlocal retry_calls
        assert isinstance(error, StorageConcurrencyError)
        retry_calls += 1
        original_log_retry(error)
        release_holder.set()
        if not holder_done.wait(timeout=5):
            raise AssertionError("concurrent discovery holder did not complete")

    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        gated_resolver,
    )
    monkeypatch.setattr(persistence_module, "_log_retry", release_after_retry)

    def discover(
        allocator: DiscoveryIdentityIdAllocator,
        *,
        holder: bool,
    ) -> IdentityApplicationResult | DiscoveryFailure:
        repository = SQLiteMetadataRepository.open(
            database_path,
            busy_timeout_ms=25,
        )
        try:
            return persist_discovered_source_identity(
                source,
                snapshot,
                request=request,
                inventory=inventory,
                repository=repository,
                allocator=allocator,
            )
        finally:
            repository.close()
            if holder:
                holder_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder_future = executor.submit(discover, holder_allocator, holder=True)
        assert holder_entered.wait(timeout=5)
        contender_future = executor.submit(
            discover,
            contender_allocator,
            holder=False,
        )
        holder_result = holder_future.result(timeout=5)
        contender_result = contender_future.result(timeout=5)

    assert isinstance(holder_result, IdentityApplicationResult)
    assert holder_result.created_document_version
    assert holder_result.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    assert isinstance(contender_result, IdentityApplicationResult)
    assert not contender_result.created_document_version
    assert contender_result.source.identity_outcome.kind is ResolutionKind.UNCHANGED
    assert contender_result.source.source_observation_id == _SOURCE_ID
    assert (
        contender_result.source.identity_outcome.document
        == holder_result.source.identity_outcome.document
    )
    assert (
        contender_result.source.identity_outcome.version
        == holder_result.source.identity_outcome.version
    )
    assert retry_calls == 1
    assert holder_allocator.counts() == (1, 1, 1)
    assert contender_allocator.counts() == (0, 0, 0)

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert len(reopened.list_document_versions(_FIRST_DOCUMENT_ID)) == 1
        assert len(reopened.list_corpus_document_version_memberships(_CORPUS_ID)) == 1
        assert (
            len(reopened.list_source_observation_history(_CORPUS_ID, _SOURCE_ID)) == 1
        )
        assert len(reopened.list_source_locator_history(_CORPUS_ID, _SOURCE_ID)) == 1
        assert (
            reopened.list_duplicate_relationships(
                _CORPUS_ID,
                _FIRST_VERSION_ID,
            )
            == ()
        )
    finally:
        reopened.close()
