"""Tests for batched identity persistence."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    IdentityApplicationResult,
    IdentityPersistenceUnit,
    TraversalInventory,
    TraversalRootScope,
    UuidDiscoveryIdentityIdAllocator,
    persist_discovered_source_identities,
)
from inference_lab.provenance import CorpusMetadata
from inference_lab.storage import SourceLocatorRevision, StorageConflictError
from inference_lab.storage import _sqlite_transaction as sqlite_transaction_module
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.ingest.identity_persistence_fixtures import (
    _CORPUS_ID,
    _HASH,
    _OTHER_HASH,
    _THIRD_HASH,
    _case,
    _persistence_unit,
    _RecordingAllocator,
    _Repository,
)


def test_identity_batch_rolls_back_one_failure_and_continues_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "sources").resolve()
    units = (
        _persistence_unit(root, ordinal=0, filename="first.txt", content_hash=_HASH),
        _persistence_unit(
            root,
            ordinal=1,
            filename="failing.txt",
            content_hash=_OTHER_HASH,
        ),
        _persistence_unit(
            root,
            ordinal=2,
            filename="third.txt",
            content_hash=_THIRD_HASH,
        ),
    )
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(root, DiscoveryInputKind.DIRECTORY),),
    )
    inventory = TraversalInventory(
        corpus_id=request.corpus_id,
        root_count=1,
        candidates=tuple(unit.source.candidate for unit in units),
        failures=(),
        complete_root_ordinals=(0,),
        root_scopes=(TraversalRootScope(0, root, DiscoveryInputKind.DIRECTORY),),
    )
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Test corpus"))

    allocator = _RecordingAllocator()
    original_add_locator = sqlite_transaction_module.add_source_locator

    def fail_middle_locator(
        connection: sqlite3.Connection,
        revision: SourceLocatorRevision,
    ) -> None:
        if revision.fingerprint.content_hash == _OTHER_HASH:
            raise StorageConflictError()
        original_add_locator(connection, revision)

    monkeypatch.setattr(
        sqlite_transaction_module,
        "add_source_locator",
        fail_middle_locator,
    )
    try:
        results = persist_discovered_source_identities(
            units,
            request=request,
            inventory=inventory,
            repository=repository,
            allocator=allocator,
        )

        assert len(results) == 3
        assert isinstance(results[0], IdentityApplicationResult)
        assert results[0].created_document_version
        assert isinstance(results[1], DiscoveryFailure)
        assert results[1].category is DiscoveryFailureCategory.IDENTITY_WRITE_FAILED
        assert isinstance(results[2], IdentityApplicationResult)
        assert results[2].created_document_version
        assert tuple(
            result.source.ordinal
            if isinstance(result, IdentityApplicationResult)
            else result.ordinal
            for result in results
        ) == (0, 1, 2)

        assert len(allocator.source_ids) == 4
        assert len(allocator.document_ids) == 4
        assert len(allocator.version_ids) == 4
        assert results[0].source.source_observation_id == allocator.source_ids[0]
        assert results[2].source.source_observation_id == allocator.source_ids[3]
        for failed_attempt in (1, 2):
            assert (
                repository.get_document(allocator.document_ids[failed_attempt]) is None
            )
            assert (
                repository.get_document_version(allocator.version_ids[failed_attempt])
                is None
            )
            assert (
                repository.get_corpus_document_version_membership(
                    _CORPUS_ID,
                    allocator.version_ids[failed_attempt],
                )
                is None
            )
            assert (
                repository.get_source_observation(
                    _CORPUS_ID,
                    allocator.source_ids[failed_attempt],
                )
                is None
            )
            assert (
                repository.get_source_locator(
                    _CORPUS_ID,
                    allocator.source_ids[failed_attempt],
                )
                is None
            )

        assert len(repository.list_corpus_document_version_memberships(_CORPUS_ID)) == 2
        assert (
            repository.find_document_versions_by_content_hash(
                _CORPUS_ID,
                _OTHER_HASH,
            )
            == ()
        )
    finally:
        repository.close()


def test_identity_batch_requires_unique_discovery_order(tmp_path: Path) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    unit = IdentityPersistenceUnit(source, snapshot)

    with pytest.raises(ValueError, match="unique ordered ordinals"):
        persist_discovered_source_identities(
            (unit, unit),
            request=request,
            inventory=inventory,
            repository=_Repository(),  # type: ignore[arg-type]
            allocator=UuidDiscoveryIdentityIdAllocator(),
        )
