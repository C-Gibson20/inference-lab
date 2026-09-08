"""Shared integration fixtures for discovery identity acceptance tests."""

from pathlib import Path

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryRequest,
    IdentityApplicationResult,
    IdentityPersistenceUnit,
    LocalTraversalFileSystem,
    TraversalInventory,
    UuidDiscoveryIdentityIdAllocator,
    VerifiedSourceSnapshot,
    classify_source_inventory,
    hash_source_candidate,
    inventory_local_sources,
    persist_discovered_source_identities,
)
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    SourceObservationId,
)
from inference_lab.storage import MetadataRepository, open_metadata_repository

from .fixtures import fixture_settings  # noqa: TID252

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_OTHER_CORPUS_ID = CorpusId("22222222-2222-4222-8222-222222222222")


def _open_test_repository(
    database_path: Path,
    *,
    apply_migrations: bool = False,
) -> MetadataRepository:
    """Open test metadata and managed-source storage under one temporary root."""
    return open_metadata_repository(
        database_path,
        source_dir=database_path.parent / "source-data",
        apply_migrations=apply_migrations,
    )


class _IncompleteScanFileSystem(LocalTraversalFileSystem):
    """Make one real directory unreadable to preserve an incomplete scan."""

    def __init__(self, inaccessible_directory: Path) -> None:
        self._inaccessible_directory = inaccessible_directory.resolve()

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        if path == self._inaccessible_directory:
            raise PermissionError("injected inaccessible directory")
        return super().list_directory(path)


class _RecordingAllocator:
    """Count allocations while delegating production-shaped UUID values."""

    def __init__(self) -> None:
        self._delegate = UuidDiscoveryIdentityIdAllocator()
        self.source_calls = 0
        self.document_calls = 0
        self.version_calls = 0

    def new_source_observation_id(self) -> SourceObservationId:
        self.source_calls += 1
        return self._delegate.new_source_observation_id()

    def new_document_id(self) -> DocumentId:
        self.document_calls += 1
        return self._delegate.new_document_id()

    def new_document_version_id(self) -> DocumentVersionId:
        self.version_calls += 1
        return self._delegate.new_document_version_id()

    @property
    def counts(self) -> tuple[int, int, int]:
        return self.source_calls, self.document_calls, self.version_calls


def _supported_persistence_units(
    request: DiscoveryRequest,
) -> tuple[TraversalInventory, tuple[IdentityPersistenceUnit, ...]]:
    traversal = inventory_local_sources(request, excluded_paths=())
    classification = classify_source_inventory(traversal)
    assert classification.failures == ()

    units: list[IdentityPersistenceUnit] = []
    for source in classification.hashable_candidates:
        snapshot = hash_source_candidate(
            source,
            max_source_bytes=fixture_settings().ingest.max_source_bytes,
        )
        assert isinstance(snapshot, VerifiedSourceSnapshot)
        units.append(IdentityPersistenceUnit(source, snapshot))
    return traversal, tuple(units)


def _persist_supported_sources(
    request: DiscoveryRequest,
    repository: MetadataRepository,
) -> tuple[IdentityApplicationResult | DiscoveryFailure, ...]:
    traversal, units = _supported_persistence_units(request)

    return persist_discovered_source_identities(
        units,
        request=request,
        inventory=traversal,
        repository=repository,
        allocator=UuidDiscoveryIdentityIdAllocator(),
    )


def _successful_results(
    results: tuple[IdentityApplicationResult | DiscoveryFailure, ...],
) -> tuple[IdentityApplicationResult, ...]:
    successes: list[IdentityApplicationResult] = []
    for result in results:
        assert isinstance(result, IdentityApplicationResult)
        successes.append(result)
    return tuple(successes)
