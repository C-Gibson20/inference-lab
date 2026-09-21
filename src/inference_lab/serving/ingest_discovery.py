"""Discovery composition for the local serving ingest facade."""

from __future__ import annotations

from inference_lab.ingest import (
    ClassifiedSourceCandidate,
    DiscoveryFailure,
    VerifiedSourceSnapshot,
    build_discovery_result,
    classify_source_inventory,
    hash_source_candidate,
    inventory_local_sources,
    persist_discovered_source_identities,
)
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.ingest.identity_application import IdentityApplicationResult
from inference_lab.ingest.identity_persistence import IdentityPersistenceUnit
from inference_lab.serving.ingest_contracts import (
    ServingIngestRequest,
    _ServingRepository,
)
from inference_lab.settings import Settings


def discover_sources(
    repository: _ServingRepository,
    settings: Settings,
    request: ServingIngestRequest,
) -> DiscoveryResult:
    """Run discovery, hashing, identity resolution, and persistence."""
    if request.discovery_request.corpus_id != request.corpus_id:
        raise ValueError("discovery request corpus must match serving request.")
    traversal = inventory_local_sources(
        request.discovery_request,
        excluded_paths=request.excluded_paths,
        filesystem=request.traversal_filesystem,
    )
    classification = classify_source_inventory(
        traversal,
        registry=request.classifier_registry,
    )
    units: list[tuple[ClassifiedSourceCandidate, VerifiedSourceSnapshot]] = []
    failures: dict[int, DiscoveryFailure] = {}
    for candidate in classification.hashable_candidates:
        hashed = hash_source_candidate(
            candidate,
            max_source_bytes=settings.ingest.max_source_bytes,
            filesystem=request.source_filesystem,
        )
        if isinstance(hashed, VerifiedSourceSnapshot):
            units.append((candidate, hashed))
        else:
            failures[candidate.candidate.ordinal] = hashed
    identity_results = list(
        persist_discovered_source_identities(
            tuple(
                IdentityPersistenceUnit(source=source, snapshot=snapshot)
                for source, snapshot in units
            ),
            request=request.discovery_request,
            inventory=traversal,
            repository=repository,
            allocator=request.identity_allocator,
        )
    )
    for result in identity_results:
        ordinal = (
            result.source.ordinal
            if isinstance(result, IdentityApplicationResult)
            else result.ordinal
        )
        failures.pop(ordinal, None)
    identity_by_ordinal = {
        (
            result.source.ordinal
            if isinstance(result, IdentityApplicationResult)
            else result.ordinal
        ): result
        for result in identity_results
    }
    ordered_results = []
    for candidate in classification.hashable_candidates:
        ordinal = candidate.candidate.ordinal
        ordered_results.append(failures.get(ordinal) or identity_by_ordinal[ordinal])
    return build_discovery_result(
        classification,
        identity_results=tuple(ordered_results),
    )
