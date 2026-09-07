"""Deterministic final result assembly for local source discovery."""

from inference_lab.ingest.classification import ClassificationInventory
from inference_lab.ingest.discovery import DiscoveredSource
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureScope,
)
from inference_lab.ingest.discovery_results import (
    DiscoveryCounts,
    DiscoveryResult,
    DiscoveryStatus,
)
from inference_lab.ingest.identity_application import IdentityApplicationResult
from inference_lab.provenance import ResolutionKind


def _validate_identity_results(results: object) -> None:
    if not isinstance(results, tuple):
        raise TypeError("identity_results must be a tuple.")
    if any(
        not isinstance(result, IdentityApplicationResult | DiscoveryFailure)
        for result in results
    ):
        raise TypeError(
            "identity_results must contain IdentityApplicationResult or "
            "DiscoveryFailure values."
        )


def _result_status(
    sources: tuple[DiscoveredSource, ...],
    failures: tuple[DiscoveryFailure, ...],
) -> DiscoveryStatus:
    if not failures:
        return DiscoveryStatus.COMPLETED
    has_request_failure = any(
        failure.scope is DiscoveryFailureScope.REQUEST for failure in failures
    )
    usable_sources = any(
        source.identity_outcome.kind
        not in {ResolutionKind.AMBIGUOUS, ResolutionKind.CONFLICT}
        for source in sources
    )
    if has_request_failure or not usable_sources:
        return DiscoveryStatus.FAILED
    return DiscoveryStatus.PARTIAL


def build_discovery_result(
    classification: ClassificationInventory,
    *,
    identity_results: tuple[IdentityApplicationResult | DiscoveryFailure, ...],
) -> DiscoveryResult:
    """Build one result bound to its classified corpus and effective policy."""
    if not isinstance(classification, ClassificationInventory):
        raise TypeError("classification must be a ClassificationInventory.")
    _validate_identity_results(identity_results)

    stage_failures = classification.failures
    expected_by_ordinal = {
        source.candidate.ordinal: source
        for source in classification.hashable_candidates
    }
    result_ordinals = tuple(
        result.source.ordinal
        if isinstance(result, IdentityApplicationResult)
        else result.ordinal
        for result in identity_results
    )
    if tuple(sorted(result_ordinals)) != tuple(expected_by_ordinal):
        raise ValueError(
            "identity results must exactly match hashable classification order."
        )

    policy = classification.policy
    corpus_id = classification.traversal.corpus_id
    for result in identity_results:
        if not isinstance(result, IdentityApplicationResult):
            continue
        source = result.source
        classified = expected_by_ordinal[source.ordinal]
        if source.corpus_id != corpus_id:
            raise ValueError(
                "discovered source corpus must match classified traversal."
            )
        if source.policy != policy:
            raise ValueError("discovered source policy must match classification.")
        if (
            source.classification != classified.classification
            or source.snapshot.locator != classified.candidate.locator
        ):
            raise ValueError("discovered source must match its classified candidate.")

    sources = tuple(
        sorted(
            (
                result.source
                for result in identity_results
                if isinstance(result, IdentityApplicationResult)
            ),
            key=lambda source: source.ordinal,
        )
    )
    failures = tuple(
        sorted(
            stage_failures
            + tuple(
                result
                for result in identity_results
                if isinstance(result, DiscoveryFailure)
            ),
            key=lambda failure: failure.ordinal,
        )
    )
    files_seen = len(classification.classified_candidates) + sum(
        failure.scope is DiscoveryFailureScope.SOURCE
        for failure in classification.traversal.failures
    )

    counts = DiscoveryCounts.from_outcomes(
        files_seen=files_seen,
        sources=sources,
        failures=failures,
    )
    return DiscoveryResult(
        corpus_id=corpus_id,
        policy=policy,
        status=_result_status(sources, failures),
        sources=sources,
        failures=failures,
        counts=counts,
    )
