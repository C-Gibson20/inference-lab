"""Deterministic final result assembly for local source discovery."""

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.discovery import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveredSource,
    DiscoveryPolicy,
    DiscoveryRequest,
)
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


def _validate_stage_failures(failures: object) -> None:
    if not isinstance(failures, tuple):
        raise TypeError("stage_failures must be a tuple.")
    if any(not isinstance(failure, DiscoveryFailure) for failure in failures):
        raise TypeError("stage_failures must contain DiscoveryFailure values.")


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
    request: DiscoveryRequest,
    *,
    files_seen: int,
    stage_failures: tuple[DiscoveryFailure, ...],
    identity_results: tuple[IdentityApplicationResult | DiscoveryFailure, ...],
    policy: DiscoveryPolicy = CURRENT_DISCOVERY_POLICY,
) -> DiscoveryResult:
    """Build one ordered result from pre-identity and identity-stage outcomes."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not is_strict_integer(files_seen):
        raise TypeError("files_seen must be an integer.")
    if files_seen < 0:
        raise ValueError("files_seen must be non-negative.")
    _validate_stage_failures(stage_failures)
    _validate_identity_results(identity_results)
    if not isinstance(policy, DiscoveryPolicy):
        raise TypeError("policy must be a DiscoveryPolicy.")

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
    accounted_files = len(sources) + sum(
        failure.scope is DiscoveryFailureScope.SOURCE for failure in failures
    )
    if files_seen != accounted_files:
        raise ValueError("files_seen must exactly match source outcomes and failures.")

    counts = DiscoveryCounts.from_outcomes(
        files_seen=files_seen,
        sources=sources,
        failures=failures,
    )
    return DiscoveryResult(
        corpus_id=request.corpus_id,
        policy=policy,
        status=_result_status(sources, failures),
        sources=sources,
        failures=failures,
        counts=counts,
    )
