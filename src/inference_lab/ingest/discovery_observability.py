"""Privacy-safe lifecycle observation for local source discovery."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from structlog.contextvars import bound_contextvars

from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.ingest.classification import (
    CURRENT_SOURCE_CLASSIFIER,
    ClassificationInventory,
    SourceClassifierRegistry,
)
from inference_lab.ingest.discovery import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveryPolicy,
    DiscoveryRequest,
)
from inference_lab.ingest.discovery_failures import DiscoveryFailure
from inference_lab.ingest.discovery_results import DiscoveryResult, DiscoveryStatus
from inference_lab.ingest.discovery_summary import build_discovery_result
from inference_lab.ingest.identity_application import IdentityApplicationResult
from inference_lab.profiling import FailureMapping, ObservedStage

_INPUT_FAILURES = frozenset(
    {
        "invalid_request",
        "invalid_root",
        "storage_root_overlap",
        "recognised_unsupported_format",
        "unknown_format",
    }
)
_DEPENDENCY_FAILURES = frozenset(
    {
        "root_not_found",
        "root_inaccessible",
        "source_not_found",
        "source_inaccessible",
        "traversal_failed",
        "source_read_failed",
        "repository_unavailable",
        "concurrent_write_conflict",
        "identity_write_failed",
    }
)


def _discovery_failure_mapping(
    failure: DiscoveryFailure,
    *,
    partial: bool,
) -> FailureMapping:
    """Map one returned discovery failure to a bounded stage failure."""
    category = failure.category.value
    if category in _INPUT_FAILURES:
        failure_category = "input"
    elif category in _DEPENDENCY_FAILURES:
        failure_category = "dependency"
    else:
        failure_category = "internal"
    return FailureMapping(
        failure_category=failure_category,
        failure_code=category,
        recoverable=failure.retryable,
        partial=partial,
    )


class _DiscoveryLifecycle:
    """Build and log one terminal result for an active discovery operation."""

    def __init__(
        self,
        request: DiscoveryRequest,
        *,
        policy: DiscoveryPolicy,
        profiling_stage: ObservedStage,
    ) -> None:
        self._request = request
        self._policy = policy
        self._profiling_stage = profiling_stage
        self._completed = False

    @property
    def completed(self) -> bool:
        """Return whether this operation has produced its terminal result."""
        return self._completed

    def complete(
        self,
        *,
        classification: ClassificationInventory,
        identity_results: tuple[IdentityApplicationResult | DiscoveryFailure, ...],
    ) -> DiscoveryResult:
        """Build and complete the single observed discovery stage."""
        if self._completed:
            raise RuntimeError("discovery lifecycle has already been completed.")

        if not isinstance(classification, ClassificationInventory):
            raise TypeError("classification must be a ClassificationInventory.")
        if classification.traversal.corpus_id != self._request.corpus_id:
            raise ValueError("classification corpus must match the observed request.")
        if classification.policy != self._policy:
            raise ValueError("classification policy must match the observed policy.")

        result = build_discovery_result(
            classification,
            identity_results=identity_results,
        )
        self._completed = True
        self._profiling_stage.set_counts(
            items_processed=result.counts.files_seen,
            records_emitted=len(result.sources) + len(result.failures),
        )
        if result.status is DiscoveryStatus.COMPLETED:
            self._profiling_stage.mark_success()
        else:
            self._profiling_stage.mark_failure(
                _discovery_failure_mapping(
                    result.failures[0],
                    partial=result.status is DiscoveryStatus.PARTIAL,
                )
            )
        return result


@contextmanager
def observe_discovery(
    request: DiscoveryRequest,
    *,
    classifier: SourceClassifierRegistry = CURRENT_SOURCE_CLASSIFIER,
    clock: Callable[[], int] = time.monotonic_ns,
    profiling: BaselineProfilingContext | None = None,
    profiling_identity: str = "request",
) -> Iterator[_DiscoveryLifecycle]:
    """Bind and observe one discovery operation until its result is complete."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not isinstance(classifier, SourceClassifierRegistry):
        raise TypeError("classifier must be a SourceClassifierRegistry.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    if profiling is not None and not isinstance(profiling, BaselineProfilingContext):
        raise TypeError("profiling must be a BaselineProfilingContext or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")

    policy = CURRENT_DISCOVERY_POLICY.with_classifier_identity(
        classifier_name=classifier.classifier_name,
        classifier_version=classifier.classifier_version,
    )
    selected_profiling = profiling or BaselineProfilingContext.create(clock=clock)
    stage_id = selected_profiling.stage_id("discovery", profiling_identity)
    with bound_contextvars(
        corpus_id=request.corpus_id.as_str(),
        corpus_version=None,
        document_id=None,
        run_id=selected_profiling.run_id,
        request_id=selected_profiling.request_id,
    ):
        with selected_profiling.observed_stage(
            stage="discovery",
            stage_id=stage_id,
            document_id=None,
            corpus_id=request.corpus_id.as_str(),
            corpus_version=None,
            parent_stage_id=None,
        ) as profiling_stage:
            lifecycle = _DiscoveryLifecycle(
                request,
                policy=policy,
                profiling_stage=profiling_stage,
            )
            try:
                yield lifecycle
            except BaseException:
                raise
            else:
                if not lifecycle.completed:
                    raise RuntimeError(
                        "discovery lifecycle exited without a terminal result."
                    )
