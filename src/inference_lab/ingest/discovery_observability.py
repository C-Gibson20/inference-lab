"""Privacy-safe lifecycle logging for local source discovery."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

from structlog.contextvars import bound_contextvars

from inference_lab._validation import is_strict_integer
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
from inference_lab.ingest.discovery_logging import emit_discovery_event
from inference_lab.ingest.discovery_results import DiscoveryResult, DiscoveryStatus
from inference_lab.ingest.discovery_summary import build_discovery_result
from inference_lab.ingest.identity_application import IdentityApplicationResult

Clock = Callable[[], int]

_COUNT_FIELDS: Final = (
    "files_seen",
    "supported",
    "unsupported",
    "hashed",
    "unchanged",
    "changed",
    "new",
    "duplicate",
    "ambiguous",
    "conflicting",
    "failed",
)


def _read_clock(clock: Clock) -> int:
    value = clock()
    if not is_strict_integer(value):
        raise TypeError("clock must return an integer nanosecond value.")
    return value


def _completion_level(status: DiscoveryStatus) -> str:
    if status is DiscoveryStatus.COMPLETED:
        return "info"
    if status is DiscoveryStatus.PARTIAL:
        return "warning"
    return "error"


class _DiscoveryLifecycle:
    """Build and log one terminal result for an active discovery operation."""

    def __init__(
        self,
        request: DiscoveryRequest,
        *,
        policy: DiscoveryPolicy,
        started_ns: int,
        clock: Clock,
    ) -> None:
        self._request = request
        self._policy = policy
        self._started_ns = started_ns
        self._clock = clock
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
        """Build and emit the single terminal discovery result."""
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
        finished_ns = _read_clock(self._clock)
        if finished_ns < self._started_ns:
            raise ValueError("clock must not move backwards during discovery.")
        duration_ms = (finished_ns - self._started_ns) / 1_000_000
        self._completed = True

        count_fields = {
            field_name: getattr(result.counts, field_name)
            for field_name in _COUNT_FIELDS
        }
        emit_discovery_event(
            __name__,
            _completion_level(result.status),
            "discovery_completed",
            policy_name=result.policy.policy_name,
            policy_version=result.policy.policy_version,
            classifier_name=result.policy.classifier_name,
            classifier_version=result.policy.classifier_version,
            status=result.status.value,
            duration_ms=duration_ms,
            **count_fields,
        )
        return result


@contextmanager
def observe_discovery(
    request: DiscoveryRequest,
    *,
    classifier: SourceClassifierRegistry = CURRENT_SOURCE_CLASSIFIER,
    clock: Clock = time.monotonic_ns,
) -> Iterator[_DiscoveryLifecycle]:
    """Bind and log one discovery operation until its result is complete."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not isinstance(classifier, SourceClassifierRegistry):
        raise TypeError("classifier must be a SourceClassifierRegistry.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

    policy = CURRENT_DISCOVERY_POLICY.with_classifier_identity(
        classifier_name=classifier.classifier_name,
        classifier_version=classifier.classifier_version,
    )
    started_ns = _read_clock(clock)
    lifecycle = _DiscoveryLifecycle(
        request,
        policy=policy,
        started_ns=started_ns,
        clock=clock,
    )
    with bound_contextvars(
        corpus_id=request.corpus_id.as_str(),
        corpus_version=None,
        document_id=None,
    ):
        emit_discovery_event(
            __name__,
            "info",
            "discovery_started",
            policy_name=policy.policy_name,
            policy_version=policy.policy_version,
            classifier_name=policy.classifier_name,
            classifier_version=policy.classifier_version,
            input_count=len(request.inputs),
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
