"""Privacy-safe lifecycle logging for local source discovery."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.discovery import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveryPolicy,
    DiscoveryRequest,
)
from inference_lab.ingest.discovery_failures import DiscoveryFailure
from inference_lab.ingest.discovery_results import DiscoveryResult, DiscoveryStatus
from inference_lab.ingest.discovery_summary import build_discovery_result
from inference_lab.ingest.identity_application import IdentityApplicationResult

Clock = Callable[[], int]

logger = structlog.get_logger(__name__)

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


def _safe_log(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded lifecycle metadata without changing discovery outcomes."""
    try:
        getattr(logger, method_name)(event, pipeline_stage="ingest", **fields)
    except Exception:
        pass


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
        files_seen: int,
        stage_failures: tuple[DiscoveryFailure, ...],
        identity_results: tuple[IdentityApplicationResult | DiscoveryFailure, ...],
    ) -> DiscoveryResult:
        """Build and emit the single terminal discovery result."""
        if self._completed:
            raise RuntimeError("discovery lifecycle has already been completed.")

        result = build_discovery_result(
            self._request,
            files_seen=files_seen,
            stage_failures=stage_failures,
            identity_results=identity_results,
            policy=self._policy,
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
        _safe_log(
            _completion_level(result.status),
            "discovery_completed",
            policy_name=result.policy.policy_name,
            policy_version=result.policy.policy_version,
            status=result.status.value,
            duration_ms=duration_ms,
            **count_fields,
        )
        return result


@contextmanager
def observe_discovery(
    request: DiscoveryRequest,
    *,
    policy: DiscoveryPolicy = CURRENT_DISCOVERY_POLICY,
    clock: Clock = time.monotonic_ns,
) -> Iterator[_DiscoveryLifecycle]:
    """Bind and log one discovery operation until its result is complete."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    if not isinstance(policy, DiscoveryPolicy):
        raise TypeError("policy must be a DiscoveryPolicy.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

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
        _safe_log(
            "info",
            "discovery_started",
            policy_name=policy.policy_name,
            policy_version=policy.policy_version,
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
