"""Shared profiling composition for the baseline ingest stages."""

import asyncio
from dataclasses import dataclass
from enum import Enum

from inference_lab.ingest.baseline_contracts import (
    BaselineFailure,
    BaselineFailureSeverity,
)
from inference_lab.profiling import (
    FailureMapper,
    FailureMapping,
    ProfilingContext,
)

_INPUT_FAILURES = frozenset(
    {
        "invalid_request",
        "incompatible_resume",
        "corpus_not_found",
    }
)
_DEPENDENCY_FAILURES = frozenset(
    {
        "discovery_failed",
        "source_acquisition_failed",
        "parser_unavailable",
        "parse_persistence_failed",
        "chunk_persistence_failed",
        "storage_unavailable",
        "publication_concurrency",
    }
)


def _as_failure_code(error: BaseException) -> str | None:
    """Read a bounded domain category without retaining exception payloads."""
    category = getattr(error, "category", None)
    if isinstance(category, Enum) and isinstance(category.value, str):
        value = category.value
        if value and value.replace("_", "").isalnum():
            return value
    return None


def map_baseline_failure(error: BaseException) -> FailureMapping:
    """Map a known baseline failure to the closed profiling vocabulary."""
    if isinstance(error, asyncio.CancelledError):
        return FailureMapping(
            failure_category="cancelled",
            failure_code="stage_cancelled",
            status="cancelled",
        )

    failure_code = _as_failure_code(error) or "stage_failed"
    if failure_code in _INPUT_FAILURES:
        failure_category = "input"
    elif failure_code in _DEPENDENCY_FAILURES:
        failure_category = "dependency"
    else:
        failure_category = "internal"
    return FailureMapping(
        failure_category=failure_category,
        failure_code=failure_code,
    )


def map_baseline_result_failure(
    failure: BaselineFailure,
    *,
    partial: bool = False,
) -> FailureMapping:
    """Map a returned bounded ingest failure without exception payloads."""
    if not isinstance(failure, BaselineFailure):
        raise TypeError("failure must be a BaselineFailure.")
    if not isinstance(partial, bool):
        raise TypeError("partial must be a boolean.")
    code = failure.category.value
    if code in _INPUT_FAILURES:
        category = "input"
    elif code in _DEPENDENCY_FAILURES:
        category = "dependency"
    else:
        category = "internal"
    return FailureMapping(
        failure_category=category,
        failure_code=code,
        recoverable=(
            failure.retryable or failure.severity is BaselineFailureSeverity.RECOVERABLE
        ),
        partial=partial,
    )


@dataclass(frozen=True, slots=True)
class BaselineProfilingContext(ProfilingContext):
    """Compatibility context retaining baseline ingest defaults."""

    module: str = "inference_lab.ingest"
    failure_mapper: FailureMapper = map_baseline_failure
