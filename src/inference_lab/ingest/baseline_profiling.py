"""Shared profiling composition for the baseline ingest stages."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4

import structlog

from inference_lab._clock import utc_now
from inference_lab.profiling import (
    BaselineStage as ProfilingBaselineStage,
)
from inference_lab.profiling import (
    FailureMapping,
    ObservedStage,
    ProfilingSink,
    StructuredLogSink,
)

_LOGGER = structlog.get_logger("inference_lab.profiling")

Clock = Callable[[], int]
ObservationTimeFactory = Callable[[], datetime]
RunIdFactory = Callable[[], str]

_INPUT_FAILURES = frozenset(
    {
        "invalid_request",
        "incompatible_resume",
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
    }
)


def _production_run_id() -> str:
    """Create an opaque run identity for one production invocation."""
    return uuid4().hex


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


@dataclass(frozen=True, slots=True)
class BaselineProfilingContext:
    """Carry operational correlation and one shared profiling sink."""

    run_id: str
    sink: ProfilingSink
    request_id: str | None = None
    condition: str = "uncontrolled"
    clock: Clock = time.monotonic_ns
    observed_at_factory: ObservationTimeFactory = utc_now

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the context's fields after initialization."""
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string.")
        if not callable(getattr(self.sink, "emit", None)):
            raise TypeError("sink must provide emit.")
        if self.request_id is not None and (
            not isinstance(self.request_id, str) or not self.request_id
        ):
            raise ValueError("request_id must be a non-empty string or None.")
        if self.condition not in {"cold", "warm", "uncontrolled"}:
            raise ValueError("condition must be cold, warm, or uncontrolled.")
        if not callable(self.clock):
            raise TypeError("clock must be callable.")
        if not callable(self.observed_at_factory):
            raise TypeError("observed_at_factory must be callable.")

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None = None,
        run_id_factory: RunIdFactory = _production_run_id,
        sink: ProfilingSink | None = None,
        request_id: str | None = None,
        condition: str = "uncontrolled",
        clock: Clock = time.monotonic_ns,
        observed_at_factory: ObservationTimeFactory = utc_now,
    ) -> "BaselineProfilingContext":
        """Create a production context or a deterministic test context."""
        selected_run_id = run_id if run_id is not None else run_id_factory()
        selected_sink = sink if sink is not None else StructuredLogSink(_LOGGER)
        return cls(
            run_id=selected_run_id,
            sink=selected_sink,
            request_id=request_id,
            condition=condition,
            clock=clock,
            observed_at_factory=observed_at_factory,
        )

    def stage_id(self, stage: str, identity: str) -> str:
        """Derive a stable, payload-free stage identity within this run."""
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string.")
        if not isinstance(identity, str) or not identity:
            raise ValueError("identity must be a non-empty string.")
        return f"{self.run_id}:{stage}:{identity}"

    def observed_stage(
        self,
        *,
        stage: ProfilingBaselineStage,
        stage_id: str,
        document_id: str | None,
        corpus_id: str,
        corpus_version: str | None,
        parent_stage_id: str | None,
    ) -> ObservedStage:
        """Build one observed stage sharing this context's clocks and sink."""
        return ObservedStage(
            module="inference_lab.ingest",
            pipeline_stage=stage,
            stage_id=stage_id,
            sink=self.sink,
            condition=self.condition,  # type: ignore[arg-type]
            clock=self.clock,
            observed_at_factory=self.observed_at_factory,
            run_id=self.run_id,
            request_id=self.request_id,
            document_id=document_id,
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            parent_stage_id=parent_stage_id,
            failure_mapper=map_baseline_failure,
        )
