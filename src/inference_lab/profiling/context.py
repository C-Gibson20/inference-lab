"""Shared profiling context composition for pipeline owners."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import uuid4

import structlog

from inference_lab._clock import utc_now
from inference_lab.profiling.contracts import BaselineStage
from inference_lab.profiling.emission import (
    ProfilingSink,
    StructuredLogSink,
)
from inference_lab.profiling.observed import FailureMapper, ObservedStage

_LOGGER = structlog.get_logger("inference_lab.profiling")

Clock = Callable[[], int]
ObservationTimeFactory = Callable[[], datetime]
RunIdFactory = Callable[[], str]


def _production_run_id() -> str:
    """Create an opaque run identity for one production invocation."""
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class ProfilingContext:
    """Carry operational correlation and one shared profiling sink."""

    run_id: str
    sink: ProfilingSink
    request_id: str | None = None
    condition: str = "uncontrolled"
    clock: Clock = time.monotonic_ns
    observed_at_factory: ObservationTimeFactory = utc_now
    module: str = "inference_lab"
    failure_mapper: FailureMapper | None = None

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
        if not isinstance(self.module, str) or not self.module:
            raise ValueError("module must be a non-empty string.")
        if self.failure_mapper is not None and not callable(self.failure_mapper):
            raise TypeError("failure_mapper must be callable or None.")

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
    ) -> Self:
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
        stage: BaselineStage,
        stage_id: str,
        document_id: str | None,
        corpus_id: str,
        corpus_version: str | None,
        parent_stage_id: str | None,
        module: str | None = None,
        failure_mapper: FailureMapper | None = None,
    ) -> ObservedStage:
        """Build one observed stage sharing this context's clocks and sink."""
        return ObservedStage(
            module=self.module if module is None else module,
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
            failure_mapper=(
                self.failure_mapper if failure_mapper is None else failure_mapper
            ),
        )
