"""Observed baseline-stage instrumentation and returned-outcome handling."""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Literal

from inference_lab.profiling._validation import is_snake_case
from inference_lab.profiling.contracts import (
    BaselineStage,
    StageCondition,
    validate_baseline_stage,
    validate_stage_condition,
)
from inference_lab.profiling.emission import ProfilingSink
from inference_lab.profiling.failures import FailureCategory, FailureEvent
from inference_lab.profiling.timing import StageTimer

FailureStatus = Literal["failed", "cancelled"]
_QUALIFIED_EXCEPTION_TYPE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*$"
)


@dataclass(frozen=True, slots=True)
class FailureMapping:
    """A bounded domain failure ready to be emitted for one stage."""

    failure_category: FailureCategory
    failure_code: str
    status: FailureStatus = "failed"
    recoverable: bool = False
    retry_count: int = 0
    partial: bool = False
    exception_type: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the failure mapping's fields for type and value."""
        if not isinstance(self.failure_category, str):
            raise TypeError("failure_category must be a string")
        if self.failure_category not in {
            "input",
            "dependency",
            "timeout",
            "resource_limit",
            "cancelled",
            "internal",
        }:
            raise ValueError("failure_category must be supported for a stage")
        if not isinstance(self.failure_code, str):
            raise TypeError("failure_code must be a string")
        if not self.failure_code or not is_snake_case(self.failure_code):
            raise ValueError("failure_code must use non-empty snake_case")
        if self.status not in {"failed", "cancelled"}:
            raise ValueError("status must be failed or cancelled")
        if self.status == "cancelled" and self.failure_category != "cancelled":
            raise ValueError("cancelled status must use the cancelled category")
        if self.failure_category == "cancelled" and self.status != "cancelled":
            raise ValueError("cancelled category must use cancelled status")
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be a boolean")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count must be a non-negative integer")
        if not isinstance(self.partial, bool):
            raise TypeError("partial must be a boolean")
        if self.exception_type is not None and (
            not isinstance(self.exception_type, str)
            or _QUALIFIED_EXCEPTION_TYPE.fullmatch(self.exception_type) is None
        ):
            raise ValueError("exception_type must be a qualified exception class name")


FailureMapper = Callable[[BaseException], FailureMapping]


def _default_failure_mapping(error: BaseException) -> FailureMapping:
    """Map an unexpected exception without retaining its message or payload."""
    if isinstance(error, asyncio.CancelledError):
        return FailureMapping(
            failure_category="cancelled",
            failure_code="stage_cancelled",
            status="cancelled",
            exception_type=_exception_type(error),
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="stage_failed",
        exception_type=_exception_type(error),
    )


def _exception_type(error: BaseException) -> str:
    """Return a privacy-safe module-qualified exception class name."""
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__name__}"


class ObservedStage:
    """Instrument one baseline stage while preserving its primary outcome.

    A normal context exit represents a successful stage, including intentional
    empty output, valid reuse, and no-change results. Callers mark a returned
    failure with :meth:`mark_failure`; raised exceptions are mapped by
    ``failure_mapper`` and re-raised unchanged. Counts are the caller's
    responsibility and remain the source of truth for reused and no-change
    distinctions.
    """

    def __init__(
        self,
        *,
        module: str,
        pipeline_stage: BaselineStage,
        stage_id: str,
        sink: ProfilingSink,
        condition: StageCondition = "uncontrolled",
        clock: Callable[[], int],
        observed_at_factory: Callable[[], datetime],
        run_id: str | None = None,
        request_id: str | None = None,
        document_id: str | None = None,
        corpus_id: str | None = None,
        corpus_version: str | None = None,
        parent_stage_id: str | None = None,
        failure_mapper: FailureMapper | None = None,
    ) -> None:
        """Configure an observed stage without starting it."""
        validate_baseline_stage(pipeline_stage)
        validate_stage_condition(condition)
        if not callable(getattr(sink, "emit", None)):
            raise TypeError("sink must provide emit")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(observed_at_factory):
            raise TypeError("observed_at_factory must be callable")
        if failure_mapper is not None and not callable(failure_mapper):
            raise TypeError("failure_mapper must be callable")

        self._sink = sink
        self._failure_mapper = failure_mapper or _default_failure_mapping
        self._timer = StageTimer(
            module=module,
            pipeline_stage=pipeline_stage,
            stage_id=stage_id,
            emit=sink.emit,
            clock=clock,
            observed_at_factory=observed_at_factory,
            run_id=run_id,
            request_id=request_id,
            document_id=document_id,
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            parent_stage_id=parent_stage_id,
            condition=condition,
        )
        self._failure: FailureMapping | None = None
        self._entered = False

    def __enter__(self) -> "ObservedStage":
        """Start the stage and emit its start record."""
        if self._entered:
            raise RuntimeError("ObservedStage instances cannot be reused")
        self._entered = True
        self._timer.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Emit terminal and failure records without masking stage failures."""
        if not self._entered:
            raise RuntimeError("ObservedStage must be entered before it can exit")

        if exc_type is not None:
            failure = self._map_exception(exc_value)
            self._timer.__exit__(exc_type, exc_value, traceback)
            self._emit_failure_best_effort(failure)
            return False

        if self._failure is not None:
            self._timer.set_terminal_status(self._failure.status)
            try:
                self._timer.__exit__(None, None, None)
            except BaseException:
                pass
            self._emit_failure_best_effort(self._failure)
            return False

        return self._timer.__exit__(None, None, None)

    def set_counts(
        self,
        *,
        items_processed: int | None = None,
        records_emitted: int | None = None,
    ) -> None:
        """Set deterministic aggregate counts for the terminal timing record."""
        self._timer.set_counts(
            items_processed=items_processed,
            records_emitted=records_emitted,
        )

    def mark_success(self) -> None:
        """Mark a returned success, reuse, empty result, or no-change result."""
        self._require_active()
        if self._failure is not None:
            raise RuntimeError("a stage cannot be marked successful after failure")

    def mark_failure(self, failure: FailureMapping) -> None:
        """Mark a typed returned failure for emission at context exit."""
        self._require_active()
        if not isinstance(failure, FailureMapping):
            raise TypeError("failure must be a FailureMapping")
        if self._failure is not None:
            raise RuntimeError("a stage failure has already been marked")
        self._failure = failure

    def _require_active(self) -> None:
        """Require that the stage is inside its context and still running."""
        if not self._entered:
            raise RuntimeError("stage must be entered before its outcome is marked")

    def _map_exception(self, error: BaseException | None) -> FailureMapping:
        """Map a raised exception, falling back without exposing its payload."""
        if error is None:
            return _default_failure_mapping(RuntimeError())
        try:
            failure = self._failure_mapper(error)
            if not isinstance(failure, FailureMapping):
                raise TypeError("failure mapper must return FailureMapping")
            if (
                isinstance(error, asyncio.CancelledError)
                and failure.status != "cancelled"
            ):
                return _default_failure_mapping(error)
            if (
                not isinstance(error, asyncio.CancelledError)
                and failure.status != "failed"
            ):
                return _default_failure_mapping(error)
            return failure
        except BaseException:
            return _default_failure_mapping(error)

    def _emit_failure_best_effort(self, failure: FailureMapping) -> None:
        """Emit one failure record while preserving the stage's primary result."""
        try:
            self._sink.emit(
                FailureEvent(
                    failure_category=failure.failure_category,
                    failure_code=failure.failure_code,
                    event="unit_failed",
                    module=self._timer.module,
                    pipeline_stage=self._timer.pipeline_stage,
                    stage_id=self._timer.stage_id,
                    run_id=self._timer.run_id,
                    request_id=self._timer.request_id,
                    document_id=self._timer.document_id,
                    corpus_id=self._timer.corpus_id,
                    corpus_version=self._timer.corpus_version,
                    status=failure.status,
                    level="warning" if failure.status == "cancelled" else "error",
                    recoverable=failure.recoverable,
                    retry_count=failure.retry_count,
                    partial=failure.partial,
                    exception_type=failure.exception_type,
                    observed_at=self._timer.observed_at_factory(),
                    record_schema_version=self._timer.record_schema_version,
                )
            )
        except BaseException:
            pass
