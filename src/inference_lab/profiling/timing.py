"""Deterministic stage timing records and context-manager instrumentation."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Self

from .records import LogLevel
from .utils import utc_now, validate_record_schema_version


Clock = Callable[[], int]
TimestampFactory = Callable[[], datetime]
RecordEmitter = Callable[["StageTiming"], None]
Event = Literal["stage_started", "stage_completed"]
Status = Literal["started", "completed", "failed", "cancelled"]

_EVENTS = frozenset({"stage_started", "stage_completed"})
_STATUSES = frozenset({"started", "completed", "failed", "cancelled"})


def _validate_non_negative_integer(value: int | None, field_name: str) -> None:
    """Validate an optional non-negative integer measurement."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_non_negative_duration(
    value: int | float | None,
    field_name: str,
) -> None:
    """Validate an optional finite, non-negative duration."""
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class StageTiming:
    """A structured start or completion record for one pipeline stage."""

    event: Event
    module: str
    pipeline_stage: str
    stage_id: str
    timestamp: datetime = field(default_factory=utc_now)
    duration_ms: float | None = None
    items_processed: int | None = None
    records_emitted: int | None = None
    status: Status = "started"
    run_id: str | None = None
    request_id: str | None = None
    document_id: str | None = None
    corpus_version: str | None = None
    parent_stage_id: str | None = None
    record_schema_version: int = 1

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the timing record invariants."""
        if self.event not in _EVENTS:
            raise ValueError("event must be a valid stage event")
        if not self.module:
            raise ValueError("module must not be empty")
        if not self.pipeline_stage:
            raise ValueError("pipeline_stage must not be empty")
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
        if self.status not in _STATUSES:
            raise ValueError("status must be a valid stage status")
        if self.event == "stage_started" and self.status != "started":
            raise ValueError("stage_started records must have started status")
        if self.event == "stage_completed" and self.status == "started":
            raise ValueError("stage_completed records must have a terminal status")
        if self.event == "stage_started" and self.duration_ms is not None:
            raise ValueError("stage_started records must not have a duration")
        if self.event == "stage_completed" and self.duration_ms is None:
            raise ValueError("stage_completed records must have a duration")
        _validate_non_negative_duration(self.duration_ms, "duration_ms")
        _validate_non_negative_integer(self.items_processed, "items_processed")
        _validate_non_negative_integer(self.records_emitted, "records_emitted")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be in UTC")
        validate_record_schema_version(self.record_schema_version)

    def to_dict(self) -> dict[str, object]:
        """Return the timing record as structured logging fields."""
        return {
            "event": self.event,
            "module": self.module,
            "level": self.level,
            "pipeline_stage": self.pipeline_stage,
            "stage_id": self.stage_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "items_processed": self.items_processed,
            "records_emitted": self.records_emitted,
            "status": self.status,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "corpus_version": self.corpus_version,
            "parent_stage_id": self.parent_stage_id,
            "record_schema_version": self.record_schema_version,
        }

    @property
    def level(self) -> LogLevel:
        """Return the logging severity implied by the stage status."""
        if self.status == "failed":
            return "error"
        if self.status == "cancelled":
            return "warning"
        return "info"


class StageTimer:
    """Emit deterministic timing records around one stage execution.

    The timer measures elapsed time with a monotonic nanosecond clock while
    records receive UTC timestamps from a separate clock. Exceptions are
    recorded and then propagated to the caller.
    """

    def __init__(
        self,
        *,
        module: str,
        pipeline_stage: str,
        stage_id: str,
        emit: RecordEmitter,
        clock: Clock = time.monotonic_ns,
        timestamp_factory: TimestampFactory = utc_now,
        run_id: str | None = None,
        request_id: str | None = None,
        document_id: str | None = None,
        corpus_version: str | None = None,
        parent_stage_id: str | None = None,
    ) -> None:
        """Configure a timer without starting the stage."""
        if not callable(emit):
            raise TypeError("emit must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(timestamp_factory):
            raise TypeError("timestamp_factory must be callable")

        self._module = module
        self._pipeline_stage = pipeline_stage
        self._stage_id = stage_id
        self._emit = emit
        self._clock = clock
        self._timestamp_factory = timestamp_factory
        self._run_id = run_id
        self._request_id = request_id
        self._document_id = document_id
        self._corpus_version = corpus_version
        self._parent_stage_id = parent_stage_id
        self._start_ns: int | None = None
        self._items_processed: int | None = None
        self._records_emitted: int | None = None
        self._entered = False
        self._finished = False

    def __enter__(self) -> Self:
        """Start timing and emit the stage-started record."""
        if self._entered:
            raise RuntimeError("StageTimer instances cannot be reused")

        self._entered = True
        self._start_ns = self._clock()
        self._emit(self._build_record("stage_started", "started"))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        """Emit completion timing and propagate any stage exception."""
        del exc_value, traceback
        if not self._entered or self._start_ns is None:
            raise RuntimeError("StageTimer must be entered before it can exit")
        if self._finished:
            raise RuntimeError("StageTimer has already finished")

        duration_ms = (self._clock() - self._start_ns) / 1_000_000
        if exc_type is None:
            status: Status = "completed"
        elif issubclass(exc_type, asyncio.CancelledError):
            status = "cancelled"
        else:
            status = "failed"

        self._finished = True
        self._emit(self._build_record("stage_completed", status, duration_ms))
        return False

    def set_counts(
        self,
        *,
        items_processed: int | None = None,
        records_emitted: int | None = None,
    ) -> None:
        """Set the counters emitted with the completion record."""
        if not self._entered or self._finished:
            raise RuntimeError("counts can only be set while the timer is active")
        _validate_non_negative_integer(items_processed, "items_processed")
        _validate_non_negative_integer(records_emitted, "records_emitted")
        self._items_processed = items_processed
        self._records_emitted = records_emitted

    def _build_record(
        self,
        event: Event,
        status: Status,
        duration_ms: float | None = None,
    ) -> StageTiming:
        """Build a timing record from the timer's shared context."""
        return StageTiming(
            event=event,
            module=self._module,
            pipeline_stage=self._pipeline_stage,
            stage_id=self._stage_id,
            timestamp=self._timestamp_factory(),
            duration_ms=duration_ms,
            items_processed=(
                self._items_processed if event == "stage_completed" else None
            ),
            records_emitted=(
                self._records_emitted if event == "stage_completed" else None
            ),
            status=status,
            run_id=self._run_id,
            request_id=self._request_id,
            document_id=self._document_id,
            corpus_version=self._corpus_version,
            parent_stage_id=self._parent_stage_id,
        )
