"""Privacy-safe failure and degraded-unit records for profiling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from .types import LogLevel
from .utils import is_snake_case, utc_now
from .versioning import CURRENT_RECORD_SCHEMA_VERSION, validate_record_schema_version


FailureEventName = Literal["unit_failed", "unit_degraded"]
FailureCategory = Literal[
    "input",
    "dependency",
    "timeout",
    "resource_limit",
    "cancelled",
    "internal",
    "measurement_unavailable",
]
FailureStatus = Literal["failed", "cancelled", "degraded"]

_FAILURE_EVENTS = frozenset({"unit_failed", "unit_degraded"})
_FAILURE_CATEGORIES = frozenset(
    {
        "input",
        "dependency",
        "timeout",
        "resource_limit",
        "cancelled",
        "internal",
        "measurement_unavailable",
    }
)
_FAILURE_STATUSES = frozenset({"failed", "cancelled", "degraded"})
_FAILURE_LEVELS = frozenset({"warning", "error"})
_QUALIFIED_EXCEPTION_TYPE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*$"
)


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """A privacy-safe record describing a failed or degraded unit of work.

    The record intentionally stores only an exception type and machine-readable
    failure code. Raw exception messages, tracebacks, paths, document content,
    queries, prompts, and model outputs must be kept out of this record.
    """

    failure_category: FailureCategory
    failure_code: str
    event: FailureEventName = "unit_failed"
    module: str = "profiling"
    pipeline_stage: str | None = None
    stage_id: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    document_id: str | None = None
    corpus_version: str | None = None
    status: FailureStatus = "failed"
    level: LogLevel = "error"
    recoverable: bool = False
    retry_count: int = 0
    partial: bool = False
    exception_type: str | None = None
    observed_at: datetime = field(default_factory=utc_now)
    record_schema_version: int = CURRENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate failure identity, severity, status, and privacy fields."""
        if self.event not in _FAILURE_EVENTS:
            raise ValueError("event must be a valid failure event")
        if self.failure_category not in _FAILURE_CATEGORIES:
            raise ValueError("failure_category must be supported")
        if not self.failure_code or not is_snake_case(self.failure_code):
            raise ValueError("failure_code must use non-empty snake_case")
        if not self.module:
            raise ValueError("module must not be empty")
        if self.pipeline_stage == "":
            raise ValueError("pipeline_stage must not be empty")
        if self.stage_id == "":
            raise ValueError("stage_id must not be empty")
        if self.status not in _FAILURE_STATUSES:
            raise ValueError("status must be valid")
        if self.level not in _FAILURE_LEVELS:
            raise ValueError("failure level must be warning or error")
        if self.status == "degraded" and self.event != "unit_degraded":
            raise ValueError("degraded records must use unit_degraded")
        if self.status != "degraded" and self.event != "unit_failed":
            raise ValueError("failed and cancelled records must use unit_failed")
        if self.status == "degraded" and self.level != "warning":
            raise ValueError("degraded records must use warning level")
        if self.status == "failed" and self.level != "error":
            raise ValueError("failed records must use error level")
        if self.status == "cancelled" and self.level != "warning":
            raise ValueError("cancelled records must use warning level")
        if self.failure_category == "cancelled" and self.status != "cancelled":
            raise ValueError("cancelled category must use cancelled status")
        if self.failure_category == "measurement_unavailable":
            if self.status != "degraded":
                raise ValueError("measurement_unavailable must use degraded status")
        if isinstance(self.recoverable, bool) is False:
            raise ValueError("recoverable must be a boolean")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count must be a non-negative integer")
        if isinstance(self.partial, bool) is False:
            raise ValueError("partial must be a boolean")
        if self.exception_type is not None:
            if not _QUALIFIED_EXCEPTION_TYPE.fullmatch(self.exception_type):
                raise ValueError(
                    "exception_type must be a qualified exception class name"
                )
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware UTC")
        if self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be in UTC")
        validate_record_schema_version(self.record_schema_version)

    def to_dict(self) -> dict[str, object]:
        """Return the failure event as structured logging fields."""
        return {
            "event": self.event,
            "failure_category": self.failure_category,
            "failure_code": self.failure_code,
            "module": self.module,
            "pipeline_stage": self.pipeline_stage,
            "stage_id": self.stage_id,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "corpus_version": self.corpus_version,
            "status": self.status,
            "level": self.level,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "partial": self.partial,
            "exception_type": self.exception_type,
            "observed_at": self.observed_at.isoformat(),
            "record_schema_version": self.record_schema_version,
        }
