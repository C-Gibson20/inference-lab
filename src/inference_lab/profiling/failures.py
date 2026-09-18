"""Privacy-safe failure and degraded-unit records for profiling."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal, get_args

from inference_lab._clock import utc_now
from inference_lab.profiling._validation import (
    is_snake_case,
    validate_optional_correlation,
    validate_utc_datetime,
)
from inference_lab.profiling.types import LogLevel
from inference_lab.profiling.versioning import (
    CURRENT_RECORD_SCHEMA_VERSION,
    validate_record_schema_version,
)

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

_FAILURE_EVENTS: Final[frozenset[FailureEventName]] = frozenset(
    get_args(FailureEventName)
)
_FAILURE_CATEGORIES: Final[frozenset[FailureCategory]] = frozenset(
    get_args(FailureCategory)
)
_FAILURE_STATUSES: Final[frozenset[FailureStatus]] = frozenset(get_args(FailureStatus))
_FAILURE_LEVELS = frozenset({"warning", "error"})
_QUALIFIED_EXCEPTION_TYPE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*$"
)


def _validate_identity_and_context(
    *,
    event: FailureEventName,
    failure_category: FailureCategory,
    failure_code: str,
    module: str,
    pipeline_stage: str | None,
    stage_id: str | None,
) -> None:
    """Validate failure discriminants and operational context fields."""
    if event not in _FAILURE_EVENTS:
        raise ValueError("event must be a valid failure event")
    if failure_category not in _FAILURE_CATEGORIES:
        raise ValueError("failure_category must be supported")
    if not failure_code or not is_snake_case(failure_code):
        raise ValueError("failure_code must use non-empty snake_case")
    if not isinstance(module, str) or not module:
        raise ValueError("module must not be empty")
    if pipeline_stage is not None and (
        not isinstance(pipeline_stage, str) or not is_snake_case(pipeline_stage)
    ):
        if pipeline_stage == "":
            raise ValueError("pipeline_stage must not be empty")
        raise ValueError("pipeline_stage must use non-empty snake_case")
    for field_name, value in (
        ("pipeline_stage", pipeline_stage),
        ("stage_id", stage_id),
    ):
        validate_optional_correlation(value, field_name)


def _validate_state_relationships(
    *,
    event: FailureEventName,
    failure_category: FailureCategory,
    status: FailureStatus,
    level: LogLevel,
) -> None:
    """Validate relationships among failure category, event, status, and level."""
    if status not in _FAILURE_STATUSES:
        raise ValueError("status must be valid")
    if level not in _FAILURE_LEVELS:
        raise ValueError("failure level must be warning or error")
    if status == "degraded" and event != "unit_degraded":
        raise ValueError("degraded records must use unit_degraded")
    if status != "degraded" and event != "unit_failed":
        raise ValueError("failed and cancelled records must use unit_failed")
    if status == "degraded" and level != "warning":
        raise ValueError("degraded records must use warning level")
    if status == "failed" and level != "error":
        raise ValueError("failed records must use error level")
    if status == "cancelled" and level != "warning":
        raise ValueError("cancelled records must use warning level")
    if failure_category == "cancelled" and status != "cancelled":
        raise ValueError("cancelled category must use cancelled status")
    if failure_category == "measurement_unavailable" and status != "degraded":
        raise ValueError("measurement_unavailable must use degraded status")


def _validate_retry_fields(
    *,
    recoverable: bool,
    retry_count: int,
    partial: bool,
) -> None:
    """Validate recovery, retry, and partial-result fields."""
    if isinstance(recoverable, bool) is False:
        raise ValueError("recoverable must be a boolean")
    if (
        isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or retry_count < 0
    ):
        raise ValueError("retry_count must be a non-negative integer")
    if isinstance(partial, bool) is False:
        raise ValueError("partial must be a boolean")


def _validate_exception_type(exception_type: str | None) -> None:
    """Validate privacy-safe exception type metadata when present."""
    if exception_type is not None and not _QUALIFIED_EXCEPTION_TYPE.fullmatch(
        exception_type
    ):
        raise ValueError("exception_type must be a qualified exception class name")


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
    record_schema_version: str = CURRENT_RECORD_SCHEMA_VERSION
    corpus_id: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate failure identity, severity, status, and privacy fields."""
        _validate_identity_and_context(
            event=self.event,
            failure_category=self.failure_category,
            failure_code=self.failure_code,
            module=self.module,
            pipeline_stage=self.pipeline_stage,
            stage_id=self.stage_id,
        )
        _validate_state_relationships(
            event=self.event,
            failure_category=self.failure_category,
            status=self.status,
            level=self.level,
        )
        _validate_retry_fields(
            recoverable=self.recoverable,
            retry_count=self.retry_count,
            partial=self.partial,
        )
        _validate_exception_type(self.exception_type)
        for field_name in (
            "run_id",
            "request_id",
            "document_id",
            "corpus_id",
            "corpus_version",
        ):
            validate_optional_correlation(getattr(self, field_name), field_name)
        validate_utc_datetime(self.observed_at, "observed_at")
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
            "corpus_id": self.corpus_id,
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
