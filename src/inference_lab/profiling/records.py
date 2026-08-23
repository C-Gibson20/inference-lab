"""Profiling records for inference-lab."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from .utils import is_snake_case, utc_now
from .versioning import CURRENT_RECORD_SCHEMA_VERSION, validate_record_schema_version


_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


LogLevel = Literal[
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(frozen=True, slots=True)
class ProfilingRecord:
    """A record of a profiling event."""

    event: str
    module: str
    level: LogLevel = "info"
    pipeline_stage: str | None = None
    document_id: str | None = None
    corpus_version: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    stage_id: str | None = None
    parent_stage_id: str | None = None
    observed_at: datetime = field(default_factory=utc_now)
    record_schema_version: int = CURRENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the record fields."""
        if not self.event:
            raise ValueError("event must not be empty")

        if not self.module:
            raise ValueError("module must not be empty")

        if not is_snake_case(self.event):
            raise ValueError("event must use snake_case")

        if self.level not in _LOG_LEVELS:
            raise ValueError(f"unsupported log level: {self.level!r}")

        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware UTC")

        if self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be in UTC")

        validate_record_schema_version(self.record_schema_version)

    def to_dict(self) -> dict[str, object]:
        """Return the record as a structured-logging field dictionary."""
        return {
            "event": self.event,
            "module": self.module,
            "level": self.level,
            "pipeline_stage": self.pipeline_stage,
            "document_id": self.document_id,
            "corpus_version": self.corpus_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "stage_id": self.stage_id,
            "parent_stage_id": self.parent_stage_id,
            "observed_at": self.observed_at.isoformat(),
            "record_schema_version": self.record_schema_version,
        }
