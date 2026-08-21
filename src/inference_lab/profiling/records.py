"""Profiling records for inference-lab."""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal


_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    record_schema_version: int = 1

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the record fields."""
        if not self.event:
            raise ValueError("event must not be empty")

        if not self.module:
            raise ValueError("module must not be empty")

        if not _SNAKE_CASE.fullmatch(self.event):
            raise ValueError("event must use snake_case")

        if self.level not in _LOG_LEVELS:
            raise ValueError(f"unsupported log level: {self.level!r}")

        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")

        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be in UTC")

        if self.record_schema_version < 1:
            raise ValueError("record_schema_version must be positive")

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
            "timestamp": self.timestamp.isoformat(),
            "record_schema_version": self.record_schema_version,
        }
