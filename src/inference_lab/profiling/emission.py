"""Emit profiling records through the application structured logger."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class ProfilingRecordLike(Protocol):
    """The serialization contract required by the profiling emitter."""

    def to_dict(self) -> dict[str, object]:
        """Return structured fields for one profiling record."""


_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})
_REQUIRED_FIELDS = (
    "timestamp",
    "module",
    "pipeline_stage",
    "document_id",
    "corpus_version",
)


def _default_level(fields: Mapping[str, object]) -> str:
    """Infer severity for record types without an explicit log level."""
    status = fields.get("status")
    if status in {"failed"}:
        return "error"
    if status in {"cancelled", "degraded", "unavailable", "unreliable"}:
        return "warning"
    return "info"


def emit_record(logger: Any, record: ProfilingRecordLike) -> None:
    """Emit one profiling record through a structlog-compatible logger.

    The adapter preserves the record's structured fields, supplies the
    conventional nullable context fields, and dispatches through the logger's
    severity method. It does not write files or create a second output stream.
    """
    fields = dict(record.to_dict())
    event = fields.pop("event", None)
    if not isinstance(event, str) or not event:
        raise ValueError("profiling records must provide a non-empty event")

    level = fields.get("level")
    if level is None:
        level = _default_level(fields)
        fields["level"] = level
    if not isinstance(level, str) or level not in _LOG_LEVELS:
        raise ValueError("profiling records must provide a valid log level")

    for field_name in _REQUIRED_FIELDS:
        fields.setdefault(field_name, None)

    log_method = getattr(logger, level, None)
    if not callable(log_method):
        raise TypeError(f"logger does not provide a {level} method")
    log_method(event, **fields)
