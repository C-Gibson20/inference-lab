"""Emit profiling records through the application structured logger."""

from typing import Final, Protocol, get_args

from inference_lab.profiling.types import LogLevel


class ProfilingRecordLike(Protocol):
    """The serialisation contract required by the profiling emitter."""

    def to_dict(self) -> dict[str, object]:
        """Return structured fields for one profiling record."""
        ...


class ProfilingLogger(Protocol):
    """The structured severity methods required by the profiling emitter."""

    def debug(self, event: str, **fields: object) -> object:
        """Emit a debug record."""
        ...

    def info(self, event: str, **fields: object) -> object:
        """Emit an informational record."""
        ...

    def warning(self, event: str, **fields: object) -> object:
        """Emit a warning record."""
        ...

    def error(self, event: str, **fields: object) -> object:
        """Emit an error record."""
        ...

    def critical(self, event: str, **fields: object) -> object:
        """Emit a critical record."""
        ...


_LOG_LEVELS: Final[frozenset[LogLevel]] = frozenset(get_args(LogLevel))
_REQUIRED_FIELDS = (
    "observed_at",
    "module",
    "pipeline_stage",
    "document_id",
    "corpus_id",
    "corpus_version",
)


def emit_record(logger: ProfilingLogger, record: ProfilingRecordLike) -> None:
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
        raise ValueError("profiling records must provide an explicit log level")
    if not isinstance(level, str) or level not in _LOG_LEVELS:
        raise ValueError("profiling records must provide a valid log level")

    for field_name in _REQUIRED_FIELDS:
        fields.setdefault(field_name, None)

    log_method = getattr(logger, level, None)
    if not callable(log_method):
        raise TypeError(f"logger does not provide the required {level} method")
    log_method(event, **fields)
