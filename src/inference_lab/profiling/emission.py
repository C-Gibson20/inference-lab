"""Emit profiling records through the application structured logger."""

from typing import Final, Protocol, get_args

from inference_lab.profiling.types import LogLevel


class ProfilingRecordLike(Protocol):
    """The serialisation contract required by the profiling emitter."""

    def to_dict(self) -> dict[str, object]:
        """Return structured fields for one profiling record."""
        ...


class ProfilingSink(Protocol):
    """Receive one immutable profiling record."""

    def emit(self, record: ProfilingRecordLike) -> None:
        """Deliver a profiling record to one consumer."""
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


class StructuredLogSink:
    """Adapt the shared structured logger to the profiling sink contract."""

    def __init__(self, logger: ProfilingLogger) -> None:
        """Create a sink that emits through the existing logging adapter."""
        if not all(callable(getattr(logger, level, None)) for level in _LOG_LEVELS):
            raise TypeError("logger must implement the profiling logger contract")
        self._logger = logger

    def emit(self, record: ProfilingRecordLike) -> None:
        """Emit one record through ``emit_record``."""
        emit_record(self._logger, record)


class FanoutSink:
    """Deliver each record to ordered consumers and report consumer failure."""

    def __init__(self, *sinks: ProfilingSink) -> None:
        """Create a non-empty, stable-order sink fan-out."""
        if not sinks:
            raise ValueError("at least one profiling sink is required")
        if any(not callable(getattr(sink, "emit", None)) for sink in sinks):
            raise TypeError("every profiling sink must provide emit")
        self._sinks = tuple(sinks)

    def emit(self, record: ProfilingRecordLike) -> None:
        """Deliver to every sink, then raise the first consumer failure."""
        first_error: BaseException | None = None
        for sink in self._sinks:
            try:
                sink.emit(record)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
