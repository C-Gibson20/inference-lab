"""Configure structured stdout logging for the application."""

import logging
import sys
from typing import Any, TextIO

import structlog

from inference_lab.settings import LoggingConfig

_HANDLER_MARKER = "_inference_lab_handler"
_CONTEXT_FIELDS = ("document_id", "corpus_version", "pipeline_stage")


def _add_record_fields(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add the stable fields required by the logging conventions."""
    event_dict["module"] = event_dict.pop("logger", event_dict.get("module"))
    for field in _CONTEXT_FIELDS:
        event_dict.setdefault(field, None)
    return event_dict


def _configure_structlog(
    format_name: str,
    *,
    colors: bool,
) -> structlog.stdlib.ProcessorFormatter:
    """Configure structlog and return the formatter for stdlib handlers."""
    renderer = (
        structlog.dev.ConsoleRenderer(colors=colors)
        if format_name == "console"
        else structlog.processors.JSONRenderer()
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
        _add_record_fields,
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    return formatter


def _configure_stdlib(
    formatter: logging.Formatter,
    config: LoggingConfig,
    stream: TextIO,
) -> logging.Handler:
    """Create the application-owned stdout handler."""
    handler = logging.StreamHandler(stream)
    handler.setLevel(config.level)
    handler.setFormatter(formatter)

    setattr(handler, _HANDLER_MARKER, True)

    return handler


def _configure_root_logger(handler: logging.Handler, level: str) -> None:
    """Replace the application-owned root handler without touching others."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for existing_handler in root_logger.handlers[:]:
        if getattr(existing_handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(existing_handler)
            existing_handler.close()

    root_logger.addHandler(handler)


def configure_logging(
    config: LoggingConfig,
    *,
    stream: TextIO | None = None,
) -> None:
    """Configure structured logging for the current process."""
    output_stream = stream if stream is not None else sys.stdout

    if config.format == "auto":
        format_name = "console" if output_stream.isatty() else "json"
    else:
        format_name = config.format

    formatter = _configure_structlog(
        format_name,
        colors=format_name == "console" and output_stream.isatty(),
    )
    handler = _configure_stdlib(formatter, config, stream=output_stream)
    _configure_root_logger(handler, config.level)
