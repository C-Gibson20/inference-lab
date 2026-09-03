"""Integration tests for structured profiling-record emission."""

import json
from datetime import UTC, datetime
from io import StringIO

import pytest
import structlog

from inference_lab.logging.logging_config import configure_logging
from inference_lab.profiling import (
    FailureEvent,
    ResourceMeasurement,
    StageTiming,
    emit_record,
)
from inference_lab.settings import LoggingConfig


def fixed_timestamp() -> datetime:
    """Return a deterministic UTC timestamp for emitted records."""
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "record",
    [
        StageTiming(
            event="stage_completed",
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            observed_at=fixed_timestamp(),
            duration_ms=2.5,
            status="completed",
        ),
        ResourceMeasurement(
            category="gpu_utilisation",
            value=75.0,
            observed_at=fixed_timestamp(),
        ),
        FailureEvent(
            failure_category="input",
            failure_code="malformed_document",
            module="ingest",
            pipeline_stage="parse",
            observed_at=fixed_timestamp(),
        ),
    ],
)
def test_emits_each_record_as_structured_json(
    record: StageTiming | ResourceMeasurement | FailureEvent,
) -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)

    emit_record(structlog.get_logger("profiling"), record)
    payload = json.loads(stream.getvalue())

    assert isinstance(payload["event"], str)
    assert isinstance(payload["timestamp"], str)
    assert payload["timestamp"].endswith("Z")
    assert payload["observed_at"] == "2026-08-21T12:00:00+00:00"
    assert payload["module"] is not None
    assert "pipeline_stage" in payload
    assert "document_id" in payload
    assert "corpus_id" in payload
    assert "corpus_version" in payload
    assert payload["level"] in {"info", "warning", "error", "critical"}
    assert payload["record_schema_version"] == "3.0.0"


def test_emitted_record_preserves_context_and_nullable_fields() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)

    emit_record(
        structlog.get_logger("profiling"),
        ResourceMeasurement(
            category="kv_cache",
            value=4.0,
            observed_at=fixed_timestamp(),
            module="runtime",
            pipeline_stage="generate",
            stage_id="generate-1",
            run_id="run-1",
            request_id="request-1",
            document_id="document-1",
            corpus_id="corpus-1",
            corpus_version="corpus-version-1",
        ),
    )

    payload = json.loads(stream.getvalue())

    assert payload["module"] == "runtime"
    assert payload["pipeline_stage"] == "generate"
    assert payload["stage_id"] == "generate-1"
    assert payload["run_id"] == "run-1"
    assert payload["request_id"] == "request-1"
    assert payload["document_id"] == "document-1"
    assert payload["corpus_id"] == "corpus-1"
    assert payload["corpus_version"] == "corpus-version-1"


def test_failure_output_contains_no_raw_payload_fields() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)

    emit_record(
        structlog.get_logger("profiling"),
        FailureEvent(
            failure_category="internal",
            failure_code="backend_failed",
            exception_type="RuntimeError",
            observed_at=fixed_timestamp(),
        ),
    )

    payload = json.loads(stream.getvalue())

    assert payload["exception_type"] == "RuntimeError"
    assert "message" not in payload
    assert "traceback" not in payload
    assert "query" not in payload
    assert "prompt" not in payload
    assert "model_output" not in payload
    assert "secret" not in payload


def test_emits_warning_for_unavailable_measurement() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    emit_record(
        structlog.get_logger("profiling"),
        ResourceMeasurement(
            category="power",
            value=None,
            status="unavailable",
            observed_at=fixed_timestamp(),
        ),
    )

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "warning"


def test_emits_error_for_failed_timing_record() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    emit_record(
        structlog.get_logger("profiling"),
        StageTiming(
            event="stage_completed",
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            observed_at=fixed_timestamp(),
            duration_ms=2.5,
            status="failed",
        ),
    )

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "error"


def test_rejects_record_without_event() -> None:
    class InvalidRecord:
        def to_dict(self) -> dict[str, object]:
            return {"module": "profiling"}

    with pytest.raises(ValueError, match="non-empty event"):
        emit_record(object(), InvalidRecord())  # type: ignore[arg-type]


def test_rejects_record_without_explicit_log_level() -> None:
    class InvalidRecord:
        def to_dict(self) -> dict[str, object]:
            return {"event": "custom_event", "module": "profiling"}

    with pytest.raises(ValueError, match="explicit log level"):
        emit_record(object(), InvalidRecord())  # type: ignore[arg-type]


@pytest.mark.parametrize("level", ["notice", 10, True])
def test_rejects_record_with_invalid_log_level(level: object) -> None:
    class InvalidRecord:
        def to_dict(self) -> dict[str, object]:
            return {
                "event": "custom_event",
                "level": level,
                "module": "profiling",
            }

    with pytest.raises(ValueError, match="valid log level"):
        emit_record(object(), InvalidRecord())  # type: ignore[arg-type]


def test_rejects_logger_without_required_severity_method() -> None:
    class ValidRecord:
        def to_dict(self) -> dict[str, object]:
            return {
                "event": "custom_event",
                "level": "info",
                "module": "profiling",
            }

    with pytest.raises(TypeError, match="required info method"):
        emit_record(object(), ValidRecord())  # type: ignore[arg-type]
