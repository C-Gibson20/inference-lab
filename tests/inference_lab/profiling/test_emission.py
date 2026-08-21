"""Integration tests for structured profiling-record emission."""

import json
from datetime import UTC, datetime
from io import StringIO

import pytest
import structlog

from inference_lab.logging.logging_config import configure_logging
from inference_lab.profiling.emission import ProfilingRecordLike
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
            timestamp=fixed_timestamp(),
            duration_ms=2.5,
            status="completed",
        ),
        ResourceMeasurement(
            category="gpu_utilization",
            unit="percent",
            value=75.0,
            timestamp=fixed_timestamp(),
        ),
        FailureEvent(
            failure_category="input",
            failure_code="malformed_document",
            module="ingest",
            pipeline_stage="parse",
            timestamp=fixed_timestamp(),
        ),
    ],
)
def test_emits_each_record_as_structured_json(record: ProfilingRecordLike) -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)

    emit_record(structlog.get_logger("profiling"), record)
    payload = json.loads(stream.getvalue())

    assert isinstance(payload["event"], str)
    assert isinstance(payload["timestamp"], str)
    assert payload["timestamp"].endswith("Z")
    assert payload["module"] is not None
    assert "pipeline_stage" in payload
    assert "document_id" in payload
    assert "corpus_version" in payload
    assert payload["level"] in {"info", "warning", "error", "critical"}


def test_infers_warning_for_unavailable_measurement() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    emit_record(
        structlog.get_logger("profiling"),
        ResourceMeasurement(
            category="power",
            unit="watt",
            value=None,
            status="unavailable",
            timestamp=fixed_timestamp(),
        ),
    )

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "warning"


def test_infers_error_for_failed_timing_record() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    emit_record(
        structlog.get_logger("profiling"),
        StageTiming(
            event="stage_completed",
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            timestamp=fixed_timestamp(),
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
        emit_record(object(), InvalidRecord())
