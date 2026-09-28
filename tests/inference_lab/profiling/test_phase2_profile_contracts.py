"""PROF-01 phase journeys for deterministic, privacy-safe measurements."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from io import StringIO

import pytest
import structlog

from inference_lab.logging.logging_config import configure_logging
from inference_lab.profiling import (
    FailureEvent,
    FailureMapping,
    ObservedStage,
    ResourceMeasurement,
    StageTiming,
    emit_record,
)
from inference_lab.settings import LoggingConfig

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, *values: int) -> None:
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


class Sink:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.records: list[object] = []
        self.fail_after = fail_after

    def emit(self, record: object) -> None:
        self.records.append(record)
        if self.fail_after == len(self.records):
            raise RuntimeError("sink failed")


def stage(sink: Sink) -> ObservedStage:
    return ObservedStage(
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        sink=sink,
        condition="cold",
        clock=Clock(1_000_000_000, 1_025_000_000),
        observed_at_factory=lambda: NOW,
        corpus_id="corpus-1",
        corpus_version="version-1",
    )


def test_observed_stage_records_success_failure_cancellation_and_sink_policy() -> None:
    successful = Sink()
    with stage(successful) as observed:
        observed.set_counts(items_processed=1, records_emitted=4)
        observed.mark_success()
    timings = [
        record for record in successful.records if isinstance(record, StageTiming)
    ]
    assert [(item.event, item.status) for item in timings] == [
        ("stage_started", "started"),
        ("stage_completed", "completed"),
    ]
    assert timings[-1].duration_ms == 25.0
    assert timings[-1].parent_stage_id is None
    assert timings[-1].corpus_version == "version-1"

    failed = Sink()
    sentinel = ValueError("private source payload")
    with pytest.raises(ValueError) as raised:
        with ObservedStage(
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-2",
            sink=failed,
            clock=Clock(0, 1_000_000),
            observed_at_factory=lambda: NOW,
            failure_mapper=lambda _error: FailureMapping(
                failure_category="input",
                failure_code="malformed_source",
                exception_type="builtins.ValueError",
            ),
        ):
            raise sentinel
    assert raised.value is sentinel
    assert "private source payload" not in repr(failed.records)
    assert any(isinstance(record, FailureEvent) for record in failed.records)

    cancelled = Sink()
    with pytest.raises(asyncio.CancelledError):
        with stage(cancelled):
            raise asyncio.CancelledError
    failures = [
        record for record in cancelled.records if isinstance(record, FailureEvent)
    ]
    assert len(failures) == 1
    assert (failures[0].status, failures[0].level) == ("cancelled", "warning")

    failing_sink = Sink(fail_after=2)
    with pytest.raises(RuntimeError, match="sink failed"):
        with stage(failing_sink):
            pass

    primary_sink_failure = Sink(fail_after=2)
    primary = ValueError("stage failure")
    with pytest.raises(ValueError) as raised:
        with stage(primary_sink_failure):
            raise primary
    assert raised.value is primary


def test_resource_records_keep_units_statuses_and_time_boundaries_strict() -> None:
    expected_units = {
        "model_weights": "bytes",
        "gpu_utilisation": "percent",
        "storage_throughput": "bytes_per_second",
        "power": "watts",
        "thermal": "celsius",
    }
    for category, unit in expected_units.items():
        assert ResourceMeasurement(  # type: ignore[arg-type]
            category=category, value=1
        ).unit == unit

    unavailable = ResourceMeasurement(
        category="power", value=None, status="unavailable", observed_at=NOW
    )
    unreliable = ResourceMeasurement(
        category="thermal", value=-20, status="unreliable", observed_at=NOW
    )
    sampled = ResourceMeasurement(
        category="cpu_utilisation",
        value=50,
        measurement_kind="sampled",
        sample_duration_ms=1000,
        observed_at=NOW,
    )
    assert unavailable.value is None
    assert unreliable.value == -20
    assert sampled.sample_duration_ms == 1000
    with pytest.raises(ValueError):
        ResourceMeasurement(category="gpu_utilisation", value=101)
    with pytest.raises(ValueError):
        ResourceMeasurement(category="power", value=float("nan"))
    with pytest.raises(ValueError):
        ResourceMeasurement(
            category="cpu_utilisation", value=10, measurement_kind="sampled"
        )
    with pytest.raises(ValueError):
        ResourceMeasurement(
            category="power",
            value=1,
            observed_at=NOW.replace(tzinfo=None),
        )


def test_failure_record_discriminants_and_serialised_exception_are_safe() -> None:
    record = FailureEvent(
        failure_category="input",
        failure_code="malformed_document",
        module="ingest",
        pipeline_stage="parse",
        stage_id="parse-1",
        status="failed",
        level="error",
        retry_count=1,
        exception_type="builtins.ValueError",
        observed_at=NOW,
    ).to_dict()
    assert record["record_schema_version"]
    assert record["exception_type"] == "builtins.ValueError"
    private_fields = {"message", "traceback", "query", "prompt", "model_output"}
    assert not private_fields & record.keys()
    for invalid in (
        {"failure_code": "BadCode"},
        {"failure_category": "unknown"},
        {"retry_count": -1},
        {"status": "degraded", "event": "unit_failed", "level": "warning"},
    ):
        values: dict[str, object] = {
            "failure_category": "internal",
            "failure_code": "bad_input",
        }
        values.update(invalid)
        with pytest.raises((TypeError, ValueError)):
            FailureEvent(**values)  # type: ignore[arg-type]


def test_records_emit_with_observation_time_context_and_severity() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    records = (
        StageTiming(
            event="stage_completed",
            module="ingest",
            pipeline_stage="parse",
            stage_id="parse-1",
            observed_at=NOW,
            duration_ms=2.5,
            status="completed",
        ),
        ResourceMeasurement(
            category="power", value=None, status="unavailable", observed_at=NOW
        ),
        FailureEvent(
            failure_category="input", failure_code="bad_input", observed_at=NOW
        ),
    )
    for record in records:
        emit_record(structlog.get_logger("profiling"), record)
    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [item["level"] for item in payloads] == ["info", "warning", "error"]
    assert all(item["observed_at"] == NOW.isoformat() for item in payloads)
    assert all(item["record_schema_version"] for item in payloads)
