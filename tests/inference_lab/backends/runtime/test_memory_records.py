import json
from datetime import UTC, datetime
from io import StringIO

import structlog

from inference_lab.backends.runtime import (
    ComponentRole,
    MemoryAdapterIdentity,
    MemoryBudget,
    MemoryMeasurement,
    MemorySnapshot,
    MemorySpace,
    emit_memory_budget_comparison,
    emit_memory_measurement,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")


def make_measurement(
    measurement_id: str = "weights",
    *,
    status: str = "measured",
    value_bytes: int | None = 1024,
) -> MemoryMeasurement:
    return MemoryMeasurement(
        measurement_id=measurement_id,
        component_id="generator",
        component_role=ComponentRole.GENERATOR,
        category="model_weights",
        memory_space=MemorySpace.ACCELERATOR,
        device_id="cuda-0",
        status=status,  # type: ignore[arg-type]
        value_bytes=value_bytes,
        adapter=ADAPTER,
        observed_at=OBSERVED_AT,
    )


def make_snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        observed_at=OBSERVED_AT,
        adapter=ADAPTER,
        measurements=(make_measurement("z-last"), make_measurement("a-first")),
        run_id="run-1",
        request_id="request-1",
    )


def test_emission_uses_safe_runtime_records_and_required_fields() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    measurement = make_measurement()
    comparison = make_snapshot().compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 2048)
    )

    emit_memory_measurement(structlog.get_logger("runtime"), measurement)
    emit_memory_budget_comparison(
        structlog.get_logger("runtime"),
        comparison,
        observed_at=OBSERVED_AT,
    )
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert lines[0]["event"] == "memory_component_measured"
    assert lines[0]["level"] == "info"
    assert lines[1]["event"] == "memory_budget_compared"
    assert lines[1]["level"] == "info"
    for line in lines:
        assert line["module"] == "runtime"
        assert line["pipeline_stage"] == "memory_accounting"
        assert line["document_id"] is None
        assert line["corpus_id"] is None
        assert line["corpus_version"] is None
        assert line["record_schema_version"] == 3
        assert line["timestamp"].endswith("Z")


def test_emission_contains_no_canary_payloads() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    emit_memory_measurement(structlog.get_logger("runtime"), make_measurement())

    output = stream.getvalue()
    for canary in (
        "/private/canary/model",
        "prompt canary",
        "query canary",
        "document text canary",
        "model output canary",
        "exception canary",
        "credential canary",
    ):
        assert canary not in output


def test_emission_levels_cover_warning_and_error() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    incomplete = MemorySnapshot(
        observed_at=OBSERVED_AT,
        adapter=ADAPTER,
        measurements=(
            make_measurement("missing", status="unavailable", value_bytes=None),
        ),
    )
    emit_memory_measurement(
        structlog.get_logger("runtime"),
        make_measurement("missing", status="unavailable", value_bytes=None),
    )
    emit_memory_budget_comparison(
        structlog.get_logger("runtime"),
        incomplete.compare_budget(
            MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 2048)
        ),
        observed_at=OBSERVED_AT,
    )
    breach = MemorySnapshot(
        observed_at=OBSERVED_AT,
        adapter=ADAPTER,
        measurements=(make_measurement("breach", value_bytes=4096),),
    ).compare_budget(MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 1))
    emit_memory_budget_comparison(
        structlog.get_logger("runtime"), breach, observed_at=OBSERVED_AT
    )

    assert [json.loads(line)["level"] for line in stream.getvalue().splitlines()] == [
        "warning",
        "warning",
        "error",
    ]
