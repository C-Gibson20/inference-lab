"""Tests for normalized resource-measurement records."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from inference_lab.profiling import ResourceMeasurement


def fixed_timestamp() -> datetime:
    """Return a deterministic UTC timestamp for record tests."""
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def test_measured_resource_serializes_all_fields() -> None:
    record = ResourceMeasurement(
        category="gpu_utilization",
        unit="percent",
        value=72.5,
        measurement_kind="sampled",
        timestamp=fixed_timestamp(),
        module="runtime",
        pipeline_stage="generate",
        stage_id="generate-1",
        sample_duration_ms=1000.0,
        run_id="run-1",
    )

    assert record.to_dict() == {
        "event": "resource_measured",
        "category": "gpu_utilization",
        "unit": "percent",
        "value": 72.5,
        "status": "measured",
        "level": "info",
        "measurement_kind": "sampled",
        "timestamp": "2026-08-21T12:00:00+00:00",
        "module": "runtime",
        "pipeline_stage": "generate",
        "stage_id": "generate-1",
        "sample_duration_ms": 1000.0,
        "run_id": "run-1",
        "request_id": None,
        "document_id": None,
        "corpus_version": None,
    }


def test_unavailable_resource_has_no_value() -> None:
    record = ResourceMeasurement(
        category="power",
        unit="watt",
        value=None,
        status="unavailable",
        timestamp=fixed_timestamp(),
    )

    assert record.value is None
    assert record.to_dict()["status"] == "unavailable"


def test_unreliable_resource_can_retain_observed_value() -> None:
    record = ResourceMeasurement(
        category="thermal",
        unit="celsius",
        value=83.0,
        status="unreliable",
        timestamp=fixed_timestamp(),
    )

    assert record.value == 83.0
    assert record.status == "unreliable"


@pytest.mark.parametrize(
    "category",
    ["unknown", "gpu_memory", "temperature"],
)
def test_unknown_category_is_rejected(category: str) -> None:
    with pytest.raises(ValueError, match="supported resource category"):
        ResourceMeasurement(
            category=category,  # type: ignore[arg-type]
            unit="unit",
            value=1,
        )


def test_measured_resource_requires_value_and_unit() -> None:
    with pytest.raises(ValueError, match="must have a value"):
        ResourceMeasurement(category="power", unit="watt", value=None)

    with pytest.raises(ValueError, match="must have a unit"):
        ResourceMeasurement(category="power", unit=None, value=100)


def test_unavailable_resource_cannot_have_value() -> None:
    with pytest.raises(ValueError, match="must not have a value"):
        ResourceMeasurement(
            category="power",
            unit="watt",
            value=100,
            status="unavailable",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_non_finite_or_boolean_value_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        ResourceMeasurement(category="power", unit="watt", value=value)


def test_sampled_measurement_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive sample duration"):
        ResourceMeasurement(
            category="cpu_utilization",
            unit="percent",
            value=50,
            measurement_kind="sampled",
        )


@pytest.mark.parametrize(
    "sample_duration_ms",
    [float("nan"), float("inf"), float("-inf"), True, "1000"],
)
def test_sampled_measurement_rejects_invalid_duration(
    sample_duration_ms: object,
) -> None:
    with pytest.raises(ValueError, match="positive sample duration"):
        ResourceMeasurement(
            category="cpu_utilization",
            unit="percent",
            value=50,
            measurement_kind="sampled",
            sample_duration_ms=sample_duration_ms,  # type: ignore[arg-type]
        )


def test_point_in_time_measurement_cannot_have_sample_duration() -> None:
    with pytest.raises(ValueError, match="must not have a sample duration"):
        ResourceMeasurement(
            category="kv_cache",
            unit="GiB",
            value=4,
            sample_duration_ms=10,
        )


def test_naive_and_non_utc_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        ResourceMeasurement(
            category="power",
            unit="watt",
            value=100,
            timestamp=datetime(2026, 8, 21, 12, 0),
        )

    non_utc = datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )
    with pytest.raises(ValueError, match="timestamp must be in UTC"):
        ResourceMeasurement(
            category="power",
            unit="watt",
            value=100,
            timestamp=non_utc,
        )
