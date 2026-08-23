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
        value=72.5,
        measurement_kind="sampled",
        observed_at=fixed_timestamp(),
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
        "observed_at": "2026-08-21T12:00:00+00:00",
        "module": "runtime",
        "pipeline_stage": "generate",
        "stage_id": "generate-1",
        "sample_duration_ms": 1000.0,
        "run_id": "run-1",
        "request_id": None,
        "document_id": None,
        "corpus_version": None,
        "record_schema_version": 1,
    }


def test_unavailable_resource_has_no_value() -> None:
    record = ResourceMeasurement(
        category="power",
        value=None,
        status="unavailable",
        observed_at=fixed_timestamp(),
    )

    assert record.value is None
    assert record.unit == "watts"
    assert record.to_dict()["status"] == "unavailable"


def test_unreliable_resource_can_retain_observed_value() -> None:
    record = ResourceMeasurement(
        category="thermal",
        value=83.0,
        status="unreliable",
        observed_at=fixed_timestamp(),
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
            value=1,
        )


@pytest.mark.parametrize(
    ("category", "unit"),
    [
        ("model_weights", "bytes"),
        ("kv_cache", "bytes"),
        ("activations", "bytes"),
        ("embedding_reranker_memory", "bytes"),
        ("index_memory", "bytes"),
        ("framework_overhead", "bytes"),
        ("pinned_host_memory", "bytes"),
        ("storage_cache", "bytes"),
        ("gpu_utilization", "percent"),
        ("cpu_utilization", "percent"),
        ("storage_throughput", "bytes_per_second"),
        ("power", "watts"),
        ("thermal", "celsius"),
    ],
)
def test_resource_category_determines_canonical_unit(
    category: str,
    unit: str,
) -> None:
    record = ResourceMeasurement(
        category=category,  # type: ignore[arg-type]
        value=0,
    )

    assert record.unit == unit
    assert record.to_dict()["unit"] == unit


@pytest.mark.parametrize(
    ("category", "value", "message"),
    [
        ("kv_cache", -1, "non-negative"),
        ("storage_throughput", -1, "non-negative"),
        ("power", -1, "non-negative"),
        ("gpu_utilization", -0.1, "between 0 and 100"),
        ("cpu_utilization", 100.1, "between 0 and 100"),
        ("thermal", -273.16, "absolute zero"),
    ],
)
def test_resource_category_rejects_out_of_range_value(
    category: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ResourceMeasurement(
            category=category,  # type: ignore[arg-type]
            value=value,
        )


def test_thermal_measurement_allows_valid_negative_celsius() -> None:
    record = ResourceMeasurement(category="thermal", value=-20.0)

    assert record.value == -20.0


def test_measured_resource_requires_value() -> None:
    with pytest.raises(ValueError, match="must have a value"):
        ResourceMeasurement(category="power", value=None)


def test_unavailable_resource_cannot_have_value() -> None:
    with pytest.raises(ValueError, match="must not have a value"):
        ResourceMeasurement(
            category="power",
            value=100,
            status="unavailable",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_non_finite_or_boolean_value_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        ResourceMeasurement(category="power", value=value)


def test_sampled_measurement_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive sample duration"):
        ResourceMeasurement(
            category="cpu_utilization",
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
            value=50,
            measurement_kind="sampled",
            sample_duration_ms=sample_duration_ms,  # type: ignore[arg-type]
        )


def test_point_in_time_measurement_cannot_have_sample_duration() -> None:
    with pytest.raises(ValueError, match="must not have a sample duration"):
        ResourceMeasurement(
            category="kv_cache",
            value=4,
            sample_duration_ms=10,
        )


def test_naive_and_non_utc_observation_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="observed_at must be timezone-aware UTC"):
        ResourceMeasurement(
            category="power",
            value=100,
            observed_at=datetime(2026, 8, 21, 12, 0),
        )

    non_utc = datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )
    with pytest.raises(ValueError, match="observed_at must be in UTC"):
        ResourceMeasurement(
            category="power",
            value=100,
            observed_at=non_utc,
        )


def test_resource_measurement_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        ResourceMeasurement(
            category="power",
            value=100,
            record_schema_version=0,
        )
