"""Tests for runtime memory serialisation."""

import json
from datetime import UTC, datetime

import pytest

from inference_lab.backends.runtime import (
    BudgetState,
    Completeness,
    ExpectedMemoryContribution,
    MemoryAdapterIdentity,
    MemoryBudget,
    MemoryCollectorIdentity,
    MemoryComponentRole,
    MemoryMeasurement,
    MemorySnapshot,
    MemorySpace,
    deserialise_budget_comparison,
    deserialise_memory_snapshot,
    serialise_budget_comparison,
    serialise_memory_snapshot,
)

OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")
COLLECTOR = MemoryCollectorIdentity("test", "1.0.0", "inventory-1")


def make_measurement(
    measurement_id: str = "weights",
    *,
    component_id: str = "generator",
    component_role: MemoryComponentRole = MemoryComponentRole.GENERATOR,
    category: str = "model_weights",
    status: str = "measured",
    value_bytes: int | None = 1024,
) -> MemoryMeasurement:
    return MemoryMeasurement(
        measurement_id=measurement_id,
        component_id=component_id,
        component_role=component_role,
        category=category,  # type: ignore[arg-type]
        memory_space=MemorySpace.ACCELERATOR,
        device_id="cuda-0",
        status=status,  # type: ignore[arg-type]
        value_bytes=value_bytes,
        adapter=ADAPTER,
        observed_at=OBSERVED_AT,
    )


def make_snapshot() -> MemorySnapshot:
    measurements = (
        make_measurement("z-last"),
        make_measurement(
            "a-first",
            component_id="framework",
            component_role=MemoryComponentRole.RUNTIME_FRAMEWORK,
            category="framework_overhead",
        ),
    )
    return MemorySnapshot(
        observed_at=OBSERVED_AT,
        collector=COLLECTOR,
        expected_contributions=tuple(
            ExpectedMemoryContribution(
                component_id=measurement.component_id,
                component_role=measurement.component_role,
                category=measurement.category,
                memory_space=measurement.memory_space,
                device_id=measurement.device_id,
                adapter=measurement.adapter,
            )
            for measurement in measurements
        ),
        measurements=measurements,
        run_id="run-1",
        request_id="request-1",
    )


def _snapshot_payload() -> dict[str, object]:
    """Return a mutable JSON snapshot payload."""
    return json.loads(serialise_memory_snapshot(make_snapshot()))


def _comparison_payload() -> dict[str, object]:
    """Return a mutable JSON budget-comparison payload."""
    comparison = make_snapshot().compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 2048)
    )
    return json.loads(serialise_budget_comparison(comparison))


def _replace_path(
    payload: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> bytes:
    """Replace one nested JSON value and encode the resulting payload."""
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return json.dumps(payload).encode()


def test_snapshot_serialisation_is_canonical_and_round_trips() -> None:
    snapshot = make_snapshot()

    serialised = serialise_memory_snapshot(snapshot)
    restored = deserialise_memory_snapshot(serialised)

    assert serialised == serialise_memory_snapshot(restored)
    assert restored == snapshot
    assert serialised.endswith(b"\n")
    assert serialised.index(b'"a-first"') < serialised.index(b'"z-last"')
    assert b" " not in serialised
    payload = json.loads(serialised)
    assert payload["schema_version"] == "2.0.0"
    assert payload["collector"]["collector_name"] == "test"
    assert payload["overlap_policy"] == "disallow"
    assert len(payload["expected_contributions"]) == 2


def test_budget_comparison_serialisation_round_trips() -> None:
    snapshot = make_snapshot()
    comparison = snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 2048)
    )

    restored = deserialise_budget_comparison(serialise_budget_comparison(comparison))

    assert restored == comparison
    assert json.loads(serialise_budget_comparison(comparison))["missing_count"] == 0
    assert restored.state is BudgetState.WITHIN_BUDGET
    assert restored.completeness is Completeness.COMPLETE


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("provisional_headroom_bytes", 1, "headroom"),
        ("completeness", "incomplete", "completeness"),
        ("state", "breached", "state"),
        ("measured_count", 0, "measured contribution"),
        ("missing_count", 1, "completeness"),
        ("unavailable_count", -1, "must not be negative"),
    ],
)
def test_budget_comparison_deserialisation_rejects_contradictory_facts(
    field_name: str,
    value: object,
    message: str,
) -> None:
    comparison = make_snapshot().compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 2048)
    )
    payload = json.loads(serialise_budget_comparison(comparison))
    payload[field_name] = value

    with pytest.raises(ValueError, match=message):
        deserialise_budget_comparison(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "data",
    [
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        b'{"schema_version":"1.0.0"}',
        b'{"schema_version":true}',
        b"not-json",
        b"\xff",
    ],
)
def test_snapshot_deserialisation_rejects_malformed_or_noncanonical_shapes(
    data: bytes,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        deserialise_memory_snapshot(data)


def test_snapshot_deserialisation_rejects_unknown_and_unsupported_fields() -> None:
    payload = json.loads(serialise_memory_snapshot(make_snapshot()))
    payload["unexpected"] = None
    with pytest.raises(ValueError, match="unknown fields"):
        deserialise_memory_snapshot(json.dumps(payload).encode())

    payload = json.loads(serialise_memory_snapshot(make_snapshot()))
    payload["schema_version"] = "3.0.0"
    with pytest.raises(ValueError, match="unsupported"):
        deserialise_memory_snapshot(json.dumps(payload).encode())


def test_nested_scalar_coercion_is_rejected() -> None:
    payload = json.loads(serialise_memory_snapshot(make_snapshot()))
    payload["measurements"][0]["value_bytes"] = "1024"

    with pytest.raises(TypeError, match="value_bytes"):
        deserialise_memory_snapshot(json.dumps(payload).encode())


def test_memory_serialisers_reject_wrong_envelope_types() -> None:
    with pytest.raises(TypeError, match="MemorySnapshot"):
        serialise_memory_snapshot("snapshot")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BudgetComparison"):
        serialise_budget_comparison("comparison")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        deserialise_memory_snapshot("snapshot")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        deserialise_budget_comparison("comparison")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "value", "error_type", "message"),
    [
        (("measurements",), {}, TypeError, "measurements"),
        (("expected_contributions",), {}, TypeError, "expected_contributions"),
        (("collector",), [], TypeError, "collector"),
        (("collector", "collector_name"), 1, TypeError, "collector_name"),
        (("run_id",), 1, TypeError, "run_id"),
        (("observed_at",), "not-a-date", ValueError, "ISO datetime"),
        (("overlap_policy",), "unknown", ValueError, "overlap policy"),
        (("expected_contributions", 0), [], TypeError, "JSON objects"),
        (
            ("expected_contributions", 0, "component_role"),
            "unknown",
            ValueError,
            "unsupported enum",
        ),
        (
            ("expected_contributions", 0, "memory_space"),
            "unknown",
            ValueError,
            "unsupported enum",
        ),
        (
            ("expected_contributions", 0, "category"),
            "unknown",
            ValueError,
            "supported memory category",
        ),
        (("expected_contributions", 0, "device_id"), 1, TypeError, "device_id"),
        (("expected_contributions", 0, "adapter"), [], TypeError, "adapter"),
        (("measurements", 0), [], TypeError, "JSON objects"),
        (
            ("measurements", 0, "covers_measurement_ids"),
            {},
            TypeError,
            "JSON array",
        ),
        (
            ("measurements", 0, "covers_measurement_ids"),
            [1],
            TypeError,
            "contain strings",
        ),
        (
            ("measurements", 0, "contribution_kind"),
            "unknown",
            ValueError,
            "unsupported enum",
        ),
        (("measurements", 0, "status"), "unknown", ValueError, "status"),
        (("measurements", 0, "observed_at"), "invalid", ValueError, "ISO datetime"),
    ],
)
def test_snapshot_deserialisation_rejects_invalid_nested_contracts(
    path: tuple[str | int, ...],
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    data = _replace_path(_snapshot_payload(), path, value)

    with pytest.raises(error_type, match=message):
        deserialise_memory_snapshot(data)


@pytest.mark.parametrize(
    ("path", "value", "error_type", "message"),
    [
        (("budget",), [], TypeError, "budget"),
        (("budget", "memory_space"), "unknown", ValueError, "memory space"),
        (("budget", "device_id"), 1, TypeError, "device_id"),
        (("budget", "budget_bytes"), True, TypeError, "budget_bytes"),
        (("state",), "unknown", ValueError, "unsupported enum"),
        (("completeness",), "unknown", ValueError, "unsupported enum"),
        (("measured_bytes",), True, TypeError, "measured_bytes"),
    ],
)
def test_budget_deserialisation_rejects_invalid_nested_contracts(
    path: tuple[str | int, ...],
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    data = _replace_path(_comparison_payload(), path, value)

    with pytest.raises(error_type, match=message):
        deserialise_budget_comparison(data)
