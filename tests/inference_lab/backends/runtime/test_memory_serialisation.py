import json
from datetime import UTC, datetime

import pytest

from inference_lab.backends.runtime import (
    BudgetState,
    Completeness,
    ComponentRole,
    ExpectedMemoryContribution,
    MemoryAdapterIdentity,
    MemoryBudget,
    MemoryCollectorIdentity,
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
    component_role: ComponentRole = ComponentRole.GENERATOR,
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
            component_role=ComponentRole.RUNTIME_FRAMEWORK,
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
    assert payload["schema_version"] == 2
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
    "data",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":1}',
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
    payload["schema_version"] = 3
    with pytest.raises(ValueError, match="unsupported"):
        deserialise_memory_snapshot(json.dumps(payload).encode())


def test_nested_scalar_coercion_is_rejected() -> None:
    payload = json.loads(serialise_memory_snapshot(make_snapshot()))
    payload["measurements"][0]["value_bytes"] = "1024"

    with pytest.raises(TypeError, match="value_bytes"):
        deserialise_memory_snapshot(json.dumps(payload).encode())
