"""RUN-03 acceptance: account for component observations through a memory budget."""

import json
from datetime import UTC, datetime
from io import StringIO

import pytest
import structlog

from inference_lab.backends.runtime import (
    BudgetState,
    ExpectedMemoryContribution,
    FakeMemoryAccountingAdapter,
    MemoryAdapterIdentity,
    MemoryBudget,
    MemoryCollectorIdentity,
    MemoryComponentRole,
    MemoryMeasurement,
    MemorySnapshot,
    MemorySpace,
    deserialise_budget_comparison,
    deserialise_memory_snapshot,
    emit_memory_budget_comparison,
    emit_memory_measurement,
    serialise_budget_comparison,
    serialise_memory_snapshot,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig

_OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
_ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")
_COLLECTOR = MemoryCollectorIdentity("test", "1.0.0", "inventory-1")


def _measurement(
    measurement_id: str,
    component_id: str,
    role: MemoryComponentRole,
    category: str,
    value_bytes: int | None,
    *,
    status: str = "measured",
) -> MemoryMeasurement:
    return MemoryMeasurement(
        measurement_id=measurement_id,
        component_id=component_id,
        component_role=role,
        category=category,  # type: ignore[arg-type]
        memory_space=MemorySpace.ACCELERATOR,
        device_id="cuda-0",
        status=status,  # type: ignore[arg-type]
        value_bytes=value_bytes,
        adapter=_ADAPTER,
        observed_at=_OBSERVED_AT,
    )


def _expected(item: MemoryMeasurement) -> ExpectedMemoryContribution:
    return ExpectedMemoryContribution(
        component_id=item.component_id,
        component_role=item.component_role,
        category=item.category,
        memory_space=item.memory_space,
        device_id=item.device_id,
        adapter=item.adapter,
    )


def _snapshot(
    records: tuple[MemoryMeasurement, ...],
    *,
    missing: tuple[ExpectedMemoryContribution, ...] = (),
) -> MemorySnapshot:
    expected = tuple(
        _expected(item)
        for item in records
    ) + missing
    return MemorySnapshot(
        observed_at=_OBSERVED_AT,
        collector=_COLLECTOR,
        expected_contributions=expected,
        measurements=records,
        run_id="run-1",
        request_id="request-1",
    )


def test_run03_observations_roll_up_to_attributed_budget_headroom() -> None:
    weights = _measurement(
        "weights", "generator", MemoryComponentRole.GENERATOR, "model_weights", 700
    )
    framework = _measurement(
        "framework",
        "runtime",
        MemoryComponentRole.RUNTIME_FRAMEWORK,
        "framework_overhead",
        200,
    )
    adapter = FakeMemoryAccountingAdapter(
        (weights, framework), clock=lambda: _OBSERVED_AT
    )

    current = _snapshot(adapter.measure_memory())
    comparison = current.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 1024)
    )
    breach = current.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 800)
    )

    assert comparison.state is BudgetState.WITHIN_BUDGET
    assert comparison.measured_bytes == 900
    assert comparison.provisional_headroom_bytes == 124
    assert breach.state is BudgetState.BREACHED
    assert {item.component_role for item in current.measurements} == {
        MemoryComponentRole.GENERATOR,
        MemoryComponentRole.RUNTIME_FRAMEWORK,
    }
    assert current.run_id == "run-1"


def test_run03_unavailable_observation_makes_budget_indeterminate() -> None:
    measured = _measurement(
        "weights", "generator", MemoryComponentRole.GENERATOR, "model_weights", 700
    )
    unavailable = _measurement(
        "kv-cache",
        "generator",
        MemoryComponentRole.GENERATOR,
        "kv_cache",
        None,
        status="unavailable",
    )
    unreliable = _measurement(
        "activations",
        "generator",
        MemoryComponentRole.ACTIVATION_WORKSPACE,
        "activations",
        40,
        status="unreliable",
    )
    absent = ExpectedMemoryContribution(
        component_id="embedder",
        component_role=MemoryComponentRole.EMBEDDER,
        category="model_weights",
        memory_space=MemorySpace.ACCELERATOR,
        device_id="cuda-0",
        adapter=_ADAPTER,
    )
    current = _snapshot((measured, unavailable, unreliable), missing=(absent,))

    comparison = current.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 1024)
    )

    assert comparison.state is BudgetState.INDETERMINATE
    assert comparison.unavailable_count == 1
    assert comparison.unreliable_count == 1
    assert comparison.missing_count == 1
    assert comparison.measured_bytes == 700


def test_run03_snapshot_round_trips_and_rejects_malformed_measurement() -> None:
    weights = _measurement(
        "weights",
        "generator",
        MemoryComponentRole.GENERATOR,
        "model_weights",
        700,
    )
    current = _snapshot((weights,))

    encoded = serialise_memory_snapshot(current)
    assert serialise_memory_snapshot(deserialise_memory_snapshot(encoded)) == encoded
    comparison = current.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 1024)
    )
    comparison_bytes = serialise_budget_comparison(comparison)
    assert (
        serialise_budget_comparison(deserialise_budget_comparison(comparison_bytes))
        == comparison_bytes
    )

    malformed = json.loads(encoded)
    malformed["measurements"][0]["value_bytes"] = True
    with pytest.raises((TypeError, ValueError)):
        deserialise_memory_snapshot(json.dumps(malformed).encode())


def test_run03_emits_safe_measurement_and_budget_records() -> None:
    stream = StringIO()
    configure_logging(LoggingConfig(level="DEBUG", format="json"), stream=stream)
    measurement = _measurement(
        "weights", "generator", MemoryComponentRole.GENERATOR, "model_weights", 700
    )
    comparison = _snapshot((measurement,)).compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 1024)
    )

    emit_memory_measurement(structlog.get_logger("runtime"), measurement)
    emit_memory_budget_comparison(
        structlog.get_logger("runtime"), comparison, observed_at=_OBSERVED_AT
    )
    records = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert [item["event"] for item in records] == [
        "memory_component_measured",
        "memory_budget_compared",
    ]
    assert all(item["pipeline_stage"] == "memory_accounting" for item in records)
    assert all(item["record_schema_version"] == "4.0.0" for item in records)
    assert all(item["document_id"] is None for item in records)
    for private_value in (
        "/private/canary/model",
        "query canary",
        "document text canary",
    ):
        assert private_value not in stream.getvalue()
