from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from inference_lab.backends.runtime import (
    BudgetState,
    Completeness,
    ComponentRole,
    ContributionKind,
    ExpectedMemoryContribution,
    MemoryAdapterIdentity,
    MemoryBudget,
    MemoryCollectorIdentity,
    MemoryMeasurement,
    MemorySnapshot,
    MemorySpace,
)

OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")
OTHER_ADAPTER = MemoryAdapterIdentity("process", "2.0.0", "rss-1")
COLLECTOR = MemoryCollectorIdentity("test", "1.0.0", "inventory-1")


def measurement(
    measurement_id: str,
    *,
    component_id: str = "generator",
    component_role: ComponentRole = ComponentRole.GENERATOR,
    category: str = "model_weights",
    memory_space: MemorySpace = MemorySpace.ACCELERATOR,
    device_id: str | None = "cuda-0",
    status: str = "measured",
    value_bytes: int | None = 1024,
    observed_at: datetime = OBSERVED_AT,
    contribution_kind: ContributionKind = ContributionKind.EXCLUSIVE,
    covers_measurement_ids: tuple[str, ...] = (),
    adapter: MemoryAdapterIdentity = ADAPTER,
) -> MemoryMeasurement:
    return MemoryMeasurement(
        measurement_id=measurement_id,
        component_id=component_id,
        component_role=component_role,
        category=category,  # type: ignore[arg-type]
        memory_space=memory_space,
        device_id=device_id,
        status=status,  # type: ignore[arg-type]
        value_bytes=value_bytes,
        adapter=adapter,
        observed_at=observed_at,
        contribution_kind=contribution_kind,
        covers_measurement_ids=covers_measurement_ids,
    )


def expected_contribution(
    record: MemoryMeasurement,
    *,
    adapter: MemoryAdapterIdentity | None = None,
) -> ExpectedMemoryContribution:
    return ExpectedMemoryContribution(
        component_id=record.component_id,
        component_role=record.component_role,
        category=record.category,
        memory_space=record.memory_space,
        device_id=record.device_id,
        adapter=adapter or record.adapter,
    )


def snapshot(*measurements: MemoryMeasurement) -> MemorySnapshot:
    return MemorySnapshot(
        observed_at=OBSERVED_AT,
        collector=COLLECTOR,
        expected_contributions=tuple(
            expected_contribution(record)
            for record in measurements
            if record.contribution_kind is ContributionKind.EXCLUSIVE
        ),
        measurements=measurements,
    )


def space_total(memory_snapshot: MemorySnapshot, space: MemorySpace):
    return next(
        total
        for total in memory_snapshot.totals
        if total.memory_space is space
        and total.category is None
        and total.component_role is None
    )


def test_snapshot_totals_are_order_independent_and_exclude_rollups() -> None:
    generator = measurement("generator-weights", value_bytes=10)
    framework = measurement(
        "framework",
        component_id="framework",
        component_role=ComponentRole.RUNTIME_FRAMEWORK,
        category="framework_overhead",
        value_bytes=3,
    )
    rollup = measurement(
        "accelerator-total",
        component_id="runtime",
        component_role=ComponentRole.RUNTIME_FRAMEWORK,
        contribution_kind=ContributionKind.ROLL_UP,
        covers_measurement_ids=("generator-weights", "framework"),
        value_bytes=13,
    )

    first = snapshot(generator, framework, rollup)
    second = snapshot(rollup, framework, generator)

    assert first.totals == second.totals
    total = space_total(first, MemorySpace.ACCELERATOR)
    assert total.measured_bytes == 13
    assert total.measured_count == 2
    assert first.reconciliations[0].reconciled is True


def test_totals_include_category_and_component_role_subtotals() -> None:
    record = measurement("weights", value_bytes=10)
    memory_snapshot = snapshot(record)

    assert any(
        total.category == "model_weights"
        and total.component_role is None
        and total.measured_bytes == 10
        for total in memory_snapshot.totals
    )
    assert any(
        total.category == "model_weights"
        and total.component_role is ComponentRole.GENERATOR
        and total.measured_bytes == 10
        for total in memory_snapshot.totals
    )


def test_all_memory_spaces_are_kept_separate() -> None:
    accelerator = measurement("accelerator", value_bytes=10)
    host = measurement(
        "host",
        component_id="index",
        component_role=ComponentRole.LEXICAL_INDEX,
        category="index_memory",
        memory_space=MemorySpace.HOST,
        device_id=None,
        value_bytes=20,
    )
    pinned = measurement(
        "pinned",
        component_id="pinned",
        component_role=ComponentRole.PINNED_HOST_ALLOCATION,
        category="pinned_host_memory",
        memory_space=MemorySpace.PINNED_HOST,
        device_id=None,
        value_bytes=30,
    )
    storage = measurement(
        "storage",
        component_id="storage",
        component_role=ComponentRole.STORAGE_CACHE,
        category="storage_cache",
        memory_space=MemorySpace.STORAGE_CACHE,
        device_id=None,
        value_bytes=40,
    )
    memory_snapshot = snapshot(accelerator, host, pinned, storage)

    assert [
        space_total(memory_snapshot, space).measured_bytes for space in MemorySpace
    ] == [
        10,
        20,
        30,
        40,
    ]


def test_snapshot_composes_distinct_measurement_adapters() -> None:
    weights = measurement("weights", value_bytes=10)
    framework = measurement(
        "framework",
        component_id="framework",
        component_role=ComponentRole.RUNTIME_FRAMEWORK,
        category="framework_overhead",
        value_bytes=3,
        adapter=OTHER_ADAPTER,
    )

    memory_snapshot = snapshot(weights, framework)

    assert {item.adapter for item in memory_snapshot.measurements} == {
        ADAPTER,
        OTHER_ADAPTER,
    }
    assert space_total(memory_snapshot, MemorySpace.ACCELERATOR).measured_bytes == 13


def test_snapshot_rejects_overlapping_or_unplanned_exclusive_contributions() -> None:
    first = measurement("first")
    second = measurement("second", adapter=OTHER_ADAPTER)
    with pytest.raises(ValueError, match="expected contribution slots"):
        MemorySnapshot(
            observed_at=OBSERVED_AT,
            collector=COLLECTOR,
            expected_contributions=(
                expected_contribution(first),
                expected_contribution(second),
            ),
            measurements=(first, second),
        )

    with pytest.raises(ValueError, match="was not expected"):
        MemorySnapshot(
            observed_at=OBSERVED_AT,
            collector=COLLECTOR,
            expected_contributions=(expected_contribution(first),),
            measurements=(
                first,
                measurement(
                    "framework",
                    component_id="framework",
                    component_role=ComponentRole.RUNTIME_FRAMEWORK,
                    category="framework_overhead",
                ),
            ),
        )


def test_missing_and_unreliable_values_make_totals_incomplete() -> None:
    memory_snapshot = snapshot(
        measurement("measured", value_bytes=10),
        measurement(
            "unreliable",
            component_id="workspace",
            component_role=ComponentRole.ACTIVATION_WORKSPACE,
            category="activations",
            status="unreliable",
            value_bytes=4,
        ),
        measurement(
            "unavailable",
            component_id="cache",
            component_role=ComponentRole.KV_CACHE,
            category="kv_cache",
            status="unavailable",
            value_bytes=None,
        ),
    )
    total = space_total(memory_snapshot, MemorySpace.ACCELERATOR)

    assert total.measured_bytes == 10
    assert total.unreliable_bytes == 4
    assert total.unreliable_count == 1
    assert total.unavailable_count == 1
    assert total.missing_count == 0
    assert total.completeness is Completeness.INCOMPLETE


def test_absent_expected_contribution_is_counted_and_never_safe() -> None:
    promised = measurement(
        "workspace",
        component_id="workspace",
        component_role=ComponentRole.ACTIVATION_WORKSPACE,
        category="activations",
    )
    memory_snapshot = MemorySnapshot(
        observed_at=OBSERVED_AT,
        collector=COLLECTOR,
        expected_contributions=(expected_contribution(promised),),
        measurements=(),
    )

    total = space_total(memory_snapshot, MemorySpace.ACCELERATOR)
    comparison = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 24_000_000_000)
    )

    assert total.measured_bytes == 0
    assert total.missing_count == 1
    assert total.completeness is Completeness.INCOMPLETE
    assert comparison.state is BudgetState.INDETERMINATE
    assert comparison.missing_count == 1


def test_unknown_accelerator_device_contaminates_each_device_budget() -> None:
    weights = measurement("weights", value_bytes=23_000_000_000)
    unknown_cache = measurement(
        "cache",
        component_id="cache",
        component_role=ComponentRole.KV_CACHE,
        category="kv_cache",
        device_id=None,
        status="unavailable",
        value_bytes=None,
    )
    memory_snapshot = snapshot(weights, unknown_cache)

    cuda_0 = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 24_000_000_000)
    )
    cuda_1 = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-1", 24_000_000_000)
    )
    cuda_0_total = next(
        total
        for total in memory_snapshot.totals
        if total.memory_space is MemorySpace.ACCELERATOR
        and total.device_id == "cuda-0"
        and total.category is None
    )

    assert cuda_0_total.completeness is Completeness.INCOMPLETE
    assert cuda_0_total.unavailable_count == 1
    assert cuda_0.measured_bytes == 23_000_000_000
    assert cuda_0.unavailable_count == 1
    assert cuda_0.state is BudgetState.INDETERMINATE
    assert cuda_1.measured_bytes == 0
    assert cuda_1.unavailable_count == 1
    assert cuda_1.state is BudgetState.INDETERMINATE


def test_mismatched_measured_rollup_is_retained_as_unreconciled_diagnostic() -> None:
    memory_snapshot = snapshot(
        measurement("child", value_bytes=10),
        measurement(
            "rollup",
            component_id="runtime",
            component_role=ComponentRole.RUNTIME_FRAMEWORK,
            contribution_kind=ContributionKind.ROLL_UP,
            covers_measurement_ids=("child",),
            value_bytes=11,
        ),
    )

    assert memory_snapshot.reconciliations[0].reconciled is False
    assert space_total(memory_snapshot, MemorySpace.ACCELERATOR).measured_bytes == 10


@pytest.mark.parametrize(
    "measurements",
    [
        (
            measurement(
                "rollup",
                contribution_kind=ContributionKind.ROLL_UP,
                covers_measurement_ids=("missing",),
            ),
        ),
        (
            measurement("one"),
            measurement("one"),
        ),
        (
            measurement(
                "rollup",
                contribution_kind=ContributionKind.ROLL_UP,
                covers_measurement_ids=("child",),
            ),
            measurement("child", memory_space=MemorySpace.HOST, device_id=None),
        ),
        (
            measurement(
                "one",
                contribution_kind=ContributionKind.ROLL_UP,
                covers_measurement_ids=("two",),
            ),
            measurement(
                "two",
                contribution_kind=ContributionKind.ROLL_UP,
                covers_measurement_ids=("one",),
            ),
        ),
    ],
)
def test_invalid_rollup_graph_is_rejected(
    measurements: tuple[MemoryMeasurement, ...],
) -> None:
    with pytest.raises(ValueError):
        snapshot(*measurements)


def test_snapshot_requires_expected_adapter_and_observation_boundary() -> None:
    other = measurement(
        "other",
        adapter=MemoryAdapterIdentity("other", "1.0.0", "1"),
    )
    with pytest.raises(ValueError, match="expected adapter"):
        MemorySnapshot(
            observed_at=OBSERVED_AT,
            collector=COLLECTOR,
            expected_contributions=(expected_contribution(other, adapter=ADAPTER),),
            measurements=(other,),
        )
    with pytest.raises(ValueError, match="after snapshot"):
        snapshot(measurement("future", observed_at=OBSERVED_AT + timedelta(seconds=1)))


def test_budget_comparison_covers_exact_fit_headroom_and_breach() -> None:
    memory_snapshot = snapshot(measurement("weights", value_bytes=100))
    exact = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 100)
    )
    positive = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 125)
    )
    breach = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 99)
    )

    assert exact.state is BudgetState.WITHIN_BUDGET
    assert exact.provisional_headroom_bytes == 0
    assert positive.provisional_headroom_bytes == 25
    assert breach.state is BudgetState.BREACHED
    assert breach.provisional_headroom_bytes == -1


def test_incomplete_budget_is_indeterminate_unless_measured_bytes_breach() -> None:
    memory_snapshot = snapshot(
        measurement("measured", value_bytes=100),
        measurement(
            "missing",
            component_id="cache",
            component_role=ComponentRole.KV_CACHE,
            category="kv_cache",
            status="unavailable",
            value_bytes=None,
        ),
    )
    indeterminate = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 101)
    )
    breach = memory_snapshot.compare_budget(
        MemoryBudget(MemorySpace.ACCELERATOR, "cuda-0", 99)
    )

    assert indeterminate.state is BudgetState.INDETERMINATE
    assert indeterminate.completeness is Completeness.INCOMPLETE
    assert breach.state is BudgetState.BREACHED


def test_decimal_gigabytes_convert_to_integer_bytes() -> None:
    budget = MemoryBudget.from_decimal_gigabytes(
        memory_space=MemorySpace.ACCELERATOR,
        device_id="cuda-0",
        gigabytes=Decimal("24"),
    )

    assert budget.budget_bytes == 24_000_000_000


def test_budget_space_or_device_mismatch_is_rejected() -> None:
    memory_snapshot = snapshot(measurement("weights", value_bytes=100))

    with pytest.raises(ValueError, match="no matching"):
        memory_snapshot.compare_budget(MemoryBudget(MemorySpace.HOST, None, 1000))
    with pytest.raises(ValueError, match="no matching"):
        memory_snapshot.compare_budget(
            MemoryBudget(MemorySpace.ACCELERATOR, "cuda-1", 1000)
        )
