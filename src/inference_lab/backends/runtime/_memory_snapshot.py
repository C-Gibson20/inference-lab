"""Validated runtime memory snapshots and their derived accounting state."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from inference_lab.backends.runtime._memory_accounting import (
    BudgetComparison,
    MemoryBudget,
    MemoryTotal,
    _build_memory_totals,
    _compare_memory_budget,
    _ContributionSlot,
    _expected_contribution_slot,
    _measurement_slot,
)
from inference_lab.backends.runtime._memory_values import (
    ContributionKind,
    ExpectedMemoryContribution,
    MemoryCollectorIdentity,
    MemoryMeasurement,
    MemoryOverlapPolicy,
    _validate_safe_identifier,
)
from inference_lab.backends.runtime._validation import validate_non_negative_integer
from inference_lab.profiling._validation import validate_utc_datetime

CURRENT_MEMORY_SNAPSHOT_SCHEMA_VERSION: Final[str] = "2.0.0"
SUPPORTED_MEMORY_SNAPSHOT_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {CURRENT_MEMORY_SNAPSHOT_SCHEMA_VERSION}
)


@dataclass(frozen=True, slots=True)
class RollupReconciliation:
    """Diagnostic comparing a measured roll-up with its direct children."""

    rollup_measurement_id: str
    rollup_value_bytes: int
    covered_value_bytes: int
    reconciled: bool

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate reconciliation types, units, and identity rules."""
        _validate_safe_identifier(
            self.rollup_measurement_id,
            "rollup_measurement_id",
        )
        for value, field_name in (
            (self.rollup_value_bytes, "rollup_value_bytes"),
            (self.covered_value_bytes, "covered_value_bytes"),
        ):
            validate_non_negative_integer(value, field_name)
        if not isinstance(self.reconciled, bool):
            raise TypeError("reconciled must be a boolean")


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Versioned, deterministic collection of component memory observations."""

    observed_at: datetime
    collector: MemoryCollectorIdentity
    expected_contributions: tuple[ExpectedMemoryContribution, ...]
    measurements: tuple[MemoryMeasurement, ...]
    overlap_policy: MemoryOverlapPolicy = MemoryOverlapPolicy.DISALLOW
    run_id: str | None = None
    request_id: str | None = None
    schema_version: str = CURRENT_MEMORY_SNAPSHOT_SCHEMA_VERSION
    _reconciliations: tuple[RollupReconciliation, ...] = field(
        init=False,
        repr=False,
    )
    _totals: tuple[MemoryTotal, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "measurements",
            tuple(sorted(self.measurements, key=lambda item: item.measurement_id)),
        )
        object.__setattr__(
            self,
            "expected_contributions",
            tuple(
                sorted(
                    self.expected_contributions,
                    key=self._expected_contribution_sort_key,
                )
            ),
        )
        object.__setattr__(self, "_reconciliations", self._build_reconciliations())
        object.__setattr__(
            self,
            "_totals",
            _build_memory_totals(self.expected_contributions, self.measurements),
        )

    def _validate(self) -> None:
        """Validate snapshot metadata, inventory, measurements, and roll-ups."""
        self._validate_metadata()
        expected_by_slot = self._validate_expected_inventory()
        measurements_by_id = self._validate_measurements(expected_by_slot)
        self._validate_rollup_coverage(measurements_by_id)
        self._validate_no_rollup_cycles(measurements_by_id)

    def _validate_metadata(self) -> None:
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        validate_utc_datetime(self.observed_at, "observed_at")
        if not isinstance(self.collector, MemoryCollectorIdentity):
            raise TypeError("collector must be MemoryCollectorIdentity")
        if not isinstance(self.expected_contributions, tuple):
            raise TypeError("expected_contributions must be a tuple")
        if not self.expected_contributions:
            raise ValueError("expected_contributions must not be empty")
        if not isinstance(self.measurements, tuple):
            raise TypeError("measurements must be a tuple")
        if not isinstance(self.overlap_policy, MemoryOverlapPolicy):
            raise TypeError("overlap_policy must be a MemoryOverlapPolicy")
        if not isinstance(self.schema_version, str):
            raise TypeError("schema_version must be a string")
        if self.schema_version not in SUPPORTED_MEMORY_SNAPSHOT_SCHEMA_VERSIONS:
            raise ValueError("unsupported memory snapshot schema version")
        for correlation_id, field_name in (
            (self.run_id, "run_id"),
            (self.request_id, "request_id"),
        ):
            if correlation_id is not None:
                _validate_safe_identifier(correlation_id, field_name)

    def _validate_expected_inventory(
        self,
    ) -> dict[_ContributionSlot, ExpectedMemoryContribution]:
        expected_by_slot: dict[_ContributionSlot, ExpectedMemoryContribution] = {}
        for expected in self.expected_contributions:
            if not isinstance(expected, ExpectedMemoryContribution):
                raise TypeError(
                    "expected_contributions must contain "
                    "ExpectedMemoryContribution values"
                )
            slot = _expected_contribution_slot(expected)
            if slot in expected_by_slot:
                raise ValueError("expected contribution slots must not overlap")
            expected_by_slot[slot] = expected
        return expected_by_slot

    def _validate_measurements(
        self,
        expected_by_slot: dict[_ContributionSlot, ExpectedMemoryContribution],
    ) -> dict[str, MemoryMeasurement]:
        measurements_by_id: dict[str, MemoryMeasurement] = {}
        observed_exclusive_slots: set[_ContributionSlot] = set()
        for measurement in self.measurements:
            if not isinstance(measurement, MemoryMeasurement):
                raise TypeError("measurements must contain MemoryMeasurement values")
            if measurement.measurement_id in measurements_by_id:
                raise ValueError("measurement IDs must be unique")
            measurements_by_id[measurement.measurement_id] = measurement
            if measurement.observed_at > self.observed_at:
                raise ValueError("measurements must not be after snapshot observed_at")
            if measurement.contribution_kind is not ContributionKind.EXCLUSIVE:
                continue
            slot = _measurement_slot(measurement)
            expected = expected_by_slot.get(slot)
            if expected is None:
                raise ValueError("exclusive measurement was not expected")
            if measurement.adapter != expected.adapter:
                raise ValueError("measurement does not use the expected adapter")
            if slot in observed_exclusive_slots:
                raise ValueError("exclusive measurement slots must not overlap")
            observed_exclusive_slots.add(slot)
        return measurements_by_id

    def _validate_rollup_coverage(
        self,
        measurements_by_id: dict[str, MemoryMeasurement],
    ) -> None:
        for measurement in self.measurements:
            if measurement.contribution_kind is not ContributionKind.ROLL_UP:
                continue
            for covered_id in measurement.covers_measurement_ids:
                covered = measurements_by_id.get(covered_id)
                if covered is None:
                    raise ValueError("roll-up covers an unknown measurement")
                if (
                    covered.memory_space is not measurement.memory_space
                    or covered.device_id != measurement.device_id
                ):
                    raise ValueError(
                        "roll-up coverage must use one memory space/device"
                    )

    @staticmethod
    def _expected_contribution_sort_key(
        expected: ExpectedMemoryContribution,
    ) -> tuple[str, str, str, str, str, str, str, str]:
        return (
            expected.memory_space.value,
            expected.device_id or "",
            expected.category,
            expected.component_role.value,
            expected.component_id,
            expected.adapter.adapter_name,
            expected.adapter.adapter_version,
            expected.adapter.calibration_revision,
        )

    @staticmethod
    def _validate_no_rollup_cycles(
        measurements_by_id: dict[str, MemoryMeasurement],
    ) -> None:
        """Reject direct and indirect roll-up cycles with bounded traversal."""
        visiting: set[str] = set()
        visited: set[str] = set()

        def _visit(measurement_id: str) -> None:
            if measurement_id in visiting:
                raise ValueError("roll-up coverage must be acyclic")
            if measurement_id in visited:
                return
            visiting.add(measurement_id)
            measurement = measurements_by_id[measurement_id]
            for covered_id in measurement.covers_measurement_ids:
                _visit(covered_id)
            visiting.remove(measurement_id)
            visited.add(measurement_id)

        for measurement_id in measurements_by_id:
            _visit(measurement_id)

    def _build_reconciliations(self) -> tuple[RollupReconciliation, ...]:
        measurements_by_id = {
            measurement.measurement_id: measurement for measurement in self.measurements
        }
        results: list[RollupReconciliation] = []
        for measurement in self.measurements:
            if (
                measurement.contribution_kind is ContributionKind.ROLL_UP
                and measurement.status == "measured"
                and all(
                    measurements_by_id[covered_id].status == "measured"
                    and measurements_by_id[covered_id].value_bytes is not None
                    for covered_id in measurement.covers_measurement_ids
                )
            ):
                covered_bytes = sum(
                    measurements_by_id[covered_id].value_bytes or 0
                    for covered_id in measurement.covers_measurement_ids
                )
                results.append(
                    RollupReconciliation(
                        rollup_measurement_id=measurement.measurement_id,
                        rollup_value_bytes=measurement.value_bytes or 0,
                        covered_value_bytes=covered_bytes,
                        reconciled=measurement.value_bytes == covered_bytes,
                    )
                )
        return tuple(sorted(results, key=lambda result: result.rollup_measurement_id))

    @property
    def reconciliations(self) -> tuple[RollupReconciliation, ...]:
        """Return deterministic roll-up reconciliation diagnostics."""
        return self._reconciliations

    @property
    def totals(self) -> tuple[MemoryTotal, ...]:
        """Return precomputed space, category, and component-role totals."""
        return self._totals

    def compare_budget(self, budget: MemoryBudget) -> BudgetComparison:
        """Compare one budget with its matching precomputed space subtotal."""
        return _compare_memory_budget(self._totals, budget)
