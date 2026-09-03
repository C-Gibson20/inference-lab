"""Optional component-level memory accounting contracts for runtime backends."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from inference_lab.backends.runtime._validation import validate_non_empty_string
from inference_lab.profiling._validation import validate_utc_datetime
from inference_lab.profiling.resources import MeasurementStatus, ResourceCategory
from inference_lab.profiling.utils import utc_now


class ComponentRole(StrEnum):
    """Bounded role identifying the owner of a memory contribution."""

    GENERATOR = "generator"
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    KV_CACHE = "kv_cache"
    ACTIVATION_WORKSPACE = "activation_workspace"
    LEXICAL_INDEX = "lexical_index"
    DENSE_INDEX = "dense_index"
    METADATA_INDEX = "metadata_index"
    STRUCTURED_INDEX = "structured_index"
    SUMMARY_INDEX = "summary_index"
    RUNTIME_FRAMEWORK = "runtime_framework"
    PINNED_HOST_ALLOCATION = "pinned_host_allocation"
    STORAGE_CACHE = "storage_cache"


class MemorySpace(StrEnum):
    """Physical memory space in which a contribution resides."""

    ACCELERATOR = "accelerator"
    HOST = "host"
    PINNED_HOST = "pinned_host"
    STORAGE_CACHE = "storage_cache"


class ContributionKind(StrEnum):
    """Whether a measurement is an exclusive contribution or a roll-up."""

    EXCLUSIVE = "exclusive"
    ROLL_UP = "roll_up"


class Completeness(StrEnum):
    """Reliability state of an authoritative memory subtotal."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class BudgetState(StrEnum):
    """Result of comparing a reliable or provisional subtotal with a budget."""

    WITHIN_BUDGET = "within_budget"
    BREACHED = "breached"
    INDETERMINATE = "indeterminate"


CURRENT_MEMORY_SNAPSHOT_SCHEMA_VERSION: Final[int] = 2
SUPPORTED_MEMORY_SNAPSHOT_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset(
    {CURRENT_MEMORY_SNAPSHOT_SCHEMA_VERSION}
)
_MEMORY_CATEGORIES = frozenset(
    {
        "model_weights",
        "kv_cache",
        "activations",
        "embedding_reranker_memory",
        "index_memory",
        "framework_overhead",
        "pinned_host_memory",
        "storage_cache",
    }
)


def _validate_safe_identifier(value: str, field_name: str) -> None:
    """Validate a bounded identifier suitable for operational correlation."""
    validate_non_empty_string(value, field_name)
    if len(value) > 128:
        raise ValueError(f"{field_name} must be at most 128 characters")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
        for character in value
    ):
        raise ValueError(f"{field_name} must contain only safe identifier characters")


@dataclass(frozen=True, slots=True)
class MemoryAdapterIdentity:
    """Stable identity for the adapter and calibration that produced a value."""

    adapter_name: str
    adapter_version: str
    calibration_revision: str

    def __post_init__(self) -> None:
        _validate_safe_identifier(self.adapter_name, "adapter_name")
        _validate_safe_identifier(self.adapter_version, "adapter_version")
        _validate_safe_identifier(self.calibration_revision, "calibration_revision")


@dataclass(frozen=True, slots=True)
class MemoryCollectorIdentity:
    """Stable identity for the policy that assembled a memory snapshot."""

    collector_name: str
    collector_version: str
    inventory_revision: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the collector identity's fields."""
        _validate_safe_identifier(self.collector_name, "collector_name")
        _validate_safe_identifier(self.collector_version, "collector_version")
        _validate_safe_identifier(self.inventory_revision, "inventory_revision")


@dataclass(frozen=True, slots=True)
class ExpectedMemoryContribution:
    """One planned exclusive component/category contribution in a memory space."""

    component_id: str
    component_role: ComponentRole
    category: ResourceCategory
    memory_space: MemorySpace
    device_id: str | None
    adapter: MemoryAdapterIdentity

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate expected contribution types, units, and identity rules."""
        _validate_safe_identifier(self.component_id, "component_id")
        if not isinstance(self.component_role, ComponentRole):
            raise TypeError("component_role must be a ComponentRole")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if self.category not in _MEMORY_CATEGORIES:
            raise ValueError("category must be a supported memory category")
        if not isinstance(self.memory_space, MemorySpace):
            raise TypeError("memory_space must be a MemorySpace")
        if self.device_id is not None:
            _validate_safe_identifier(self.device_id, "device_id")
        if not isinstance(self.adapter, MemoryAdapterIdentity):
            raise TypeError("adapter must be MemoryAdapterIdentity")


class MemoryOverlapPolicy(StrEnum):
    """Policy for preventing exclusive contributions from double counting."""

    DISALLOW = "disallow"


@dataclass(frozen=True, slots=True)
class MemoryMeasurement:
    """One point-in-time, component-attributed memory observation.

    ``value_bytes`` may be retained for an unreliable observation, but an
    unavailable observation never carries a value. Roll-up coverage is
    validated for shape here; graph and reconciliation validation belongs to
    the snapshot accountant.
    """

    measurement_id: str
    component_id: str
    component_role: ComponentRole
    category: ResourceCategory
    memory_space: MemorySpace
    device_id: str | None
    status: MeasurementStatus
    value_bytes: int | None
    adapter: MemoryAdapterIdentity
    contribution_kind: ContributionKind = ContributionKind.EXCLUSIVE
    covers_measurement_ids: tuple[str, ...] = ()
    observed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate measurement types, units, identity, and status rules."""
        _validate_safe_identifier(self.measurement_id, "measurement_id")
        _validate_safe_identifier(self.component_id, "component_id")
        if not isinstance(self.component_role, ComponentRole):
            raise TypeError("component_role must be a ComponentRole")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if self.category not in {
            "model_weights",
            "kv_cache",
            "activations",
            "embedding_reranker_memory",
            "index_memory",
            "framework_overhead",
            "pinned_host_memory",
            "storage_cache",
        }:
            raise ValueError("category must be a supported memory category")
        if not isinstance(self.memory_space, MemorySpace):
            raise TypeError("memory_space must be a MemorySpace")
        if self.device_id is not None:
            _validate_safe_identifier(self.device_id, "device_id")
        if not isinstance(self.status, str):
            raise TypeError("status must be a string")
        if self.status not in {"measured", "unavailable", "unreliable"}:
            raise ValueError("status must be a supported measurement status")
        if self.value_bytes is not None:
            if isinstance(self.value_bytes, bool) or not isinstance(
                self.value_bytes, int
            ):
                raise TypeError("value_bytes must be an integer or None")
            if self.value_bytes < 0:
                raise ValueError("value_bytes must not be negative")
        if self.status == "measured" and self.value_bytes is None:
            raise ValueError("measured memory must have a value")
        if self.status == "unavailable" and self.value_bytes is not None:
            raise ValueError("unavailable memory must not have a value")
        if (
            self.memory_space is MemorySpace.ACCELERATOR
            and self.device_id is None
            and self.status == "measured"
        ):
            raise ValueError("measured accelerator memory must identify its device")
        if not isinstance(self.adapter, MemoryAdapterIdentity):
            raise TypeError("adapter must be MemoryAdapterIdentity")
        if not isinstance(self.contribution_kind, ContributionKind):
            raise TypeError("contribution_kind must be a ContributionKind")
        if not isinstance(self.covers_measurement_ids, tuple):
            raise TypeError("covers_measurement_ids must be a tuple")
        for covered_id in self.covers_measurement_ids:
            _validate_safe_identifier(covered_id, "covered measurement ID")
        if len(set(self.covers_measurement_ids)) != len(self.covers_measurement_ids):
            raise ValueError("covers_measurement_ids must not contain duplicates")
        if self.contribution_kind is ContributionKind.EXCLUSIVE:
            if self.covers_measurement_ids:
                raise ValueError("exclusive measurements must not cover contributions")
        elif not self.covers_measurement_ids:
            raise ValueError("roll-up measurements must cover contributions")
        if self.measurement_id in self.covers_measurement_ids:
            raise ValueError("a measurement cannot cover itself")
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        validate_utc_datetime(self.observed_at, "observed_at")


@runtime_checkable
class MemoryAccountingAdapter(Protocol):
    """Optional one-shot capability for component memory observations."""

    def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
        """Return one immutable point-in-time memory observation set."""
        ...


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
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isinstance(self.reconciled, bool):
            raise TypeError("reconciled must be a boolean")


@dataclass(frozen=True, slots=True)
class MemoryTotal:
    """Authoritative or diagnostic subtotal for one memory-space grouping."""

    memory_space: MemorySpace
    device_id: str | None
    category: ResourceCategory | None
    component_role: ComponentRole | None
    measured_bytes: int
    unreliable_bytes: int
    measured_count: int
    unreliable_count: int
    unavailable_count: int
    missing_count: int
    completeness: Completeness

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the memory total's fields."""
        if not isinstance(self.memory_space, MemorySpace):
            raise TypeError("memory_space must be a MemorySpace")
        if self.device_id is not None:
            _validate_safe_identifier(self.device_id, "device_id")
        if self.category is not None and not isinstance(self.category, str):
            raise TypeError("category must be a string or None")
        if self.component_role is not None and not isinstance(
            self.component_role, ComponentRole
        ):
            raise TypeError("component_role must be a ComponentRole or None")
        if self.category is None and self.component_role is not None:
            raise ValueError("component role subtotals require a category")
        if self.category is not None and self.category not in _MEMORY_CATEGORIES:
            raise ValueError("category must be a supported memory category")
        for value, field_name in (
            (self.measured_bytes, "measured_bytes"),
            (self.unreliable_bytes, "unreliable_bytes"),
            (self.measured_count, "measured_count"),
            (self.unreliable_count, "unreliable_count"),
            (self.unavailable_count, "unavailable_count"),
            (self.missing_count, "missing_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isinstance(self.completeness, Completeness):
            raise TypeError("completeness must be a Completeness value")
        expected = (
            Completeness.COMPLETE
            if self.unreliable_count == 0
            and self.unavailable_count == 0
            and self.missing_count == 0
            else Completeness.INCOMPLETE
        )
        if self.completeness is not expected:
            raise ValueError("completeness does not match contribution statuses")


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """Integer-byte budget bound to one memory space and device."""

    memory_space: MemorySpace
    device_id: str | None
    budget_bytes: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the memory budget's fields."""
        if not isinstance(self.memory_space, MemorySpace):
            raise TypeError("memory_space must be a MemorySpace")
        if self.device_id is not None:
            _validate_safe_identifier(self.device_id, "device_id")
        if isinstance(self.budget_bytes, bool) or not isinstance(
            self.budget_bytes, int
        ):
            raise TypeError("budget_bytes must be an integer")
        if self.budget_bytes < 0:
            raise ValueError("budget_bytes must not be negative")

    @classmethod
    def from_decimal_gigabytes(
        cls,
        *,
        memory_space: MemorySpace,
        device_id: str | None,
        gigabytes: int | float | Decimal,
    ) -> "MemoryBudget":
        """Convert decimal gigabytes to canonical integer bytes exactly once."""
        if isinstance(gigabytes, bool) or not isinstance(
            gigabytes, (int, float, Decimal)
        ):
            raise TypeError("gigabytes must be an integer, float, or Decimal")
        try:
            decimal_gigabytes = Decimal(str(gigabytes))
        except (InvalidOperation, ValueError) as error:
            raise ValueError("gigabytes must be finite and non-negative") from error
        if not decimal_gigabytes.is_finite() or decimal_gigabytes < 0:
            raise ValueError("gigabytes must be finite and non-negative")
        bytes_value = decimal_gigabytes * Decimal(1_000_000_000)
        if bytes_value != bytes_value.to_integral_value():
            raise ValueError("gigabytes conversion must produce whole bytes")
        return cls(
            memory_space=memory_space,
            device_id=device_id,
            budget_bytes=int(bytes_value),
        )


@dataclass(frozen=True, slots=True)
class BudgetComparison:
    """Typed budget comparison including incomplete-measurement diagnostics."""

    state: BudgetState
    budget: MemoryBudget
    measured_bytes: int
    provisional_headroom_bytes: int
    measured_count: int
    unreliable_count: int
    unavailable_count: int
    missing_count: int
    completeness: Completeness

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the budget comparison's fields."""
        if not isinstance(self.state, BudgetState):
            raise TypeError("state must be a BudgetState value")
        if not isinstance(self.budget, MemoryBudget):
            raise TypeError("budget must be a MemoryBudget")
        for value, field_name in (
            (self.measured_bytes, "measured_bytes"),
            (self.provisional_headroom_bytes, "provisional_headroom_bytes"),
            (self.measured_count, "measured_count"),
            (self.unreliable_count, "unreliable_count"),
            (self.unavailable_count, "unavailable_count"),
            (self.missing_count, "missing_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if field_name != "provisional_headroom_bytes" and value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isinstance(self.completeness, Completeness):
            raise TypeError("completeness must be a Completeness value")
        expected_headroom = self.budget.budget_bytes - self.measured_bytes
        if self.provisional_headroom_bytes != expected_headroom:
            raise ValueError("provisional headroom does not match budget and bytes")
        expected_completeness = self._completeness_for_counts(
            unreliable_count=self.unreliable_count,
            unavailable_count=self.unavailable_count,
            missing_count=self.missing_count,
        )
        if self.completeness is not expected_completeness:
            raise ValueError("completeness does not match contribution counts")
        if self.completeness is Completeness.COMPLETE and self.measured_count == 0:
            raise ValueError("complete comparison requires a measured contribution")
        if self.measured_bytes > 0 and self.measured_count == 0:
            raise ValueError("positive measured bytes require a measured contribution")
        expected_state = self._state_for(
            measured_bytes=self.measured_bytes,
            budget_bytes=self.budget.budget_bytes,
            completeness=self.completeness,
        )
        if self.state is not expected_state:
            raise ValueError("state does not match budget and completeness")

    @classmethod
    def from_accounting(
        cls,
        *,
        budget: MemoryBudget,
        measured_bytes: int,
        measured_count: int,
        unreliable_count: int,
        unavailable_count: int,
        missing_count: int,
    ) -> "BudgetComparison":
        """Construct a comparison from authoritative accounting inputs."""
        if not isinstance(budget, MemoryBudget):
            raise TypeError("budget must be a MemoryBudget")
        for value, field_name in (
            (measured_bytes, "measured_bytes"),
            (measured_count, "measured_count"),
            (unreliable_count, "unreliable_count"),
            (unavailable_count, "unavailable_count"),
            (missing_count, "missing_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        completeness = cls._completeness_for_counts(
            unreliable_count=unreliable_count,
            unavailable_count=unavailable_count,
            missing_count=missing_count,
        )
        return cls(
            state=cls._state_for(
                measured_bytes=measured_bytes,
                budget_bytes=budget.budget_bytes,
                completeness=completeness,
            ),
            budget=budget,
            measured_bytes=measured_bytes,
            provisional_headroom_bytes=budget.budget_bytes - measured_bytes,
            measured_count=measured_count,
            unreliable_count=unreliable_count,
            unavailable_count=unavailable_count,
            missing_count=missing_count,
            completeness=completeness,
        )

    @staticmethod
    def _completeness_for_counts(
        *,
        unreliable_count: int,
        unavailable_count: int,
        missing_count: int,
    ) -> Completeness:
        if unreliable_count == 0 and unavailable_count == 0 and missing_count == 0:
            return Completeness.COMPLETE
        return Completeness.INCOMPLETE

    @staticmethod
    def _state_for(
        *,
        measured_bytes: int,
        budget_bytes: int,
        completeness: Completeness,
    ) -> BudgetState:
        if measured_bytes > budget_bytes:
            return BudgetState.BREACHED
        if completeness is Completeness.INCOMPLETE:
            return BudgetState.INDETERMINATE
        return BudgetState.WITHIN_BUDGET


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
    schema_version: int = CURRENT_MEMORY_SNAPSHOT_SCHEMA_VERSION
    _reconciliations: tuple[RollupReconciliation, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "measurements",
            tuple(sorted(self.measurements, key=lambda m: m.measurement_id)),
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

    def _validate(self) -> None:
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
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be an integer")
        if self.schema_version not in SUPPORTED_MEMORY_SNAPSHOT_SCHEMA_VERSIONS:
            raise ValueError("unsupported memory snapshot schema version")
        for correlation_id, field_name in (
            (self.run_id, "run_id"),
            (self.request_id, "request_id"),
        ):
            if correlation_id is not None:
                _validate_safe_identifier(correlation_id, field_name)
        expected_by_slot: dict[
            tuple[str, ComponentRole, ResourceCategory, MemorySpace, str | None],
            ExpectedMemoryContribution,
        ] = {}
        for expected in self.expected_contributions:
            if not isinstance(expected, ExpectedMemoryContribution):
                raise TypeError(
                    "expected_contributions must contain "
                    "ExpectedMemoryContribution values"
                )
            slot = self._expected_contribution_slot(expected)
            if slot in expected_by_slot:
                raise ValueError("expected contribution slots must not overlap")
            expected_by_slot[slot] = expected

        measurements_by_id: dict[str, MemoryMeasurement] = {}
        observed_exclusive_slots: set[
            tuple[str, ComponentRole, ResourceCategory, MemorySpace, str | None]
        ] = set()
        for measurement in self.measurements:
            if not isinstance(measurement, MemoryMeasurement):
                raise TypeError("measurements must contain MemoryMeasurement values")
            if measurement.measurement_id in measurements_by_id:
                raise ValueError("measurement IDs must be unique")
            measurements_by_id[measurement.measurement_id] = measurement
            if measurement.observed_at > self.observed_at:
                raise ValueError("measurements must not be after snapshot observed_at")
            if measurement.contribution_kind is ContributionKind.EXCLUSIVE:
                slot = self._measurement_slot(measurement)
                expected = expected_by_slot.get(slot)
                if expected is None:
                    raise ValueError("exclusive measurement was not expected")
                if measurement.adapter != expected.adapter:
                    raise ValueError("measurement does not use the expected adapter")
                if slot in observed_exclusive_slots:
                    raise ValueError("exclusive measurement slots must not overlap")
                observed_exclusive_slots.add(slot)

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

        self._validate_no_rollup_cycles(measurements_by_id)

    @staticmethod
    def _expected_contribution_slot(
        expected: ExpectedMemoryContribution,
    ) -> tuple[str, ComponentRole, ResourceCategory, MemorySpace, str | None]:
        return (
            expected.component_id,
            expected.component_role,
            expected.category,
            expected.memory_space,
            expected.device_id,
        )

    @staticmethod
    def _measurement_slot(
        measurement: MemoryMeasurement,
    ) -> tuple[str, ComponentRole, ResourceCategory, MemorySpace, str | None]:
        return (
            measurement.component_id,
            measurement.component_role,
            measurement.category,
            measurement.memory_space,
            measurement.device_id,
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

        def visit(measurement_id: str) -> None:
            if measurement_id in visiting:
                raise ValueError("roll-up coverage must be acyclic")
            if measurement_id in visited:
                return
            visiting.add(measurement_id)
            measurement = measurements_by_id[measurement_id]
            for covered_id in measurement.covers_measurement_ids:
                visit(covered_id)
            visiting.remove(measurement_id)
            visited.add(measurement_id)

        for measurement_id in measurements_by_id:
            visit(measurement_id)

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
        """Return per-space, category, and component-role counted totals."""
        counted_by_slot = {
            self._measurement_slot(measurement): measurement
            for measurement in self.measurements
            if measurement.contribution_kind is ContributionKind.EXCLUSIVE
        }
        keys: set[
            tuple[
                MemorySpace, str | None, ResourceCategory | None, ComponentRole | None
            ]
        ] = set()
        for expected in self.expected_contributions:
            base = (
                expected.memory_space,
                expected.device_id,
            )
            keys.add((*base, None, None))
            keys.add((*base, expected.category, None))
            keys.add((*base, expected.category, expected.component_role))

        totals: list[MemoryTotal] = []
        for memory_space, device_id, category, component_role in keys:
            selected_expected = tuple(
                expected
                for expected in self.expected_contributions
                if expected.memory_space is memory_space
                and (
                    expected.device_id == device_id
                    or (
                        memory_space is MemorySpace.ACCELERATOR
                        and device_id is not None
                        and expected.device_id is None
                    )
                )
                and (category is None or expected.category == category)
                and (
                    component_role is None or expected.component_role is component_role
                )
            )
            selected = tuple(
                counted_by_slot[self._expected_contribution_slot(expected)]
                for expected in selected_expected
                if self._expected_contribution_slot(expected) in counted_by_slot
            )
            measured = tuple(item for item in selected if item.status == "measured")
            unreliable = tuple(item for item in selected if item.status == "unreliable")
            unavailable_count = sum(item.status == "unavailable" for item in selected)
            missing_count = len(selected_expected) - len(selected)
            totals.append(
                MemoryTotal(
                    memory_space=memory_space,
                    device_id=device_id,
                    category=category,
                    component_role=component_role,
                    measured_bytes=sum(item.value_bytes or 0 for item in measured),
                    unreliable_bytes=sum(item.value_bytes or 0 for item in unreliable),
                    measured_count=len(measured),
                    unreliable_count=len(unreliable),
                    unavailable_count=unavailable_count,
                    missing_count=missing_count,
                    completeness=(
                        Completeness.COMPLETE
                        if not unreliable
                        and unavailable_count == 0
                        and missing_count == 0
                        else Completeness.INCOMPLETE
                    ),
                )
            )
        return tuple(
            sorted(
                totals,
                key=lambda total: (
                    total.memory_space.value,
                    total.device_id or "",
                    total.category or "",
                    total.component_role.value if total.component_role else "",
                ),
            )
        )

    def compare_budget(self, budget: MemoryBudget) -> BudgetComparison:
        """Compare one budget with its matching per-space subtotal."""
        if not isinstance(budget, MemoryBudget):
            raise TypeError("budget must be a MemoryBudget")
        totals = self.totals
        matching = tuple(
            total
            for total in totals
            if total.memory_space is budget.memory_space
            and total.device_id == budget.device_id
            and total.category is None
            and total.component_role is None
        )
        unresolved = tuple(
            total
            for total in totals
            if budget.memory_space is MemorySpace.ACCELERATOR
            and budget.device_id is not None
            and total.memory_space is MemorySpace.ACCELERATOR
            and total.device_id is None
            and total.category is None
            and total.component_role is None
        )
        if not matching:
            available_spaces = {
                (total.memory_space, total.device_id)
                for total in totals
                if total.category is None and total.component_role is None
            }
            if (
                not unresolved
                and (
                    budget.memory_space,
                    budget.device_id,
                )
                not in available_spaces
            ):
                raise ValueError("budget has no matching memory space/device subtotal")
            measured_bytes = 0
            measured_count = 0
            if unresolved:
                unresolved_total = unresolved[0]
                unreliable_count = unresolved_total.unreliable_count
                unavailable_count = unresolved_total.unavailable_count
                missing_count = unresolved_total.missing_count
            else:
                unreliable_count = 0
                unavailable_count = 0
                missing_count = 0
        else:
            total = matching[0]
            measured_bytes = total.measured_bytes
            measured_count = total.measured_count
            unreliable_count = total.unreliable_count
            unavailable_count = total.unavailable_count
            missing_count = total.missing_count
        return BudgetComparison.from_accounting(
            budget=budget,
            measured_bytes=measured_bytes,
            measured_count=measured_count,
            unreliable_count=unreliable_count,
            unavailable_count=unavailable_count,
            missing_count=missing_count,
        )
