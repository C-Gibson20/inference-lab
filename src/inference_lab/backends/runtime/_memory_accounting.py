"""Derived totals and budgets for runtime memory accounting."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from inference_lab.backends.runtime._memory_values import (
    ContributionKind,
    ExpectedMemoryContribution,
    MemoryComponentRole,
    MemoryMeasurement,
    MemorySpace,
    _validate_safe_identifier,
)
from inference_lab.backends.runtime._validation import (
    validate_integer,
    validate_non_negative_integer,
)
from inference_lab.profiling.resources import (
    MEMORY_RESOURCE_CATEGORIES,
    MemoryResourceCategory,
)

type _ContributionSlot = tuple[
    str,
    MemoryComponentRole,
    MemoryResourceCategory,
    MemorySpace,
    str | None,
]
type _TotalKey = tuple[
    MemorySpace,
    str | None,
    MemoryResourceCategory | None,
    MemoryComponentRole | None,
]
type _AccountingCounts = tuple[int, int, int, int, int]


def _expected_contribution_slot(
    expected: ExpectedMemoryContribution,
) -> _ContributionSlot:
    return (
        expected.component_id,
        expected.component_role,
        expected.category,
        expected.memory_space,
        expected.device_id,
    )


def _measurement_slot(measurement: MemoryMeasurement) -> _ContributionSlot:
    return (
        measurement.component_id,
        measurement.component_role,
        measurement.category,
        measurement.memory_space,
        measurement.device_id,
    )


class Completeness(StrEnum):
    """Reliability state of an authoritative memory subtotal."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class BudgetState(StrEnum):
    """Result of comparing a reliable or provisional subtotal with a budget."""

    WITHIN_BUDGET = "within_budget"
    BREACHED = "breached"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class MemoryTotal:
    """Authoritative or diagnostic subtotal for one memory-space grouping."""

    memory_space: MemorySpace
    device_id: str | None
    category: MemoryResourceCategory | None
    component_role: MemoryComponentRole | None
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
            self.component_role, MemoryComponentRole
        ):
            raise TypeError("component_role must be a MemoryComponentRole or None")
        if self.category is None and self.component_role is not None:
            raise ValueError("component role subtotals require a category")
        if (
            self.category is not None
            and self.category not in MEMORY_RESOURCE_CATEGORIES
        ):
            raise ValueError("category must be a supported memory category")
        for value, field_name in (
            (self.measured_bytes, "measured_bytes"),
            (self.unreliable_bytes, "unreliable_bytes"),
            (self.measured_count, "measured_count"),
            (self.unreliable_count, "unreliable_count"),
            (self.unavailable_count, "unavailable_count"),
            (self.missing_count, "missing_count"),
        ):
            validate_non_negative_integer(value, field_name)
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


def _build_memory_totals(
    expected_contributions: tuple[ExpectedMemoryContribution, ...],
    measurements: tuple[MemoryMeasurement, ...],
) -> tuple[MemoryTotal, ...]:
    """Build deterministic totals from a validated contribution inventory."""
    counted_by_slot = {
        _measurement_slot(measurement): measurement
        for measurement in measurements
        if measurement.contribution_kind is ContributionKind.EXCLUSIVE
    }
    totals = (
        _build_memory_total(key, expected_contributions, counted_by_slot)
        for key in _total_keys(expected_contributions)
    )
    return tuple(sorted(totals, key=_memory_total_sort_key))


def _total_keys(
    expected_contributions: tuple[ExpectedMemoryContribution, ...],
) -> set[_TotalKey]:
    keys: set[_TotalKey] = set()
    for expected in expected_contributions:
        base = (expected.memory_space, expected.device_id)
        keys.add((*base, None, None))
        keys.add((*base, expected.category, None))
        keys.add((*base, expected.category, expected.component_role))
    return keys


def _build_memory_total(
    key: _TotalKey,
    expected_contributions: tuple[ExpectedMemoryContribution, ...],
    counted_by_slot: dict[_ContributionSlot, MemoryMeasurement],
) -> MemoryTotal:
    memory_space, device_id, category, component_role = key
    selected_expected = tuple(
        expected
        for expected in expected_contributions
        if _expected_matches_total(expected, key)
    )
    selected = tuple(
        counted_by_slot[_expected_contribution_slot(expected)]
        for expected in selected_expected
        if _expected_contribution_slot(expected) in counted_by_slot
    )
    measured = tuple(item for item in selected if item.status == "measured")
    unreliable = tuple(item for item in selected if item.status == "unreliable")
    unavailable_count = sum(item.status == "unavailable" for item in selected)
    missing_count = len(selected_expected) - len(selected)
    return MemoryTotal(
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
            if not unreliable and unavailable_count == 0 and missing_count == 0
            else Completeness.INCOMPLETE
        ),
    )


def _expected_matches_total(
    expected: ExpectedMemoryContribution,
    key: _TotalKey,
) -> bool:
    memory_space, device_id, category, component_role = key
    return (
        expected.memory_space is memory_space
        and (
            expected.device_id == device_id
            or (
                memory_space is MemorySpace.ACCELERATOR
                and device_id is not None
                and expected.device_id is None
            )
        )
        and (category is None or expected.category == category)
        and (component_role is None or expected.component_role is component_role)
    )


def _memory_total_sort_key(total: MemoryTotal) -> tuple[str, str, str, str]:
    return (
        total.memory_space.value,
        total.device_id or "",
        total.category or "",
        total.component_role.value if total.component_role else "",
    )


def _compare_memory_budget(
    totals: tuple[MemoryTotal, ...],
    budget: "MemoryBudget",
) -> "BudgetComparison":
    """Compare a budget with precomputed space totals."""
    if not isinstance(budget, MemoryBudget):
        raise TypeError("budget must be a MemoryBudget")
    (
        measured_bytes,
        measured_count,
        unreliable_count,
        unavailable_count,
        missing_count,
    ) = _accounting_counts_for_budget(totals, budget)
    return BudgetComparison.from_accounting(
        budget=budget,
        measured_bytes=measured_bytes,
        measured_count=measured_count,
        unreliable_count=unreliable_count,
        unavailable_count=unavailable_count,
        missing_count=missing_count,
    )


def _accounting_counts_for_budget(
    totals: tuple[MemoryTotal, ...],
    budget: "MemoryBudget",
) -> _AccountingCounts:
    matching = next(
        (
            total
            for total in totals
            if total.memory_space is budget.memory_space
            and total.device_id == budget.device_id
            and total.category is None
            and total.component_role is None
        ),
        None,
    )
    if matching is not None:
        return (
            matching.measured_bytes,
            matching.measured_count,
            matching.unreliable_count,
            matching.unavailable_count,
            matching.missing_count,
        )

    unresolved = next(
        (
            total
            for total in totals
            if budget.memory_space is MemorySpace.ACCELERATOR
            and budget.device_id is not None
            and total.memory_space is MemorySpace.ACCELERATOR
            and total.device_id is None
            and total.category is None
            and total.component_role is None
        ),
        None,
    )
    if unresolved is not None:
        return (
            0,
            0,
            unresolved.unreliable_count,
            unresolved.unavailable_count,
            unresolved.missing_count,
        )

    available_spaces = {
        (total.memory_space, total.device_id)
        for total in totals
        if total.category is None and total.component_role is None
    }
    if (budget.memory_space, budget.device_id) not in available_spaces:
        raise ValueError("budget has no matching memory space/device subtotal")
    return (0, 0, 0, 0, 0)


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
        validate_non_negative_integer(self.budget_bytes, "budget_bytes")

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
            if field_name == "provisional_headroom_bytes":
                validate_integer(value, field_name)
            else:
                validate_non_negative_integer(value, field_name)
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
            validate_non_negative_integer(value, field_name)
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
