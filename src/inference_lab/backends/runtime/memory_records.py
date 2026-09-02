"""Privacy-safe profiling-shaped records for runtime memory accounting."""

from dataclasses import dataclass
from datetime import datetime

from inference_lab.backends.runtime.memory import (
    BudgetComparison,
    BudgetState,
    Completeness,
    MemoryMeasurement,
)
from inference_lab.profiling import ProfilingLogger, emit_record
from inference_lab.profiling.types import LogLevel
from inference_lab.profiling.versioning import CURRENT_RECORD_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MemoryComponentRecord:
    """Safe structured record describing one component observation."""

    measurement: MemoryMeasurement

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MemoryMeasurement):
            raise TypeError("measurement must be a MemoryMeasurement")

    def to_dict(self) -> dict[str, object]:
        """Return bounded operational fields for the profiling emitter."""
        measurement = self.measurement
        return {
            "event": "memory_component_measured",
            "level": self.level,
            "module": "runtime",
            "pipeline_stage": "memory_accounting",
            "document_id": None,
            "corpus_id": None,
            "corpus_version": None,
            "measurement_id": measurement.measurement_id,
            "component_id": measurement.component_id,
            "component_role": measurement.component_role.value,
            "category": measurement.category,
            "memory_space": measurement.memory_space.value,
            "device_id": measurement.device_id,
            "status": measurement.status,
            "value_bytes": measurement.value_bytes,
            "contribution_kind": measurement.contribution_kind.value,
            "covered_measurement_count": len(measurement.covers_measurement_ids),
            "adapter_name": measurement.adapter.adapter_name,
            "adapter_version": measurement.adapter.adapter_version,
            "calibration_revision": measurement.adapter.calibration_revision,
            "observed_at": measurement.observed_at.isoformat(),
            "record_schema_version": CURRENT_RECORD_SCHEMA_VERSION,
        }

    @property
    def level(self) -> LogLevel:
        """Return warning for unavailable or unreliable observations."""
        if self.measurement.status in {"unavailable", "unreliable"}:
            return "warning"
        return "info"


@dataclass(frozen=True, slots=True)
class MemoryBudgetRecord:
    """Safe structured record describing one budget comparison."""

    comparison: BudgetComparison
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, BudgetComparison):
            raise TypeError("comparison must be a BudgetComparison")
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")

    def to_dict(self) -> dict[str, object]:
        """Return bounded operational fields for the profiling emitter."""
        comparison = self.comparison
        budget = comparison.budget
        return {
            "event": "memory_budget_compared",
            "level": self.level,
            "module": "runtime",
            "pipeline_stage": "memory_accounting",
            "document_id": None,
            "corpus_id": None,
            "corpus_version": None,
            "state": comparison.state.value,
            "completeness": comparison.completeness.value,
            "memory_space": budget.memory_space.value,
            "device_id": budget.device_id,
            "budget_bytes": budget.budget_bytes,
            "measured_bytes": comparison.measured_bytes,
            "provisional_headroom_bytes": comparison.provisional_headroom_bytes,
            "measured_count": comparison.measured_count,
            "unreliable_count": comparison.unreliable_count,
            "unavailable_count": comparison.unavailable_count,
            "observed_at": self.observed_at.isoformat(),
            "record_schema_version": CURRENT_RECORD_SCHEMA_VERSION,
        }

    @property
    def level(self) -> LogLevel:
        """Return severity implied by budget and measurement completeness."""
        if self.comparison.state is BudgetState.BREACHED:
            return "error"
        if self.comparison.completeness is Completeness.INCOMPLETE:
            return "warning"
        return "info"


def emit_memory_measurement(
    logger: ProfilingLogger,
    measurement: MemoryMeasurement,
) -> None:
    """Emit one memory observation through the shared profiling emitter."""
    emit_record(logger, MemoryComponentRecord(measurement))


def emit_memory_budget_comparison(
    logger: ProfilingLogger,
    comparison: BudgetComparison,
    *,
    observed_at: datetime,
) -> None:
    """Emit one budget comparison through the shared profiling emitter."""
    emit_record(logger, MemoryBudgetRecord(comparison, observed_at))
