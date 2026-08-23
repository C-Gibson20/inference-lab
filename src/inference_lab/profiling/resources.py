"""Normalised resource-measurement records for profiling."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal, get_args

from inference_lab.profiling._validation import validate_utc_datetime
from inference_lab.profiling.types import LogLevel
from inference_lab.profiling.utils import utc_now
from inference_lab.profiling.versioning import (
    CURRENT_RECORD_SCHEMA_VERSION,
    validate_record_schema_version,
)

ResourceCategory = Literal[
    "model_weights",
    "kv_cache",
    "activations",
    "embedding_reranker_memory",
    "index_memory",
    "framework_overhead",
    "pinned_host_memory",
    "storage_cache",
    "gpu_utilisation",
    "cpu_utilisation",
    "storage_throughput",
    "power",
    "thermal",
]
MeasurementStatus = Literal["measured", "unavailable", "unreliable"]
MeasurementKind = Literal["point_in_time", "sampled", "interval"]
ResourceUnit = Literal[
    "bytes",
    "percent",
    "bytes_per_second",
    "watts",
    "celsius",
]

_RESOURCE_CATEGORIES: Final[frozenset[ResourceCategory]] = frozenset(
    get_args(ResourceCategory)
)
_MEASUREMENT_STATUSES: Final[frozenset[MeasurementStatus]] = frozenset(
    get_args(MeasurementStatus)
)
_MEASUREMENT_KINDS: Final[frozenset[MeasurementKind]] = frozenset(
    get_args(MeasurementKind)
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
_RESOURCE_UNITS: dict[ResourceCategory, ResourceUnit] = {
    "model_weights": "bytes",
    "kv_cache": "bytes",
    "activations": "bytes",
    "embedding_reranker_memory": "bytes",
    "index_memory": "bytes",
    "framework_overhead": "bytes",
    "pinned_host_memory": "bytes",
    "storage_cache": "bytes",
    "gpu_utilisation": "percent",
    "cpu_utilisation": "percent",
    "storage_throughput": "bytes_per_second",
    "power": "watts",
    "thermal": "celsius",
}

if _RESOURCE_UNITS.keys() != _RESOURCE_CATEGORIES:
    raise RuntimeError("resource units must cover every resource category")


def _validate_identity_and_status(
    *,
    category: ResourceCategory,
    module: str,
    pipeline_stage: str | None,
    stage_id: str | None,
    status: MeasurementStatus,
    measurement_kind: MeasurementKind,
) -> None:
    """Validate resource discriminants and operational context fields."""
    if category not in _RESOURCE_CATEGORIES:
        raise ValueError("category must be a supported resource category")
    if not module:
        raise ValueError("module must not be empty")
    if pipeline_stage == "":
        raise ValueError("pipeline_stage must not be empty")
    if stage_id == "":
        raise ValueError("stage_id must not be empty")
    if status not in _MEASUREMENT_STATUSES:
        raise ValueError("status must be a valid measurement status")
    if measurement_kind not in _MEASUREMENT_KINDS:
        raise ValueError("measurement_kind must be valid")


def _validate_measurement_value(
    value: int | float | None,
    status: MeasurementStatus,
) -> None:
    """Validate numeric shape and presence rules for a measurement status."""
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be a finite number or None")
        if not math.isfinite(float(value)):
            raise ValueError("value must be a finite number or None")
    if status == "measured" and value is None:
        raise ValueError("measured resources must have a value")
    if status == "unavailable" and value is not None:
        raise ValueError("unavailable resources must not have a value")


def _validate_value_range(
    category: ResourceCategory,
    value: int | float | None,
) -> None:
    """Validate the physical range implied by the resource category."""
    if value is None:
        return
    if category in {"gpu_utilisation", "cpu_utilisation"}:
        if not 0 <= value <= 100:
            raise ValueError("utilisation percentage must be between 0 and 100")
    elif category == "thermal":
        if value < -273.15:
            raise ValueError("thermal measurement cannot be below absolute zero")
    elif category in _MEMORY_CATEGORIES or category in {
        "storage_throughput",
        "power",
    }:
        if value < 0:
            raise ValueError(f"{category} measurement must be non-negative")


def _validate_sampling(
    measurement_kind: MeasurementKind,
    sample_duration_ms: float | None,
) -> None:
    """Validate the sample duration required by the measurement kind."""
    if measurement_kind == "point_in_time" and sample_duration_ms is not None:
        raise ValueError("point-in-time measurements must not have a sample duration")
    if measurement_kind != "point_in_time" and (
        isinstance(sample_duration_ms, bool)
        or not isinstance(sample_duration_ms, (int, float))
        or not math.isfinite(sample_duration_ms)
        or sample_duration_ms <= 0
    ):
        raise ValueError(
            "sampled and interval measurements require a positive sample "
            "duration expressed as a finite number"
        )


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """One normalised measurement for one runtime resource category.

    A record may contain a value for an unreliable measurement, allowing
    consumers to retain the observation without treating it as authoritative.
    An unavailable measurement has no value. Every category has a canonical
    unit, including when a measurement is unavailable.
    """

    category: ResourceCategory
    value: int | float | None
    status: MeasurementStatus = "measured"
    measurement_kind: MeasurementKind = "point_in_time"
    observed_at: datetime = field(default_factory=utc_now)
    module: str = "profiling"
    pipeline_stage: str | None = None
    stage_id: str | None = None
    sample_duration_ms: float | None = None
    run_id: str | None = None
    request_id: str | None = None
    document_id: str | None = None
    corpus_version: str | None = None
    record_schema_version: int = CURRENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate resource category, value, status, and timing invariants."""
        _validate_identity_and_status(
            category=self.category,
            module=self.module,
            pipeline_stage=self.pipeline_stage,
            stage_id=self.stage_id,
            status=self.status,
            measurement_kind=self.measurement_kind,
        )
        _validate_measurement_value(self.value, self.status)
        _validate_value_range(self.category, self.value)
        _validate_sampling(self.measurement_kind, self.sample_duration_ms)
        validate_utc_datetime(self.observed_at, "observed_at")
        validate_record_schema_version(self.record_schema_version)

    def to_dict(self) -> dict[str, object]:
        """Return the measurement as structured logging fields."""
        return {
            "event": "resource_measured",
            "category": self.category,
            "unit": self.unit,
            "value": self.value,
            "status": self.status,
            "level": self.level,
            "measurement_kind": self.measurement_kind,
            "observed_at": self.observed_at.isoformat(),
            "module": self.module,
            "pipeline_stage": self.pipeline_stage,
            "stage_id": self.stage_id,
            "sample_duration_ms": self.sample_duration_ms,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "corpus_version": self.corpus_version,
            "record_schema_version": self.record_schema_version,
        }

    @property
    def level(self) -> LogLevel:
        """Return the logging severity implied by measurement status."""
        if self.status in {"unavailable", "unreliable"}:
            return "warning"
        return "info"

    @property
    def unit(self) -> ResourceUnit:
        """Return the canonical unit required by the resource category."""
        return _RESOURCE_UNITS[self.category]
