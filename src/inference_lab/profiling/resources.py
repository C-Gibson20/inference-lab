"""Normalized resource-measurement records for profiling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from .types import LogLevel
from .utils import utc_now
from .versioning import CURRENT_RECORD_SCHEMA_VERSION, validate_record_schema_version


ResourceCategory = Literal[
    "model_weights",
    "kv_cache",
    "activations",
    "embedding_reranker_memory",
    "index_memory",
    "framework_overhead",
    "pinned_host_memory",
    "storage_cache",
    "gpu_utilization",
    "cpu_utilization",
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

_RESOURCE_CATEGORIES = frozenset(
    {
        "model_weights",
        "kv_cache",
        "activations",
        "embedding_reranker_memory",
        "index_memory",
        "framework_overhead",
        "pinned_host_memory",
        "storage_cache",
        "gpu_utilization",
        "cpu_utilization",
        "storage_throughput",
        "power",
        "thermal",
    }
)
_MEASUREMENT_STATUSES = frozenset({"measured", "unavailable", "unreliable"})
_MEASUREMENT_KINDS = frozenset({"point_in_time", "sampled", "interval"})
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
    "gpu_utilization": "percent",
    "cpu_utilization": "percent",
    "storage_throughput": "bytes_per_second",
    "power": "watts",
    "thermal": "celsius",
}


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """One normalized measurement for one runtime resource category.

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
        if self.category not in _RESOURCE_CATEGORIES:
            raise ValueError("category must be a supported resource category")
        if not self.module:
            raise ValueError("module must not be empty")
        if self.pipeline_stage == "":
            raise ValueError("pipeline_stage must not be empty")
        if self.stage_id == "":
            raise ValueError("stage_id must not be empty")
        if self.status not in _MEASUREMENT_STATUSES:
            raise ValueError("status must be a valid measurement status")
        if self.measurement_kind not in _MEASUREMENT_KINDS:
            raise ValueError("measurement_kind must be valid")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("value must be a finite number or None")
            if not math.isfinite(float(self.value)):
                raise ValueError("value must be a finite number or None")
        if self.status == "measured" and self.value is None:
            raise ValueError("measured resources must have a value")
        if self.status == "unavailable" and self.value is not None:
            raise ValueError("unavailable resources must not have a value")
        self._validate_value_range()
        if (
            self.measurement_kind == "point_in_time"
            and self.sample_duration_ms is not None
        ):
            raise ValueError(
                "point-in-time measurements must not have a sample duration"
            )
        if self.measurement_kind != "point_in_time":
            if (
                isinstance(self.sample_duration_ms, bool)
                or not isinstance(self.sample_duration_ms, (int, float))
                or not math.isfinite(self.sample_duration_ms)
                or self.sample_duration_ms <= 0
            ):
                raise ValueError(
                    "sampled and interval measurements require a positive sample "
                    "duration expressed as a finite number"
                )
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware UTC")
        if self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be in UTC")
        validate_record_schema_version(self.record_schema_version)

    def _validate_value_range(self) -> None:
        """Validate the physical range implied by the resource category."""
        if self.value is None:
            return
        if self.category in {"gpu_utilization", "cpu_utilization"}:
            if not 0 <= self.value <= 100:
                raise ValueError("utilization percentage must be between 0 and 100")
        elif self.category == "thermal":
            if self.value < -273.15:
                raise ValueError("thermal measurement cannot be below absolute zero")
        elif self.category in _MEMORY_CATEGORIES or self.category in {
            "storage_throughput",
            "power",
        }:
            if self.value < 0:
                raise ValueError(f"{self.category} measurement must be non-negative")

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
