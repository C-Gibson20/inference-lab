"""Normalized resource-measurement records for profiling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from .utils import utc_now


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


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """One normalized measurement for one runtime resource category.

    A record may contain a value for an unreliable measurement, allowing
    consumers to retain the observation without treating it as authoritative.
    An unavailable measurement has no value and can still retain its expected
    unit when that is known.
    """

    category: ResourceCategory
    unit: str | None
    value: int | float | None
    status: MeasurementStatus = "measured"
    measurement_kind: MeasurementKind = "point_in_time"
    timestamp: datetime = field(default_factory=utc_now)
    module: str = "profiling"
    pipeline_stage: str | None = None
    stage_id: str | None = None
    sample_duration_ms: float | None = None
    run_id: str | None = None
    request_id: str | None = None
    document_id: str | None = None
    corpus_version: str | None = None

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
        if self.value is not None and not self.unit:
            raise ValueError("measured resources must have a unit")
        if (
            self.measurement_kind == "point_in_time"
            and self.sample_duration_ms is not None
        ):
            raise ValueError(
                "point-in-time measurements must not have a sample duration"
            )
        if self.measurement_kind != "point_in_time":
            if self.sample_duration_ms is None or self.sample_duration_ms <= 0:
                raise ValueError(
                    "sampled and interval measurements require a positive sample duration"
                )
        if self.sample_duration_ms is not None and not math.isfinite(
            self.sample_duration_ms
        ):
            raise ValueError("sample_duration_ms must be finite")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be in UTC")

    def to_dict(self) -> dict[str, object]:
        """Return the measurement as structured logging fields."""
        return {
            "event": "resource_measured",
            "category": self.category,
            "unit": self.unit,
            "value": self.value,
            "status": self.status,
            "measurement_kind": self.measurement_kind,
            "timestamp": self.timestamp.isoformat(),
            "module": self.module,
            "pipeline_stage": self.pipeline_stage,
            "stage_id": self.stage_id,
            "sample_duration_ms": self.sample_duration_ms,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "corpus_version": self.corpus_version,
        }
