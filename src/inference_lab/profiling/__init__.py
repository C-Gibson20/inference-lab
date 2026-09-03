"""Reusable profiling helpers; experiment outputs belong in top-level benchmarks/."""

from inference_lab.profiling.emission import ProfilingLogger, emit_record
from inference_lab.profiling.failures import FailureEvent
from inference_lab.profiling.resources import (
    MEASUREMENT_STATUSES,
    MEMORY_RESOURCE_CATEGORIES,
    RESOURCE_CATEGORIES,
    MeasurementStatus,
    MemoryResourceCategory,
    ResourceCategory,
    ResourceMeasurement,
)
from inference_lab.profiling.timing import StageTimer, StageTiming
from inference_lab.profiling.types import LogLevel
from inference_lab.profiling.versioning import (
    CURRENT_RECORD_SCHEMA_VERSION,
    SUPPORTED_RECORD_SCHEMA_VERSIONS,
)

__all__ = [
    "CURRENT_RECORD_SCHEMA_VERSION",
    "MEASUREMENT_STATUSES",
    "MEMORY_RESOURCE_CATEGORIES",
    "RESOURCE_CATEGORIES",
    "SUPPORTED_RECORD_SCHEMA_VERSIONS",
    "FailureEvent",
    "LogLevel",
    "MeasurementStatus",
    "MemoryResourceCategory",
    "ProfilingLogger",
    "ResourceCategory",
    "ResourceMeasurement",
    "StageTimer",
    "StageTiming",
    "emit_record",
]
