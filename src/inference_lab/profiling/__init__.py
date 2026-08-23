"""Reusable profiling helpers; experiment outputs belong in top-level benchmarks/."""

from inference_lab.profiling.emission import emit_record
from inference_lab.profiling.failures import FailureEvent
from inference_lab.profiling.records import LogLevel, ProfilingRecord
from inference_lab.profiling.resources import ResourceMeasurement
from inference_lab.profiling.timing import StageTimer, StageTiming
from inference_lab.profiling.versioning import (
    CURRENT_RECORD_SCHEMA_VERSION,
    SUPPORTED_RECORD_SCHEMA_VERSIONS,
)

__all__ = [
    "LogLevel",
    "CURRENT_RECORD_SCHEMA_VERSION",
    "FailureEvent",
    "ProfilingRecord",
    "ResourceMeasurement",
    "StageTimer",
    "StageTiming",
    "SUPPORTED_RECORD_SCHEMA_VERSIONS",
    "emit_record",
]
