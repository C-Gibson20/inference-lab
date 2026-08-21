"""Reusable profiling helpers; experiment outputs belong in top-level benchmarks/."""

from inference_lab.profiling.emission import emit_record
from inference_lab.profiling.failures import FailureEvent
from inference_lab.profiling.records import LogLevel, ProfilingRecord
from inference_lab.profiling.resources import ResourceMeasurement
from inference_lab.profiling.timing import StageTimer, StageTiming

__all__ = [
    "LogLevel",
    "FailureEvent",
    "ProfilingRecord",
    "ResourceMeasurement",
    "StageTimer",
    "StageTiming",
    "emit_record",
]
