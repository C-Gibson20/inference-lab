"""Reusable profiling helpers; experiment outputs belong in top-level benchmarks/."""

from inference_lab.profiling.context import ProfilingContext
from inference_lab.profiling.contracts import (
    BASELINE_STAGES,
    STAGE_CONDITIONS,
    BaselineStage,
    StageCondition,
    validate_baseline_stage,
    validate_stage_condition,
)
from inference_lab.profiling.emission import (
    FanoutSink,
    ProfilingLogger,
    ProfilingSink,
    StructuredLogSink,
    emit_record,
)
from inference_lab.profiling.failures import FailureEvent
from inference_lab.profiling.observed import (
    FailureMapper,
    FailureMapping,
    ObservedStage,
)
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
    "BASELINE_STAGES",
    "CURRENT_RECORD_SCHEMA_VERSION",
    "MEASUREMENT_STATUSES",
    "MEMORY_RESOURCE_CATEGORIES",
    "RESOURCE_CATEGORIES",
    "STAGE_CONDITIONS",
    "SUPPORTED_RECORD_SCHEMA_VERSIONS",
    "BaselineStage",
    "FailureEvent",
    "FailureMapper",
    "FailureMapping",
    "FanoutSink",
    "LogLevel",
    "MeasurementStatus",
    "MemoryResourceCategory",
    "ObservedStage",
    "ProfilingContext",
    "ProfilingLogger",
    "ProfilingSink",
    "ResourceCategory",
    "ResourceMeasurement",
    "StageCondition",
    "StageTimer",
    "StageTiming",
    "StructuredLogSink",
    "emit_record",
    "validate_baseline_stage",
    "validate_stage_condition",
]
