"""Closed stage and run-condition contracts for baseline profiling."""

from typing import Final, Literal, get_args

BaselineStage = Literal[
    "discovery",
    "source_acquisition",
    "parse",
    "parse_persistence",
    "chunk",
    "chunk_persistence",
    "membership",
    "publication",
    "lexical_index",
    "lexical_retrieval",
    "metadata_filter",
    "serving",
]
StageCondition = Literal["cold", "warm", "uncontrolled"]

BASELINE_STAGES: Final[frozenset[BaselineStage]] = frozenset(get_args(BaselineStage))
STAGE_CONDITIONS: Final[frozenset[StageCondition]] = frozenset(get_args(StageCondition))


def validate_baseline_stage(value: object) -> BaselineStage:
    """Return a valid baseline stage name or reject it."""
    if not isinstance(value, str):
        raise TypeError("pipeline_stage must be a string")
    if value not in BASELINE_STAGES:
        raise ValueError("pipeline_stage must be a supported baseline stage")
    return value  # type: ignore[return-value]


def validate_stage_condition(value: object) -> StageCondition:
    """Return a valid caller-provided cold/warm condition."""
    if not isinstance(value, str):
        raise TypeError("condition must be a string")
    if value not in STAGE_CONDITIONS:
        raise ValueError("condition must be cold, warm, or uncontrolled")
    return value  # type: ignore[return-value]
