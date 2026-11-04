"""Closed stage and run-condition contracts for baseline profiling."""

from dataclasses import dataclass
from typing import Final, Literal, get_args

from inference_lab.profiling._validation import validate_optional_correlation

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
    "dense_index",
    "lexical_retrieval",
    "dense_retrieval",
    "retrieval_fusion",
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
    return value


def validate_stage_condition(value: object) -> StageCondition:
    """Return a valid caller-provided cold/warm condition."""
    if not isinstance(value, str):
        raise TypeError("condition must be a string")
    if value not in STAGE_CONDITIONS:
        raise ValueError("condition must be cold, warm, or uncontrolled")
    return value


_CORRELATION_FIELDS: Final = (
    "run_id",
    "request_id",
    "document_id",
    "corpus_id",
    "corpus_version",
    "parent_stage_id",
)


@dataclass(frozen=True, slots=True)
class StageCorrelation:
    """Opaque identifiers tying one observed stage to the work around it."""

    run_id: str | None = None
    request_id: str | None = None
    document_id: str | None = None
    corpus_id: str | None = None
    corpus_version: str | None = None
    parent_stage_id: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Require every supplied value to be opaque and payload-free."""
        for field_name in _CORRELATION_FIELDS:
            validate_optional_correlation(getattr(self, field_name), field_name)
