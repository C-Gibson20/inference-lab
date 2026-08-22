"""Evaluation dataset manifest schema."""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from inference_lab.evaluation.utils import (
    CaseId,
    DatasetId,
    HELD_OUT_SPLIT,
    SchemaVersion,
    SplitName,
)
from inference_lab.evaluation.schemas import EvaluationPolicy
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION
from inference_lab.provenance.identifiers import CorpusId, CorpusVersionId


class DatasetManifest(BaseModel):
    """Metadata describing one immutable evaluation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = CURRENT_SCHEMA_VERSION
    dataset_id: DatasetId
    dataset_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    corpus_id: str
    corpus_version: str
    case_file: str = Field(min_length=1)
    case_count: int = Field(ge=1)
    splits: dict[SplitName, tuple[CaseId, ...]]
    provenance: dict[str, str] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    evaluation_policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_splits(self) -> "DatasetManifest":
        """Validate corpus identifiers and explicit split membership."""
        CorpusId.from_str(self.corpus_id)
        CorpusVersionId.from_str(self.corpus_version)

        if HELD_OUT_SPLIT not in self.splits:
            raise ValueError("splits must include 'held_out'")

        split_ids = [case_id for ids in self.splits.values() for case_id in ids]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("Each case ID must belong to exactly one split")
        if len(split_ids) != self.case_count:
            raise ValueError("case_count must match total split IDs")

        return self
