"""Evaluation dataset manifest schema."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from inference_lab.evaluation.schemas import EvaluationPolicy
from inference_lab.evaluation.utils import (
    HELD_OUT_SPLIT,
    CaseId,
    DatasetId,
    SchemaVersion,
    SplitName,
    is_sha256_pattern,
)
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION
from inference_lab.provenance import CorpusId, CorpusVersionId


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
    case_count: StrictInt = Field(ge=1)
    splits: Mapping[SplitName, tuple[CaseId, ...]]
    provenance: Mapping[str, str] = Field(default_factory=dict)
    checksums: Mapping[str, str] = Field(default_factory=dict)
    evaluation_policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @field_serializer("splits", "provenance", "checksums")
    def serialize_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        """Serialize immutable mappings as ordinary JSON objects."""
        return dict(value)

    @field_validator("dataset_version", "description", "case_file")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        """Reject manifest strings without meaningful content."""
        if not value.strip():
            raise ValueError("Manifest text fields must not be blank")
        return value

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
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")

        if self.checksums:
            if set(self.checksums) != {self.case_file}:
                raise ValueError("checksums must contain only the case file")
            if not is_sha256_pattern(self.checksums[self.case_file]):
                raise ValueError("Case-file checksum must use SHA256 format")

        object.__setattr__(self, "splits", MappingProxyType(dict(self.splits)))
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )
        object.__setattr__(
            self,
            "checksums",
            MappingProxyType(dict(self.checksums)),
        )

        return self
