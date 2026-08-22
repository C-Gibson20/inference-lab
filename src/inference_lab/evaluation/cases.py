"""Evaluation-case schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_lab.evaluation.utils import CaseId, SchemaVersion, SplitName
from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    EvidenceReference,
    ExpectedAnswer,
)
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION


class EvaluationCase(BaseModel):
    """One annotated question used by an evaluation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = CURRENT_SCHEMA_VERSION
    case_id: CaseId
    question: str = Field(min_length=1)
    question_type: Literal[
        "lookup",
        "comparison",
        "thematic",
        "aggregation",
        "follow_up",
        "difficult_terminology",
    ]
    context: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    expected_answer: ExpectedAnswer
    evidence: tuple[EvidenceReference, ...] = ()
    abstention: AbstentionExpectation
    split: SplitName

    @model_validator(mode="after")
    def validate_semantics(self) -> "EvaluationCase":
        """Validate evidence, abstention, and follow-up semantics."""
        authoritative = tuple(
            item for item in self.evidence if item.role == "authoritative"
        )

        if not self.abstention.expected and not authoritative:
            raise ValueError("Answerable cases require authoritative evidence")
        if self.abstention.expected and authoritative:
            raise ValueError(
                "Expected-abstention cases cannot have authoritative evidence"
            )
        if self.question_type == "follow_up" and not self.context:
            raise ValueError("Follow-up cases require conversation context")

        return self
