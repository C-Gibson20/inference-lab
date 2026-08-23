"""Evaluation-case schema."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    EvidenceReference,
    ExpectedAnswer,
)
from inference_lab.evaluation.utils import CaseId, SchemaVersion, SplitName
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION

QuestionType = Literal[
    "lookup",
    "comparison",
    "thematic",
    "aggregation",
    "follow_up",
    "difficult_terminology",
]

QUESTION_TYPES: Final[tuple[QuestionType, ...]] = (
    "lookup",
    "comparison",
    "thematic",
    "aggregation",
    "follow_up",
    "difficult_terminology",
)


class EvaluationCase(BaseModel):
    """One annotated question used by an evaluation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = CURRENT_SCHEMA_VERSION
    case_id: CaseId
    question: str = Field(min_length=1)
    question_type: QuestionType
    context: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    expected_answer: ExpectedAnswer
    evidence: tuple[EvidenceReference, ...] = ()
    abstention: AbstentionExpectation
    split: SplitName

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject questions without meaningful text."""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank conversation turns."""
        if any(not item.strip() for item in value):
            raise ValueError("context must not contain blank turns")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, value: object) -> object:
        """Reject blank or duplicate tags before conversion to a frozenset."""
        if isinstance(value, (list, tuple, set, frozenset)):
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError("tags must contain non-blank strings")
            if len(value) != len(set(value)):
                raise ValueError("tags must be unique")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> "EvaluationCase":
        """Validate evidence, abstention, and follow-up semantics."""
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique within a case")

        locations = tuple(item.location_key for item in self.evidence)
        if len(locations) != len(set(locations)):
            raise ValueError("Evidence locations must be unique within a case")

        referenced_ids = set(self.expected_answer.evidence_ids)
        authoritative_ids = {
            item.evidence_id for item in self.evidence if item.role == "authoritative"
        }

        if self.abstention.reason in {
            "evidence_absent",
            "deliberately_unanswerable",
        }:
            if authoritative_ids:
                raise ValueError(
                    f"{self.abstention.reason} cases cannot have authoritative evidence"
                )
            if self.expected_answer.has_expectations:
                raise ValueError(
                    f"{self.abstention.reason} cases cannot define expected answers"
                )

        if self.abstention.reason == "evidence_insufficient" and not self.evidence:
            raise ValueError("evidence_insufficient cases require available evidence")

        if self.abstention.reason == "evidence_ambiguous" and not authoritative_ids:
            raise ValueError("evidence_ambiguous cases require authoritative evidence")

        if not self.abstention.expected and not authoritative_ids:
            raise ValueError("Answerable cases require authoritative evidence")
        if not self.abstention.expected and not self.expected_answer.has_expectations:
            raise ValueError("Answerable cases require expected-answer characteristics")

        unknown_ids = referenced_ids - evidence_by_id.keys()
        if unknown_ids:
            names = ", ".join(sorted(unknown_ids))
            raise ValueError(f"Expected answer references unknown evidence: {names}")

        non_authoritative_ids = {
            evidence_id
            for evidence_id in referenced_ids
            if evidence_by_id[evidence_id].role != "authoritative"
        }
        if non_authoritative_ids:
            names = ", ".join(sorted(non_authoritative_ids))
            raise ValueError(
                f"Expected answer references non-authoritative evidence: {names}"
            )

        if not self.abstention.expected:
            unused_ids = authoritative_ids - referenced_ids
            if unused_ids:
                names = ", ".join(sorted(unused_ids))
                raise ValueError(f"Authoritative evidence is not referenced: {names}")

        if self.abstention.partial_answer_allowed:
            if not self.expected_answer.has_expectations:
                raise ValueError(
                    "Partial-answer cases require expected-answer characteristics"
                )

        if self.question_type == "follow_up" and not self.context:
            raise ValueError("Follow-up cases require conversation context")

        return self
