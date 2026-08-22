"""Reusable schemas for evaluation datasets."""

from __future__ import annotations

import math
from typing import Literal
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_lab.provenance.identifiers import (
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)


def _canonical_id(value: str, identifier_type: type) -> str:
    """Validate an ID using a PROV-01 identifier type."""
    return identifier_type.from_str(value).as_str()


class EvidenceReference(BaseModel):
    """Evidence associated with an evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["source_span", "record", "document"] = "source_span"
    role: Literal["authoritative", "context"] = "authoritative"

    corpus_version: str
    document_id: str
    document_version_id: str | None = None
    source_span_id: str | None = None
    record_id: str | None = None

    @model_validator(mode="after")
    def validate_ids_and_shape(self) -> "EvidenceReference":
        """Validate provenance identifiers and evidence granularity."""
        _canonical_id(self.corpus_version, CorpusVersionId)
        _canonical_id(self.document_id, DocumentId)

        if self.document_version_id is not None:
            _canonical_id(self.document_version_id, DocumentVersionId)
        if self.source_span_id is not None:
            _canonical_id(self.source_span_id, SourceSpanId)
        if self.record_id is not None:
            _canonical_id(self.record_id, RecordId)

        if self.kind == "source_span":
            if self.document_version_id is None or self.source_span_id is None:
                raise ValueError(
                    "source_span evidence requires document_version_id "
                    "and source_span_id"
                )
        elif self.kind == "record" and self.record_id is None:
            raise ValueError("record evidence requires record_id")
        elif self.kind == "document" and self.role == "authoritative":
            raise ValueError("document-level evidence cannot be authoritative")

        return self


class AnswerClaim(BaseModel):
    """A required or optional expected answer claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    required: bool = True
    acceptable_values: tuple[str, ...] = ()
    normalized_values: tuple[str, ...] = ()
    description: str | None = None


class DateExpectation(BaseModel):
    """A date expectation with an optional inclusive tolerance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    value: date
    tolerance_days: int = Field(default=0, ge=0)


class NumericExpectation(BaseModel):
    """A numeric expected answer value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    value: float
    absolute_tolerance: float | None = Field(default=None, ge=0)
    relative_tolerance: float | None = Field(default=None, ge=0)
    unit: str | None = None

    @model_validator(mode="after")
    def validate_finite_values(self) -> "NumericExpectation":
        """Reject non-finite expected values and tolerances."""

        values = (
            self.value,
            self.absolute_tolerance,
            self.relative_tolerance,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("Numeric expectations must be finite")
        return self


class ExpectedAnswer(BaseModel):
    """Machine-checkable expectations for an answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str | None = None
    claims: tuple[AnswerClaim, ...] = ()
    numeric_values: tuple[NumericExpectation, ...] = ()
    date_values: tuple[DateExpectation, ...] = ()
    comparison_order: tuple[str, ...] | None = None


class AbstentionExpectation(BaseModel):
    """Expected abstention behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: bool
    reason: Literal[
        "evidence_absent",
        "evidence_insufficient",
        "evidence_ambiguous",
        "deliberately_unanswerable",
        "not_applicable",
    ]
    partial_answer_allowed: bool = False
