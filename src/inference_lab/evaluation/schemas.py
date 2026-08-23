"""Reusable schemas for evaluation datasets."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from inference_lab.evaluation.utils import EvidenceId, ExpectationKey, normalise_text
from inference_lab.provenance import (
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)


def _canonical_id(value: str, identifier_type: type) -> str:
    """Validate an ID using a PROV-01 identifier type."""
    return identifier_type.from_str(value).as_str()


def _require_non_blank(value: str | None, *, field_name: str) -> str | None:
    """Reject a present string that contains no meaningful text."""
    if value is not None and not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _validate_unique_strings(values: tuple[str, ...], *, field_name: str) -> None:
    """Reject blank or normalized-duplicate strings."""
    normalized = tuple(normalise_text(value) for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} must not contain blank values")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique values")


def _validate_unique_ids(values: tuple[str, ...], *, field_name: str) -> None:
    """Reject duplicate identifiers where order is semantically meaningful."""
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique evidence IDs")


def _reject_boolean_numbers(value: Any) -> Any:
    """Reject booleans before Pydantic can coerce them to numbers."""
    values = value if isinstance(value, (list, tuple)) else (value,)
    if any(isinstance(item, bool) for item in values if item is not None):
        raise ValueError("Boolean values are not numeric expectations")
    return value


class EvidenceReference(BaseModel):
    """One case-local reference to expected source evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: EvidenceId
    kind: Literal["source_span", "record", "document"] = "source_span"
    role: Literal["authoritative", "context"] = "authoritative"
    corpus_version: str
    document_id: str
    document_version_id: str | None = None
    source_span_id: str | None = None
    record_id: str | None = None

    @model_validator(mode="after")
    def validate_ids_and_shape(self) -> "EvidenceReference":
        """Validate provenance identifiers and exact evidence granularity."""
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
            if self.record_id is not None:
                raise ValueError("source_span evidence cannot contain record_id")
        elif self.kind == "record":
            if self.record_id is None:
                raise ValueError("record evidence requires record_id")
            if self.source_span_id is not None:
                raise ValueError("record evidence cannot contain source_span_id")
        else:
            if self.role == "authoritative":
                raise ValueError("document-level evidence cannot be authoritative")
            if any(
                value is not None
                for value in (
                    self.document_version_id,
                    self.source_span_id,
                    self.record_id,
                )
            ):
                raise ValueError(
                    "document evidence cannot contain version, span, or record IDs"
                )

        return self

    @property
    def location_key(self) -> tuple[str, ...]:
        """Return the provenance location independent of its case-local ID."""
        return (
            self.kind,
            self.corpus_version,
            self.document_id,
            self.document_version_id or "",
            self.source_span_id or "",
            self.record_id or "",
        )


class AnswerClaim(BaseModel):
    """A required or optional expected answer claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ExpectationKey
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    required: StrictBool = True
    acceptable_values: tuple[str, ...] = ()
    normalized_values: tuple[str, ...] = ()
    description: str | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "AnswerClaim":
        """Require a meaningful, unambiguous claim annotation."""
        _validate_unique_ids(self.evidence_ids, field_name="evidence_ids")
        _validate_unique_strings(
            self.acceptable_values,
            field_name="acceptable_values",
        )
        _validate_unique_strings(
            self.normalized_values,
            field_name="normalized_values",
        )
        _require_non_blank(self.description, field_name="description")
        if not (self.acceptable_values or self.normalized_values or self.description):
            raise ValueError(
                "Claims require accepted values, normalized values, or a description"
            )
        return self


class DateExpectation(BaseModel):
    """A date expectation with an optional inclusive tolerance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ExpectationKey
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    value: date
    tolerance_days: StrictInt = Field(default=0, ge=0)

    @field_validator("value", mode="before")
    @classmethod
    def validate_date_value(cls, value: Any) -> Any:
        """Accept dates and ISO date strings without datetime coercion."""
        if isinstance(value, datetime):
            raise ValueError("Date expectations cannot contain datetimes")
        if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date expectations must use YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "DateExpectation":
        """Reject duplicate evidence links."""
        _validate_unique_ids(self.evidence_ids, field_name="evidence_ids")
        return self


class NumericExpectation(BaseModel):
    """A numeric expected answer value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ExpectationKey
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    value: StrictFloat
    acceptable_values: tuple[StrictFloat, ...] = ()
    absolute_tolerance: StrictFloat | None = Field(default=None, ge=0)
    relative_tolerance: StrictFloat | None = Field(default=None, ge=0)
    unit: str | None = None

    @field_validator(
        "value",
        "acceptable_values",
        "absolute_tolerance",
        "relative_tolerance",
        mode="before",
    )
    @classmethod
    def reject_boolean_numbers(cls, value: Any) -> Any:
        """Reject booleans rather than coercing them to zero or one."""
        return _reject_boolean_numbers(value)

    @model_validator(mode="after")
    def validate_values(self) -> "NumericExpectation":
        """Validate finite values, unique alternatives, and evidence links."""
        _validate_unique_ids(self.evidence_ids, field_name="evidence_ids")
        values = (
            self.value,
            *self.acceptable_values,
            self.absolute_tolerance,
            self.relative_tolerance,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("Numeric expectations must be finite")
        if len(self.acceptable_values) != len(set(self.acceptable_values)):
            raise ValueError("acceptable_values must contain unique numbers")
        if self.value in self.acceptable_values:
            raise ValueError("acceptable_values cannot repeat the primary value")
        _require_non_blank(self.unit, field_name="unit")
        return self


class ExpectedAnswer(BaseModel):
    """Evidence-linked expectations; summary text is illustrative only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str | None = None
    claims: tuple[AnswerClaim, ...] = ()
    numeric_values: tuple[NumericExpectation, ...] = ()
    date_values: tuple[DateExpectation, ...] = ()
    comparison_order: tuple[str, ...] | None = None
    acceptable_comparison_orders: tuple[tuple[str, ...], ...] = ()
    comparison_evidence_ids: tuple[EvidenceId, ...] = ()

    @model_validator(mode="after")
    def validate_expectations(self) -> "ExpectedAnswer":
        """Reject ambiguous keys and contradictory comparison annotations."""
        _require_non_blank(self.summary, field_name="summary")

        keys = tuple(
            item.key for item in (*self.claims, *self.numeric_values, *self.date_values)
        )
        if len(keys) != len(set(keys)):
            raise ValueError("Expected-answer keys must be unique across value types")

        if self.comparison_order is None:
            if self.acceptable_comparison_orders:
                raise ValueError(
                    "acceptable comparison orders require comparison_order"
                )
            if self.comparison_evidence_ids:
                raise ValueError("comparison evidence requires comparison_order")
            return self

        _validate_unique_strings(
            self.comparison_order,
            field_name="comparison_order",
        )
        if len(self.comparison_order) < 2:
            raise ValueError("comparison_order requires at least two values")
        _validate_unique_ids(
            self.comparison_evidence_ids,
            field_name="comparison_evidence_ids",
        )
        if not self.comparison_evidence_ids:
            raise ValueError("comparison_order requires comparison evidence")

        expected_members = set(self.comparison_order)
        seen_orders: set[tuple[str, ...]] = {self.comparison_order}
        for order in self.acceptable_comparison_orders:
            _validate_unique_strings(order, field_name="acceptable_comparison_orders")
            if set(order) != expected_members or len(order) != len(
                self.comparison_order
            ):
                raise ValueError(
                    "Acceptable comparison orders must contain the expected values"
                )
            if order in seen_orders:
                raise ValueError("Comparison orders must be unique")
            seen_orders.add(order)

        return self

    @property
    def has_expectations(self) -> bool:
        """Return whether the answer has a machine-checkable characteristic."""
        return bool(
            self.claims
            or self.numeric_values
            or self.date_values
            or self.comparison_order
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return evidence IDs referenced by all expected characteristics."""
        return (
            tuple(
                evidence_id
                for item in (*self.claims, *self.numeric_values, *self.date_values)
                for evidence_id in item.evidence_ids
            )
            + self.comparison_evidence_ids
        )


class EvaluationPolicy(BaseModel):
    """Governance rules for using the held-out split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    held_out_tuning_allowed: Literal[False] = False
    held_out_model_selection_allowed: Literal[False] = False
    held_out_membership_immutable: Literal[True] = True

    @field_validator(
        "held_out_tuning_allowed",
        "held_out_model_selection_allowed",
        "held_out_membership_immutable",
        mode="before",
    )
    @classmethod
    def reject_boolean_coercion(cls, value: Any) -> Any:
        """Require actual booleans for held-out governance fields."""
        if not isinstance(value, bool):
            raise ValueError("Evaluation policy values must be booleans")
        return value


class AbstentionExpectation(BaseModel):
    """Expected abstention behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: StrictBool
    reason: Literal[
        "evidence_absent",
        "evidence_insufficient",
        "evidence_ambiguous",
        "deliberately_unanswerable",
        "not_applicable",
    ]
    partial_answer_allowed: StrictBool = False

    @model_validator(mode="after")
    def validate_semantics(self) -> "AbstentionExpectation":
        """Keep the reason and partial-answer policy internally consistent."""
        if self.expected:
            if self.reason == "not_applicable":
                raise ValueError("Expected abstention requires a substantive reason")
        else:
            if self.reason != "not_applicable":
                raise ValueError(
                    "Non-abstention cases must use reason 'not_applicable'"
                )
            if self.partial_answer_allowed:
                raise ValueError(
                    "partial_answer_allowed applies only when abstention is expected"
                )
        return self
