"""Tests for reusable evaluation schemas."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    AnswerClaim,
    DateExpectation,
    EvaluationPolicy,
    EvidenceReference,
    ExpectedAnswer,
    NumericExpectation,
)

from .fixtures import EVIDENCE_IDS  # noqa: TID252


def test_source_span_evidence_is_serialisable() -> None:
    evidence = EvidenceReference(**EVIDENCE_IDS)

    assert evidence.model_dump() == EVIDENCE_IDS | {
        "kind": "source_span",
        "role": "authoritative",
        "record_id": None,
    }


def test_evidence_kinds_require_exact_identifier_shapes() -> None:
    with pytest.raises(ValidationError, match="source_span evidence"):
        EvidenceReference(
            evidence_id="evidence-primary",
            corpus_version=EVIDENCE_IDS["corpus_version"],
            document_id=EVIDENCE_IDS["document_id"],
        )

    with pytest.raises(ValidationError, match="cannot contain record_id"):
        EvidenceReference(
            **EVIDENCE_IDS,
            record_id="00000000-0000-4000-8000-000000000006",
        )

    with pytest.raises(ValidationError, match="cannot contain source_span_id"):
        EvidenceReference(
            **EVIDENCE_IDS,
            kind="record",
            record_id="00000000-0000-4000-8000-000000000006",
        )

    with pytest.raises(ValidationError, match="record evidence requires record_id"):
        EvidenceReference(
            evidence_id="evidence-primary",
            corpus_version=EVIDENCE_IDS["corpus_version"],
            document_id=EVIDENCE_IDS["document_id"],
            kind="record",
        )

    with pytest.raises(ValidationError, match="cannot contain version"):
        EvidenceReference(**EVIDENCE_IDS, kind="document", role="context")


@pytest.mark.parametrize(
    "field_name",
    [
        "corpus_version",
        "document_id",
        "document_version_id",
        "source_span_id",
        "record_id",
    ],
)
def test_provenance_ids_must_be_canonical_uuids(field_name: str) -> None:
    invalid = dict(EVIDENCE_IDS)
    invalid[field_name] = "not-a-canonical-id"

    with pytest.raises(ValidationError, match="valid UUID"):
        EvidenceReference.model_validate(invalid)


def test_authoritative_document_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be authoritative"):
        EvidenceReference(
            evidence_id="evidence-primary",
            corpus_version=EVIDENCE_IDS["corpus_version"],
            document_id=EVIDENCE_IDS["document_id"],
            kind="document",
        )


def test_expectations_require_meaningful_unique_values_and_evidence() -> None:
    with pytest.raises(ValidationError, match="Claims require"):
        AnswerClaim(key="assay", evidence_ids=("evidence-primary",))

    with pytest.raises(ValidationError, match="unique values"):
        AnswerClaim(
            key="assay",
            evidence_ids=("evidence-primary",),
            acceptable_values=("Cell Based", " cell   based "),
        )

    with pytest.raises(ValidationError, match="blank values"):
        AnswerClaim(
            key="assay",
            evidence_ids=("evidence-primary",),
            acceptable_values=(" ",),
        )

    with pytest.raises(ValidationError, match="description must not be blank"):
        AnswerClaim(
            key="assay",
            evidence_ids=("evidence-primary",),
            description=" ",
        )

    with pytest.raises(ValidationError, match="unique evidence IDs"):
        DateExpectation(
            key="publication_date",
            evidence_ids=("evidence-primary", "evidence-primary"),
            value=date(2024, 3, 15),
        )

    with pytest.raises(ValidationError, match="cannot contain datetimes"):
        DateExpectation(
            key="publication_date",
            evidence_ids=("evidence-primary",),
            value=datetime(2024, 3, 15),
        )

    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        DateExpectation.model_validate(
            {
                "key": "publication_date",
                "evidence_ids": ("evidence-primary",),
                "value": "2024-03-15T00:00:00Z",
            }
        )


def test_numeric_expectations_reject_booleans_and_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="Boolean values"):
        NumericExpectation(
            key="activity",
            evidence_ids=("evidence-primary",),
            value=True,
        )

    with pytest.raises(ValidationError, match="finite"):
        NumericExpectation(
            key="activity",
            evidence_ids=("evidence-primary",),
            value=float("inf"),
        )

    with pytest.raises(ValidationError):
        NumericExpectation.model_validate(
            {
                "key": "activity",
                "evidence_ids": ("evidence-primary",),
                "value": "42",
            }
        )

    with pytest.raises(ValidationError, match="unique numbers"):
        NumericExpectation(
            key="activity",
            evidence_ids=("evidence-primary",),
            value=42,
            acceptable_values=(41, 41),
        )

    with pytest.raises(ValidationError, match="primary value"):
        NumericExpectation(
            key="activity",
            evidence_ids=("evidence-primary",),
            value=42,
            acceptable_values=(42,),
        )

    with pytest.raises(ValidationError, match="unit must not be blank"):
        NumericExpectation(
            key="activity",
            evidence_ids=("evidence-primary",),
            value=42,
            unit=" ",
        )


def test_evaluation_policy_rejects_boolean_coercion() -> None:
    with pytest.raises(ValidationError, match="must be booleans"):
        EvaluationPolicy.model_validate({"held_out_tuning_allowed": 0})


def test_expected_answer_rejects_duplicate_keys_and_invalid_comparisons() -> None:
    claim = AnswerClaim(
        key="activity",
        evidence_ids=("evidence-primary",),
        description="Reported activity",
    )
    numeric = NumericExpectation(
        key="activity",
        evidence_ids=("evidence-primary",),
        value=42,
    )

    with pytest.raises(ValidationError, match="keys must be unique"):
        ExpectedAnswer(claims=(claim,), numeric_values=(numeric,))

    with pytest.raises(ValidationError, match="requires comparison evidence"):
        ExpectedAnswer(comparison_order=("compound-a", "compound-b"))

    with pytest.raises(ValidationError, match="require comparison_order"):
        ExpectedAnswer(acceptable_comparison_orders=(("compound-a", "compound-b"),))

    with pytest.raises(ValidationError, match="requires comparison_order"):
        ExpectedAnswer(comparison_evidence_ids=("evidence-primary",))

    with pytest.raises(ValidationError, match="at least two"):
        ExpectedAnswer(
            comparison_order=("compound-a",),
            comparison_evidence_ids=("evidence-primary",),
        )

    with pytest.raises(ValidationError, match="expected values"):
        ExpectedAnswer(
            comparison_order=("compound-a", "compound-b"),
            acceptable_comparison_orders=(("compound-a", "compound-c"),),
            comparison_evidence_ids=("evidence-primary",),
        )

    with pytest.raises(ValidationError, match="orders must be unique"):
        ExpectedAnswer(
            comparison_order=("compound-a", "compound-b"),
            acceptable_comparison_orders=(("compound-a", "compound-b"),),
            comparison_evidence_ids=("evidence-primary",),
        )

    with pytest.raises(ValidationError, match="summary must not be blank"):
        ExpectedAnswer(summary=" ")


def test_expected_answer_accepts_distinct_comparison_alternatives() -> None:
    expected = ExpectedAnswer(
        comparison_order=("compound-a", "compound-b"),
        acceptable_comparison_orders=(("compound-b", "compound-a"),),
        comparison_evidence_ids=("evidence-primary",),
    )

    assert expected.has_expectations
    assert expected.evidence_ids == ("evidence-primary",)


@pytest.mark.parametrize(
    "payload",
    [
        {"expected": False, "reason": "evidence_absent"},
        {
            "expected": False,
            "reason": "not_applicable",
            "partial_answer_allowed": True,
        },
        {"expected": True, "reason": "not_applicable"},
        {"expected": 1, "reason": "evidence_absent"},
    ],
)
def test_abstention_contract_rejects_contradictory_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AbstentionExpectation.model_validate(payload)
