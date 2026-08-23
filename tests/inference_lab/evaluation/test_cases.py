"""Tests for evaluation-case semantics."""

from typing import Literal

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.cases import EvaluationCase

from .fixtures import EVIDENCE_IDS


def _evidence(
    *,
    role: Literal["authoritative", "context"] = "authoritative",
    evidence_id: str = "evidence-primary",
) -> tuple[dict[str, object], ...]:
    return (dict(EVIDENCE_IDS, evidence_id=evidence_id, role=role),)


def _answer(*, evidence_id: str = "evidence-primary") -> dict[str, object]:
    return {
        "claims": (
            {
                "key": "assay",
                "description": "The assay used by the study",
                "evidence_ids": (evidence_id,),
            },
        )
    }


def _case_payload() -> dict[str, object]:
    return {
        "case_id": "case-001",
        "question": "What assay was used?",
        "question_type": "lookup",
        "expected_answer": _answer(),
        "evidence": _evidence(),
        "abstention": {"expected": False, "reason": "not_applicable"},
        "split": "development",
    }


def test_answerable_case_requires_linked_authoritative_evidence() -> None:
    case = EvaluationCase.model_validate(_case_payload())

    assert case.expected_answer.evidence_ids == ("evidence-primary",)

    without_evidence = _case_payload()
    without_evidence["evidence"] = ()
    with pytest.raises(ValidationError, match="authoritative evidence"):
        EvaluationCase.model_validate(without_evidence)

    context_only = _case_payload()
    context_only["evidence"] = _evidence(role="context")
    with pytest.raises(ValidationError, match="authoritative evidence"):
        EvaluationCase.model_validate(context_only)


def test_case_rejects_duplicate_or_unreferenced_evidence() -> None:
    duplicate_id = _case_payload()
    duplicate_id["evidence"] = (*_evidence(), *_evidence())
    with pytest.raises(ValidationError, match="Evidence IDs must be unique"):
        EvaluationCase.model_validate(duplicate_id)

    duplicate_location = _case_payload()
    duplicate_location["evidence"] = (
        *_evidence(),
        dict(EVIDENCE_IDS, evidence_id="evidence-secondary"),
    )
    with pytest.raises(ValidationError, match="locations must be unique"):
        EvaluationCase.model_validate(duplicate_location)

    unreferenced = _case_payload()
    second = dict(EVIDENCE_IDS)
    second.update(
        {
            "evidence_id": "evidence-secondary",
            "document_id": "00000000-0000-4000-8000-000000000006",
            "document_version_id": "00000000-0000-4000-8000-000000000007",
            "source_span_id": "00000000-0000-4000-8000-000000000008",
        }
    )
    unreferenced["evidence"] = (*_evidence(), second)
    with pytest.raises(ValidationError, match="not referenced"):
        EvaluationCase.model_validate(unreferenced)


def test_case_rejects_missing_or_invalid_evidence_links() -> None:
    unknown = _case_payload()
    unknown["expected_answer"] = _answer(evidence_id="evidence-missing")
    with pytest.raises(ValidationError, match="unknown evidence"):
        EvaluationCase.model_validate(unknown)

    context_link = _case_payload()
    context_link.update(
        {
            "evidence": _evidence(role="context"),
            "abstention": {
                "expected": True,
                "reason": "evidence_insufficient",
                "partial_answer_allowed": True,
            },
        }
    )
    with pytest.raises(ValidationError, match="non-authoritative evidence"):
        EvaluationCase.model_validate(context_link)


def test_absent_evidence_abstention_has_no_expected_answer() -> None:
    payload = _case_payload()
    payload.update(
        {
            "expected_answer": {},
            "evidence": _evidence(role="context"),
            "abstention": {"expected": True, "reason": "evidence_absent"},
        }
    )

    case = EvaluationCase.model_validate(payload)
    assert case.abstention.expected

    payload["expected_answer"] = _answer()
    with pytest.raises(ValidationError, match="cannot define expected answers"):
        EvaluationCase.model_validate(payload)

    payload["expected_answer"] = {}
    payload["evidence"] = _evidence()
    with pytest.raises(ValidationError, match="cannot have authoritative evidence"):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize("reason", ["evidence_insufficient", "evidence_ambiguous"])
def test_abstention_can_retain_authoritative_partial_evidence(reason: str) -> None:
    payload = _case_payload()
    payload["abstention"] = {
        "expected": True,
        "reason": reason,
        "partial_answer_allowed": True,
    }

    case = EvaluationCase.model_validate(payload)

    assert case.expected_answer.evidence_ids == ("evidence-primary",)
    assert case.abstention.partial_answer_allowed


def test_insufficient_and_ambiguous_reasons_require_available_evidence() -> None:
    insufficient = _case_payload()
    insufficient.update(
        {
            "expected_answer": {},
            "evidence": (),
            "abstention": {
                "expected": True,
                "reason": "evidence_insufficient",
            },
        }
    )
    with pytest.raises(ValidationError, match="require available evidence"):
        EvaluationCase.model_validate(insufficient)

    ambiguous = dict(insufficient)
    ambiguous["evidence"] = _evidence(role="context")
    ambiguous["abstention"] = {
        "expected": True,
        "reason": "evidence_ambiguous",
    }
    with pytest.raises(ValidationError, match="require authoritative evidence"):
        EvaluationCase.model_validate(ambiguous)

    partial = dict(insufficient)
    partial["evidence"] = _evidence(role="context")
    partial["abstention"] = {
        "expected": True,
        "reason": "evidence_insufficient",
        "partial_answer_allowed": True,
    }
    with pytest.raises(ValidationError, match="expected-answer characteristics"):
        EvaluationCase.model_validate(partial)


def test_follow_up_requires_non_blank_context() -> None:
    payload = _case_payload()
    payload["question_type"] = "follow_up"
    with pytest.raises(ValidationError, match="conversation context"):
        EvaluationCase.model_validate(payload)

    payload["context"] = (" ",)
    with pytest.raises(ValidationError, match="blank turns"):
        EvaluationCase.model_validate(payload)


def test_unknown_case_fields_and_duplicate_tags_are_rejected() -> None:
    payload = _case_payload()
    payload["unexpected"] = "value"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationCase.model_validate(payload)

    payload = _case_payload()
    payload["tags"] = ["lookup", "lookup"]
    with pytest.raises(ValidationError, match="tags must be unique"):
        EvaluationCase.model_validate(payload)

    payload = _case_payload()
    payload["tags"] = {" "}
    with pytest.raises(ValidationError, match="non-blank strings"):
        EvaluationCase.model_validate(payload)

    payload = _case_payload()
    payload["question"] = " "
    with pytest.raises(ValidationError, match="question must not be blank"):
        EvaluationCase.model_validate(payload)


def test_answerable_case_requires_expected_characteristics() -> None:
    payload = _case_payload()
    payload["expected_answer"] = {"summary": "Illustrative text only"}

    with pytest.raises(ValidationError, match="expected-answer characteristics"):
        EvaluationCase.model_validate(payload)
