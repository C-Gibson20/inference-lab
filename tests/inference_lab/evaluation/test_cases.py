"""Tests for evaluation-case semantics."""

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.cases import EvaluationCase

from .test_schemas import IDS


def _evidence() -> tuple[dict[str, str], ...]:
    return (dict(IDS),)


def _answer() -> dict[str, object]:
    return {"claims": ({"key": "assay"},)}


def test_answerable_case_requires_authoritative_evidence() -> None:
    case = EvaluationCase(
        case_id="case-001",
        question="What assay was used?",
        question_type="lookup",
        expected_answer=_answer(),
        evidence=_evidence(),
        abstention={"expected": False, "reason": "not_applicable"},
        split="development",
    )

    assert case.case_id == "case-001"


def test_answerable_case_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="authoritative evidence"):
        EvaluationCase(
            case_id="case-001",
            question="What assay was used?",
            question_type="lookup",
            expected_answer=_answer(),
            abstention={"expected": False, "reason": "not_applicable"},
            split="development",
        )


def test_follow_up_requires_context() -> None:
    with pytest.raises(ValidationError, match="conversation context"):
        EvaluationCase(
            case_id="case-001",
            question="What about the previous result?",
            question_type="follow_up",
            expected_answer=_answer(),
            evidence=_evidence(),
            abstention={"expected": False, "reason": "not_applicable"},
            split="development",
        )


def test_unknown_case_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationCase(
            case_id="case-001",
            question="What assay was used?",
            question_type="lookup",
            expected_answer=_answer(),
            evidence=_evidence(),
            abstention={"expected": False, "reason": "not_applicable"},
            split="development",
            unexpected="value",
        )
