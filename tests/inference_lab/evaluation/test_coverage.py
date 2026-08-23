"""Tests for benchmark coverage validation."""

import pytest

from inference_lab.evaluation.coverage import validate_benchmark_coverage

from .test_loader import _case
from inference_lab.evaluation.cases import EvaluationCase


def _coverage_case(
    case_id: str,
    question_type: str,
    *,
    split: str = "development",
    abstention: bool = False,
) -> EvaluationCase:
    payload = _case(case_id, split)
    payload["question_type"] = question_type
    payload["abstention"] = {
        "expected": abstention,
        "reason": "evidence_absent" if abstention else "not_applicable",
    }
    if abstention:
        payload["evidence"][0]["role"] = "context"
        payload["expected_answer"] = {}
    return EvaluationCase.model_validate(payload)


def test_coverage_report_counts_cases() -> None:
    cases = tuple(
        _coverage_case(
            f"case-{index:03d}",
            "lookup",
            split="held_out" if index == 10 else "development",
            abstention=index == 9,
        )
        for index in range(1, 11)
    )

    report = validate_benchmark_coverage(
        cases,
        minimum_cases=10,
        minimum_answerable_cases=9,
        minimum_abstention_cases=1,
        minimum_held_out_cases=1,
        minimum_question_types={"lookup": 10},
    )

    assert report.total_cases == 10
    assert report.answerable_cases == 9
    assert report.abstention_cases == 1
    assert report.held_out_cases == 1


def test_coverage_requirements_report_failures() -> None:
    cases = (_coverage_case("case-001", "lookup"),)

    with pytest.raises(ValueError, match="coverage requirements failed"):
        validate_benchmark_coverage(cases)
