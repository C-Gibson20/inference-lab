"""Tests for benchmark coverage validation."""

from collections.abc import Mapping
from typing import cast

import pytest

from inference_lab.evaluation.cases import EvaluationCase, QuestionType
from inference_lab.evaluation.coverage import (
    CoveragePolicy,
    validate_benchmark_coverage,
    validate_final_benchmark,
)
from inference_lab.evaluation.utils import SplitName

from .test_loader import _case


def _repeat_question_type(
    question_type: QuestionType,
    count: int,
) -> tuple[QuestionType, ...]:
    return (question_type,) * count


QUESTION_TYPE_DISTRIBUTION: tuple[QuestionType, ...] = (
    *_repeat_question_type("lookup", 6),
    *_repeat_question_type("comparison", 4),
    *_repeat_question_type("thematic", 4),
    *_repeat_question_type("aggregation", 5),
    *_repeat_question_type("follow_up", 4),
    *_repeat_question_type("difficult_terminology", 3),
    *_repeat_question_type("lookup", 4),
)

DEFAULT_HELD_OUT_INDEXES = frozenset({1, 7, 11, 15, 20, 24})
DEFAULT_ABSTENTION_INDEXES = frozenset({2, 8, 12, 16, 21, 24})


def _coverage_case(
    index: int,
    question_type: QuestionType,
    *,
    split: SplitName = "development",
    abstention: bool = False,
) -> EvaluationCase:
    case_id = f"case-{index:03d}"
    payload = _case(case_id, split)
    payload["question"] = f"What result is recorded for benchmark item {index}?"
    payload["question_type"] = question_type
    payload["context"] = (
        (f"Previous question for benchmark item {index}",)
        if question_type == "follow_up"
        else ()
    )
    payload["abstention"] = {
        "expected": abstention,
        "reason": "evidence_absent" if abstention else "not_applicable",
    }
    payload["evidence"][0].update(
        {
            "document_id": f"00000000-0000-4000-8000-{index:012x}",
            "document_version_id": f"00000000-0000-4001-8000-{index:012x}",
            "source_span_id": f"00000000-0000-4002-8000-{index:012x}",
        }
    )
    if abstention:
        payload["evidence"][0]["role"] = "context"
        payload["expected_answer"] = {}
    else:
        payload["expected_answer"] = {
            "claims": (
                {
                    "key": "result",
                    "description": f"Expected result for benchmark item {index}",
                    "evidence_ids": ("evidence-primary",),
                },
            )
        }
    return EvaluationCase.model_validate(payload)


def _benchmark_cases(
    *,
    held_out_indexes: frozenset[int] = DEFAULT_HELD_OUT_INDEXES,
    abstention_indexes: frozenset[int] = DEFAULT_ABSTENTION_INDEXES,
    extra_cases: int = 0,
) -> tuple[EvaluationCase, ...]:
    question_types = QUESTION_TYPE_DISTRIBUTION + ("lookup",) * extra_cases
    return tuple(
        _coverage_case(
            index,
            question_type,
            split="held_out" if index in held_out_indexes else "development",
            abstention=index in abstention_indexes,
        )
        for index, question_type in enumerate(question_types, start=1)
    )


def test_coverage_report_counts_complete_and_held_out_composition() -> None:
    cases = tuple(
        _coverage_case(
            index,
            "lookup",
            split="held_out" if index == 10 else "development",
            abstention=index == 9,
        )
        for index in range(1, 11)
    )
    policy = CoveragePolicy(
        minimum_cases=10,
        maximum_cases=10,
        minimum_answerable_cases=9,
        minimum_abstention_cases=1,
        minimum_held_out_cases=1,
        minimum_held_out_answerable_cases=1,
        minimum_held_out_abstention_cases=0,
        minimum_question_types={"lookup": 10},
        minimum_held_out_question_types={"lookup": 1},
    )

    report = validate_benchmark_coverage(cases, policy=policy)

    assert report.total_cases == 10
    assert report.answerable_cases == 9
    assert report.abstention_cases == 1
    assert report.held_out_cases == 1
    assert report.held_out_answerable_cases == 1
    assert report.held_out_abstention_cases == 0
    assert report.question_type_counts["lookup"] == 10
    assert report.question_type_counts["comparison"] == 0
    assert report.held_out_question_type_counts["lookup"] == 1
    with pytest.raises(TypeError):
        report.question_type_counts["lookup"] = 0  # type: ignore[index]


def test_coverage_policy_is_defensive_and_rejects_invalid_thresholds() -> None:
    minima: dict[QuestionType, int] = {"lookup": 1}
    policy = CoveragePolicy(
        minimum_question_types=minima,
        minimum_held_out_question_types={},
    )
    minima["lookup"] = 2
    assert policy.minimum_question_types["lookup"] == 1
    with pytest.raises(TypeError):
        policy.minimum_question_types["lookup"] = 2  # type: ignore[index]

    with pytest.raises(TypeError, match="minimum_cases must be an integer"):
        CoveragePolicy(minimum_cases=True)
    with pytest.raises(ValueError, match="non-negative"):
        CoveragePolicy(minimum_abstention_cases=-1)
    with pytest.raises(ValueError, match="cannot exceed"):
        CoveragePolicy(minimum_cases=51, maximum_cases=50)
    with pytest.raises(ValueError, match="exceeds maximum_cases"):
        CoveragePolicy(minimum_answerable_cases=50, minimum_abstention_cases=1)

    unknown_minima = cast(Mapping[QuestionType, int], {"unsupported": 1})
    with pytest.raises(ValueError, match="unknown question types"):
        CoveragePolicy(minimum_question_types=unknown_minima)

    invalid_minima = cast(Mapping[QuestionType, int], ())
    with pytest.raises(TypeError, match="must be a mapping"):
        CoveragePolicy(minimum_question_types=invalid_minima)


def test_final_benchmark_accepts_complete_representative_question_set() -> None:
    report = validate_final_benchmark(_benchmark_cases())

    assert report.total_cases == 30
    assert report.answerable_cases == 24
    assert report.abstention_cases == 6
    assert report.held_out_cases == 6
    assert report.held_out_answerable_cases == 5
    assert report.held_out_abstention_cases == 1
    assert set(report.held_out_question_type_counts.values()) == {1}


def test_final_benchmark_rejects_more_than_fifty_cases() -> None:
    with pytest.raises(ValueError, match="at most 50 cases allowed"):
        validate_benchmark_coverage(_benchmark_cases(extra_cases=21))


def test_final_benchmark_rejects_unrepresentative_held_out_question_types() -> None:
    held_out_indexes = frozenset({1, 3, 7, 11, 15, 20})

    with pytest.raises(ValueError, match="held-out difficult_terminology"):
        validate_benchmark_coverage(_benchmark_cases(held_out_indexes=held_out_indexes))


def test_final_benchmark_rejects_held_out_without_abstention_coverage() -> None:
    abstention_indexes = frozenset({2, 8, 12, 16, 21, 25})

    with pytest.raises(ValueError, match="abstention held-out"):
        validate_benchmark_coverage(
            _benchmark_cases(abstention_indexes=abstention_indexes)
        )


def test_coverage_requirements_report_failures() -> None:
    cases = (_coverage_case(1, "lookup"),)

    with pytest.raises(ValueError, match="coverage requirements failed"):
        validate_benchmark_coverage(cases)

    with pytest.raises(ValueError, match="At least one evaluation case"):
        validate_benchmark_coverage(())
