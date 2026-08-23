"""Tests for benchmark coverage validation."""

from collections.abc import Mapping
from typing import cast

import pytest

from inference_lab.evaluation.cases import QuestionType
from inference_lab.evaluation.coverage import (
    CoveragePolicy,
    validate_benchmark_coverage,
    validate_final_benchmark,
)

from .fixtures import benchmark_cases, coverage_case


def test_coverage_report_counts_complete_and_held_out_composition() -> None:
    cases = tuple(
        coverage_case(
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
    report = validate_final_benchmark(benchmark_cases())

    assert report.total_cases == 30
    assert report.answerable_cases == 24
    assert report.abstention_cases == 6
    assert report.held_out_cases == 6
    assert report.held_out_answerable_cases == 5
    assert report.held_out_abstention_cases == 1
    assert set(report.held_out_question_type_counts.values()) == {1}


def test_final_benchmark_rejects_more_than_fifty_cases() -> None:
    with pytest.raises(ValueError, match="at most 50 cases allowed"):
        validate_benchmark_coverage(benchmark_cases(extra_cases=21))


def test_final_benchmark_rejects_unrepresentative_held_out_question_types() -> None:
    held_out_indexes = frozenset({1, 3, 7, 11, 15, 20})

    with pytest.raises(ValueError, match="held-out difficult_terminology"):
        validate_benchmark_coverage(benchmark_cases(held_out_indexes=held_out_indexes))


def test_final_benchmark_rejects_held_out_without_abstention_coverage() -> None:
    abstention_indexes = frozenset({2, 8, 12, 16, 21, 25})

    with pytest.raises(ValueError, match="abstention held-out"):
        validate_benchmark_coverage(
            benchmark_cases(abstention_indexes=abstention_indexes)
        )


def test_coverage_requirements_report_failures() -> None:
    cases = (coverage_case(1, "lookup"),)

    with pytest.raises(ValueError, match="coverage requirements failed"):
        validate_benchmark_coverage(cases)

    with pytest.raises(ValueError, match="At least one evaluation case"):
        validate_benchmark_coverage(())


def test_coverage_failures_are_complete_and_deterministically_ordered() -> None:
    policy = CoveragePolicy(
        minimum_cases=2,
        maximum_cases=3,
        minimum_answerable_cases=2,
        minimum_abstention_cases=1,
        minimum_held_out_cases=1,
        minimum_held_out_answerable_cases=1,
        minimum_held_out_abstention_cases=1,
        minimum_question_types={"comparison": 1},
        minimum_held_out_question_types={"lookup": 1},
    )

    with pytest.raises(ValueError) as caught:
        validate_benchmark_coverage((coverage_case(1, "lookup"),), policy=policy)

    assert str(caught.value) == (
        "Benchmark coverage requirements failed: at least 2 cases required; "
        "at least 2 answerable cases required; at least 1 abstention cases required; "
        "at least 1 held-out cases required; at least 1 answerable held-out cases "
        "required; at least 1 abstention held-out cases required; at least 1 "
        "comparison cases required; found 0; at least 1 held-out lookup cases "
        "required; found 0"
    )
