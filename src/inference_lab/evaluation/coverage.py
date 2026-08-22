"""Benchmark coverage policy and validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from inference_lab.evaluation.cases import EvaluationCase


DEFAULT_MINIMUM_QUESTION_TYPES: Mapping[str, int] = MappingProxyType(
    {
        "lookup": 6,
        "comparison": 4,
        "thematic": 4,
        "aggregation": 5,
        "follow_up": 4,
        "difficult_terminology": 3,
    }
)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Coverage counts produced by benchmark validation."""

    total_cases: int
    answerable_cases: int
    abstention_cases: int
    held_out_cases: int
    question_type_counts: Mapping[str, int]


def validate_benchmark_coverage(
    cases: tuple[EvaluationCase, ...],
    *,
    minimum_cases: int = 30,
    minimum_answerable_cases: int = 24,
    minimum_abstention_cases: int = 6,
    minimum_held_out_cases: int = 6,
    minimum_question_types: Mapping[str, int] = DEFAULT_MINIMUM_QUESTION_TYPES,
) -> CoverageReport:
    """Validate category, answerability, and held-out coverage."""
    if not cases:
        raise ValueError("At least one evaluation case is required")
    if minimum_cases < 1:
        raise ValueError("minimum_cases must be positive")

    question_type_counts = Counter(case.question_type for case in cases)
    answerable_cases = sum(not case.abstention.expected for case in cases)
    abstention_cases = sum(case.abstention.expected for case in cases)
    held_out_cases = sum(case.split == "held_out" for case in cases)

    failures: list[str] = []
    if len(cases) < minimum_cases:
        failures.append(f"at least {minimum_cases} cases required")
    if answerable_cases < minimum_answerable_cases:
        failures.append(
            f"at least {minimum_answerable_cases} answerable cases required"
        )
    if abstention_cases < minimum_abstention_cases:
        failures.append(
            f"at least {minimum_abstention_cases} abstention cases required"
        )
    if held_out_cases < minimum_held_out_cases:
        failures.append(f"at least {minimum_held_out_cases} held-out cases required")

    for question_type, minimum in minimum_question_types.items():
        actual = question_type_counts[question_type]
        if actual < minimum:
            failures.append(
                f"at least {minimum} {question_type} cases required; found {actual}"
            )

    if failures:
        raise ValueError(
            "Benchmark coverage requirements failed: " + "; ".join(failures)
        )

    return CoverageReport(
        total_cases=len(cases),
        answerable_cases=answerable_cases,
        abstention_cases=abstention_cases,
        held_out_cases=held_out_cases,
        question_type_counts=MappingProxyType(dict(question_type_counts)),
    )
