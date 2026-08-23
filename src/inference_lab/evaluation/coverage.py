"""Benchmark coverage policy and validation."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from inference_lab.evaluation.cases import (
    QUESTION_TYPES,
    EvaluationCase,
    QuestionType,
)
from inference_lab.evaluation.splitting import validate_no_leakage

DEFAULT_MINIMUM_QUESTION_TYPES: Mapping[QuestionType, int] = MappingProxyType(
    {
        "lookup": 6,
        "comparison": 4,
        "thematic": 4,
        "aggregation": 5,
        "follow_up": 4,
        "difficult_terminology": 3,
    }
)

DEFAULT_MINIMUM_HELD_OUT_QUESTION_TYPES: Mapping[QuestionType, int] = MappingProxyType(
    dict.fromkeys(QUESTION_TYPES, 1)
)


def _validate_count(name: str, value: int, *, positive: bool = False) -> None:
    """Validate one strict integer coverage threshold."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_question_type_minima(
    name: str,
    minima: Mapping[QuestionType, int],
) -> Mapping[QuestionType, int]:
    """Validate and defensively freeze question-type minima."""
    if not isinstance(minima, Mapping):
        raise TypeError(f"{name} must be a mapping")

    unknown_types = set(minima) - set(QUESTION_TYPES)
    if unknown_types:
        names = ", ".join(sorted(unknown_types))
        raise ValueError(f"{name} contains unknown question types: {names}")

    copied = dict(minima)
    for question_type, minimum in copied.items():
        _validate_count(f"{name}[{question_type}]", minimum)
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
    """Immutable acceptance thresholds for an evaluation question set."""

    minimum_cases: int = 30
    maximum_cases: int = 50
    minimum_answerable_cases: int = 24
    minimum_abstention_cases: int = 6
    minimum_held_out_cases: int = 6
    minimum_held_out_answerable_cases: int = 1
    minimum_held_out_abstention_cases: int = 1
    minimum_question_types: Mapping[QuestionType, int] = DEFAULT_MINIMUM_QUESTION_TYPES
    minimum_held_out_question_types: Mapping[QuestionType, int] = (
        DEFAULT_MINIMUM_HELD_OUT_QUESTION_TYPES
    )

    def __post_init__(self) -> None:
        """Reject invalid or internally impossible threshold combinations."""
        _validate_count("minimum_cases", self.minimum_cases, positive=True)
        _validate_count("maximum_cases", self.maximum_cases, positive=True)
        for name in (
            "minimum_answerable_cases",
            "minimum_abstention_cases",
            "minimum_held_out_cases",
            "minimum_held_out_answerable_cases",
            "minimum_held_out_abstention_cases",
        ):
            _validate_count(name, getattr(self, name))

        minimum_question_types = _validate_question_type_minima(
            "minimum_question_types",
            self.minimum_question_types,
        )
        minimum_held_out_question_types = _validate_question_type_minima(
            "minimum_held_out_question_types",
            self.minimum_held_out_question_types,
        )
        object.__setattr__(
            self,
            "minimum_question_types",
            minimum_question_types,
        )
        object.__setattr__(
            self,
            "minimum_held_out_question_types",
            minimum_held_out_question_types,
        )

        if self.minimum_cases > self.maximum_cases:
            raise ValueError("minimum_cases cannot exceed maximum_cases")

        required_totals = {
            "answerability minima": (
                self.minimum_answerable_cases + self.minimum_abstention_cases
            ),
            "question-type minima": sum(minimum_question_types.values()),
            "held-out case minimum": self.minimum_held_out_cases,
            "held-out answerability minima": (
                self.minimum_held_out_answerable_cases
                + self.minimum_held_out_abstention_cases
            ),
            "held-out question-type minima": sum(
                minimum_held_out_question_types.values()
            ),
        }
        impossible = [
            name
            for name, required in required_totals.items()
            if required > self.maximum_cases
        ]
        if impossible:
            names = ", ".join(impossible)
            raise ValueError(f"Coverage policy exceeds maximum_cases: {names}")


FINAL_BENCHMARK_POLICY: Final[CoveragePolicy] = CoveragePolicy()


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Coverage counts produced by benchmark validation."""

    total_cases: int
    answerable_cases: int
    abstention_cases: int
    held_out_cases: int
    held_out_answerable_cases: int
    held_out_abstention_cases: int
    question_type_counts: Mapping[QuestionType, int]
    held_out_question_type_counts: Mapping[QuestionType, int]

    def __post_init__(self) -> None:
        """Defensively freeze the report's nested count mappings."""
        object.__setattr__(
            self,
            "question_type_counts",
            MappingProxyType(dict(self.question_type_counts)),
        )
        object.__setattr__(
            self,
            "held_out_question_type_counts",
            MappingProxyType(dict(self.held_out_question_type_counts)),
        )


def validate_benchmark_coverage(
    cases: tuple[EvaluationCase, ...],
    *,
    policy: CoveragePolicy = FINAL_BENCHMARK_POLICY,
) -> CoverageReport:
    """Validate category, answerability, and held-out coverage."""
    if not cases:
        raise ValueError("At least one evaluation case is required")

    question_type_counts: Counter[QuestionType] = Counter(
        case.question_type for case in cases
    )
    answerable_cases = sum(not case.abstention.expected for case in cases)
    abstention_cases = sum(case.abstention.expected for case in cases)
    held_out = tuple(case for case in cases if case.split == "held_out")
    held_out_answerable_cases = sum(not case.abstention.expected for case in held_out)
    held_out_abstention_cases = sum(case.abstention.expected for case in held_out)
    held_out_question_type_counts: Counter[QuestionType] = Counter(
        case.question_type for case in held_out
    )

    failures: list[str] = []
    if len(cases) < policy.minimum_cases:
        failures.append(f"at least {policy.minimum_cases} cases required")
    if len(cases) > policy.maximum_cases:
        failures.append(f"at most {policy.maximum_cases} cases allowed")
    if answerable_cases < policy.minimum_answerable_cases:
        failures.append(
            f"at least {policy.minimum_answerable_cases} answerable cases required"
        )
    if abstention_cases < policy.minimum_abstention_cases:
        failures.append(
            f"at least {policy.minimum_abstention_cases} abstention cases required"
        )
    if len(held_out) < policy.minimum_held_out_cases:
        failures.append(
            f"at least {policy.minimum_held_out_cases} held-out cases required"
        )
    if held_out_answerable_cases < policy.minimum_held_out_answerable_cases:
        failures.append(
            "at least "
            f"{policy.minimum_held_out_answerable_cases} answerable held-out cases "
            "required"
        )
    if held_out_abstention_cases < policy.minimum_held_out_abstention_cases:
        failures.append(
            "at least "
            f"{policy.minimum_held_out_abstention_cases} abstention held-out cases "
            "required"
        )

    for question_type, minimum in policy.minimum_question_types.items():
        actual = question_type_counts[question_type]
        if actual < minimum:
            failures.append(
                f"at least {minimum} {question_type} cases required; found {actual}"
            )

    for question_type, minimum in policy.minimum_held_out_question_types.items():
        actual = held_out_question_type_counts[question_type]
        if actual < minimum:
            failures.append(
                f"at least {minimum} held-out {question_type} cases required; "
                f"found {actual}"
            )

    if failures:
        raise ValueError(
            "Benchmark coverage requirements failed: " + "; ".join(failures)
        )

    return CoverageReport(
        total_cases=len(cases),
        answerable_cases=answerable_cases,
        abstention_cases=abstention_cases,
        held_out_cases=len(held_out),
        held_out_answerable_cases=held_out_answerable_cases,
        held_out_abstention_cases=held_out_abstention_cases,
        question_type_counts={
            question_type: question_type_counts[question_type]
            for question_type in QUESTION_TYPES
        },
        held_out_question_type_counts={
            question_type: held_out_question_type_counts[question_type]
            for question_type in QUESTION_TYPES
        },
    )


def validate_final_benchmark(
    cases: tuple[EvaluationCase, ...],
) -> CoverageReport:
    """Validate final coverage and ensure no leakage group crosses splits."""
    report = validate_benchmark_coverage(cases)
    validate_no_leakage(cases)
    return report
