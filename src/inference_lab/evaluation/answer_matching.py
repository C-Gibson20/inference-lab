"""Deterministic matching of typed evaluation expectations."""

import math
from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType
from typing import Mapping

from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    AnswerClaim,
    DateExpectation,
    ExpectedAnswer,
    NumericExpectation,
)
from inference_lab.evaluation.utils import normalise_text


@dataclass(frozen=True, slots=True)
class AnswerObservation:
    """Structured answer values extracted for evaluation."""

    claims: Mapping[str, str]
    numeric_values: Mapping[str, float]
    date_values: Mapping[str, date]
    comparison_order: tuple[str, ...] | None = None
    abstained: bool = False

    def __post_init__(self) -> None:
        """Validate and defensively copy observed answer values."""

        self._validate()

        object.__setattr__(self, "claims", MappingProxyType(dict(self.claims)))
        object.__setattr__(
            self,
            "numeric_values",
            MappingProxyType(dict(self.numeric_values)),
        )
        object.__setattr__(
            self,
            "date_values",
            MappingProxyType(dict(self.date_values)),
        )

    def _validate(self) -> None:
        """Validate observed answer values."""

        if any(not key for key in self.claims):
            raise ValueError("Claim keys must not be empty")
        if any(not key for key in self.numeric_values):
            raise ValueError("Numeric value keys must not be empty")
        if any(not key for key in self.date_values):
            raise ValueError("Date value keys must not be empty")
        if any(not math.isfinite(value) for value in self.numeric_values.values()):
            raise ValueError("Numeric observations must be finite")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of matching an observed answer against expectations."""

    matched: bool
    missing_required_claims: tuple[str, ...] = ()
    mismatched_claims: tuple[str, ...] = ()
    mismatched_numeric_values: tuple[str, ...] = ()
    mismatched_dates: tuple[str, ...] = ()
    mismatched_comparison: bool = False
    abstention_matched: bool = True


def claim_matches(claim: AnswerClaim, observed: str) -> bool:
    """Return whether an observed claim matches its accepted alternatives."""

    accepted = {
        normalise_text(value)
        for value in (*claim.acceptable_values, *claim.normalized_values)
    }
    return not accepted or normalise_text(observed) in accepted


def numeric_matches(expected: NumericExpectation, observed: float) -> bool:
    """Return whether an observed numeric value matches its expectation."""

    if not math.isfinite(observed):
        return False

    candidates = (expected.value, *expected.acceptable_values)
    return any(
        _numeric_value_matches(expected, candidate, observed)
        for candidate in candidates
    )


def _numeric_value_matches(
    expected: NumericExpectation,
    expected_value: float,
    observed: float,
) -> bool:
    """Match one numeric candidate against an observation."""
    difference = abs(observed - expected_value)

    if expected.absolute_tolerance is not None and (
        difference <= expected.absolute_tolerance
    ):
        return True

    if expected.relative_tolerance is not None:
        denominator = max(abs(expected_value), 1e-12)
        if difference / denominator <= expected.relative_tolerance:
            return True

    return (
        expected.absolute_tolerance is None
        and expected.relative_tolerance is None
        and difference == 0.0
    )


def date_matches(expected: DateExpectation, observed: date) -> bool:
    """Return whether an observed date is within the inclusive tolerance."""
    return abs(observed - expected.value) <= timedelta(days=expected.tolerance_days)


def comparison_matches(
    expected: tuple[str, ...] | None,
    observed: tuple[str, ...] | None,
    acceptable: tuple[tuple[str, ...], ...] = (),
) -> bool:
    """Return whether an observed comparison order matches its expectation."""
    accepted = acceptable or (() if expected is None else (expected,))
    return not accepted or observed in accepted


def match_expected_answer(
    expected: ExpectedAnswer,
    abstention: AbstentionExpectation,
    observed: AnswerObservation,
) -> MatchResult:
    """Match a typed observation against answer and abstention expectations."""

    abstention_matched = abstention.expected == observed.abstained

    if observed.abstained:
        has_partial_answer = bool(
            observed.claims
            or observed.numeric_values
            or observed.date_values
            or observed.comparison_order
        )
        partial_allowed = not has_partial_answer or abstention.partial_answer_allowed
        return MatchResult(
            matched=abstention.expected and partial_allowed,
            abstention_matched=abstention_matched,
        )

    missing_claims: list[str] = []
    mismatched_claims: list[str] = []
    for claim in expected.claims:
        observed_value = observed.claims.get(claim.key)
        if observed_value is None:
            if claim.required:
                missing_claims.append(claim.key)
        elif not claim_matches(claim, observed_value):
            mismatched_claims.append(claim.key)

    mismatched_numeric = [
        expectation.key
        for expectation in expected.numeric_values
        if expectation.key not in observed.numeric_values
        or not numeric_matches(
            expectation,
            observed.numeric_values[expectation.key],
        )
    ]
    mismatched_dates = [
        expectation.key
        for expectation in expected.date_values
        if expectation.key not in observed.date_values
        or not date_matches(
            expectation,
            observed.date_values[expectation.key],
        )
    ]
    comparison_mismatch = not comparison_matches(
        expected.comparison_order,
        observed.comparison_order,
        expected.acceptable_comparison_orders,
    )

    required_claim_mismatches = {
        claim.key
        for claim in expected.claims
        if claim.required and claim.key in mismatched_claims
    }
    matched = (
        abstention_matched
        and not missing_claims
        and not required_claim_mismatches
        and not mismatched_numeric
        and not mismatched_dates
        and not comparison_mismatch
    )

    return MatchResult(
        matched=matched,
        missing_required_claims=tuple(sorted(missing_claims)),
        mismatched_claims=tuple(sorted(mismatched_claims)),
        mismatched_numeric_values=tuple(sorted(mismatched_numeric)),
        mismatched_dates=tuple(sorted(mismatched_dates)),
        mismatched_comparison=comparison_mismatch,
        abstention_matched=abstention_matched,
    )
