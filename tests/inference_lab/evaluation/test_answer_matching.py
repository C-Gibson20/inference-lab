"""Tests for deterministic typed answer matching."""

from datetime import date

import pytest

from inference_lab.evaluation.answer_matching import (
    AnswerObservation,
    claim_matches,
    date_matches,
    match_expected_answer,
    numeric_matches,
)
from inference_lab.evaluation.schemas import (
    AbstentionExpectation,
    AnswerClaim,
    DateExpectation,
    ExpectedAnswer,
    NumericExpectation,
)


def test_claim_matching_accepts_normalized_alternatives() -> None:
    claim = AnswerClaim(
        key="assay",
        acceptable_values=("Cell Based",),
        normalized_values=("cell-based",),
    )

    assert claim_matches(claim, "  CELL-BASED ")
    assert not claim_matches(claim, "animal model")


def test_claim_without_alternatives_matches_presence_only() -> None:
    assert claim_matches(AnswerClaim(key="assay"), "any value")


def test_numeric_matching_supports_absolute_and_relative_tolerance() -> None:
    absolute = NumericExpectation(key="value", value=10, absolute_tolerance=0.5)
    relative = NumericExpectation(key="value", value=100, relative_tolerance=0.1)

    assert numeric_matches(absolute, 10.5)
    assert not numeric_matches(absolute, 10.51)
    assert numeric_matches(relative, 109)
    assert not numeric_matches(relative, 111)


def test_numeric_matching_accepts_alternative_values() -> None:
    expected = NumericExpectation(
        key="value",
        value=10,
        acceptable_values=(20, 30),
    )

    assert numeric_matches(expected, 20)
    assert not numeric_matches(expected, 21)


def test_date_matching_includes_tolerance_boundary() -> None:
    expected = DateExpectation(
        key="publication_date",
        value=date(2024, 3, 15),
        tolerance_days=1,
    )

    assert date_matches(expected, date(2024, 3, 16))
    assert not date_matches(expected, date(2024, 3, 17))


def test_match_expected_answer_reports_required_failures() -> None:
    expected = ExpectedAnswer(
        claims=(
            AnswerClaim(
                key="assay",
                acceptable_values=("cell-based",),
            ),
            AnswerClaim(key="optional_note", required=False),
        ),
        numeric_values=(
            NumericExpectation(key="activity", value=10, absolute_tolerance=1),
        ),
        date_values=(DateExpectation(key="date", value=date(2024, 3, 15)),),
        comparison_order=("compound-a", "compound-b"),
    )
    observed = AnswerObservation(
        claims={"assay": "cell based"},
        numeric_values={"activity": 12},
        date_values={"date": date(2024, 3, 15)},
        comparison_order=("compound-b", "compound-a"),
    )

    result = match_expected_answer(
        expected,
        AbstentionExpectation(expected=False, reason="not_applicable"),
        observed,
    )

    assert not result.matched
    assert result.mismatched_numeric_values == ("activity",)
    assert result.mismatched_comparison


def test_match_expected_answer_accepts_alternative_comparison_orders() -> None:
    expected = ExpectedAnswer(
        acceptable_comparison_orders=(
            ("compound-a", "compound-b"),
            ("compound-b", "compound-a"),
        )
    )
    observed = AnswerObservation(
        claims={},
        numeric_values={},
        date_values={},
        comparison_order=("compound-b", "compound-a"),
    )

    result = match_expected_answer(
        expected,
        AbstentionExpectation(expected=False, reason="not_applicable"),
        observed,
    )

    assert result.matched


def test_expected_answer_matches_typed_observation() -> None:
    expected = ExpectedAnswer(
        claims=(AnswerClaim(key="assay", acceptable_values=("cell-based",)),),
        numeric_values=(NumericExpectation(key="activity", value=10),),
        date_values=(DateExpectation(key="date", value=date(2024, 3, 15)),),
    )
    observed = AnswerObservation(
        claims={"assay": "CELL-BASED"},
        numeric_values={"activity": 10},
        date_values={"date": date(2024, 3, 15)},
    )

    result = match_expected_answer(
        expected,
        AbstentionExpectation(expected=False, reason="not_applicable"),
        observed,
    )

    assert result.matched


def test_abstention_expectations_are_explicit() -> None:
    abstention = AbstentionExpectation(
        expected=True,
        reason="evidence_absent",
    )
    observed = AnswerObservation(
        claims={}, numeric_values={}, date_values={}, abstained=True
    )

    assert match_expected_answer(ExpectedAnswer(), abstention, observed).matched
    assert not match_expected_answer(
        ExpectedAnswer(),
        abstention,
        AnswerObservation(claims={}, numeric_values={}, date_values={}),
    ).matched


def test_partial_abstention_respects_policy() -> None:
    expected = ExpectedAnswer(
        claims=(AnswerClaim(key="known_fact"),),
    )
    observed = AnswerObservation(
        claims={"known_fact": "known"},
        numeric_values={},
        date_values={},
        abstained=True,
    )

    disallowed = AbstentionExpectation(
        expected=True,
        reason="evidence_insufficient",
        partial_answer_allowed=False,
    )
    allowed = disallowed.model_copy(update={"partial_answer_allowed": True})

    assert not match_expected_answer(expected, disallowed, observed).matched
    assert match_expected_answer(expected, allowed, observed).matched


def test_observations_reject_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="finite"):
        AnswerObservation(
            claims={},
            numeric_values={"value": float("nan")},
            date_values={},
        )
