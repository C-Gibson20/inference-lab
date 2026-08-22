"""Tests for deterministic splitting and leakage prevention."""

from __future__ import annotations

import math

import pytest

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.splitting import (
    SplitPolicy,
    build_leakage_groups,
    generate_splits,
    jaccard_similarity,
    stable_digest,
    validate_no_leakage,
)

from .test_loader import (
    CORPUS_VERSION,
    DOCUMENT_ID,
    _case,
)


def _evaluation_case(
    case_id: str,
    question: str,
    *,
    document_id: str = DOCUMENT_ID,
    split: str = "development",
) -> EvaluationCase:
    payload = _case(case_id, split)
    payload["question"] = question
    payload["expected_answer"] = {"summary": case_id}
    suffix = document_id[-1]
    payload["evidence"][0].update(
        {
            "document_id": document_id,
            "corpus_version": CORPUS_VERSION,
            "document_version_id": (f"00000000-0000-4000-8000-00000000000{suffix}"),
            "source_span_id": f"00000000-0000-4000-8000-00000000000{suffix}",
        }
    )
    return EvaluationCase.model_validate(payload)


def test_split_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitPolicy(0.5, 0.25, 0.1)

    with pytest.raises(ValueError, match="finite"):
        SplitPolicy(math.nan, 0.25, 0.75)

    with pytest.raises(ValueError, match="near_duplicate_threshold"):
        SplitPolicy(0.5, 0.25, 0.25, near_duplicate_threshold=1.1)


def test_stable_digest_and_jaccard_are_deterministic() -> None:
    assert stable_digest("value") == stable_digest("value")
    assert jaccard_similarity(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1 / 3


def test_shared_document_cases_form_one_leakage_group() -> None:
    cases = (
        _evaluation_case("case-001", "What was measured?"),
        _evaluation_case("case-002", "Which method was used?"),
    )

    groups = build_leakage_groups(cases)

    assert len(groups) == 1
    assert {case.case_id for case in groups[0].cases} == {"case-001", "case-002"}


def test_near_duplicate_grouping_is_transitive() -> None:
    cases = (
        _evaluation_case("case-001", "alpha beta gamma"),
        _evaluation_case(
            "case-002",
            "alpha beta gamma delta",
            document_id="00000000-0000-4000-8000-000000000006",
        ),
        _evaluation_case(
            "case-003",
            "alpha beta delta",
            document_id="00000000-0000-4000-8000-000000000007",
        ),
    )

    groups = build_leakage_groups(cases, near_duplicate_threshold=0.7)

    assert len(groups) == 1


def test_cross_split_leakage_is_rejected() -> None:
    cases = (
        _evaluation_case("case-001", "What was measured?", split="train"),
        _evaluation_case("case-002", "Which method was used?", split="held_out"),
    )

    with pytest.raises(ValueError, match="crosses splits"):
        validate_no_leakage(cases)


def test_split_generation_is_stable_and_preserves_input_order() -> None:
    cases = tuple(
        _evaluation_case(
            f"case-00{index}",
            f"Question {index}",
            document_id=f"00000000-0000-4000-8000-00000000000{index + 5}",
        )
        for index in range(1, 5)
    )
    policy = SplitPolicy(0.5, 0.25, 0.25)

    first = generate_splits(cases, policy)
    second = generate_splits(tuple(reversed(cases)), policy)

    first_assignments = {case.case_id: case.split for case in first}
    second_assignments = {case.case_id: case.split for case in second}
    assert first_assignments == second_assignments
    assert [case.case_id for case in first] == [
        "case-001",
        "case-002",
        "case-003",
        "case-004",
    ]


def test_zero_fraction_splits_are_never_selected() -> None:
    cases = (
        _evaluation_case(
            "case-001",
            "Question one",
            document_id="00000000-0000-4000-8000-000000000006",
        ),
        _evaluation_case(
            "case-002",
            "Question two",
            document_id="00000000-0000-4000-8000-000000000007",
        ),
    )

    assigned = generate_splits(cases, SplitPolicy(1.0, 0.0, 0.0))

    assert {case.split for case in assigned} == {"train"}
