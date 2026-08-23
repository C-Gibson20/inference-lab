"""Tests for deterministic splitting and leakage prevention."""

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

from .fixtures import (
    CORPUS_VERSION,
    DOCUMENT_ID,
    case_payload,
)


def _evaluation_case(
    case_id: str,
    question: str,
    *,
    document_id: str = DOCUMENT_ID,
    split: str = "development",
    evidence_id: str = "evidence-primary",
    answer_description: str | None = None,
) -> EvaluationCase:
    payload = case_payload(case_id, split)
    payload["question"] = question
    payload["expected_answer"] = {
        "summary": answer_description or case_id,
        "claims": (
            {
                "key": "result",
                "description": answer_description or f"Expected result for {case_id}",
                "evidence_ids": (evidence_id,),
            },
        ),
    }
    suffix = document_id[-1]
    payload["evidence"][0].update(
        {
            "evidence_id": evidence_id,
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

    with pytest.raises(ValueError, match="finite"):
        SplitPolicy(True, 0.0, 0.0)

    with pytest.raises(ValueError, match="near_duplicate_threshold"):
        SplitPolicy(0.5, 0.25, 0.25, near_duplicate_threshold=False)


def test_stable_digest_and_jaccard_are_deterministic() -> None:
    assert stable_digest("value") == stable_digest("value")
    assert jaccard_similarity(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1 / 3


@pytest.mark.parametrize(
    "threshold",
    [True, math.nan, -0.1, 1.1],
)
def test_leakage_group_threshold_is_validated(threshold: object) -> None:
    case = _evaluation_case("case-001", "What was measured?")

    with pytest.raises(ValueError, match="near_duplicate_threshold"):
        build_leakage_groups(
            (case,),
            near_duplicate_threshold=threshold,  # type: ignore[arg-type]
        )


def test_leakage_group_cases_must_be_present_and_unique() -> None:
    case = _evaluation_case("case-001", "What was measured?")

    with pytest.raises(ValueError, match="case IDs must be unique"):
        build_leakage_groups((case, case))

    with pytest.raises(ValueError, match="At least one case"):
        build_leakage_groups(())


def test_leakage_group_output_is_independent_of_input_order() -> None:
    cases = (
        _evaluation_case("case-001", "alpha", document_id=DOCUMENT_ID),
        _evaluation_case(
            "case-002",
            "beta",
            document_id="00000000-0000-4000-8000-000000000006",
        ),
        _evaluation_case(
            "case-003",
            "gamma",
            document_id="00000000-0000-4000-8000-000000000007",
        ),
    )

    assert build_leakage_groups(cases) == build_leakage_groups(reversed(cases))


def test_shared_document_cases_form_one_leakage_group() -> None:
    cases = (
        _evaluation_case("case-001", "What was measured?"),
        _evaluation_case("case-002", "Which method was used?"),
    )

    groups = build_leakage_groups(cases)

    assert len(groups) == 1
    assert {case.case_id for case in groups[0].cases} == {"case-001", "case-002"}


def test_shared_answers_group_independently_of_case_local_evidence_ids() -> None:
    cases = (
        _evaluation_case(
            "case-001",
            "What was measured?",
            document_id="00000000-0000-4000-8000-000000000006",
            evidence_id="evidence-first",
            answer_description="The shared expected result",
        ),
        _evaluation_case(
            "case-002",
            "Which result was reported?",
            document_id="00000000-0000-4000-8000-000000000007",
            evidence_id="evidence-second",
            answer_description="The shared expected result",
        ),
    )

    groups = build_leakage_groups(cases)

    assert len(groups) == 1


def test_empty_abstention_answers_do_not_create_a_global_leakage_group() -> None:
    answerable = (
        _evaluation_case(
            "case-001",
            "Is result one available?",
            document_id="00000000-0000-4000-8000-000000000006",
            evidence_id="evidence-first",
        ),
        _evaluation_case(
            "case-002",
            "Is result two available?",
            document_id="00000000-0000-4000-8000-000000000007",
            evidence_id="evidence-second",
        ),
    )
    cases = []
    for case in answerable:
        payload = case.model_dump(mode="python")
        payload["expected_answer"] = {}
        payload["abstention"] = {
            "expected": True,
            "reason": "evidence_absent",
        }
        payload["evidence"][0]["role"] = "context"
        cases.append(EvaluationCase.model_validate(payload))

    groups = build_leakage_groups(cases)

    assert len(groups) == 2


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
