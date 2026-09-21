"""Tests for bounded repeat-run comparison."""

from pathlib import Path

from inference_lab.evaluation import (
    RetrievalComparisonCategory,
    RetrievalEvaluationRequest,
    RetrievalEvaluationRunResult,
    compare_retrieval_evaluations,
    run_retrieval_evaluation,
)

from .test_retrieval_execution import (  # noqa: TID252
    FakeAdapter,
    request_for,
    system_identity,
)


def run_fixture(
    tmp_path: Path,
    *,
    first_behavior: str = "completed",
    identity_seed: str = "stable",
) -> RetrievalEvaluationRunResult:
    """Build one reproducible run with controlled semantic changes."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    adapter = FakeAdapter(
        system_identity(identity_seed),
        {"case-001": first_behavior, "case-002": "empty"},
    )
    request: RetrievalEvaluationRequest = request_for(
        tmp_path,
        adapter,
        case_ids=("case-001", "case-002"),
    )
    return run_retrieval_evaluation(request)


def test_identical_runs_compare_equal_across_different_output_directories(
    tmp_path: Path,
) -> None:
    """Output location and execution time are absent from semantic comparison."""
    left = run_fixture(tmp_path / "left")
    right = run_fixture(tmp_path / "right")

    comparison = compare_retrieval_evaluations(left, right)

    assert comparison.equivalent is True
    assert comparison.mismatch_categories == ()
    assert comparison.left_run_fingerprint == comparison.right_run_fingerprint
    assert comparison.left_result_digest == comparison.right_result_digest
    assert comparison.as_dict()["mismatch_categories"] == []


def test_case_and_aggregate_changes_are_bounded_mismatches(
    tmp_path: Path,
) -> None:
    """Changed ranked outcomes fail comparison without exposing payloads."""
    left = run_fixture(tmp_path / "left")
    right = run_fixture(tmp_path / "right", first_behavior="empty")

    comparison = compare_retrieval_evaluations(left, right)

    assert comparison.equivalent is False
    assert comparison.mismatch_categories == (
        RetrievalComparisonCategory.CASE_RESULTS,
        RetrievalComparisonCategory.AGGREGATE,
        RetrievalComparisonCategory.RESULT_DIGEST,
    )
    assert "candidate content sentinel" not in repr(comparison)
    assert "question and candidate content sentinel" not in repr(comparison)


def test_identity_changes_are_reported_separately(
    tmp_path: Path,
) -> None:
    """Changed system identity is classified without dumping semantic payloads."""
    left = run_fixture(tmp_path / "left")
    right = run_fixture(tmp_path / "right", identity_seed="changed")

    comparison = compare_retrieval_evaluations(left, right)

    assert comparison.equivalent is False
    assert comparison.mismatch_categories == (
        RetrievalComparisonCategory.RUN_IDENTITY,
        RetrievalComparisonCategory.CASE_RESULTS,
        RetrievalComparisonCategory.RESULT_DIGEST,
    )
