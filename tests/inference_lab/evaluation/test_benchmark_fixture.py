"""Tests for the deterministic benchmark fixture."""

from pathlib import Path

from inference_lab.evaluation import load_dataset


FIXTURE_MANIFEST = (
    Path(__file__).parents[3] / "benchmarks" / "evaluation" / "manifest.json"
)


def test_benchmark_fixture_loads_with_required_coverage() -> None:
    dataset = load_dataset(FIXTURE_MANIFEST)

    assert len(dataset.cases) == 10
    assert len(dataset.manifest.splits["held_out"]) == 2
    assert not dataset.manifest.evaluation_policy.held_out_tuning_allowed
    assert not dataset.manifest.evaluation_policy.held_out_model_selection_allowed
    assert dataset.manifest.evaluation_policy.held_out_membership_immutable
    assert {case.question_type for case in dataset.cases} >= {
        "lookup",
        "comparison",
        "thematic",
        "aggregation",
        "follow_up",
        "difficult_terminology",
    }
    assert any(case.abstention.expected for case in dataset.cases)


def test_benchmark_fixture_has_numeric_and_date_expectations() -> None:
    dataset = load_dataset(FIXTURE_MANIFEST)

    assert any(case.expected_answer.numeric_values for case in dataset.cases)
    assert any(case.expected_answer.date_values for case in dataset.cases)
