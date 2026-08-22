"""Tests for final benchmark acceptance behavior."""

from pathlib import Path

import pytest

from inference_lab.evaluation.coverage import validate_final_benchmark
from inference_lab.evaluation.loader import load_dataset


FIXTURE_MANIFEST = (
    Path(__file__).parents[3] / "benchmarks" / "evaluation" / "manifest.json"
)


def test_contract_fixture_is_not_accepted_as_final_benchmark() -> None:
    dataset = load_dataset(FIXTURE_MANIFEST)

    with pytest.raises(ValueError, match="coverage requirements failed"):
        validate_final_benchmark(dataset.cases)
