"""Tests for final benchmark acceptance behavior."""

import subprocess
import sys
from pathlib import Path

import pytest

from inference_lab.evaluation import (
    DatasetManifest,
    load_dataset,
    validate_final_benchmark,
    write_dataset,
)

from .fixtures import CORPUS_ID, CORPUS_VERSION, benchmark_cases  # noqa: TID252

REPOSITORY_ROOT = Path(__file__).parents[3]
FIXTURE_MANIFEST = REPOSITORY_ROOT / "benchmarks" / "evaluation" / "manifest.json"
VALIDATOR = (
    REPOSITORY_ROOT
    / "scripts"
    / "inference_lab"
    / "evaluation"
    / "validate-evaluation-benchmark.py"
)


def test_contract_fixture_is_not_accepted_as_final_benchmark() -> None:
    dataset = load_dataset(FIXTURE_MANIFEST)

    with pytest.raises(ValueError, match="coverage requirements failed"):
        validate_final_benchmark(dataset.cases)


def test_documented_validator_rejects_contract_fixture_without_traceback() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(FIXTURE_MANIFEST)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("benchmark validation failed:")
    assert "at least 30 cases required" in result.stderr
    assert "Traceback" not in result.stderr


def test_documented_validator_accepts_complete_benchmark(tmp_path: Path) -> None:
    cases = benchmark_cases()
    manifest = DatasetManifest.model_validate(
        {
            "dataset_id": "accepted-benchmark",
            "dataset_version": "1",
            "description": "Complete final benchmark fixture",
            "corpus_id": CORPUS_ID,
            "corpus_version": CORPUS_VERSION,
            "case_file": "cases.jsonl",
            "case_count": len(cases),
            "splits": {
                "development": tuple(
                    case.case_id for case in cases if case.split == "development"
                ),
                "held_out": tuple(
                    case.case_id for case in cases if case.split == "held_out"
                ),
            },
            "created_at": "2026-08-23T00:00:00Z",
            "updated_at": "2026-08-23T00:00:00Z",
        }
    )
    destination = tmp_path / "accepted-benchmark"
    write_dataset(destination, manifest, cases)

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(destination / "manifest.json")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "validated 30 cases: 24 answerable, 6 abstention, "
        "6 held-out (5 answerable, 1 abstention)\n"
    )
    assert result.stderr == ""
