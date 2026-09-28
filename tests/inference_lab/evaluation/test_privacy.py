"""Tests for evaluation local-only filesystem boundaries."""

import traceback
from pathlib import Path

import pytest

from inference_lab.evaluation.loader import load_dataset

from .fixtures import case_payload, write_dataset_fixture  # noqa: TID252


def test_loader_errors_do_not_repeat_sensitive_absolute_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "confidential-corpus-name" / "manifest.json"

    with pytest.raises(FileNotFoundError) as caught:
        load_dataset(manifest_path)

    assert str(manifest_path) not in str(caught.value)
    assert "confidential-corpus-name" not in str(caught.value)


def test_case_validation_errors_suppress_sensitive_payloads(tmp_path: Path) -> None:
    sensitive_question = "Confidential compound codename AURORA"
    invalid_case = case_payload()
    invalid_case["question"] = sensitive_question
    invalid_case["expected_answer"]["claims"][0]["evidence_ids"] = ("evidence-missing",)
    manifest_path = write_dataset_fixture(tmp_path, cases=[invalid_case])

    with pytest.raises(ValueError, match="Invalid evaluation case") as caught:
        load_dataset(manifest_path)

    rendered = "".join(traceback.format_exception(caught.value))
    assert sensitive_question not in rendered
