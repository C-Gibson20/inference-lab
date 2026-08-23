"""Tests for evaluation local-only filesystem boundaries."""

import traceback
from pathlib import Path

import pytest

from inference_lab.evaluation.loader import load_dataset
from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.serialisation import atomic_write, write_json_schema

from .fixtures import case_payload, write_dataset_fixture  # noqa: TID252


@pytest.mark.parametrize(
    "path",
    [
        "https://example.test/dataset.json",
        Path("https://example.test/dataset.json"),
        "s3://private-bucket/dataset.json",
        "file:/tmp/dataset.json",
        "mailto:dataset@example.test",
        "//remote-host/share/dataset.json",
        r"\\remote-host\share\dataset.json",
    ],
)
def test_remote_shaped_paths_are_rejected_after_path_coercion(
    path: str | Path,
) -> None:
    with pytest.raises(ValueError, match="local filesystem path"):
        validate_local_path(path)


@pytest.mark.parametrize("path", ["", "   ", "invalid\x00path"])
def test_empty_or_invalid_local_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        validate_local_path(path)


def test_wrong_path_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="string or Path"):
        validate_local_path(42)  # type: ignore[arg-type]


def test_relative_local_paths_are_resolved(tmp_path: Path) -> None:
    path = validate_local_path(tmp_path / "dataset" / "manifest.json")

    assert path.is_absolute()
    assert path == tmp_path / "dataset" / "manifest.json"


@pytest.mark.parametrize(
    "operation",
    [
        lambda: load_dataset("https://example.test/manifest.json"),
        lambda: atomic_write("s3://bucket/output.json", b"content"),
        lambda: write_json_schema("file:/tmp/evaluation-schema.json"),
    ],
)
def test_all_evaluation_io_entry_points_enforce_local_paths(operation) -> None:
    with pytest.raises(ValueError, match="local filesystem path"):
        operation()


def test_atomic_write_requires_bytes(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="content must be bytes"):
        atomic_write(tmp_path / "output.json", "content")  # type: ignore[arg-type]


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
