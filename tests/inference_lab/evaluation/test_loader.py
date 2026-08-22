"""Tests for evaluation dataset loading and consistency validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from inference_lab.evaluation.loader import load_dataset


CORPUS_ID = "00000000-0000-4000-8000-000000000001"
CORPUS_VERSION = "00000000-0000-4000-8000-000000000002"
DOCUMENT_ID = "00000000-0000-4000-8000-000000000003"
DOCUMENT_VERSION_ID = "00000000-0000-4000-8000-000000000004"
SOURCE_SPAN_ID = "00000000-0000-4000-8000-000000000005"


def _case(case_id: str = "case-001", split: str = "development") -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "case_id": case_id,
        "question": "What assay was used?",
        "question_type": "lookup",
        "expected_answer": {"claims": ({"key": "assay"},)},
        "evidence": [
            {
                "kind": "source_span",
                "role": "authoritative",
                "corpus_version": CORPUS_VERSION,
                "document_id": DOCUMENT_ID,
                "document_version_id": DOCUMENT_VERSION_ID,
                "source_span_id": SOURCE_SPAN_ID,
            }
        ],
        "abstention": {"expected": False, "reason": "not_applicable"},
        "split": split,
    }


def _write_dataset(
    directory: Path,
    cases: list[dict[str, Any]] | None = None,
    *,
    manifest_overrides: dict[str, Any] | None = None,
    case_text: str | None = None,
) -> Path:
    case_path = directory / "cases.jsonl"
    case_records = cases if cases is not None else [_case()]
    contents = case_text
    if contents is None:
        contents = "".join(
            f"{json.dumps(case, sort_keys=True)}\n" for case in case_records
        )
    case_path.write_text(contents, encoding="utf-8", newline="")

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": "baseline-v1",
        "dataset_version": "1",
        "description": "Evaluation fixture",
        "corpus_id": CORPUS_ID,
        "corpus_version": CORPUS_VERSION,
        "case_file": "cases.jsonl",
        "case_count": len(case_records),
        "splits": {
            "development": [case["case_id"] for case in case_records],
            "held_out": [],
        },
        "checksums": {
            "cases.jsonl": (
                "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
            )
        },
        "created_at": "2026-08-22T00:00:00Z",
        "updated_at": "2026-08-22T00:00:00Z",
    }
    manifest.update(manifest_overrides or {})
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_load_dataset_returns_ordered_validated_dataset(tmp_path: Path) -> None:
    manifest_path = _write_dataset(
        tmp_path,
        cases=[_case("case-001"), _case("case-002")],
        manifest_overrides={
            "splits": {
                "development": ["case-001", "case-002"],
                "held_out": [],
            }
        },
    )

    loaded = load_dataset(manifest_path)

    assert [case.case_id for case in loaded.cases] == ["case-001", "case-002"]
    assert loaded.manifest.dataset_id == "baseline-v1"


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing-manifest.json")


def test_malformed_manifest_json_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read dataset manifest"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    ("case_file", "message"),
    [
        ("/tmp/cases.jsonl", "relative"),
        ("../cases.jsonl", "inside"),
        ("https://example.test/cases.jsonl", "local"),
    ],
)
def test_case_file_must_be_a_safe_relative_path(
    tmp_path: Path, case_file: str, message: str
) -> None:
    manifest_path = _write_dataset(
        tmp_path, manifest_overrides={"case_file": case_file}
    )

    with pytest.raises(ValueError, match=message):
        load_dataset(manifest_path)


def test_missing_case_file_is_rejected(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)
    (tmp_path / "cases.jsonl").unlink()

    with pytest.raises(ValueError, match="does not exist"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    "checksums",
    [
        {},
        {"cases.jsonl": "sha256:not-a-digest"},
        {"cases.jsonl": "sha256:" + "0" * 64},
    ],
)
def test_checksum_failures_are_rejected(
    tmp_path: Path, checksums: dict[str, str]
) -> None:
    manifest_path = _write_dataset(
        tmp_path, manifest_overrides={"checksums": checksums}
    )

    with pytest.raises(ValueError, match="checksum|Checksum"):
        load_dataset(manifest_path)


def test_malformed_jsonl_is_rejected(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, case_text="not-json\n")
    case_path = tmp_path / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["cases.jsonl"] = (
        "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        load_dataset(manifest_path)


def test_blank_jsonl_record_is_rejected(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, case_text="\n")
    case_path = tmp_path / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["cases.jsonl"] = (
        "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Blank JSONL record"):
        load_dataset(manifest_path)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    manifest_path = _write_dataset(
        tmp_path,
        cases=[_case("case-001"), _case("case-001")],
        manifest_overrides={
            "splits": {
                "development": ["case-001", "case-002"],
                "held_out": [],
            }
        },
    )

    with pytest.raises(ValueError, match="case IDs must be unique"):
        load_dataset(manifest_path)


def test_case_count_must_match_loaded_cases(tmp_path: Path) -> None:
    manifest_path = _write_dataset(
        tmp_path,
        manifest_overrides={
            "case_count": 2,
            "splits": {
                "development": ["case-001", "case-002"],
                "held_out": [],
            },
        },
    )

    with pytest.raises(ValueError, match="case_count"):
        load_dataset(manifest_path)


def test_manifest_must_contain_every_loaded_case(tmp_path: Path) -> None:
    manifest_path = _write_dataset(
        tmp_path,
        cases=[_case("case-001"), _case("case-002")],
        manifest_overrides={
            "splits": {
                "development": ["case-001", "case-003"],
                "held_out": [],
            }
        },
    )

    with pytest.raises(ValueError, match="every loaded case"):
        load_dataset(manifest_path)


def test_case_split_must_match_manifest(tmp_path: Path) -> None:
    manifest_path = _write_dataset(
        tmp_path,
        manifest_overrides={"splits": {"held_out": ["case-001"]}},
    )

    with pytest.raises(ValueError, match="Case split"):
        load_dataset(manifest_path)


def test_evidence_corpus_version_must_match_manifest(tmp_path: Path) -> None:
    invalid_case = _case()
    invalid_case["evidence"][0]["corpus_version"] = (
        "00000000-0000-4000-8000-000000000006"
    )
    manifest_path = _write_dataset(tmp_path, cases=[invalid_case])

    with pytest.raises(ValueError, match="corpus version"):
        load_dataset(manifest_path)
