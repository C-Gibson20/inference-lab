"""Tests for evaluation dataset loading and consistency validation."""

import hashlib
import json
from pathlib import Path

import pytest

from inference_lab.evaluation.loader import load_dataset

from .fixtures import case_payload, write_dataset_fixture


def test_load_dataset_returns_ordered_validated_dataset(tmp_path: Path) -> None:
    manifest_path = write_dataset_fixture(
        tmp_path,
        cases=[case_payload("case-001"), case_payload("case-002")],
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
    manifest_path = write_dataset_fixture(
        tmp_path,
        manifest_overrides={
            "case_file": case_file,
            "checksums": {},
        },
    )

    with pytest.raises(ValueError, match=message):
        load_dataset(manifest_path)


def test_missing_case_file_is_rejected(tmp_path: Path) -> None:
    manifest_path = write_dataset_fixture(tmp_path)
    (tmp_path / "cases.jsonl").unlink()

    with pytest.raises(ValueError, match="does not exist"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    ("checksums", "message"),
    [
        ({}, "checksum|Checksum"),
        ({"cases.jsonl": "sha256:not-a-digest"}, "manifest"),
        ({"cases.jsonl": "sha256:" + "0" * 64}, "checksum|Checksum"),
    ],
)
def test_checksum_failures_are_rejected(
    tmp_path: Path,
    checksums: dict[str, str],
    message: str,
) -> None:
    manifest_path = write_dataset_fixture(
        tmp_path, manifest_overrides={"checksums": checksums}
    )

    with pytest.raises(ValueError, match=message):
        load_dataset(manifest_path)


def test_malformed_jsonl_is_rejected(tmp_path: Path) -> None:
    manifest_path = write_dataset_fixture(tmp_path, case_text="not-json\n")
    case_path = tmp_path / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["cases.jsonl"] = (
        "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        load_dataset(manifest_path)


def test_blank_jsonl_record_is_rejected(tmp_path: Path) -> None:
    manifest_path = write_dataset_fixture(tmp_path, case_text="\n")
    case_path = tmp_path / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["cases.jsonl"] = (
        "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Blank JSONL record"):
        load_dataset(manifest_path)


def test_case_file_must_be_valid_utf8(tmp_path: Path) -> None:
    manifest_path = write_dataset_fixture(tmp_path)
    case_path = tmp_path / "cases.jsonl"
    case_path.write_bytes(b"\xff")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["cases.jsonl"] = (
        "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="valid UTF-8"):
        load_dataset(manifest_path)


def test_checksum_and_parsing_use_one_case_file_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = write_dataset_fixture(tmp_path)
    case_path = tmp_path / "cases.jsonl"
    original_read_bytes = Path.read_bytes
    case_reads = 0

    def mutate_after_read(path: Path) -> bytes:
        nonlocal case_reads
        content = original_read_bytes(path)
        if path == case_path:
            case_reads += 1
            path.write_bytes(b"changed after verified read")
        return content

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)

    loaded = load_dataset(manifest_path)

    assert len(loaded.cases) == 1
    assert case_reads == 1


def test_case_file_read_failures_are_privacy_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = write_dataset_fixture(tmp_path)

    def fail_read(_path: Path) -> bytes:
        raise OSError("/sensitive/path/cases.jsonl")

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    with pytest.raises(ValueError, match="Unable to read case file") as caught:
        load_dataset(manifest_path)

    assert "/sensitive/path" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    manifest_path = write_dataset_fixture(
        tmp_path,
        cases=[case_payload("case-001"), case_payload("case-001")],
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
    manifest_path = write_dataset_fixture(
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
    manifest_path = write_dataset_fixture(
        tmp_path,
        cases=[case_payload("case-001"), case_payload("case-002")],
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
    manifest_path = write_dataset_fixture(
        tmp_path,
        manifest_overrides={"splits": {"held_out": ["case-001"]}},
    )

    with pytest.raises(ValueError, match="Case split"):
        load_dataset(manifest_path)


def test_evidence_corpus_version_must_match_manifest(tmp_path: Path) -> None:
    invalid_case = case_payload()
    invalid_case["evidence"][0]["corpus_version"] = (
        "00000000-0000-4000-8000-000000000006"
    )
    manifest_path = write_dataset_fixture(tmp_path, cases=[invalid_case])

    with pytest.raises(ValueError, match="corpus version"):
        load_dataset(manifest_path)
