"""Tests for deterministic evaluation serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import load_dataset
from inference_lab.evaluation.serialisation import (
    atomic_write,
    evaluation_json_schema,
    serialise_cases,
    serialise_manifest,
    write_dataset,
    write_json_schema,
)

from .test_loader import CORPUS_ID, CORPUS_VERSION, _case


def _evaluation_case(case_id: str = "case-001") -> EvaluationCase:
    case = EvaluationCase.model_validate(_case(case_id))
    return case.model_copy(update={"tags": frozenset({"zeta", "alpha"})})


def _manifest(case_ids: tuple[str, ...] = ("case-001",)) -> DatasetManifest:
    return DatasetManifest(
        dataset_id="baseline-v1",
        dataset_version="1",
        description="Evaluation fixture",
        corpus_id=CORPUS_ID,
        corpus_version=CORPUS_VERSION,
        case_file="cases.jsonl",
        case_count=len(case_ids),
        splits={"development": case_ids, "held_out": ()},
        created_at="2026-08-22T00:00:00Z",
        updated_at="2026-08-22T00:00:00Z",
    )


def test_case_serialization_is_deterministic_and_preserves_order() -> None:
    first = _evaluation_case("case-001")
    second = _evaluation_case("case-002")

    serialized = serialise_cases((first, second))

    assert serialized == serialise_cases((first, second))
    assert serialized.endswith(b"\n")
    assert serialized.splitlines()[0].find(b'"case_id":"case-001"') >= 0
    assert b'["alpha","zeta"]' in serialized

    restored = tuple(
        EvaluationCase.model_validate(json.loads(line))
        for line in serialized.splitlines()
    )
    assert restored == (first, second)


def test_empty_case_serialization_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        serialise_cases(())


def test_manifest_serialization_round_trips() -> None:
    manifest = _manifest()

    restored = DatasetManifest.model_validate(json.loads(serialise_manifest(manifest)))

    assert restored == manifest


def test_write_dataset_calculates_checksum_and_can_be_loaded(
    tmp_path: Path,
) -> None:
    cases = (_evaluation_case(),)
    final_manifest = write_dataset(tmp_path, _manifest(), cases)

    case_bytes = (tmp_path / "cases.jsonl").read_bytes()
    expected_checksum = "sha256:" + hashlib.sha256(case_bytes).hexdigest()

    assert final_manifest.checksums == {"cases.jsonl": expected_checksum}
    loaded = load_dataset(tmp_path / "manifest.json")
    assert loaded.cases == cases


def test_write_dataset_rejects_case_path_escape(tmp_path: Path) -> None:
    manifest = _manifest().model_copy(update={"case_file": "../cases.jsonl"})

    with pytest.raises(ValueError, match="inside"):
        write_dataset(tmp_path, manifest, (_evaluation_case(),))


def test_atomic_write_cleans_up_failed_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "output.json"

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(
        "inference_lab.evaluation.serialisation.os.replace", fail_replace
    )

    with pytest.raises(OSError, match="replacement failed"):
        atomic_write(destination, b"content")

    assert not destination.exists()
    assert list(tmp_path.glob(".output.json.*.tmp")) == []


def test_schema_can_be_written_as_json(tmp_path: Path) -> None:
    path = tmp_path / "evaluation-schema.json"

    write_json_schema(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == evaluation_json_schema()
    assert payload["schema_version"] == "1.0.0"
