"""Tests for deterministic evaluation serialization."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import LoadedDataset, load_dataset
from inference_lab.evaluation.serialisation import (
    atomic_write,
    evaluation_json_schema,
    serialise_cases,
    serialise_manifest,
    write_dataset,
    write_json_schema,
)
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION

from .test_loader import CORPUS_ID, CORPUS_VERSION, _case


REPOSITORY_ROOT = Path(__file__).parents[3]
SCHEMA_GENERATOR = (
    REPOSITORY_ROOT
    / "scripts"
    / "inference_lab"
    / "evaluation"
    / "generate-evaluation-schema.py"
)


def _evaluation_case(case_id: str = "case-001") -> EvaluationCase:
    case = EvaluationCase.model_validate(_case(case_id))
    return EvaluationCase.model_validate(
        case.model_dump(mode="python") | {"tags": frozenset({"zeta", "alpha"})}
    )


def _manifest(case_ids: tuple[str, ...] = ("case-001",)) -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "dataset_id": "baseline-v1",
            "dataset_version": "1",
            "description": "Evaluation fixture",
            "corpus_id": CORPUS_ID,
            "corpus_version": CORPUS_VERSION,
            "case_file": "cases.jsonl",
            "case_count": len(case_ids),
            "splits": {"development": case_ids, "held_out": ()},
            "created_at": "2026-08-22T00:00:00Z",
            "updated_at": "2026-08-22T00:00:00Z",
        }
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
    destination = tmp_path / "dataset"
    final_manifest = write_dataset(destination, _manifest(), cases)

    case_bytes = (destination / "cases.jsonl").read_bytes()
    expected_checksum = "sha256:" + hashlib.sha256(case_bytes).hexdigest()

    assert final_manifest.checksums == {"cases.jsonl": expected_checksum}
    with pytest.raises(TypeError):
        final_manifest.checksums["cases.jsonl"] = "changed"  # type: ignore[index]
    loaded = load_dataset(destination / "manifest.json")
    assert loaded.cases == cases


def test_write_dataset_rejects_case_path_escape(tmp_path: Path) -> None:
    manifest = DatasetManifest.model_validate(
        _manifest().model_dump(mode="python") | {"case_file": "../cases.jsonl"}
    )

    with pytest.raises(ValueError, match="inside"):
        write_dataset(tmp_path / "dataset", manifest, (_evaluation_case(),))

    absolute = DatasetManifest.model_validate(
        _manifest().model_dump(mode="python") | {"case_file": "/tmp/cases.jsonl"}
    )
    with pytest.raises(ValueError, match="relative"):
        write_dataset(tmp_path / "dataset", absolute, (_evaluation_case(),))

    manifest_collision = DatasetManifest.model_validate(
        _manifest().model_dump(mode="python") | {"case_file": "manifest.json"}
    )
    with pytest.raises(ValueError, match="cannot be manifest.json"):
        write_dataset(
            tmp_path / "dataset",
            manifest_collision,
            (_evaluation_case(),),
        )


def test_write_dataset_rejects_remote_destination() -> None:
    with pytest.raises(ValueError, match="local filesystem path"):
        write_dataset(
            "https://example.test/dataset",
            _manifest(),
            (_evaluation_case(),),
        )


def test_write_dataset_never_overwrites_an_existing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    write_dataset(destination, _manifest(), (_evaluation_case(),))
    original_manifest = (destination / "manifest.json").read_bytes()
    original_cases = (destination / "cases.jsonl").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_dataset(destination, _manifest(), (_evaluation_case(),))

    assert (destination / "manifest.json").read_bytes() == original_manifest
    assert (destination / "cases.jsonl").read_bytes() == original_cases
    assert not tuple(tmp_path.glob(".dataset.*.staging"))


def test_write_dataset_cleans_staging_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dataset"
    call_count = 0
    real_atomic_write = atomic_write

    def fail_second_write(path: Path, content: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("manifest write failed")
        real_atomic_write(path, content)

    monkeypatch.setattr(
        "inference_lab.evaluation.serialisation.atomic_write",
        fail_second_write,
    )

    with pytest.raises(OSError, match="manifest write failed"):
        write_dataset(destination, _manifest(), (_evaluation_case(),))

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".dataset.*.staging"))


def test_write_dataset_cleans_staging_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dataset"
    real_replace = os.replace

    def fail_publication(source: Path, target: Path) -> None:
        if Path(target) == destination:
            raise OSError("publication failed")
        real_replace(source, target)

    monkeypatch.setattr(
        "inference_lab.evaluation.serialisation.os.replace",
        fail_publication,
    )

    with pytest.raises(OSError, match="publication failed"):
        write_dataset(destination, _manifest(), (_evaluation_case(),))

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".dataset.*.staging"))


def test_write_dataset_preserves_destination_created_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dataset"
    real_load_dataset = load_dataset

    def create_competing_destination(manifest_path: Path) -> LoadedDataset:
        loaded = real_load_dataset(manifest_path)
        destination.mkdir()
        (destination / "owner.txt").write_text("external", encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        "inference_lab.evaluation.serialisation.load_dataset",
        create_competing_destination,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_dataset(destination, _manifest(), (_evaluation_case(),))

    assert (destination / "owner.txt").read_text(encoding="utf-8") == "external"
    assert not tuple(tmp_path.glob(".dataset.*.staging"))


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
    assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert payload["$id"].endswith(CURRENT_SCHEMA_VERSION)
    assert payload["oneOf"] == [
        {"$ref": "#/$defs/DatasetManifest"},
        {"$ref": "#/$defs/EvaluationCase"},
    ]

    definitions = payload["$defs"]
    for model_name in ("DatasetManifest", "EvaluationCase"):
        model_schema = definitions[model_name]
        assert model_schema["properties"]["schema_version"]["const"] == (
            CURRENT_SCHEMA_VERSION
        )

    references: list[str] = []

    def collect_references(value: object) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str):
                references.append(reference)
            for nested in value.values():
                collect_references(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_references(nested)

    collect_references(payload)
    assert references
    assert all(
        reference.startswith("#/$defs/")
        and reference.removeprefix("#/$defs/") in definitions
        for reference in references
    )


def test_documented_schema_generator_writes_public_schema(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation-schema.json"

    result = subprocess.run(
        [sys.executable, str(SCHEMA_GENERATOR), str(destination)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        evaluation_json_schema()
    )
