"""Acceptance tests for authored, published, and loaded evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from inference_lab.evaluation import (
    DatasetManifest,
    EvaluationCase,
    LoadedDataset,
    evaluation_json_schema,
    load_dataset,
    validate_final_benchmark,
    write_dataset,
)
from tests.inference_lab.evaluation.fixtures import (
    CORPUS_ID,
    CORPUS_VERSION,
    benchmark_cases,
    case_payload,
)


def _manifest(cases: tuple[EvaluationCase, ...]) -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "dataset_id": "dataset-acceptance",
            "dataset_version": "1.0.0",
            "description": "Evaluation dataset acceptance fixture",
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
            "created_at": "2026-08-22T00:00:00Z",
            "updated_at": "2026-08-22T00:00:00Z",
        }
    )


def test_benchmark_publication_load_and_held_out_gate_round_trip(
    tmp_path: Path,
) -> None:
    cases = benchmark_cases()
    manifest = _manifest(cases)
    destination = tmp_path / "dataset"
    published = write_dataset(destination, manifest, cases)
    case_bytes = (destination / "cases.jsonl").read_bytes()

    assert len(cases) == 30
    assert published.checksums["cases.jsonl"] == (
        "sha256:" + hashlib.sha256(case_bytes).hexdigest()
    )
    loaded = load_dataset(destination / "manifest.json")
    assert loaded.cases == cases
    assert validate_final_benchmark(loaded.cases).total_cases == 30
    with pytest.raises(FileExistsError):
        write_dataset(destination, manifest, cases)


def test_dataset_schema_and_mutable_annotations_are_checked() -> None:
    schema = evaluation_json_schema()
    validator = Draft202012Validator(schema)
    # JSON Schema validates JSON values; the builder uses tuples to make the
    # Python-side fixture convenient, so round-trip it through JSON first.
    payload = json.loads(json.dumps(case_payload()))
    validator.validate(payload)

    invalid = dict(payload, question="")
    assert list(validator.iter_errors(invalid))
    for missing_value in (
        dict(payload, schema_version="99.0.0"),
        dict(payload, evidence=[]),
    ):
        with pytest.raises(ValueError):
            EvaluationCase.model_validate(missing_value)


def test_dataset_loader_fails_closed_on_checksum_split_and_local_path_errors(
    tmp_path: Path,
) -> None:
    case = EvaluationCase.model_validate(case_payload())
    manifest_path = tmp_path / "dataset" / "manifest.json"
    destination = manifest_path.parent
    write_dataset(destination, _manifest((case,)), (case,))

    case_path = destination / "cases.jsonl"
    case_path.write_text(case_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_dataset(manifest_path)

    with pytest.raises(ValueError, match="local filesystem path"):
        load_dataset("https://example.test/manifest.json")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["case_file"] = "../outside.jsonl"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid dataset manifest"):
        load_dataset(manifest_path)


def test_publication_rejects_cross_split_evidence_leakage_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = EvaluationCase.model_validate(case_payload("case-001"))
    second_payload = case_payload("case-002", split="held_out")
    second_payload["evidence"][0]["source_span_id"] = first.evidence[0].source_span_id
    second = type(first).model_validate(second_payload)
    cases = (first, second)
    manifest = _manifest(cases)
    with pytest.raises(ValueError, match="Leakage group crosses splits"):
        write_dataset(tmp_path / "leaked", manifest, cases)
    assert not (tmp_path / "leaked").exists()

    destination = tmp_path / "failed"

    def fail_write(_path: Path, _content: bytes) -> None:
        raise OSError("private path details")

    monkeypatch.setattr(
        "inference_lab.evaluation.serialisation.atomic_write",
        fail_write,
    )
    with pytest.raises(OSError, match="private path details"):
        write_dataset(destination, _manifest((first,)), (first,))
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".failed.*.staging"))


def test_competing_publication_is_preserved_and_staging_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = EvaluationCase.model_validate(case_payload())
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
        write_dataset(destination, _manifest((case,)), (case,))
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "external"
    assert not tuple(tmp_path.glob(".dataset.*.staging"))


def test_published_dataset_schema_is_json_serialisable() -> None:
    encoded = json.dumps(evaluation_json_schema(), sort_keys=True)
    assert '"$schema": "https://json-schema.org/draft/2020-12/schema"' in encoded
    assert "EvaluationCase" in encoded
    assert "DatasetManifest" in encoded
