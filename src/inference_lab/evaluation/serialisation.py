"""Deterministic serialization utilities for evaluation contracts."""

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import _validate_dataset_consistency
from inference_lab.evaluation.splitting import validate_no_leakage
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION


def _canonical_payload(model: BaseModel) -> dict[str, Any]:
    """Convert a model to a JSON-compatible canonical payload."""

    payload = model.model_dump(mode="json")

    if "tags" in payload:
        payload["tags"] = sorted(payload["tags"])

    return payload


def _json_bytes(payload: dict[str, Any], *, indent: int | None) -> bytes:
    """Encode a payload using deterministic UTF-8 JSON."""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
    )
    return f"{text}\n".encode("utf-8")


def serialise_cases(cases: Iterable[EvaluationCase]) -> bytes:
    """Serialize cases as deterministic JSONL in the supplied order."""

    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise ValueError("Evaluation datasets must contain at least one case")

    lines = [
        json.dumps(
            _canonical_payload(case),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in ordered_cases
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialise_manifest(manifest: DatasetManifest) -> bytes:
    """Serialize a manifest as human-reviewable deterministic JSON."""

    return _json_bytes(_canonical_payload(manifest), indent=2)


def atomic_write(path: Path, content: bytes) -> None:
    """Write bytes atomically within the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _output_case_path(destination: Path, case_file: str) -> Path:
    """Resolve a case output path below the dataset destination."""

    if Path(case_file).is_absolute():
        raise ValueError("case_file must be relative to the dataset destination")

    root = destination.resolve()
    resolved = (root / case_file).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "case_file must remain inside the dataset destination"
        ) from error

    manifest_path = root / "manifest.json"
    if resolved == manifest_path:
        raise ValueError("case_file cannot be manifest.json")

    return resolved


def write_dataset(
    destination: Path,
    manifest: DatasetManifest,
    cases: tuple[EvaluationCase, ...],
) -> DatasetManifest:
    """Write a checksummed dataset and return its final manifest."""

    ordered_cases = tuple(cases)
    _validate_dataset_consistency(manifest, ordered_cases)
    validate_no_leakage(ordered_cases)

    destination = destination.resolve()
    case_path = _output_case_path(destination, manifest.case_file)
    case_bytes = serialise_cases(ordered_cases)
    checksum = f"sha256:{hashlib.sha256(case_bytes).hexdigest()}"

    checksums = dict(manifest.checksums)
    checksums[manifest.case_file] = checksum
    final_manifest = manifest.model_copy(update={"checksums": checksums})

    atomic_write(case_path, case_bytes)
    atomic_write(destination / "manifest.json", serialise_manifest(final_manifest))
    return final_manifest


def evaluation_json_schema() -> dict[str, object]:
    """Return the published evaluation contract schema."""

    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "manifest": DatasetManifest.model_json_schema(),
        "case": EvaluationCase.model_json_schema(),
    }


def write_json_schema(path: Path) -> None:
    """Write the published evaluation contract schema atomically."""

    atomic_write(path, _json_bytes(evaluation_json_schema(), indent=2))
