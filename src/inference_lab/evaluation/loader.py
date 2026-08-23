"""Load and validate evaluation datasets."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
from io import StringIO
import json

from pydantic import ValidationError

from .datasets import DatasetManifest
from .cases import EvaluationCase
from .privacy import validate_local_path
from .versioning import (
    validate_case_payload_version,
    validate_manifest_payload_version,
)


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A validated evaluation manifest and ordered cases"""

    manifest: DatasetManifest
    cases: tuple[EvaluationCase, ...]


def _resolve_case_file(manifest_path: Path, case_file: str) -> Path:
    """Resolve a manifest-owned case file without allowing traversal"""

    if Path(case_file).is_absolute():
        raise ValueError("Case file must be relative to the manifest")
    validate_local_path(case_file, field_name="case_file")

    dataset_root = manifest_path.parent.resolve()
    resolved_path = (dataset_root / case_file).resolve()

    try:
        resolved_path.relative_to(dataset_root)
    except ValueError:
        raise ValueError("Case file must remain inside the dataset directory") from None

    if not resolved_path.is_file():
        raise ValueError("Case file does not exist")

    return resolved_path


def _verify_checksum(case_bytes: bytes, expected: str | None) -> None:
    """Verify a case-file checksum when one is declared."""

    if expected is None:
        raise ValueError("Missing checksum for case file")

    digest = hashlib.sha256(case_bytes).hexdigest()
    actual = f"sha256:{digest}"

    if actual != expected:
        raise ValueError("Checksum mismatch for case file")


def _load_cases(case_bytes: bytes) -> tuple[EvaluationCase, ...]:
    """Load ordered evaluation cases from a JSONL file."""

    cases: list[EvaluationCase] = []
    try:
        case_text = case_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Case file must contain valid UTF-8") from None

    for line_number, line in enumerate(StringIO(case_text, newline=""), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at line {line_number}")

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSONL record at line {line_number}") from None

        try:
            case = EvaluationCase.model_validate(validate_case_payload_version(payload))
        except (ValidationError, ValueError):
            raise ValueError(f"Invalid evaluation case at line {line_number}") from None

        cases.append(case)

    return tuple(cases)


def _validate_dataset_consistency(
    manifest: DatasetManifest, cases: tuple[EvaluationCase, ...]
) -> None:
    """Validate relationships between the manifest and loaded cases."""

    if len(cases) != manifest.case_count:
        raise ValueError("Manifest case_count does not match the number of JSONL cases")

    case_ids = tuple(case.case_id for case in cases)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Evaluation case IDs must be unique")

    manifest_ids = {
        case_id
        for split_case_ids in manifest.splits.values()
        for case_id in split_case_ids
    }

    if set(case_ids) != manifest_ids:
        raise ValueError(
            "Manifest split membership must contain every loaded case exactly once"
        )

    expected_split_by_case = {
        case_id: split
        for split, split_case_ids in manifest.splits.items()
        for case_id in split_case_ids
    }

    for case in cases:
        if expected_split_by_case[case.case_id] != case.split:
            raise ValueError(
                f"Case split does not match manifest for case {case.case_id}"
            )

        for evidence in case.evidence:
            if evidence.corpus_version != manifest.corpus_version:
                raise ValueError(
                    f"Evidence corpus version does not match manifest for case {case.case_id}"
                )


def load_dataset(manifest_path: str | Path) -> LoadedDataset:
    """Load and validate one evaluation dataset."""

    manifest_path = validate_local_path(manifest_path, field_name="manifest_path")

    if not manifest_path.is_file():
        raise FileNotFoundError("Dataset manifest does not exist")

    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Unable to read dataset manifest") from None

    try:
        manifest = DatasetManifest.model_validate(
            validate_manifest_payload_version(manifest_payload)
        )
    except (ValidationError, ValueError):
        raise ValueError("Invalid dataset manifest") from None

    case_path = _resolve_case_file(manifest_path, manifest.case_file)
    try:
        case_bytes = case_path.read_bytes()
    except OSError:
        raise ValueError("Unable to read case file") from None
    _verify_checksum(case_bytes, manifest.checksums.get(manifest.case_file))
    cases = _load_cases(case_bytes)
    _validate_dataset_consistency(manifest, cases)

    return LoadedDataset(manifest=manifest, cases=cases)
