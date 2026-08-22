"""Load and validate evaluation datasets."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from pydantic import ValidationError

from .datasets import DatasetManifest
from .cases import EvaluationCase
from .utils import is_sha256_pattern
from .versioning import migrate_case_payload, migrate_manifest_payload


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A validated evaluation manifest and ordered cases"""

    manifest: DatasetManifest
    cases: tuple[EvaluationCase, ...]


def _resolve_case_file(manifest_path: Path, case_file: str) -> Path:
    """Resolve a manifest-owned case file without allowing traversal"""

    if Path(case_file).is_absolute():
        raise ValueError(f"Case file {case_file} must be relative to the manifest")

    dataset_root = manifest_path.parent.resolve()
    resolved_path = (dataset_root / case_file).resolve()

    try:
        resolved_path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            f"Case file {case_file} must remain inside the dataset directory"
        ) from error

    if not resolved_path.is_file():
        raise ValueError(f"Case file {case_file} does not exist")

    return resolved_path


def _verify_checksum(case_path: Path, expected: str | None) -> None:
    """Verify a case-file checksum when one is declared."""

    if expected is None:
        raise ValueError(f"Missing checksum for case file: {case_path.name}")

    if not is_sha256_pattern(expected):
        raise ValueError(
            f"Checksum for case file {case_path.name} must use SHA256 format (sha256:<64 hex digits>)"
        )

    digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    actual = f"sha256:{digest}"

    if actual != expected:
        raise ValueError(f"Checksum mismatch for case file {case_path.name}")


def _load_cases(case_path: Path) -> tuple[EvaluationCase, ...]:
    """Load ordered evaluation cases from a JSONL file."""

    cases: list[EvaluationCase] = []

    with case_path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank JSONL record at {case_path.name}:{line_number}"
                )

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL record at {case_path.name}:{line_number}"
                ) from error

            try:
                case = EvaluationCase.model_validate(migrate_case_payload(payload))
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    f"Invalid evaluation case at {case_path.name}:{line_number}"
                ) from error

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


def load_dataset(manifest_path: Path) -> LoadedDataset:
    """Load and validate one evaluation dataset."""

    manifest_path = manifest_path.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read dataset manifest: {manifest_path}") from error

    try:
        manifest = DatasetManifest.model_validate(
            migrate_manifest_payload(manifest_payload)
        )
    except (ValidationError, ValueError) as error:
        raise ValueError(f"Invalid dataset manifest: {manifest_path}") from error

    case_path = _resolve_case_file(manifest_path, manifest.case_file)
    _verify_checksum(case_path, manifest.checksums.get(manifest.case_file))
    cases = _load_cases(case_path)
    _validate_dataset_consistency(manifest, cases)

    return LoadedDataset(manifest=manifest, cases=cases)
