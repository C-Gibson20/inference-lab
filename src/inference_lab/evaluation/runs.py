"""Immutable, reproducible identities for evaluation runs."""

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.loader import LoadedDataset
from inference_lab.evaluation.utils import DatasetId, SplitName, is_sha256_pattern
from inference_lab.provenance import CorpusId, CorpusVersionId


EvaluationPurpose = Literal["development", "final", "reporting"]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
_RUN_ID_PATTERN = re.compile(r"^run-[0-9a-f]{64}$")


class EvaluationRun(BaseModel):
    """Immutable identity and inputs for one evaluation run.

    The record contains provenance and control-plane metadata only. It must
    never contain questions, expected answers, prompts, model outputs, or
    evidence text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^run-[0-9a-f]{64}$")
    dataset_id: DatasetId
    dataset_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    corpus_id: str
    corpus_version: str
    split: SplitName
    purpose: EvaluationPurpose
    tuning: bool = False
    model_selection: bool = False
    system_version: str = Field(min_length=1)
    configuration_digest: Digest
    algorithm_versions: dict[str, str]
    dataset_manifest_digest: Digest
    case_file_digest: Digest
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_identity_metadata(self) -> "EvaluationRun":
        """Validate typed provenance IDs and internal invariants."""

        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must use the format run-<64 lowercase hex digits>")
        CorpusId.from_str(self.corpus_id)
        CorpusVersionId.from_str(self.corpus_version)
        if not self.algorithm_versions:
            raise ValueError("algorithm_versions must not be empty")
        if any(
            not name.strip() or not version.strip()
            for name, version in self.algorithm_versions.items()
        ):
            raise ValueError("algorithm_versions names and values must be non-empty")
        _validate_purpose(self.split, self.purpose, self.tuning, self.model_selection)
        expected_run_id = compute_run_id(
            run_identity_payload(
                dataset_id=self.dataset_id,
                dataset_version=self.dataset_version,
                schema_version=self.schema_version,
                corpus_id=self.corpus_id,
                corpus_version=self.corpus_version,
                split=self.split,
                purpose=self.purpose,
                tuning=self.tuning,
                model_selection=self.model_selection,
                system_version=self.system_version,
                configuration_digest=self.configuration_digest,
                algorithm_versions=self.algorithm_versions,
                dataset_manifest_digest=self.dataset_manifest_digest,
                case_file_digest=self.case_file_digest,
            )
        )
        if self.run_id != expected_run_id:
            raise ValueError("run_id does not match the run's semantic inputs")
        return self


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode a JSON object deterministically, rejecting unsafe values."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Configuration and run identity inputs must be finite JSON values"
        ) from error


def run_identity_payload(
    *,
    dataset_id: str,
    dataset_version: str,
    schema_version: str,
    corpus_id: str,
    corpus_version: str,
    split: SplitName,
    purpose: EvaluationPurpose,
    tuning: bool,
    model_selection: bool,
    system_version: str,
    configuration_digest: str,
    algorithm_versions: Mapping[str, str],
    dataset_manifest_digest: str,
    case_file_digest: str,
) -> dict[str, object]:
    """Return the semantic, privacy-safe payload used to derive a run ID."""
    return {
        "algorithm_versions": dict(sorted(algorithm_versions.items())),
        "case_file_digest": case_file_digest,
        "configuration_digest": configuration_digest,
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "dataset_id": dataset_id,
        "dataset_manifest_digest": dataset_manifest_digest,
        "dataset_version": dataset_version,
        "model_selection": model_selection,
        "purpose": purpose,
        "schema_version": schema_version,
        "split": split,
        "system_version": system_version,
        "tuning": tuning,
    }


def compute_run_id(payload: Mapping[str, object]) -> str:
    """Compute a stable identifier from canonical run inputs."""
    return f"run-{hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()}"


class EvaluationRunRequest(BaseModel):
    """Requested scope and purpose for an evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: SplitName
    purpose: EvaluationPurpose
    tuning: bool = False
    model_selection: bool = False

    @model_validator(mode="after")
    def validate_purpose(self) -> "EvaluationRunRequest":
        """Reject invalid purpose/split combinations at the API boundary."""
        _validate_purpose(self.split, self.purpose, self.tuning, self.model_selection)
        return self


def _validate_purpose(
    split: SplitName,
    purpose: EvaluationPurpose,
    tuning: bool,
    model_selection: bool,
) -> None:
    """Enforce governance rules independent of a dataset."""

    if split == "held_out" and (tuning or model_selection):
        raise ValueError("held-out split cannot be used for tuning or model selection")
    if purpose == "final" and split != "held_out":
        raise ValueError("final evaluation must use the held-out split")


def validate_run_purpose(
    request: EvaluationRunRequest,
    *,
    held_out_tuning_allowed: bool = False,
    held_out_model_selection_allowed: bool = False,
) -> None:
    """Enforce a request against the dataset's held-out policy."""

    if request.split != "held_out":
        return
    if request.tuning and not held_out_tuning_allowed:
        raise ValueError("held-out split cannot be used for tuning")
    if request.model_selection and not held_out_model_selection_allowed:
        raise ValueError("held-out split cannot be used for model selection")


def validate_run_inputs(
    dataset: LoadedDataset,
    request: EvaluationRunRequest,
) -> tuple[EvaluationCase, ...]:
    """Validate and return cases selected for a run in file order."""

    if request.split not in dataset.manifest.splits:
        raise ValueError(f"Unknown evaluation split: {request.split}")
    selected_ids = set(dataset.manifest.splits[request.split])
    selected_cases = tuple(
        case for case in dataset.cases if case.case_id in selected_ids
    )
    if {case.case_id for case in selected_cases} != selected_ids:
        raise ValueError("Manifest split membership does not match case data")
    return selected_cases


def configuration_digest(configuration: Mapping[str, object]) -> Digest:
    """Hash a finite JSON-compatible configuration deterministically."""
    return f"sha256:{hashlib.sha256(_canonical_json_bytes(configuration)).hexdigest()}"


def _manifest_digest(dataset: LoadedDataset) -> Digest:
    """Digest the canonical semantic manifest associated with a loaded dataset."""
    payload = dataset.manifest.model_dump(mode="json")
    return f"sha256:{hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()}"


def create_evaluation_run(
    dataset: LoadedDataset,
    request: EvaluationRunRequest,
    *,
    system_version: str,
    configuration_digest: str,
    algorithm_versions: Mapping[str, str],
    created_at: AwareDatetime | datetime,
    dataset_manifest_digest: str | None = None,
    case_file_digest: str | None = None,
) -> EvaluationRun:
    """Validate inputs and create a reproducible evaluation-run identity.

    Dataset and case digests are derived from the validated dataset. Optional
    caller-provided values are accepted only as assertions and must match.
    """

    validate_run_purpose(
        request,
        held_out_tuning_allowed=dataset.manifest.evaluation_policy.held_out_tuning_allowed,
        held_out_model_selection_allowed=dataset.manifest.evaluation_policy.held_out_model_selection_allowed,
    )
    validate_run_inputs(dataset, request)

    manifest_digest = _manifest_digest(dataset)
    case_digest = dataset.manifest.checksums.get(dataset.manifest.case_file)
    if case_digest is None or not is_sha256_pattern(case_digest):
        raise ValueError("Loaded dataset must contain a valid case-file checksum")
    if (
        dataset_manifest_digest is not None
        and dataset_manifest_digest != manifest_digest
    ):
        raise ValueError("dataset_manifest_digest does not match the loaded dataset")
    if case_file_digest is not None and case_file_digest != case_digest:
        raise ValueError("case_file_digest does not match the loaded dataset")

    if not algorithm_versions:
        raise ValueError("algorithm_versions must not be empty")
    if any(
        not name.strip() or not version.strip()
        for name, version in algorithm_versions.items()
    ):
        raise ValueError("algorithm_versions names and values must be non-empty")
    if not system_version.strip():
        raise ValueError("system_version must not be empty")
    if not is_sha256_pattern(configuration_digest):
        raise ValueError("configuration_digest must use SHA256 format")

    payload = run_identity_payload(
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.dataset_version,
        schema_version=dataset.manifest.schema_version,
        corpus_id=dataset.manifest.corpus_id,
        corpus_version=dataset.manifest.corpus_version,
        split=request.split,
        purpose=request.purpose,
        tuning=request.tuning,
        model_selection=request.model_selection,
        system_version=system_version,
        configuration_digest=configuration_digest,
        algorithm_versions=algorithm_versions,
        dataset_manifest_digest=manifest_digest,
        case_file_digest=case_digest,
    )

    return EvaluationRun(
        run_id=compute_run_id(payload),
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.dataset_version,
        schema_version=dataset.manifest.schema_version,
        corpus_id=dataset.manifest.corpus_id,
        corpus_version=dataset.manifest.corpus_version,
        split=request.split,
        purpose=request.purpose,
        tuning=request.tuning,
        model_selection=request.model_selection,
        system_version=system_version,
        configuration_digest=configuration_digest,
        algorithm_versions=dict(sorted(algorithm_versions.items())),
        dataset_manifest_digest=manifest_digest,
        case_file_digest=case_digest,
        created_at=created_at,
    )
