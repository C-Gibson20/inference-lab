"""Immutable local publication for aggregate held-out baseline reports."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import cast

from inference_lab._json import (
    JsonObject,
    canonical_json_bytes,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_integer,
    require_json_object,
    require_json_string,
)
from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.heldout_reporting import (
    HeldOutBaselineReport,
    render_held_out_baseline_report,
    serialise_held_out_baseline_report,
)
from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.serialisation import (
    _fsync_directory,
    _fsync_tree_directories,
    _rename_directory_without_replacement,
    atomic_write,
)

HELDOUT_REPORT_SCHEMA_VERSION = "1.0.0"
HELDOUT_REPORT_ARTIFACT_NAMES = ("report.json", "report.md")
_MANIFEST_ARTIFACT_NAMES = (*HELDOUT_REPORT_ARTIFACT_NAMES, "manifest.json")


class HeldOutPublicationError(OSError):
    """Report publication failure without exposing filesystem details."""

    def __init__(self) -> None:
        """Create a fixed-message publication error."""
        super().__init__("held-out baseline publication failed")
        self.category = "publication_failure"
        self.stage = "publication"
        self.retryable = False


class HeldOutPublicationMismatchError(ValueError):
    """Report that a prior publication differs from the supplied run."""

    def __init__(self) -> None:
        """Create a fixed-message repeatability error."""
        super().__init__("held-out baseline publication does not match this run")


@dataclass(frozen=True, slots=True)
class HeldOutReportArtifact:
    """Checksum and byte length for one public aggregate report artifact."""

    name: str
    size_bytes: int
    checksum: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the held-out report artifact fields."""
        if self.name not in HELDOUT_REPORT_ARTIFACT_NAMES:
            raise ValueError("unsupported held-out report artifact name.")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be non-negative.")
        if not is_sha256_checksum(self.checksum):
            raise ValueError("artifact checksum must be SHA-256.")


@dataclass(frozen=True, slots=True)
class HeldOutReportManifest:
    """Versioned identity and checksum inventory for a held-out report."""

    schema_version: str
    run_fingerprint: str
    report_digest: str
    artifacts: tuple[HeldOutReportArtifact, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the held-out report manifest fields."""
        if self.schema_version != HELDOUT_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported held-out report manifest version.")
        if not is_sha256_checksum(self.run_fingerprint):
            raise ValueError("run_fingerprint must be SHA-256.")
        if not is_sha256_checksum(self.report_digest):
            raise ValueError("report_digest must be SHA-256.")
        if not isinstance(self.artifacts, tuple) or any(
            not isinstance(item, HeldOutReportArtifact) for item in self.artifacts
        ):
            raise TypeError("artifacts must be a tuple of report artifacts.")
        if tuple(item.name for item in self.artifacts) != HELDOUT_REPORT_ARTIFACT_NAMES:
            raise ValueError("manifest artifacts must use canonical order.")


def serialise_heldout_report_manifest(manifest: HeldOutReportManifest) -> bytes:
    """Encode a held-out publication manifest as canonical JSON."""
    if not isinstance(manifest, HeldOutReportManifest):
        raise TypeError("manifest must be a HeldOutReportManifest.")
    return canonical_json_bytes(cast(JsonObject, asdict(manifest)))


def deserialise_heldout_report_manifest(data: bytes) -> HeldOutReportManifest:
    """Decode and strictly validate a canonical held-out report manifest."""
    payload = load_json_object(data)
    if canonical_json_bytes(payload) != data:
        raise ValueError("held-out report manifest is not canonical JSON.")
    require_exact_json_fields(
        payload,
        frozenset({"schema_version", "run_fingerprint", "report_digest", "artifacts"}),
    )
    artifacts: list[HeldOutReportArtifact] = []
    for value in require_json_array(payload["artifacts"]):
        artifact = require_json_object(value)
        require_exact_json_fields(
            artifact, frozenset({"name", "size_bytes", "checksum"})
        )
        artifacts.append(
            HeldOutReportArtifact(
                name=require_json_string(artifact["name"]),
                size_bytes=require_json_integer(artifact["size_bytes"]),
                checksum=require_json_string(artifact["checksum"]),
            )
        )
    return HeldOutReportManifest(
        schema_version=require_json_string(payload["schema_version"]),
        run_fingerprint=require_json_string(payload["run_fingerprint"]),
        report_digest=require_json_string(payload["report_digest"]),
        artifacts=tuple(artifacts),
    )


def publish_heldout_baseline_report(
    destination: str | Path,
    report: HeldOutBaselineReport,
) -> HeldOutReportManifest:
    """Atomically publish an aggregate-only report to a new local directory."""
    if not isinstance(report, HeldOutBaselineReport):
        raise TypeError("report must be a HeldOutBaselineReport.")
    raw_destination = Path(destination).expanduser()
    destination = validate_local_path(destination, field_name="destination")
    if raw_destination.is_symlink() or destination.exists() or destination.is_symlink():
        raise FileExistsError("Held-out report destination already exists")

    artifact_bytes = {
        "report.json": serialise_held_out_baseline_report(report),
        "report.md": render_held_out_baseline_report(report).encode("utf-8"),
    }
    manifest = _build_manifest(report, artifact_bytes)
    manifest_bytes = serialise_heldout_report_manifest(manifest)
    staging_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_path = Path(
            mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".staging",
            )
        )
        for name in HELDOUT_REPORT_ARTIFACT_NAMES:
            atomic_write(staging_path / name, artifact_bytes[name])
        atomic_write(staging_path / "manifest.json", manifest_bytes)
        _verify_staged(staging_path, artifact_bytes, manifest_bytes, report)
        _fsync_tree_directories(staging_path)
        _rename_directory_without_replacement(staging_path, destination)
        staging_path = None
        _fsync_directory(destination.parent)
        return manifest
    except FileExistsError:
        _cleanup_staging(staging_path)
        raise FileExistsError("Held-out report destination already exists") from None
    except (OSError, ValueError, TypeError):
        _cleanup_staging(staging_path)
        raise HeldOutPublicationError from None


def verify_heldout_baseline_publication(
    destination: str | Path,
) -> HeldOutReportManifest:
    """Validate inventory, checksums, and report identity of a local bundle."""
    raw_destination = Path(destination).expanduser()
    if raw_destination.is_symlink():
        raise ValueError("destination must identify a held-out report directory.")
    destination = validate_local_path(destination, field_name="destination")
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("destination must identify a held-out report directory.")
    try:
        actual_names = {path.name for path in destination.iterdir()}
        if actual_names != set(_MANIFEST_ARTIFACT_NAMES):
            raise ValueError("held-out report directory has an invalid inventory.")
        manifest = deserialise_heldout_report_manifest(
            (destination / "manifest.json").read_bytes()
        )
        contents: dict[str, bytes] = {}
        for artifact in manifest.artifacts:
            path = destination / artifact.name
            if path.is_symlink() or not path.is_file():
                raise ValueError("held-out report artifact is not a regular file.")
            content = path.read_bytes()
            if (
                len(content) != artifact.size_bytes
                or _sha256(content) != artifact.checksum
            ):
                raise ValueError("held-out report artifact checksum mismatch.")
            contents[artifact.name] = content
        report_payload = load_json_object(contents["report.json"])
        if canonical_json_bytes(report_payload) != contents["report.json"]:
            raise ValueError("held-out report is not canonical JSON.")
        run_contract = require_json_object(report_payload.get("run_contract"))
        if (
            require_json_string(run_contract.get("fingerprint"))
            != manifest.run_fingerprint
            or _sha256(contents["report.json"]) != manifest.report_digest
        ):
            raise ValueError("held-out report identity does not match its manifest.")
        if not contents["report.md"].decode("utf-8"):
            raise ValueError("held-out Markdown report must not be empty.")
        return manifest
    except (OSError, UnicodeDecodeError, TypeError):
        raise ValueError("held-out report publication is invalid.") from None


def compare_heldout_baseline_publication(
    destination: str | Path,
    report: HeldOutBaselineReport,
) -> bool:
    """Return true only when a prior bundle is byte-identical to this report."""
    if not isinstance(report, HeldOutBaselineReport):
        raise TypeError("report must be a HeldOutBaselineReport.")
    try:
        manifest = verify_heldout_baseline_publication(destination)
        artifact_bytes = {
            "report.json": serialise_held_out_baseline_report(report),
            "report.md": render_held_out_baseline_report(report).encode("utf-8"),
        }
        expected_manifest = _build_manifest(report, artifact_bytes)
        if manifest != expected_manifest:
            raise HeldOutPublicationMismatchError
        root = validate_local_path(destination, field_name="destination")
        if any(
            (root / name).read_bytes() != content
            for name, content in artifact_bytes.items()
        ):
            raise HeldOutPublicationMismatchError
        return True
    except HeldOutPublicationMismatchError:
        raise
    except (OSError, ValueError, TypeError):
        raise HeldOutPublicationMismatchError from None


def _build_manifest(
    report: HeldOutBaselineReport,
    artifact_bytes: dict[str, bytes],
) -> HeldOutReportManifest:
    if tuple(artifact_bytes) != HELDOUT_REPORT_ARTIFACT_NAMES:
        raise ValueError("report artifacts are not in canonical order.")
    fingerprint = report.contract.fingerprint
    if fingerprint is None:
        raise ValueError("held-out report contract must have a fingerprint.")
    return HeldOutReportManifest(
        schema_version=HELDOUT_REPORT_SCHEMA_VERSION,
        run_fingerprint=fingerprint,
        report_digest=report.digest,
        artifacts=tuple(
            HeldOutReportArtifact(
                name=name,
                size_bytes=len(artifact_bytes[name]),
                checksum=_sha256(artifact_bytes[name]),
            )
            for name in HELDOUT_REPORT_ARTIFACT_NAMES
        ),
    )


def _verify_staged(
    staging_path: Path,
    artifact_bytes: dict[str, bytes],
    manifest_bytes: bytes,
    report: HeldOutBaselineReport,
) -> None:
    for name, expected in (*artifact_bytes.items(), ("manifest.json", manifest_bytes)):
        if (staging_path / name).read_bytes() != expected:
            raise OSError("staged held-out report bytes changed")
    verified = verify_heldout_baseline_publication(staging_path)
    if verified != _build_manifest(report, artifact_bytes):
        raise OSError("staged held-out report failed validation")


def _cleanup_staging(staging_path: Path | None) -> None:
    if staging_path is not None:
        shutil.rmtree(staging_path, ignore_errors=True)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
