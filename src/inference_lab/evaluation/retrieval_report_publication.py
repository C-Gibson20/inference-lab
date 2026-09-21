"""Atomic local publication of retrieval-evaluation artefacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.retrieval_contracts import (
    PUBLISHED_ARTIFACT_NAMES,
    PublishedArtifact,
    PublishedRetrievalEvaluationManifest,
)
from inference_lab.evaluation.retrieval_execution import RetrievalEvaluationRunResult
from inference_lab.evaluation.retrieval_report_codec import (
    RetrievalEvaluationRunArtifact,
    deserialise_case_results,
    deserialise_retrieval_manifest,
    deserialise_retrieval_report,
    deserialise_run_artifact,
    retrieval_result_digest,
    serialise_case_results,
    serialise_retrieval_manifest,
    serialise_retrieval_report,
    serialise_run_artifact,
)
from inference_lab.evaluation.retrieval_report_contracts import (
    RetrievalEvaluationReport,
)
from inference_lab.evaluation.serialisation import (
    _fsync_directory,
    _fsync_tree_directories,
    atomic_write,
)

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


class RetrievalEvaluationPublicationError(OSError):
    """Report a local publication failure without exposing filesystem details."""

    def __init__(self) -> None:
        """Create a fixed-message publication error."""
        super().__init__("retrieval evaluation publication failed")
        self.category = "publication_failure"
        self.stage = "publication"
        self.retryable = False


def publish_retrieval_evaluation(
    destination: str | Path,
    run: RetrievalEvaluationRunResult,
    report_markdown: str,
    *,
    limitations: tuple[str, ...] = (),
) -> PublishedRetrievalEvaluationManifest:
    """Publish one complete retrieval evaluation into a new directory.

    The Markdown text is supplied by the deterministic renderer so this
    publication boundary remains independent of presentation concerns.
    """
    if not isinstance(run, RetrievalEvaluationRunResult):
        raise TypeError("run must be a RetrievalEvaluationRunResult.")
    if not isinstance(report_markdown, str):
        raise TypeError("report_markdown must be a string.")
    markdown_bytes = report_markdown.encode("utf-8")
    if not markdown_bytes:
        raise ValueError("report_markdown must not be empty.")

    raw_destination = Path(destination).expanduser()
    destination = validate_local_path(destination, field_name="destination")
    if raw_destination.is_symlink() or destination.exists():
        raise FileExistsError("Retrieval evaluation destination already exists")

    result_digest = retrieval_result_digest(run)
    if not isinstance(run.identity.run_fingerprint, str):
        raise ValueError("run identity must contain a fingerprint.")
    report = RetrievalEvaluationReport(
        run_fingerprint=run.identity.run_fingerprint,
        result_digest=result_digest,
        aggregate=run.aggregate,
        limitations=limitations,
    )
    artefact_bytes = {
        "run.json": serialise_run_artifact(
            run.identity,
            result_digest,
            limitations,
        ),
        "cases.jsonl": serialise_case_results(run),
        "report.json": serialise_retrieval_report(report),
        "report.md": markdown_bytes,
    }
    manifest = _build_manifest(run, result_digest, artefact_bytes)
    manifest_bytes = serialise_retrieval_manifest(manifest)

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
        for name in PUBLISHED_ARTIFACT_NAMES:
            atomic_write(staging_path / name, artefact_bytes[name])
        atomic_write(staging_path / "manifest.json", manifest_bytes)
        _verify_staged_files(
            staging_path,
            run,
            report,
            manifest,
            artefact_bytes,
            manifest_bytes,
        )
        _fsync_tree_directories(staging_path)
        _rename_directory_without_replacement(staging_path, destination)
        staging_path = None
        _fsync_directory(destination.parent)
        return manifest
    except FileExistsError:
        _cleanup_staging(staging_path)
        raise FileExistsError(
            "Retrieval evaluation destination already exists"
        ) from None
    except (OSError, ValueError, TypeError):
        _cleanup_staging(staging_path)
        raise RetrievalEvaluationPublicationError from None


def load_published_retrieval_evaluation(
    destination: str | Path,
) -> RetrievalEvaluationRunResult:
    """Load and cross-validate one previously published evaluation bundle."""
    destination = validate_local_path(destination, field_name="destination")
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("destination must identify a published evaluation directory.")
    expected_names = {*PUBLISHED_ARTIFACT_NAMES, "manifest.json"}
    actual_names = {path.name for path in destination.iterdir()}
    if actual_names != expected_names:
        raise ValueError("published evaluation directory has an invalid inventory.")
    manifest = deserialise_retrieval_manifest(
        (destination / "manifest.json").read_bytes()
    )
    artefact_contents: dict[str, bytes] = {}
    for artefact in manifest.artefacts:
        path = destination / artefact.name
        if path.is_symlink() or not path.is_file():
            raise ValueError("published evaluation artefact is not a regular file.")
        content = path.read_bytes()
        if len(content) != artefact.size_bytes or _sha256(content) != artefact.checksum:
            raise ValueError("published evaluation artefact checksum mismatch.")
        artefact_contents[artefact.name] = content

    run_artifact: RetrievalEvaluationRunArtifact = deserialise_run_artifact(
        artefact_contents["run.json"]
    )
    cases = deserialise_case_results(
        artefact_contents["cases.jsonl"],
        run_artifact.identity,
    )
    report = deserialise_retrieval_report(artefact_contents["report.json"])
    if (
        report.run_fingerprint != manifest.run_fingerprint
        or report.result_digest != manifest.result_digest
        or run_artifact.identity.run_fingerprint != manifest.run_fingerprint
        or run_artifact.result_digest != manifest.result_digest
        or report.aggregate.case_count != len(cases)
    ):
        raise ValueError("published evaluation identities do not reconcile.")
    run = RetrievalEvaluationRunResult(
        identity=run_artifact.identity,
        case_results=cases,
        aggregate=report.aggregate,
    )
    if retrieval_result_digest(run) != manifest.result_digest:
        raise ValueError("published evaluation result digest mismatch.")
    return run


def _build_manifest(
    run: RetrievalEvaluationRunResult,
    result_digest: str,
    artefact_bytes: dict[str, bytes],
) -> PublishedRetrievalEvaluationManifest:
    """Build the ordered checksum and size inventory for publication."""
    if tuple(artefact_bytes) != PUBLISHED_ARTIFACT_NAMES:
        raise ValueError("publication artefacts are not in canonical order.")
    if not isinstance(run.identity.run_fingerprint, str):
        raise ValueError("run identity must contain a fingerprint.")
    return PublishedRetrievalEvaluationManifest(
        run_fingerprint=run.identity.run_fingerprint,
        result_digest=result_digest,
        artefacts=tuple(
            PublishedArtifact(
                name=name,
                size_bytes=len(artefact_bytes[name]),
                checksum=_sha256(artefact_bytes[name]),
            )
            for name in PUBLISHED_ARTIFACT_NAMES
        ),
    )


def _verify_staged_files(
    staging_path: Path,
    run: RetrievalEvaluationRunResult,
    report: RetrievalEvaluationReport,
    manifest: PublishedRetrievalEvaluationManifest,
    artefact_bytes: dict[str, bytes],
    manifest_bytes: bytes,
) -> None:
    """Verify bytes, checksums, and public readers before promotion."""
    for name in (*PUBLISHED_ARTIFACT_NAMES, "manifest.json"):
        path = staging_path / name
        expected = manifest_bytes if name == "manifest.json" else artefact_bytes[name]
        if path.read_bytes() != expected:
            raise OSError("staged retrieval evaluation bytes changed")

    restored_run = deserialise_run_artifact((staging_path / "run.json").read_bytes())
    restored_cases = deserialise_case_results(
        (staging_path / "cases.jsonl").read_bytes(),
        run.identity,
    )
    restored_report = deserialise_retrieval_report(
        (staging_path / "report.json").read_bytes()
    )
    restored_manifest = deserialise_retrieval_manifest(
        (staging_path / "manifest.json").read_bytes()
    )
    if (
        restored_run.identity != run.identity
        or restored_run.result_digest != manifest.result_digest
    ):
        raise OSError("staged retrieval evaluation run failed validation")
    if restored_cases != run.case_results:
        raise OSError("staged retrieval evaluation cases failed validation")
    if restored_report != report or restored_manifest != manifest:
        raise OSError("staged retrieval evaluation report failed validation")
    for artefact in manifest.artefacts:
        content = (staging_path / artefact.name).read_bytes()
        if len(content) != artefact.size_bytes or _sha256(content) != artefact.checksum:
            raise OSError("staged retrieval evaluation checksum failed validation")


def _rename_directory_without_replacement(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination is absent."""
    if sys.platform == "darwin":
        _call_libc_rename(
            "renamex_np",
            os.fsencode(source),
            os.fsencode(destination),
            _RENAME_EXCL,
            destination=destination,
        )
        return
    if sys.platform.startswith("linux"):
        _call_libc_rename(
            "renameat2",
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
            destination=destination,
        )
        return
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory rename is unsupported",
        destination,
    )


def _call_libc_rename(
    function_name: str,
    *arguments: bytes | int,
    destination: Path,
) -> None:
    """Call a platform rename primitive and preserve its errno semantics."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = getattr(libc, function_name)
    except AttributeError:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory rename is unsupported",
            destination,
        ) from None
    if function_name == "renamex_np":
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    else:
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
    rename.restype = ctypes.c_int
    if rename(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), destination)


def _cleanup_staging(staging_path: Path | None) -> None:
    """Remove only this operation's writer-owned staging directory."""
    if staging_path is not None:
        shutil.rmtree(staging_path, ignore_errors=True)


def _sha256(data: bytes) -> str:
    """Return a canonical SHA-256 checksum for one artefact."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


__all__ = [
    "RetrievalEvaluationPublicationError",
    "load_published_retrieval_evaluation",
    "publish_retrieval_evaluation",
]
