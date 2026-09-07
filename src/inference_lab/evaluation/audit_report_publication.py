"""Atomic local publication of immutable corpus-audit report artefacts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

from inference_lab.evaluation.audit_errors import AuditPublicationError
from inference_lab.evaluation.audit_report_serialisation import (
    AuditReportManifest,
    build_audit_report_manifest,
    deserialise_audit_report,
    render_audit_report,
    serialise_audit_report,
    serialise_audit_report_manifest,
)
from inference_lab.evaluation.audit_reports import CorpusAuditReport
from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.serialisation import (
    _fsync_directory,
    _fsync_tree_directories,
    atomic_write,
)


def publish_audit_report(
    destination: str | Path,
    report: CorpusAuditReport,
) -> AuditReportManifest:
    """Publish one validated report directory without overwriting a destination."""
    if not isinstance(report, CorpusAuditReport):
        raise TypeError("report must be a CorpusAuditReport.")
    raw_destination = Path(destination).expanduser()
    destination = validate_local_path(destination, field_name="destination")
    if raw_destination.is_symlink() or destination.exists():
        raise FileExistsError("Audit report destination already exists")

    report_json = serialise_audit_report(report)
    report_markdown = render_audit_report(report)
    manifest = build_audit_report_manifest(report, report_json, report_markdown)
    manifest_json = serialise_audit_report_manifest(manifest)

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
        report_path = staging_path / "report.json"
        markdown_path = staging_path / "report.md"
        manifest_path = staging_path / "manifest.json"
        atomic_write(report_path, report_json)
        atomic_write(markdown_path, report_markdown.encode("utf-8"))
        atomic_write(manifest_path, manifest_json)
        _verify_staged_files(
            report_path,
            markdown_path,
            manifest_path,
            report_json,
            report_markdown,
            manifest_json,
            report,
        )
        _fsync_tree_directories(staging_path)

        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Audit report destination already exists")
        os.replace(staging_path, destination)
        staging_path = None
        _fsync_directory(destination.parent)
        return manifest
    except FileExistsError:
        _cleanup_staging(staging_path)
        raise FileExistsError("Audit report destination already exists") from None
    except OSError as error:
        _cleanup_staging(staging_path, error)
        raise AuditPublicationError from None


def _cleanup_staging(
    staging_path: Path | None, error: BaseException | None = None
) -> None:
    """Remove only this operation's staging directory after failure."""
    if staging_path is None:
        return
    try:
        shutil.rmtree(staging_path)
    except OSError as cleanup_error:
        if error is not None:
            error.add_note(
                f"Staging cleanup also failed: {type(cleanup_error).__name__}"
            )


def _verify_staged_files(
    report_path: Path,
    markdown_path: Path,
    manifest_path: Path,
    report_json: bytes,
    report_markdown: str,
    manifest_json: bytes,
    report: CorpusAuditReport,
) -> None:
    """Verify staged bytes before the directory becomes visible."""
    if report_path.read_bytes() != report_json:
        raise OSError("staged audit report bytes changed")
    if markdown_path.read_bytes() != report_markdown.encode("utf-8"):
        raise OSError("staged audit Markdown bytes changed")
    if manifest_path.read_bytes() != manifest_json:
        raise OSError("staged audit manifest bytes changed")
    if deserialise_audit_report(report_path.read_bytes()) != report:
        raise OSError("staged audit report failed validation")
