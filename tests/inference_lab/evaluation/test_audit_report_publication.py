"""Tests for immutable local corpus-audit report publication."""

import hashlib
import json
from pathlib import Path

import pytest

from inference_lab._fs import rename_directory_without_replacement
from inference_lab.evaluation import (
    AuditErrorCategory,
    AuditPublicationError,
    CorpusAuditReport,
    build_audit_report_manifest,
    deserialise_audit_report,
    publish_audit_report,
    render_audit_report,
)
from inference_lab.evaluation.audit_report_publication import (
    atomic_write,
)

from .audit_fixtures import (  # noqa: TID252
    _DOCUMENT_ONE,
    _VERSION_ONE,
    _membership,
    _observation,
    _request,
)


def _report() -> CorpusAuditReport:
    membership = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    return CorpusAuditReport.from_request(
        _request((membership,), (_observation(membership),))
    )


def test_publish_writes_verified_immutable_report_directory(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "audit"

    manifest = publish_audit_report(destination, report)

    report_bytes = (destination / "report.json").read_bytes()
    markdown_bytes = (destination / "report.md").read_bytes()
    manifest_bytes = (destination / "manifest.json").read_bytes()
    assert tuple(sorted(path.name for path in destination.iterdir())) == (
        "manifest.json",
        "report.json",
        "report.md",
    )
    assert deserialise_audit_report(report_bytes) == report
    assert markdown_bytes.decode("utf-8") == render_audit_report(report)
    assert json.loads(manifest_bytes)["report_id"] == report.report_id
    assert (
        manifest.report_digest == "sha256:" + hashlib.sha256(report_bytes).hexdigest()
    )
    assert manifest == build_audit_report_manifest(
        report,
        report_bytes,
        markdown_bytes.decode("utf-8"),
    )


def test_publish_rejects_remote_and_existing_destinations(tmp_path: Path) -> None:
    report = _report()

    with pytest.raises(ValueError, match="local filesystem path"):
        publish_audit_report("https://example.test/audit", report)

    destination = tmp_path / "audit"
    destination.mkdir()
    marker = destination / "marker"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        publish_audit_report(destination, report)
    assert marker.read_text(encoding="utf-8") == "existing"


def test_publish_cleans_staging_after_write_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    destination = tmp_path / "audit"
    real_atomic_write = atomic_write

    def corrupt_report(path: str | Path, content: bytes) -> None:
        real_atomic_write(
            path, b"corrupted" if Path(path).name == "report.json" else content
        )

    monkeypatch.setattr(
        "inference_lab.evaluation.audit_report_publication.atomic_write",
        corrupt_report,
    )

    with pytest.raises(AuditPublicationError, match="publication failed") as captured:
        publish_audit_report(destination, report)

    assert captured.value.category is AuditErrorCategory.PUBLICATION_FAILURE
    assert captured.value.stage == "publication"
    assert captured.value.retryable is False
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".audit.*.staging"))


def test_publish_cleans_staging_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    destination = tmp_path / "audit"

    def fail_publication(source: Path, target: Path) -> None:
        if Path(target) == destination:
            raise OSError("publication failed")
        rename_directory_without_replacement(source, target)

    monkeypatch.setattr(
        "inference_lab.evaluation.audit_report_publication."
        "rename_directory_without_replacement",
        fail_publication,
    )

    with pytest.raises(AuditPublicationError, match="publication failed"):
        publish_audit_report(destination, report)

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".audit.*.staging"))


def test_publish_preserves_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    destination = tmp_path / "audit"
    real_deserialise = deserialise_audit_report

    def create_competitor(data: bytes) -> CorpusAuditReport:
        restored = real_deserialise(data)
        destination.mkdir()
        (destination / "owner").write_text("external", encoding="utf-8")
        return restored

    monkeypatch.setattr(
        "inference_lab.evaluation.audit_report_publication.deserialise_audit_report",
        create_competitor,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        publish_audit_report(destination, report)

    assert (destination / "owner").read_text(encoding="utf-8") == "external"
    assert not tuple(tmp_path.glob(".audit.*.staging"))


def test_publish_preserves_empty_competing_destination_at_atomic_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    destination = tmp_path / "audit"

    def publish_after_competitor(source: Path, target: Path) -> None:
        destination.mkdir()
        rename_directory_without_replacement(source, target)

    monkeypatch.setattr(
        "inference_lab.evaluation.audit_report_publication."
        "rename_directory_without_replacement",
        publish_after_competitor,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        publish_audit_report(destination, report)

    assert destination.is_dir()
    assert not tuple(destination.iterdir())
    assert not tuple(tmp_path.glob(".audit.*.staging"))
