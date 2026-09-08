"""Checksum manifest contracts and construction for corpus-audit reports."""

import hashlib
from dataclasses import asdict, dataclass
from typing import cast

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.evaluation.audit_report_rendering import render_audit_report
from inference_lab.evaluation.audit_report_serialisation import serialise_audit_report
from inference_lab.evaluation.audit_reports import (
    AUDIT_REPORT_SCHEMA_VERSION,
    CorpusAuditReport,
)


@dataclass(frozen=True, slots=True)
class AuditReportManifest:
    """Checksum inventory for one immutable report's logical artefacts."""

    schema_version: str
    report_id: str
    input_digest: str
    report_digest: str
    artifacts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the manifest's schema version, identities, and artefact ordering."""
        if self.schema_version != AUDIT_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported audit manifest schema version.")
        for value in (self.report_id, self.input_digest, self.report_digest):
            if not isinstance(value, str) or not _is_sha256(value):
                raise ValueError("manifest identities must be SHA-256 values.")
        if self.artifacts != tuple(sorted(self.artifacts)):
            raise ValueError("manifest artifacts must be ordered.")
        if len(self.artifacts) != len({name for name, _ in self.artifacts}):
            raise ValueError("manifest artifacts must be unique.")
        for name, checksum in self.artifacts:
            if not isinstance(name, str) or not name:
                raise ValueError("manifest artifact names must be non-empty strings.")
            if not _is_sha256(checksum):
                raise ValueError("manifest artifact checksums must be SHA-256 values.")


def build_audit_report_manifest(
    report: CorpusAuditReport,
    report_json: bytes | None = None,
    report_markdown: str | None = None,
) -> AuditReportManifest:
    """Build the deterministic checksum inventory used by publication."""
    if not isinstance(report, CorpusAuditReport):
        raise TypeError("report must be a CorpusAuditReport.")
    json_bytes = serialise_audit_report(report) if report_json is None else report_json
    markdown = (
        render_audit_report(report) if report_markdown is None else report_markdown
    )
    if not isinstance(json_bytes, bytes) or not isinstance(markdown, str):
        raise TypeError("report artefacts must be bytes and text.")
    expected_json = serialise_audit_report(report)
    expected_markdown = render_audit_report(report)
    if json_bytes != expected_json or markdown != expected_markdown:
        raise ValueError("report artefacts do not match the validated report.")
    json_digest = _sha256(json_bytes)
    markdown_bytes = markdown.encode("utf-8")
    return AuditReportManifest(
        schema_version=AUDIT_REPORT_SCHEMA_VERSION,
        report_id=report.report_id,
        input_digest=report.input_digest,
        report_digest=json_digest,
        artifacts=(
            ("report.json", json_digest),
            ("report.md", _sha256(markdown_bytes)),
        ),
    )


def serialise_audit_report_manifest(manifest: AuditReportManifest) -> bytes:
    """Return canonical UTF-8 JSON bytes for a report checksum manifest."""
    if not isinstance(manifest, AuditReportManifest):
        raise TypeError("manifest must be an AuditReportManifest.")
    return canonical_json_bytes(cast(JsonObject, asdict(manifest)))


def _is_sha256(value: str) -> bool:
    return (
        value.startswith("sha256:")
        and len(value) == len("sha256:") + 64
        and all(
            character in "0123456789abcdef"
            for character in value.removeprefix("sha256:")
        )
    )


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
