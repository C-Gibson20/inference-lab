"""Operator composition for running and publishing one corpus audit."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inference_lab.evaluation.audit_contracts import CorpusAuditRequest
from inference_lab.evaluation.audit_observability import run_corpus_audit
from inference_lab.evaluation.audit_report_publication import publish_audit_report
from inference_lab.evaluation.audit_report_serialisation import AuditReportManifest
from inference_lab.evaluation.audit_reports import CorpusAuditReport

Clock = Callable[[], int]


@dataclass(frozen=True, slots=True)
class AuditRunResult:
    """Return the report and verified publication manifest for one run."""

    report: CorpusAuditReport
    manifest: AuditReportManifest

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the report and manifest are consistent with each other."""
        if not isinstance(self.report, CorpusAuditReport):
            raise TypeError("report must be a CorpusAuditReport.")
        if not isinstance(self.manifest, AuditReportManifest):
            raise TypeError("manifest must be an AuditReportManifest.")
        if self.manifest.report_id != self.report.report_id:
            raise ValueError("manifest must identify the returned report.")
        if self.manifest.input_digest != self.report.input_digest:
            raise ValueError("manifest must identify the returned input.")


def run_and_publish_audit(
    request: CorpusAuditRequest,
    destination: str | Path,
    *,
    clock: Clock = time.monotonic_ns,
) -> AuditRunResult:
    """Run one validated audit and publish its immutable local artefacts.

    The caller owns construction of the complete request, including any
    storage-backed snapshot or source-intake adapter.  This boundary performs
    no discovery, repository selection, source reading, or network access.
    """
    if not isinstance(request, CorpusAuditRequest):
        raise TypeError("request must be a CorpusAuditRequest.")
    if not isinstance(destination, (str, Path)):
        raise TypeError("destination must be a string or Path.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

    report = run_corpus_audit(request, clock=clock)
    manifest = publish_audit_report(destination, report)
    return AuditRunResult(report=report, manifest=manifest)
