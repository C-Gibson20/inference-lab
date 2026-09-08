"""Focused tests for corpus-audit report validation phases."""

from dataclasses import replace

import pytest

from inference_lab.evaluation import AuditOverallStatus

from .test_audit_report_publication import _report  # noqa: TID252


def test_report_identity_phase_rejects_unsupported_schema() -> None:
    report = _report()

    with pytest.raises(ValueError, match="unsupported audit report schema"):
        replace(report, schema_version="2.0.0")


def test_report_population_phase_rejects_negative_counts() -> None:
    report = _report()
    populations = replace(report.populations, member_versions=-1)

    with pytest.raises(ValueError, match="member_versions must be a non-negative"):
        replace(report, populations=populations)


def test_report_canonical_shape_phase_rejects_unordered_limitations() -> None:
    report = _report()

    with pytest.raises(ValueError, match="limitations must be ordered and unique"):
        replace(report, limitations=("z-limitation", "a-limitation"))


def test_report_reconciliation_phase_rejects_source_total_drift() -> None:
    report = _report()
    source = replace(report.source, files_seen=report.source.files_seen + 1)

    with pytest.raises(ValueError, match="does not reconcile files seen"):
        replace(report, source=source)


def test_report_digest_phase_rejects_content_drift() -> None:
    report = _report()
    changed_status = (
        AuditOverallStatus.COMPLETE
        if report.overall_status is not AuditOverallStatus.COMPLETE
        else AuditOverallStatus.PARTIAL
    )

    with pytest.raises(ValueError, match="digest does not match"):
        replace(report, overall_status=changed_status)
