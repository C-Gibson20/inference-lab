"""EVAL-02 report validation through publication invariants."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from inference_lab.evaluation import AuditOverallStatus

from .test_audit_report_publication import _report  # noqa: TID252


def test_report_rejects_incoherent_or_unpublishable_mutations() -> None:
    report = _report()
    changed_status = (
        AuditOverallStatus.COMPLETE
        if report.overall_status is not AuditOverallStatus.COMPLETE
        else AuditOverallStatus.PARTIAL
    )

    mutations: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: replace(report, schema_version="2.0.0"),
            "unsupported audit report schema",
        ),
        (
            lambda: replace(
                report,
                populations=replace(report.populations, member_versions=-1),
            ),
            "member_versions must be a non-negative",
        ),
        (
            lambda: replace(report, limitations=("z-limitation", "a-limitation")),
            "limitations must be ordered and unique",
        ),
        (
            lambda: replace(
                report,
                source=replace(report.source, files_seen=report.source.files_seen + 1),
            ),
            "does not reconcile files seen",
        ),
        (
            lambda: replace(report, overall_status=changed_status),
            "digest does not match",
        ),
    )
    for mutation, message in mutations:
        with pytest.raises(ValueError, match=message):
            mutation()
