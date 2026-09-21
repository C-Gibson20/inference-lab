"""Tests for deterministic retrieval-evaluation Markdown reports."""

from dataclasses import replace
from pathlib import Path

import pytest

from inference_lab.evaluation import (
    DuplicateDiagnostics,
    RetrievalEvaluationReport,
    RetrievalEvaluationRunResult,
    render_retrieval_report,
    retrieval_result_digest,
)

from .test_retrieval_report_publication import publication_run  # noqa: TID252


def report_for(
    tmp_path: Path,
) -> tuple[RetrievalEvaluationRunResult, RetrievalEvaluationReport]:
    """Build one report fixture with explicit aggregate diagnostics."""
    run = publication_run(tmp_path / "fixture")
    assert isinstance(run.identity.run_fingerprint, str)
    aggregate = replace(
        run.aggregate,
        duplicate_diagnostics=DuplicateDiagnostics(
            raw_hit_count=1,
            distinct_record_count=1,
            distinct_document_count=1,
            distinct_duplicate_group_count=1,
            distinct_family_count=0,
            unknown_family_document_count=1,
            exact_document_repeat_excess=0,
            byte_identical_group_repeat_excess=0,
            family_member_repeat_excess=0,
            relevant_non_gold_duplicate_copy_count=0,
        ),
    )
    return run, RetrievalEvaluationReport(
        run_fingerprint=run.identity.run_fingerprint,
        result_digest=retrieval_result_digest(run),
        aggregate=aggregate,
        limitations=("candidate depth was exhausted for one or more cases",),
    )


def test_render_report_is_deterministic_and_covers_required_sections(
    tmp_path: Path,
) -> None:
    """Rendering exposes safe metrics and identities in stable order."""
    run, report = report_for(tmp_path)

    rendered = render_retrieval_report(report, run.identity)

    assert rendered == render_retrieval_report(report, run.identity)
    assert rendered.startswith("# Retrieval evaluation report\n")
    for section in (
        "## Headline macro metrics",
        "## Micro diagnostics",
        "## Outcome reconciliation",
        "## Duplicate and family diagnostics",
        "## Pinned identities and policy",
        "## Limitations",
    ):
        assert section in rendered
    assert rendered.index("document_recall_at_10") < rendered.index(
        "passage_recall_at_1"
    )
    assert rendered.index("passage_recall_at_1") < rendered.index("passage_recall_at_5")
    assert rendered.index("passage_recall_at_5") < rendered.index(
        "passage_recall_at_10"
    )
    assert "matched targets" in rendered
    assert "unknown family document count" in rendered
    assert "candidate depth was exhausted for one or more cases" in rendered
    assert isinstance(run.identity.run_fingerprint, str)
    assert run.identity.run_fingerprint in rendered
    assert report.result_digest in rendered


def test_render_report_rejects_mixed_identity_and_never_emits_payload_text(
    tmp_path: Path,
) -> None:
    """Rendering enforces run linkage and only exposes safe contract values."""
    run, report = report_for(tmp_path)

    with pytest.raises(ValueError, match="fingerprints do not match"):
        render_retrieval_report(
            replace(report, run_fingerprint="sha256:" + "0" * 64),
            run.identity,
        )

    rendered = render_retrieval_report(report, run.identity)
    assert "question and candidate content sentinel" not in rendered
    assert "candidate content sentinel" not in rendered
    assert "/private/" not in rendered
