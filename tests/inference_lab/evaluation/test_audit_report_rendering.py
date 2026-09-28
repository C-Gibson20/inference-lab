"""EVAL-02 rendered report acceptance."""

from inference_lab.evaluation import render_audit_report

from .test_audit_report_publication import _report  # noqa: TID252


def test_public_report_contains_ordered_sections_and_owned_content() -> None:
    rendered = render_audit_report(_report())
    headings = (
        "## Identity",
        "## Tools and policies",
        "## Populations",
        "## Source intake",
        "## Parsing and fields",
        "## Acquisition status",
        "## Relationships",
        "## Structure and provenance",
        "## Limitations",
        "## Reconciliation",
    )

    positions = tuple(rendered.index(heading) for heading in headings)
    assert positions == tuple(sorted(positions))
    assert "Report ID" in rendered
    assert "source files seen" in rendered
    assert "known-only" in rendered
    assert "acquisition_status_incomplete" in rendered
