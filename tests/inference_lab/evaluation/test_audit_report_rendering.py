"""Focused tests for deterministic corpus-audit Markdown sections."""

from collections.abc import Callable

import pytest

from inference_lab.evaluation import CorpusAuditReport
from inference_lab.evaluation.audit_report_rendering import (
    _render_acquisition,
    _render_identity,
    _render_limitations,
    _render_parsing,
    _render_populations,
    _render_reconciliation,
    _render_relationships,
    _render_source_intake,
    _render_structure_and_provenance,
    _render_tools_and_policies,
)

from .test_audit_report_publication import _report  # noqa: TID252

SectionRenderer = Callable[[CorpusAuditReport], tuple[str, ...]]


@pytest.mark.parametrize(
    ("renderer", "heading", "content_marker"),
    (
        (_render_identity, "## Identity", "Report ID"),
        (_render_tools_and_policies, "## Tools and policies", "audit policy"),
        (_render_populations, "## Populations", "source files seen"),
        (_render_source_intake, "## Source intake", "### Source failures"),
        (_render_parsing, "## Parsing and fields", "### Field coverage"),
        (_render_acquisition, "## Acquisition status", "known-only"),
        (_render_relationships, "## Relationships", "duplicate edges"),
        (
            _render_structure_and_provenance,
            "## Structure and provenance",
            "source spans",
        ),
        (_render_limitations, "## Limitations", "acquisition_status_incomplete"),
        (_render_reconciliation, "## Reconciliation", "section populations"),
    ),
)
def test_section_renderer_has_stable_boundary_and_owned_content(
    renderer: SectionRenderer,
    heading: str,
    content_marker: str,
) -> None:
    lines = renderer(_report())

    assert lines[0] == heading
    assert lines[-1] == ""
    assert content_marker in "\n".join(lines)
