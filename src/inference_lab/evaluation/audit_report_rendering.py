"""Deterministic aggregate-only Markdown rendering for corpus-audit reports."""

from decimal import ROUND_HALF_UP, Decimal

from inference_lab.evaluation.audit_contracts import AuditRate
from inference_lab.evaluation.audit_reports import (
    AuditCoverageSummary,
    AuditIntegerSummary,
    CorpusAuditReport,
)


def render_audit_report(report: CorpusAuditReport) -> str:
    """Render one validated aggregate-only report as deterministic Markdown."""
    if not isinstance(report, CorpusAuditReport):
        raise TypeError("report must be a CorpusAuditReport.")
    lines = (
        "# Corpus audit report",
        "",
        *_render_identity(report),
        *_render_tools_and_policies(report),
        *_render_populations(report),
        *_render_source_intake(report),
        *_render_parsing(report),
        *_render_acquisition(report),
        *_render_relationships(report),
        *_render_structure_and_provenance(report),
        *_render_limitations(report),
        *_render_reconciliation(report),
    )
    return "\n".join(lines)


def _render_identity(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render report and corpus identities."""
    return (
        "## Identity",
        "",
        f"- Schema version: `{report.schema_version}`",
        f"- Report ID: `{report.report_id}`",
        f"- Input digest: `{report.input_digest}`",
        f"- Report digest: `{report.report_digest}`",
        f"- Corpus ID: `{report.corpus_id}`",
        f"- Corpus version: `{report.corpus_version}`",
        f"- Overall status: `{report.overall_status.value}`",
        "",
    )


def _render_tools_and_policies(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render toolchain and policy identities."""
    return (
        "## Tools and policies",
        "",
        _table(
            ("identity", "value"),
            (
                (
                    "implementation",
                    f"{report.tools.implementation}@{report.tools.implementation_version}",
                ),
                ("contract", report.tools.contract_version),
                (
                    "audit policy",
                    f"{report.tools.policy.policy_name}@{report.tools.policy.policy_version}",
                ),
                ("availability policy", report.tools.policy.availability_version),
                (
                    "discovery policy",
                    f"{report.tools.discovery.policy_name}@{report.tools.discovery.policy_version}",
                ),
                (
                    "classifier",
                    f"{report.tools.discovery.classifier_name}@{report.tools.discovery.classifier_version}",
                ),
                ("parser versions", ", ".join(report.tools.parser_versions) or "none"),
                (
                    "capability profiles",
                    ", ".join(report.tools.capability_profiles) or "none",
                ),
            ),
        ),
        "",
    )


def _render_populations(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render audited population counts."""
    return (
        "## Populations",
        "",
        _table(
            ("population", "count"),
            (
                ("source files seen", report.populations.source_files_seen),
                ("classified source files", report.populations.source_classified_files),
                (
                    "source traversal failures",
                    report.populations.source_traversal_failures,
                ),
                ("corpus member versions", report.populations.member_versions),
            ),
        ),
        "",
    )


def _render_source_intake(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render source format, size, and failure aggregates."""
    return (
        "## Source intake",
        "",
        _table(
            ("media type", "support", "files", "bytes", "hashed", "identity"),
            tuple(
                (
                    item.media_type,
                    item.support.value,
                    item.file_count,
                    item.total_bytes,
                    item.hashed_count,
                    item.identity_count,
                )
                for item in report.source.formats
            ),
        ),
        "",
        "### Source sizes",
        "",
        _distribution_table(report.source.size_summary),
        "",
        "### Source failures",
        "",
        _count_table(report.source.failure_counts, "failure"),
        "",
    )


def _render_parsing(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render parser outcomes, field coverage, and distributions."""
    return (
        "## Parsing and fields",
        "",
        "### Parse statuses",
        "",
        _count_table(report.parse.status_counts, "status"),
        "",
        "### Parser versions",
        "",
        _count_table(report.parse.parser_counts, "parser"),
        "",
        "### Media types",
        "",
        _count_table(report.parse.media_type_counts, "media type"),
        "",
        "### Parse failures",
        "",
        _count_table(report.parse.failure_counts, "failure"),
        "",
        "### Field coverage",
        "",
        _coverage_table(report.parse.field_coverage),
        "",
        "### Numeric distributions",
        "",
        _summary_table(report.parse.numeric_summaries),
        "",
        "### Languages",
        "",
        _count_table(report.parse.language_counts, "language"),
        "",
        "### Token distributions by parser",
        "",
        _summary_table(report.parse.token_summaries),
        "",
    )


def _render_acquisition(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render acquisition coverage and rates."""
    return (
        "## Acquisition status",
        "",
        _count_table(report.acquisition.status_counts, "status"),
        "",
        _coverage_table((("acquisition", report.acquisition.coverage),)),
        "",
        _table(
            ("rate", "numerator", "denominator", "percentage"),
            (
                (
                    "known-only",
                    report.acquisition.known_only_rate.numerator,
                    report.acquisition.known_only_rate.denominator,
                    _percentage(report.acquisition.known_only_rate, report),
                ),
                (
                    "complete-population",
                    report.acquisition.population_rate.numerator,
                    report.acquisition.population_rate.denominator,
                    _percentage(report.acquisition.population_rate, report),
                ),
            ),
        ),
        "",
    )


def _render_relationships(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render duplicate and family relationship aggregates."""
    return (
        "## Relationships",
        "",
        _table(
            ("metric", "value"),
            (
                ("duplicate edges", report.relationships.duplicate_edge_count),
                (
                    "duplicate participating members",
                    report.relationships.duplicate_participating_members,
                ),
                ("duplicate groups", report.relationships.duplicate_group_count),
                (
                    "duplicate excess copies",
                    report.relationships.duplicate_excess_copies,
                ),
                ("family count", report.relationships.family_count),
                ("family documents", report.relationships.family_document_count),
                ("family versions", report.relationships.family_version_count),
                (
                    "unassigned documents",
                    report.relationships.unassigned_document_count,
                ),
                ("family candidates", report.relationships.family_candidate_count),
                (
                    "candidate families",
                    report.relationships.family_candidate_family_count,
                ),
            ),
        ),
        "",
        _table(
            ("rate", "numerator", "denominator", "percentage"),
            (
                (
                    "duplicate participating members",
                    report.relationships.duplicate_participating_rate.numerator,
                    report.relationships.duplicate_participating_rate.denominator,
                    _percentage(
                        report.relationships.duplicate_participating_rate, report
                    ),
                ),
                (
                    "duplicate excess copies",
                    report.relationships.duplicate_excess_rate.numerator,
                    report.relationships.duplicate_excess_rate.denominator,
                    _percentage(report.relationships.duplicate_excess_rate, report),
                ),
            ),
        ),
        "",
        "### Relationship distributions",
        "",
        _summary_table(
            (
                ("duplicate_group_size", report.relationships.duplicate_group_summary),
                ("family_size", report.relationships.family_size_summary),
            )
        ),
        "",
    )


def _render_structure_and_provenance(
    report: CorpusAuditReport,
) -> tuple[str, ...]:
    """Render structural coverage and provenance aggregates."""
    return (
        "## Structure and provenance",
        "",
        _coverage_table(
            tuple((item.kind.value, item.coverage) for item in report.structures)
        ),
        "",
        _summary_table(
            tuple((item.kind.value, item.unit_summary) for item in report.structures)
        ),
        "",
        _table(
            ("structure", "present documents", "absent documents", "units"),
            tuple(
                (
                    item.kind.value,
                    item.present_documents,
                    item.absent_documents,
                    item.unit_summary.distribution.count,
                )
                for item in report.structures
            ),
        ),
        "",
        _table(
            ("provenance metric", "value"),
            (
                ("source spans", report.spans.source_span_count),
                ("page numbers", report.spans.page_number_count),
                ("bounding boxes", report.spans.bounding_box_count),
                ("zero-text pages", report.ocr.zero_text_page_count),
                ("mixed-text documents", report.ocr.mixed_text_documents),
                ("pages with layout", report.ocr.pages_with_layout),
                ("pages without layout", report.ocr.pages_without_layout),
                ("pages with boxes", report.ocr.pages_with_boxes),
                ("pages without boxes", report.ocr.pages_without_boxes),
            ),
        ),
        "",
        "### Provenance distributions",
        "",
        _summary_table(
            (
                ("blocks", report.spans.block_summary),
                ("page_characters", report.ocr.page_character_summary),
            )
        ),
        "",
    )


def _render_limitations(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render bounded limitation codes."""
    limitations = (
        tuple(f"- `{code}`" for code in report.limitations)
        if report.limitations
        else ("- None recorded.",)
    )
    return ("## Limitations", "", *limitations, "")


def _render_reconciliation(report: CorpusAuditReport) -> tuple[str, ...]:
    """Render cross-section reconciliation checks."""
    return (
        "## Reconciliation",
        "",
        _table(
            ("check", "value"),
            (
                ("source files", _yes_no(report.reconciliation.source_files_reconcile)),
                ("source format files", report.reconciliation.source_format_files),
                (
                    "source traversal failures",
                    report.reconciliation.source_traversal_failures,
                ),
                (
                    "member versions",
                    _yes_no(report.reconciliation.member_versions_reconcile),
                ),
                (
                    "parse observation slots",
                    report.reconciliation.parse_observation_slots,
                ),
                (
                    "acquisition observation slots",
                    report.reconciliation.acquisition_observation_slots,
                ),
                (
                    "section populations",
                    _yes_no(report.reconciliation.section_populations_reconcile),
                ),
            ),
        ),
        "",
    )


def _table(headers: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> str:
    head = "| " + " | ".join(_markdown(value) for value in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_markdown(value) for value in row) + " |" for row in rows]
    return "\n".join((head, separator, *body))


def _count_table(values: tuple[tuple[str, int], ...], label: str) -> str:
    return _table((label, "count"), tuple((key, value) for key, value in values))


def _coverage_table(
    values: tuple[tuple[str, AuditCoverageSummary], ...],
) -> str:
    return _table(
        (
            "field",
            "observed",
            "absent",
            "unavailable",
            "missing",
            "failed",
            "invalid",
            "population",
            "reasons",
        ),
        tuple(
            (
                key,
                summary.coverage.observed,
                summary.coverage.observed_absent,
                summary.coverage.unavailable,
                summary.coverage.missing,
                summary.coverage.failed,
                summary.coverage.invalid,
                summary.coverage.population,
                ", ".join(f"{reason}={count}" for reason, count in summary.reasons)
                or "none",
            )
            for key, summary in sorted(values, key=lambda item: item[0])
        ),
    )


def _summary_table(values: tuple[tuple[str, AuditIntegerSummary], ...]) -> str:
    return _table(
        (
            "field",
            "count",
            "total",
            "minimum",
            "maximum",
            "zero",
            "p50",
            "p90",
            "p95",
            "p99",
        ),
        tuple(
            (
                key,
                summary.distribution.count,
                summary.distribution.total,
                summary.distribution.minimum,
                summary.distribution.maximum,
                summary.distribution.zero_count,
                *dict(summary.quantiles).values(),
            )
            for key, summary in sorted(values, key=lambda item: item[0])
        ),
    )


def _distribution_table(summary: AuditIntegerSummary) -> str:
    return _summary_table((("source_bytes", summary),))


def _markdown(value: object) -> str:
    if value is None:
        return "null"
    return str(value).replace("|", "\\|")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _percentage(rate: AuditRate, report: CorpusAuditReport) -> str:
    if rate.denominator == 0:
        return "null"
    places = report.tools.policy.rate_decimal_places
    quantum = Decimal(1).scaleb(-places)
    value = (Decimal(rate.numerator * 100) / Decimal(rate.denominator)).quantize(
        quantum,
        rounding=ROUND_HALF_UP,
    )
    return f"{value:.{places}f}%"
