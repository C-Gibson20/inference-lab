"""Deterministic Markdown rendering for retrieval-evaluation reports."""

from __future__ import annotations

from collections.abc import Iterable

from inference_lab.evaluation.retrieval_contracts import (
    AggregateMetric,
    DuplicateDiagnostics,
    MetricFraction,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
)
from inference_lab.evaluation.retrieval_report_contracts import (
    RetrievalEvaluationReport,
)


def render_retrieval_report(
    report: RetrievalEvaluationReport,
    identity: RetrievalEvaluationRunIdentity,
) -> str:
    """Render one validated aggregate report as stable, payload-free Markdown."""
    if not isinstance(report, RetrievalEvaluationReport):
        raise TypeError("report must be a RetrievalEvaluationReport.")
    if not isinstance(identity, RetrievalEvaluationRunIdentity):
        raise TypeError("identity must be a RetrievalEvaluationRunIdentity.")
    if report.run_fingerprint != identity.run_fingerprint:
        raise ValueError("report and run identity fingerprints do not match.")

    lines = (
        "# Retrieval evaluation report",
        "",
        *_render_headline(report, identity.policy),
        *_render_micro_diagnostics(report, identity.policy),
        *_render_outcome_reconciliation(report),
        *_render_duplicate_diagnostics(report.aggregate.duplicate_diagnostics),
        *_render_identities(report, identity),
        *_render_limitations(report.limitations),
    )
    return "\n".join(lines)


def _render_headline(
    report: RetrievalEvaluationReport,
    policy: RetrievalEvaluationPolicy,
) -> tuple[str, ...]:
    """Render the named macro headline metrics in policy order."""
    rows = [
        (
            "document_recall_at_10",
            _format_fraction(report.aggregate.document.macro, policy),
        )
    ]
    rows.extend(
        (
            f"passage_recall_at_{metric.cutoff}",
            _format_fraction(metric.macro, policy),
        )
        for metric in report.aggregate.passage
    )
    return (
        "## Headline macro metrics",
        "",
        _table(("metric", "value"), rows),
        "",
    )


def _render_micro_diagnostics(
    report: RetrievalEvaluationReport,
    policy: RetrievalEvaluationPolicy,
) -> tuple[str, ...]:
    """Render exact micro diagnostics without making them headline metrics."""
    metrics = _named_metrics(report)
    rows = [
        (
            name,
            _format_fraction(metric.micro, policy),
            metric.matched_target_count,
            metric.relevant_target_count,
        )
        for name, metric in metrics
    ]
    return (
        "## Micro diagnostics",
        "",
        _table(
            ("metric", "value", "matched targets", "relevant targets"),
            rows,
        ),
        "",
    )


def _render_outcome_reconciliation(
    report: RetrievalEvaluationReport,
) -> tuple[str, ...]:
    """Render case populations and terminal outcome reconciliation."""
    rows = [
        (
            name,
            report.aggregate.case_count,
            metric.applicable_case_count,
            metric.non_applicable_case_count,
            metric.completed_case_count,
            metric.partial_case_count,
            metric.failed_case_count,
            metric.empty_case_count,
        )
        for name, metric in _named_metrics(report)
    ]
    return (
        "## Outcome reconciliation",
        "",
        _table(
            (
                "metric",
                "case count",
                "applicable",
                "non-applicable",
                "completed",
                "partial",
                "failed",
                "empty",
            ),
            rows,
        ),
        "",
    )


def _render_duplicate_diagnostics(
    diagnostics: DuplicateDiagnostics | None,
) -> tuple[str, ...]:
    """Render bounded duplicate and family crowding diagnostics."""
    if diagnostics is None:
        return (
            "## Duplicate and family diagnostics",
            "",
            "No duplicate or family diagnostics were recorded.",
            "",
        )
    labels = (
        ("raw hit count", diagnostics.raw_hit_count),
        ("distinct record count", diagnostics.distinct_record_count),
        ("distinct document count", diagnostics.distinct_document_count),
        (
            "distinct byte-identical group count",
            diagnostics.distinct_duplicate_group_count,
        ),
        ("distinct family count", diagnostics.distinct_family_count),
        (
            "unknown family document count",
            diagnostics.unknown_family_document_count,
        ),
        ("exact document repeat excess", diagnostics.exact_document_repeat_excess),
        (
            "byte-identical group repeat excess",
            diagnostics.byte_identical_group_repeat_excess,
        ),
        ("family-member repeat excess", diagnostics.family_member_repeat_excess),
        (
            "relevant non-gold duplicate copy count",
            diagnostics.relevant_non_gold_duplicate_copy_count,
        ),
    )
    return (
        "## Duplicate and family diagnostics",
        "",
        _table(("diagnostic", "count"), labels),
        "",
        "Duplicate and family relationships are diagnostics only; they never substitute for relevance.",
        "",
    )


def _render_identities(
    report: RetrievalEvaluationReport,
    identity: RetrievalEvaluationRunIdentity,
) -> tuple[str, ...]:
    """Render every pinned identity and policy that defines the run."""
    system = identity.system_identity
    policy = identity.policy
    identity_rows = (
        ("run fingerprint", identity.run_fingerprint),
        ("result digest", report.result_digest),
        ("dataset", f"{identity.dataset_id}@{identity.dataset_version}"),
        ("dataset schema", identity.dataset_schema_version),
        ("case file checksum", identity.case_file_checksum),
        ("selected split", identity.selected_split),
        ("selected case count", len(identity.selected_case_ids)),
        ("selected case IDs", ", ".join(identity.selected_case_ids)),
        ("corpus", f"{identity.corpus_id}/{identity.corpus_version}"),
        ("relationship snapshot", identity.relationship_snapshot_digest),
        (
            "evaluation implementation",
            f"{identity.implementation}@{identity.implementation_version}",
        ),
        ("result schema", identity.result_schema_version),
        (
            "adapter",
            f"{system.adapter_implementation}@{system.adapter_contract_version}",
        ),
        (
            "query construction",
            f"{system.query_construction_algorithm}@{system.query_construction_version}",
        ),
        (
            "retrieval algorithm",
            f"{system.retrieval_algorithm}@{system.retrieval_algorithm_version}",
        ),
        ("retrieval configuration", system.configuration_digest),
        ("system model identity", system.model_identity or "not recorded"),
        ("system model revision", system.model_revision or "not recorded"),
    )
    component_rows = tuple(
        (
            component.component,
            f"{component.implementation}@{component.format_version}",
            component.contract_version,
            component.configuration_digest,
            component.generation_id or "not recorded",
            component.source_manifest_digest or "not recorded",
            component.model_identity or "not recorded",
            component.model_revision or "not recorded",
        )
        for component in system.components
    )
    policy_rows = (
        ("document cutoff", policy.document_cutoff),
        (
            "passage cutoffs",
            ", ".join(str(cutoff) for cutoff in policy.passage_cutoffs),
        ),
        ("candidate depth", policy.candidate_depth),
        ("document policy", policy.document_policy_version),
        ("passage policy", policy.passage_policy_version),
        ("duplicate policy", policy.duplicate_policy_version),
        ("family policy", policy.family_policy_version),
        ("aggregation policy", policy.aggregation_policy_version),
        ("decimal precision", policy.decimal_precision),
        ("rounding mode", policy.rounding_mode),
    )
    return (
        "## Pinned identities and policy",
        "",
        _table(("identity", "value"), identity_rows),
        "",
        "### Retrieval components",
        "",
        _table(
            (
                "component",
                "implementation",
                "contract",
                "configuration",
                "generation",
                "source manifest",
                "model identity",
                "model revision",
            ),
            component_rows,
        ),
        "",
        "### Evaluation policy",
        "",
        _table(("policy", "value"), policy_rows),
        "",
    )


def _render_limitations(limitations: tuple[str, ...]) -> tuple[str, ...]:
    """Render explicit bounded limitations, including the empty case."""
    values = tuple(_markdown_cell(limitation) for limitation in limitations)
    return (
        "## Limitations",
        "",
        *(f"- {limitation}" for limitation in values or ("None recorded.",)),
        "",
    )


def _named_metrics(
    report: RetrievalEvaluationReport,
) -> tuple[tuple[str, AggregateMetric], ...]:
    """Return document then passage metrics in canonical cutoff order."""
    return (
        ("document_recall_at_10", report.aggregate.document),
        *tuple(
            (f"passage_recall_at_{metric.cutoff}", metric)
            for metric in report.aggregate.passage
        ),
    )


def _format_fraction(
    fraction: MetricFraction | None,
    policy: RetrievalEvaluationPolicy,
) -> str:
    """Format an exact fraction with its policy-controlled decimal view."""
    if fraction is None:
        return "not applicable"
    return (
        f"{fraction.numerator}/{fraction.denominator} "
        f"({fraction.decimal_value(precision=policy.decimal_precision, rounding_mode=policy.rounding_mode)})"
    )


def _table(headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    """Render a stable GitHub-Flavored Markdown table."""
    materialised_rows = tuple(rows)
    header_line = "| " + " | ".join(_markdown_cell(header) for header in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = tuple(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in materialised_rows
    )
    return "\n".join((header_line, separator_line, *body))


def _markdown_cell(value: object) -> str:
    """Keep table cells and limitation bullets structurally safe."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
