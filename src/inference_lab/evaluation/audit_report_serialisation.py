"""Canonical JSON, Markdown, and checksum manifests for audit reports."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import cast

from inference_lab._json import (
    JsonObject,
    JsonStructureError,
    JsonValue,
    canonical_json_bytes,
    canonical_json_text,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_boolean,
    require_json_integer,
    require_json_object,
    require_json_string,
)
from inference_lab.evaluation.audit_contracts import (
    AuditCoverage,
    AuditDistribution,
    AuditRate,
    AuditSourceFormatCoverage,
    AuditStructureKind,
    CorpusAuditPolicy,
)
from inference_lab.evaluation.audit_reports import (
    AuditAcquisitionReport,
    AuditCoverageSummary,
    AuditDiscoveryIdentity,
    AuditIntegerSummary,
    AuditOcrReport,
    AuditOverallStatus,
    AuditParseReport,
    AuditPopulationCounts,
    AuditReconciliation,
    AuditRelationshipReport,
    AuditSourceReport,
    AuditSpanReport,
    AuditStructureReport,
    AuditToolIdentities,
    CorpusAuditReport,
    report_content_digest,
)
from inference_lab.ingest.discovery import (
    DirectorySnapshotPolicy,
    DiscoveryPolicy,
    HiddenEntryPolicy,
    NestedMountPolicy,
    SourceSupport,
    StableReadPolicy,
    SymlinkPolicy,
)


def serialise_audit_report(report: CorpusAuditReport) -> bytes:
    """Return canonical UTF-8 JSON bytes for one validated report."""
    if not isinstance(report, CorpusAuditReport):
        raise TypeError("report must be a CorpusAuditReport.")
    return canonical_json_bytes(cast(JsonObject, asdict(report)))


def deserialise_audit_report(data: bytes) -> CorpusAuditReport:
    """Decode and strictly validate canonical report JSON bytes."""
    payload = load_json_object(data)
    report = _decode_report(payload)
    if serialise_audit_report(report) != data:
        raise ValueError("audit report JSON is not canonical.")
    if report.report_id != _expected_report_id(report):
        raise ValueError("audit report identity digest does not match its contents.")
    if report.report_digest != report_content_digest(report):
        raise ValueError("audit report content digest does not match its contents.")
    return report


def _decode_report(payload: JsonObject) -> CorpusAuditReport:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "schema_version",
                "report_id",
                "input_digest",
                "report_digest",
                "corpus_id",
                "corpus_version",
                "tools",
                "overall_status",
                "populations",
                "source",
                "parse",
                "acquisition",
                "relationships",
                "structures",
                "spans",
                "ocr",
                "limitations",
                "reconciliation",
            }
        ),
    )
    return CorpusAuditReport(
        schema_version=_string(payload["schema_version"]),
        report_id=_string(payload["report_id"]),
        input_digest=_string(payload["input_digest"]),
        report_digest=_string(payload["report_digest"]),
        corpus_id=_string(payload["corpus_id"]),
        corpus_version=_string(payload["corpus_version"]),
        tools=_decode_tools(_object(payload["tools"])),
        overall_status=AuditOverallStatus(_string(payload["overall_status"])),
        populations=_decode_populations(_object(payload["populations"])),
        source=_decode_source(_object(payload["source"])),
        parse=_decode_parse(_object(payload["parse"])),
        acquisition=_decode_acquisition(_object(payload["acquisition"])),
        relationships=_decode_relationships(_object(payload["relationships"])),
        structures=tuple(
            _decode_structure(_object(item)) for item in _array(payload["structures"])
        ),
        spans=_decode_spans(_object(payload["spans"])),
        ocr=_decode_ocr(_object(payload["ocr"])),
        limitations=tuple(_string(item) for item in _array(payload["limitations"])),
        reconciliation=_decode_reconciliation(_object(payload["reconciliation"])),
    )


def _decode_tools(payload: JsonObject) -> AuditToolIdentities:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "implementation",
                "implementation_version",
                "contract_version",
                "policy",
                "discovery",
                "parser_versions",
                "capability_profiles",
            }
        ),
    )
    return AuditToolIdentities(
        implementation=_string(payload["implementation"]),
        implementation_version=_string(payload["implementation_version"]),
        contract_version=_string(payload["contract_version"]),
        policy=_decode_policy(_object(payload["policy"])),
        discovery=_decode_discovery(_object(payload["discovery"])),
        parser_versions=tuple(
            _string(item) for item in _array(payload["parser_versions"])
        ),
        capability_profiles=tuple(
            _string(item) for item in _array(payload["capability_profiles"])
        ),
    )


def _decode_policy(payload: JsonObject) -> CorpusAuditPolicy:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "policy_name",
                "policy_version",
                "availability_version",
                "rate_decimal_places",
                "rate_rounding",
                "quantile_method",
                "duplicate_grouping_algorithm",
                "rendering_version",
            }
        ),
    )
    return CorpusAuditPolicy(
        policy_name=_string(payload["policy_name"]),
        policy_version=_string(payload["policy_version"]),
        availability_version=_string(payload["availability_version"]),
        rate_decimal_places=_integer(payload["rate_decimal_places"]),
        rate_rounding=_string(payload["rate_rounding"]),
        quantile_method=_string(payload["quantile_method"]),
        duplicate_grouping_algorithm=_string(payload["duplicate_grouping_algorithm"]),
        rendering_version=_string(payload["rendering_version"]),
    )


def _decode_discovery(payload: JsonObject) -> AuditDiscoveryIdentity:
    fields = (
        "policy_name",
        "policy_version",
        "symlink_policy",
        "nested_mount_policy",
        "hidden_entry_policy",
        "directory_snapshot_policy",
        "stable_read_policy",
        "classifier_name",
        "classifier_version",
    )
    require_exact_json_fields(payload, frozenset(fields))
    policy = DiscoveryPolicy(
        policy_name=_string(payload["policy_name"]),
        policy_version=_string(payload["policy_version"]),
        symlink_policy=SymlinkPolicy(_string(payload["symlink_policy"])),
        nested_mount_policy=NestedMountPolicy(_string(payload["nested_mount_policy"])),
        hidden_entry_policy=HiddenEntryPolicy(_string(payload["hidden_entry_policy"])),
        directory_snapshot_policy=DirectorySnapshotPolicy(
            _string(payload["directory_snapshot_policy"])
        ),
        stable_read_policy=StableReadPolicy(_string(payload["stable_read_policy"])),
        classifier_name=_string(payload["classifier_name"]),
        classifier_version=_string(payload["classifier_version"]),
    )
    return AuditDiscoveryIdentity.from_policy(policy)


def _decode_source(payload: JsonObject) -> AuditSourceReport:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "files_seen",
                "classified_files",
                "traversal_failures",
                "formats",
                "size_summary",
                "failure_counts",
                "reconciles",
            }
        ),
    )
    return AuditSourceReport(
        files_seen=_integer(payload["files_seen"]),
        classified_files=_integer(payload["classified_files"]),
        traversal_failures=_integer(payload["traversal_failures"]),
        formats=tuple(
            _decode_format(_object(item)) for item in _array(payload["formats"])
        ),
        size_summary=_decode_summary(_object(payload["size_summary"])),
        failure_counts=_decode_count_pairs(payload["failure_counts"]),
        reconciles=_boolean(payload["reconciles"]),
    )


def _decode_format(payload: JsonObject) -> AuditSourceFormatCoverage:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "media_type",
                "support",
                "file_count",
                "total_bytes",
                "hashed_count",
                "identity_count",
            }
        ),
    )
    return AuditSourceFormatCoverage(
        media_type=_string(payload["media_type"]),
        support=SourceSupport(_string(payload["support"])),
        file_count=_integer(payload["file_count"]),
        total_bytes=_integer(payload["total_bytes"]),
        hashed_count=_integer(payload["hashed_count"]),
        identity_count=_integer(payload["identity_count"]),
    )


def _decode_parse(payload: JsonObject) -> AuditParseReport:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "status_counts",
                "parser_counts",
                "media_type_counts",
                "failure_counts",
                "field_coverage",
                "language_counts",
                "numeric_summaries",
                "token_summaries",
            }
        ),
    )
    return AuditParseReport(
        status_counts=_decode_count_pairs(payload["status_counts"]),
        parser_counts=_decode_count_pairs(payload["parser_counts"]),
        media_type_counts=_decode_count_pairs(payload["media_type_counts"]),
        failure_counts=_decode_count_pairs(payload["failure_counts"]),
        field_coverage=_decode_coverage_pairs(payload["field_coverage"]),
        language_counts=_decode_count_pairs(payload["language_counts"]),
        numeric_summaries=_decode_summary_pairs(payload["numeric_summaries"]),
        token_summaries=_decode_summary_pairs(payload["token_summaries"]),
    )


def _decode_acquisition(payload: JsonObject) -> AuditAcquisitionReport:
    require_exact_json_fields(
        payload,
        frozenset({"status_counts", "coverage", "known_only_rate", "population_rate"}),
    )
    return AuditAcquisitionReport(
        status_counts=_decode_count_pairs(payload["status_counts"]),
        coverage=_decode_coverage_summary(_object(payload["coverage"])),
        known_only_rate=_decode_rate(_object(payload["known_only_rate"])),
        population_rate=_decode_rate(_object(payload["population_rate"])),
    )


def _decode_structure(payload: JsonObject) -> AuditStructureReport:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "kind",
                "coverage",
                "present_documents",
                "absent_documents",
                "unit_summary",
            }
        ),
    )
    return AuditStructureReport(
        kind=AuditStructureKind(_string(payload["kind"])),
        coverage=_decode_coverage_summary(_object(payload["coverage"])),
        present_documents=_integer(payload["present_documents"]),
        absent_documents=_integer(payload["absent_documents"]),
        unit_summary=_decode_summary(_object(payload["unit_summary"])),
    )


def _decode_spans(payload: JsonObject) -> AuditSpanReport:
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "block_summary",
                "source_span_count",
                "page_number_count",
                "bounding_box_count",
                "document_coverage",
            }
        ),
    )
    return AuditSpanReport(
        block_summary=_decode_summary(_object(payload["block_summary"])),
        source_span_count=_integer(payload["source_span_count"]),
        page_number_count=_integer(payload["page_number_count"]),
        bounding_box_count=_integer(payload["bounding_box_count"]),
        document_coverage=_decode_coverage_summary(
            _object(payload["document_coverage"])
        ),
    )


def _decode_ocr(payload: JsonObject) -> AuditOcrReport:
    fields = frozenset(
        {
            "coverage",
            "zero_text_page_count",
            "page_character_summary",
            "mixed_text_documents",
            "pages_with_layout",
            "pages_without_layout",
            "pages_with_boxes",
            "pages_without_boxes",
        }
    )
    require_exact_json_fields(payload, fields)
    return AuditOcrReport(
        coverage=_decode_coverage_summary(_object(payload["coverage"])),
        zero_text_page_count=_integer(payload["zero_text_page_count"]),
        page_character_summary=_decode_summary(
            _object(payload["page_character_summary"])
        ),
        mixed_text_documents=_integer(payload["mixed_text_documents"]),
        pages_with_layout=_integer(payload["pages_with_layout"]),
        pages_without_layout=_integer(payload["pages_without_layout"]),
        pages_with_boxes=_integer(payload["pages_with_boxes"]),
        pages_without_boxes=_integer(payload["pages_without_boxes"]),
    )


def _decode_relationships(payload: JsonObject) -> AuditRelationshipReport:
    fields = frozenset(
        {
            "duplicate_edge_count",
            "duplicate_participating_members",
            "duplicate_group_count",
            "duplicate_group_summary",
            "duplicate_excess_copies",
            "duplicate_participating_rate",
            "duplicate_excess_rate",
            "family_count",
            "family_document_count",
            "family_version_count",
            "unassigned_document_count",
            "family_size_summary",
            "document_level_memberships",
            "version_level_memberships",
            "family_candidate_count",
            "family_candidate_family_count",
        }
    )
    require_exact_json_fields(payload, fields)
    return AuditRelationshipReport(
        duplicate_edge_count=_integer(payload["duplicate_edge_count"]),
        duplicate_participating_members=_integer(
            payload["duplicate_participating_members"]
        ),
        duplicate_group_count=_integer(payload["duplicate_group_count"]),
        duplicate_group_summary=_decode_summary(
            _object(payload["duplicate_group_summary"])
        ),
        duplicate_excess_copies=_integer(payload["duplicate_excess_copies"]),
        duplicate_participating_rate=_decode_rate(
            _object(payload["duplicate_participating_rate"])
        ),
        duplicate_excess_rate=_decode_rate(_object(payload["duplicate_excess_rate"])),
        family_count=_integer(payload["family_count"]),
        family_document_count=_integer(payload["family_document_count"]),
        family_version_count=_integer(payload["family_version_count"]),
        unassigned_document_count=_integer(payload["unassigned_document_count"]),
        family_size_summary=_decode_summary(_object(payload["family_size_summary"])),
        document_level_memberships=_integer(payload["document_level_memberships"]),
        version_level_memberships=_integer(payload["version_level_memberships"]),
        family_candidate_count=_integer(payload["family_candidate_count"]),
        family_candidate_family_count=_integer(
            payload["family_candidate_family_count"]
        ),
    )


def _decode_populations(payload: JsonObject) -> AuditPopulationCounts:
    fields = frozenset(
        {
            "source_files_seen",
            "source_classified_files",
            "source_traversal_failures",
            "member_versions",
        }
    )
    require_exact_json_fields(payload, fields)
    return AuditPopulationCounts(
        source_files_seen=_integer(payload["source_files_seen"]),
        source_classified_files=_integer(payload["source_classified_files"]),
        source_traversal_failures=_integer(payload["source_traversal_failures"]),
        member_versions=_integer(payload["member_versions"]),
    )


def _decode_reconciliation(payload: JsonObject) -> AuditReconciliation:
    fields = frozenset(
        {
            "source_files_reconcile",
            "source_format_files",
            "source_traversal_failures",
            "member_versions_reconcile",
            "parse_observation_slots",
            "acquisition_observation_slots",
            "section_populations_reconcile",
        }
    )
    require_exact_json_fields(payload, fields)
    return AuditReconciliation(
        source_files_reconcile=_boolean(payload["source_files_reconcile"]),
        source_format_files=_integer(payload["source_format_files"]),
        source_traversal_failures=_integer(payload["source_traversal_failures"]),
        member_versions_reconcile=_boolean(payload["member_versions_reconcile"]),
        parse_observation_slots=_integer(payload["parse_observation_slots"]),
        acquisition_observation_slots=_integer(
            payload["acquisition_observation_slots"]
        ),
        section_populations_reconcile=_boolean(
            payload["section_populations_reconcile"]
        ),
    )


def _decode_summary(payload: JsonObject) -> AuditIntegerSummary:
    require_exact_json_fields(payload, frozenset({"distribution", "quantiles"}))
    quantiles = []
    for item in _array(payload["quantiles"]):
        pair = _array(item)
        if len(pair) != 2:
            raise ValueError("quantile entries must contain two values.")
        quantiles.append((_string(pair[0]), _optional_integer(pair[1])))
    return AuditIntegerSummary(
        distribution=_decode_distribution(_object(payload["distribution"])),
        quantiles=tuple(quantiles),
    )


def _decode_distribution(payload: JsonObject) -> AuditDistribution:
    require_exact_json_fields(
        payload, frozenset({"count", "total", "minimum", "maximum", "zero_count"})
    )
    return AuditDistribution(
        count=_integer(payload["count"]),
        total=_integer(payload["total"]),
        minimum=_optional_integer(payload["minimum"]),
        maximum=_optional_integer(payload["maximum"]),
        zero_count=_integer(payload["zero_count"]),
    )


def _decode_coverage_summary(payload: JsonObject) -> AuditCoverageSummary:
    require_exact_json_fields(payload, frozenset({"coverage", "reasons"}))
    return AuditCoverageSummary(
        coverage=_decode_coverage(_object(payload["coverage"])),
        reasons=_decode_count_pairs(payload["reasons"]),
    )


def _decode_coverage_pairs(
    value: JsonValue,
) -> tuple[tuple[str, AuditCoverageSummary], ...]:
    pairs = []
    for item in _array(value):
        pair = _array(item)
        if len(pair) != 2:
            raise ValueError("coverage entries must contain two values.")
        pairs.append((_string(pair[0]), _decode_coverage_summary(_object(pair[1]))))
    return tuple(pairs)


def _decode_summary_pairs(
    value: JsonValue,
) -> tuple[tuple[str, AuditIntegerSummary], ...]:
    pairs = []
    for item in _array(value):
        pair = _array(item)
        if len(pair) != 2:
            raise ValueError("summary entries must contain two values.")
        pairs.append((_string(pair[0]), _decode_summary(_object(pair[1]))))
    return tuple(pairs)


def _decode_count_pairs(value: JsonValue) -> tuple[tuple[str, int], ...]:
    pairs = []
    for item in _array(value):
        pair = _array(item)
        if len(pair) != 2:
            raise ValueError("count entries must contain two values.")
        pairs.append((_string(pair[0]), _integer(pair[1])))
    return tuple(pairs)


def _decode_coverage(payload: JsonObject) -> AuditCoverage:
    fields = frozenset(
        {
            "observed",
            "observed_absent",
            "unavailable",
            "missing",
            "failed",
            "invalid",
            "population",
        }
    )
    require_exact_json_fields(payload, fields)
    return AuditCoverage(
        observed=_integer(payload["observed"]),
        observed_absent=_integer(payload["observed_absent"]),
        unavailable=_integer(payload["unavailable"]),
        missing=_integer(payload["missing"]),
        failed=_integer(payload["failed"]),
        invalid=_integer(payload["invalid"]),
        population=_integer(payload["population"]),
    )


def _decode_rate(payload: JsonObject) -> AuditRate:
    require_exact_json_fields(payload, frozenset({"numerator", "denominator"}))
    return AuditRate(
        numerator=_integer(payload["numerator"]),
        denominator=_integer(payload["denominator"]),
    )


def _object(value: JsonValue) -> JsonObject:
    try:
        return require_json_object(value)
    except JsonStructureError:
        raise ValueError("value must be a JSON object.") from None


def _array(value: JsonValue) -> list[JsonValue]:
    try:
        return require_json_array(value)
    except JsonStructureError:
        raise ValueError("value must be a JSON array.") from None


def _string(value: JsonValue) -> str:
    try:
        return require_json_string(value)
    except JsonStructureError:
        raise ValueError("value must be a JSON string.") from None


def _integer(value: JsonValue) -> int:
    try:
        return require_json_integer(value)
    except JsonStructureError:
        raise ValueError("value must be a JSON integer.") from None


def _optional_integer(value: JsonValue) -> int | None:
    return None if value is None else _integer(value)


def _boolean(value: JsonValue) -> bool:
    try:
        return require_json_boolean(value)
    except JsonStructureError:
        raise ValueError("value must be a JSON boolean.") from None


def _expected_report_id(report: CorpusAuditReport) -> str:
    payload = {
        "schema_version": report.schema_version,
        "corpus_id": report.corpus_id,
        "corpus_version": report.corpus_version,
        "input_digest": report.input_digest,
        "implementation": report.tools.implementation,
        "implementation_version": report.tools.implementation_version,
        "contract_version": report.tools.contract_version,
        "policy": asdict(report.tools.policy),
        "capability_profiles": list(report.tools.capability_profiles),
    }
    encoded = canonical_json_text(cast(JsonObject, payload)).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
