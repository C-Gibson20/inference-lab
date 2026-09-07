"""Immutable aggregate corpus-audit reports and deterministic identities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Final

from inference_lab.evaluation.audit_aggregation import (
    _AcquisitionAggregates,
    _CorpusAuditAggregates,
    _CoverageSummary,
    _IntegerSummary,
    _OcrAggregates,
    _ParseAggregates,
    _RelationshipAggregates,
    _SourceAggregates,
    _SpanAggregates,
    _StructureAggregate,
    aggregate_corpus_audit,
)
from inference_lab.evaluation.audit_capabilities import ParserCapabilityProfile
from inference_lab.evaluation.audit_contracts import (
    AUDIT_CONTRACT_VERSION,
    AuditAcquisitionObservation,
    AuditCoverage,
    AuditDistribution,
    AuditParsedDocumentFacts,
    AuditParseEvidence,
    AuditParseObservation,
    AuditRate,
    AuditSourceFailure,
    AuditSourceFormatCoverage,
    AuditStructureKind,
    CorpusAuditPolicy,
    CorpusAuditRequest,
)
from inference_lab.ingest.discovery import DiscoveryPolicy
from inference_lab.provenance import CorpusId, CorpusVersionId

AUDIT_REPORT_SCHEMA_VERSION: Final = "1.0.0"
AUDIT_IMPLEMENTATION: Final = "inference-lab"
AUDIT_IMPLEMENTATION_VERSION: Final = "0.1.0"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class AuditOverallStatus(StrEnum):
    """Describe whether the report has complete or limited measurements."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_OBSERVATIONS = "no_observations"


@dataclass(frozen=True, slots=True)
class AuditIntegerSummary:
    """Expose an exact integer distribution and named quantiles."""

    distribution: AuditDistribution
    quantiles: tuple[tuple[str, int | None], ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate distribution and quantile names and values."""
        if not isinstance(self.distribution, AuditDistribution):
            raise TypeError("distribution must be an AuditDistribution.")
        if not isinstance(self.quantiles, tuple):
            raise TypeError("quantiles must be a tuple.")
        names = tuple(name for name, _ in self.quantiles)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("quantile names must be ordered and unique.")
        for name, value in self.quantiles:
            if not isinstance(name, str) or not name:
                raise ValueError("quantile names must be non-empty strings.")
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError("quantile values must be non-negative integers.")


@dataclass(frozen=True, slots=True)
class AuditCoverageSummary:
    """Pair reconciled coverage counts with bounded state counts."""

    coverage: AuditCoverage
    reasons: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate coverage and reason values."""
        if not isinstance(self.coverage, AuditCoverage):
            raise TypeError("coverage must be an AuditCoverage.")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple.")
        keys = tuple(key for key, _ in self.reasons)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("coverage reasons must be ordered and unique.")
        for key, count in self.reasons:
            if not isinstance(key, str) or not key:
                raise ValueError("coverage reason keys must be non-empty strings.")
            if not isinstance(count, int) or count <= 0:
                raise ValueError("coverage reason counts must be positive integers.")


@dataclass(frozen=True, slots=True)
class AuditDiscoveryIdentity:
    """Record the path-free discovery and classifier policy identity."""

    policy_name: str
    policy_version: str
    symlink_policy: str
    nested_mount_policy: str
    hidden_entry_policy: str
    directory_snapshot_policy: str
    stable_read_policy: str
    classifier_name: str
    classifier_version: str

    @classmethod
    def from_policy(cls, policy: DiscoveryPolicy) -> "AuditDiscoveryIdentity":
        """Project a discovery policy into report-safe scalar fields."""
        if not isinstance(policy, DiscoveryPolicy):
            raise TypeError("policy must be a DiscoveryPolicy.")
        return cls(
            policy_name=policy.policy_name,
            policy_version=policy.policy_version,
            symlink_policy=policy.symlink_policy.value,
            nested_mount_policy=policy.nested_mount_policy.value,
            hidden_entry_policy=policy.hidden_entry_policy.value,
            directory_snapshot_policy=policy.directory_snapshot_policy.value,
            stable_read_policy=policy.stable_read_policy.value,
            classifier_name=policy.classifier_name,
            classifier_version=policy.classifier_version,
        )


@dataclass(frozen=True, slots=True)
class AuditToolIdentities:
    """Identify every versioned tool and policy used by the report."""

    implementation: str
    implementation_version: str
    contract_version: str
    policy: CorpusAuditPolicy
    discovery: AuditDiscoveryIdentity
    parser_versions: tuple[str, ...]
    capability_profiles: tuple[str, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate tool identity fields and version uniqueness."""
        if not isinstance(self.policy, CorpusAuditPolicy):
            raise TypeError("policy must be a CorpusAuditPolicy.")
        for field_name in (
            "implementation",
            "implementation_version",
            "contract_version",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string.")
        if self.implementation != AUDIT_IMPLEMENTATION:
            raise ValueError("unsupported audit implementation.")
        if self.implementation_version != AUDIT_IMPLEMENTATION_VERSION:
            raise ValueError("unsupported audit implementation version.")
        if self.contract_version != AUDIT_CONTRACT_VERSION:
            raise ValueError("unsupported audit contract version.")
        if not isinstance(self.discovery, AuditDiscoveryIdentity):
            raise TypeError("discovery must be an AuditDiscoveryIdentity.")
        if not isinstance(self.parser_versions, tuple):
            raise TypeError("parser_versions must be a tuple.")
        if any(not isinstance(value, str) for value in self.parser_versions):
            raise TypeError("parser_versions must contain strings.")
        if tuple(self.parser_versions) != tuple(sorted(set(self.parser_versions))):
            raise ValueError("parser_versions must be ordered and unique.")
        if not isinstance(self.capability_profiles, tuple):
            raise TypeError("capability_profiles must be a tuple.")
        if any(not isinstance(value, str) for value in self.capability_profiles):
            raise TypeError("capability_profiles must contain strings.")
        if tuple(self.capability_profiles) != tuple(
            sorted(set(self.capability_profiles))
        ):
            raise ValueError("capability_profiles must be ordered and unique.")


@dataclass(frozen=True, slots=True)
class AuditSourceReport:
    """Aggregate source-intake format, size, and failure evidence."""

    files_seen: int
    classified_files: int
    traversal_failures: int
    formats: tuple[AuditSourceFormatCoverage, ...]
    size_summary: AuditIntegerSummary
    failure_counts: tuple[tuple[str, int], ...]
    reconciles: bool


@dataclass(frozen=True, slots=True)
class AuditParseReport:
    """Aggregate parser outcomes and field-level availability."""

    status_counts: tuple[tuple[str, int], ...]
    parser_counts: tuple[tuple[str, int], ...]
    media_type_counts: tuple[tuple[str, int], ...]
    failure_counts: tuple[tuple[str, int], ...]
    field_coverage: tuple[tuple[str, AuditCoverageSummary], ...]
    language_counts: tuple[tuple[str, int], ...]
    numeric_summaries: tuple[tuple[str, AuditIntegerSummary], ...]
    token_summaries: tuple[tuple[str, AuditIntegerSummary], ...]


@dataclass(frozen=True, slots=True)
class AuditAcquisitionReport:
    """Aggregate known and unknown acquisition-status observations."""

    status_counts: tuple[tuple[str, int], ...]
    coverage: AuditCoverageSummary
    known_only_rate: AuditRate
    population_rate: AuditRate


@dataclass(frozen=True, slots=True)
class AuditStructureReport:
    """Aggregate one capability-aware structural observation kind."""

    kind: AuditStructureKind
    coverage: AuditCoverageSummary
    present_documents: int
    absent_documents: int
    unit_summary: AuditIntegerSummary


@dataclass(frozen=True, slots=True)
class AuditSpanReport:
    """Aggregate source-span and page/box provenance coverage."""

    block_summary: AuditIntegerSummary
    source_span_count: int
    page_number_count: int
    bounding_box_count: int
    document_coverage: AuditCoverageSummary


@dataclass(frozen=True, slots=True)
class AuditOcrReport:
    """Aggregate raw page indicators without assigning scan labels."""

    coverage: AuditCoverageSummary
    zero_text_page_count: int
    page_character_summary: AuditIntegerSummary
    mixed_text_documents: int
    pages_with_layout: int
    pages_without_layout: int
    pages_with_boxes: int
    pages_without_boxes: int


@dataclass(frozen=True, slots=True)
class AuditRelationshipReport:
    """Aggregate in-scope duplicates, families, and advisory candidates."""

    duplicate_edge_count: int
    duplicate_participating_members: int
    duplicate_group_count: int
    duplicate_group_summary: AuditIntegerSummary
    duplicate_excess_copies: int
    duplicate_participating_rate: AuditRate
    duplicate_excess_rate: AuditRate
    family_count: int
    family_document_count: int
    family_version_count: int
    unassigned_document_count: int
    family_size_summary: AuditIntegerSummary
    document_level_memberships: int
    version_level_memberships: int
    family_candidate_count: int
    family_candidate_family_count: int


@dataclass(frozen=True, slots=True)
class AuditPopulationCounts:
    """State both report denominators explicitly."""

    source_files_seen: int
    source_classified_files: int
    source_traversal_failures: int
    member_versions: int


@dataclass(frozen=True, slots=True)
class AuditReconciliation:
    """Prove that source and member observations were retained in sections."""

    source_files_reconcile: bool
    source_format_files: int
    source_traversal_failures: int
    member_versions_reconcile: bool
    parse_observation_slots: int
    acquisition_observation_slots: int
    section_populations_reconcile: bool


@dataclass(frozen=True, slots=True, repr=False)
class CorpusAuditReport:
    """Validated aggregate-only report for exactly one corpus version."""

    schema_version: str
    report_id: str
    input_digest: str
    corpus_id: str
    corpus_version: str
    tools: AuditToolIdentities
    overall_status: AuditOverallStatus
    populations: AuditPopulationCounts
    source: AuditSourceReport
    parse: AuditParseReport
    acquisition: AuditAcquisitionReport
    relationships: AuditRelationshipReport
    structures: tuple[AuditStructureReport, ...]
    spans: AuditSpanReport
    ocr: AuditOcrReport
    limitations: tuple[str, ...]
    reconciliation: AuditReconciliation
    report_digest: str = ""

    def __post_init__(self) -> None:
        if self.report_digest == "":
            object.__setattr__(self, "report_digest", _report_content_digest(self))
        self._validate()

    def _validate(self) -> None:
        """Validate report identity, denominators, and section reconciliation."""
        if self.schema_version != AUDIT_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported audit report schema version.")
        for field_name in ("report_id", "input_digest", "report_digest"):
            value = getattr(self, field_name)
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a SHA-256 identity.")
        if not isinstance(self.corpus_id, str) or not isinstance(
            self.corpus_version, str
        ):
            raise TypeError("corpus and corpus-version identities must be strings.")
        CorpusId.from_str(self.corpus_id)
        CorpusVersionId.from_str(self.corpus_version)
        if not isinstance(self.tools, AuditToolIdentities):
            raise TypeError("tools must be AuditToolIdentities.")
        if not isinstance(self.overall_status, AuditOverallStatus):
            raise TypeError("overall_status must be an AuditOverallStatus.")
        if not isinstance(self.populations, AuditPopulationCounts):
            raise TypeError("populations must be AuditPopulationCounts.")
        if not isinstance(self.source, AuditSourceReport):
            raise TypeError("source must be AuditSourceReport.")
        if not isinstance(self.parse, AuditParseReport):
            raise TypeError("parse must be AuditParseReport.")
        if not isinstance(self.acquisition, AuditAcquisitionReport):
            raise TypeError("acquisition must be AuditAcquisitionReport.")
        if not isinstance(self.relationships, AuditRelationshipReport):
            raise TypeError("relationships must be AuditRelationshipReport.")
        if not isinstance(self.spans, AuditSpanReport):
            raise TypeError("spans must be AuditSpanReport.")
        if not isinstance(self.ocr, AuditOcrReport):
            raise TypeError("ocr must be AuditOcrReport.")
        if not isinstance(self.reconciliation, AuditReconciliation):
            raise TypeError("reconciliation must be AuditReconciliation.")
        for field_name in (
            "source_files_seen",
            "source_classified_files",
            "source_traversal_failures",
            "member_versions",
        ):
            _validate_nonnegative(
                getattr(self.populations, field_name), field_name=field_name
            )
        _validate_count_pairs(self.source.failure_counts, field_name="source failures")
        _validate_count_pairs(self.parse.status_counts, field_name="parse statuses")
        _validate_count_pairs(self.parse.parser_counts, field_name="parser counts")
        _validate_count_pairs(
            self.parse.media_type_counts, field_name="media type counts"
        )
        _validate_count_pairs(self.parse.failure_counts, field_name="parse failures")
        _validate_count_pairs(self.parse.language_counts, field_name="language counts")
        _validate_summary_pairs(
            self.parse.numeric_summaries, field_name="numeric summaries"
        )
        _validate_summary_pairs(
            self.parse.token_summaries, field_name="token summaries"
        )
        if any(
            not isinstance(name, str) or not isinstance(summary, AuditCoverageSummary)
            for name, summary in self.parse.field_coverage
        ):
            raise TypeError("parse field coverage has invalid values.")
        field_names = tuple(name for name, _ in self.parse.field_coverage)
        if field_names != (
            "language",
            "page_count",
            "character_count",
            "token_count",
        ):
            raise ValueError("parse field coverage must use canonical fields.")
        if tuple(name for name, _ in self.parse.numeric_summaries) != (
            "page_count",
            "character_count",
            "token_count",
        ):
            raise ValueError("numeric summaries must use canonical fields.")
        if len({item.kind for item in self.structures}) != len(self.structures):
            raise ValueError("structure reports must be unique.")
        if any(not isinstance(item, AuditStructureReport) for item in self.structures):
            raise TypeError("structures must contain AuditStructureReport values.")
        if tuple(item.kind for item in self.structures) != tuple(AuditStructureKind):
            raise ValueError("structure reports must use canonical ordering.")
        if any(
            not isinstance(item, AuditSourceFormatCoverage)
            for item in self.source.formats
        ):
            raise TypeError("source formats must contain coverage values.")
        format_keys = tuple(item.key() for item in self.source.formats)
        if format_keys != tuple(sorted(format_keys)):
            raise ValueError("source formats must use canonical ordering.")
        _validate_count_pairs(
            self.acquisition.status_counts, field_name="acquisition statuses"
        )
        if not isinstance(self.limitations, tuple):
            raise TypeError("limitations must be a tuple.")
        if any(not isinstance(item, str) or not item for item in self.limitations):
            raise ValueError("limitations must contain non-empty strings.")
        if self.limitations != tuple(sorted(set(self.limitations))):
            raise ValueError("limitations must be ordered and unique.")
        for field_name in (
            "source_files_reconcile",
            "member_versions_reconcile",
            "section_populations_reconcile",
        ):
            if not isinstance(getattr(self.reconciliation, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")
        for field_name in (
            "source_format_files",
            "source_traversal_failures",
            "parse_observation_slots",
            "acquisition_observation_slots",
        ):
            _validate_nonnegative(
                getattr(self.reconciliation, field_name), field_name=field_name
            )
        if self.source.files_seen != (
            self.source.classified_files + self.source.traversal_failures
        ):
            raise ValueError("source report does not reconcile files seen.")
        if sum(item.file_count for item in self.source.formats) != (
            self.source.classified_files
        ):
            raise ValueError("source report does not reconcile format files.")
        member_count = self.populations.member_versions
        if any(
            summary.coverage.population != member_count
            for _, summary in self.parse.field_coverage
        ):
            raise ValueError("parse field coverage has the wrong denominator.")
        if self.acquisition.coverage.coverage.population != member_count:
            raise ValueError("acquisition coverage has the wrong denominator.")
        if any(
            item.coverage.coverage.population != member_count
            for item in self.structures
        ):
            raise ValueError("structure coverage has the wrong denominator.")
        if self.spans.document_coverage.coverage.population != member_count:
            raise ValueError("span coverage has the wrong denominator.")
        if self.ocr.coverage.coverage.population != member_count:
            raise ValueError("OCR coverage has the wrong denominator.")
        if not self.reconciliation.source_files_reconcile:
            raise ValueError("source reconciliation failed.")
        if not self.reconciliation.member_versions_reconcile:
            raise ValueError("member reconciliation failed.")
        if not self.reconciliation.section_populations_reconcile:
            raise ValueError("section reconciliation failed.")
        if self.reconciliation.source_format_files != self.source.classified_files:
            raise ValueError("source reconciliation totals disagree.")
        if self.reconciliation.source_traversal_failures != (
            self.source.traversal_failures
        ):
            raise ValueError("source failure totals disagree.")
        if self.reconciliation.parse_observation_slots != member_count:
            raise ValueError("parse observation slots do not reconcile.")
        if self.report_digest != _report_content_digest(self):
            raise ValueError("audit report digest does not match its contents.")

    @classmethod
    def from_request(cls, request: CorpusAuditRequest) -> "CorpusAuditReport":
        """Aggregate and identity one validated audit request."""
        if not isinstance(request, CorpusAuditRequest):
            raise TypeError("request must be a CorpusAuditRequest.")
        aggregates = aggregate_corpus_audit(request)
        if not aggregates.source.reconciles:
            raise ValueError("source inventory does not reconcile.")
        input_digest = _semantic_digest(request)
        report_id = _report_identity(request, input_digest)
        member_count = len(request.scope.memberships)
        parser_versions = tuple(
            sorted(
                {
                    observation.parser_version.value
                    for observation in request.parse_observations
                    if observation.parser_version is not None
                }
            )
        )
        # A complete request with no parser observations is valid for an empty
        # corpus. Keep the identity shape explicit without inventing a parser.
        capability_profiles = tuple(
            sorted(profile.value for profile in request.capability_profiles)
        )
        source = _public_source(aggregates.source)
        parse = _public_parse(aggregates.parse)
        acquisition = _public_acquisition(aggregates.acquisition)
        structures = tuple(_public_structure(item) for item in aggregates.structures)
        spans = _public_spans(aggregates.spans)
        ocr = _public_ocr(aggregates.ocr)
        relationships = _public_relationships(aggregates.relationships)
        section_populations_reconcile = (
            all(
                summary.coverage.population == member_count
                for _, summary in parse.field_coverage
            )
            and acquisition.coverage.coverage.population == member_count
            and all(
                item.coverage.coverage.population == member_count for item in structures
            )
            and spans.document_coverage.coverage.population == member_count
            and ocr.coverage.coverage.population == member_count
        )
        reconciliation = AuditReconciliation(
            source_files_reconcile=source.reconciles,
            source_format_files=source.classified_files,
            source_traversal_failures=source.traversal_failures,
            member_versions_reconcile=(len(request.parse_observations) == member_count),
            parse_observation_slots=len(request.parse_observations),
            acquisition_observation_slots=len(request.acquisition_observations),
            section_populations_reconcile=section_populations_reconcile,
        )
        status = _overall_status(request, aggregates)
        return cls(
            schema_version=AUDIT_REPORT_SCHEMA_VERSION,
            report_id=report_id,
            input_digest=input_digest,
            corpus_id=request.scope.corpus_id.as_str(),
            corpus_version=request.scope.corpus_version.as_str(),
            tools=AuditToolIdentities(
                implementation=AUDIT_IMPLEMENTATION,
                implementation_version=AUDIT_IMPLEMENTATION_VERSION,
                contract_version=AUDIT_CONTRACT_VERSION,
                policy=request.policy,
                discovery=AuditDiscoveryIdentity.from_policy(
                    request.source_inventory.policy
                ),
                parser_versions=parser_versions,
                capability_profiles=capability_profiles,
            ),
            overall_status=status,
            populations=AuditPopulationCounts(
                source_files_seen=source.files_seen,
                source_classified_files=source.classified_files,
                source_traversal_failures=source.traversal_failures,
                member_versions=member_count,
            ),
            source=source,
            parse=parse,
            acquisition=acquisition,
            relationships=relationships,
            structures=structures,
            spans=spans,
            ocr=ocr,
            limitations=aggregates.limitations,
            reconciliation=reconciliation,
        )

    def __repr__(self) -> str:
        """Return only safe report identity and denominator metadata."""
        return (
            "CorpusAuditReport("
            f"report_id={self.report_id!r}, "
            f"corpus_id={self.corpus_id!r}, "
            f"corpus_version={self.corpus_version!r}, "
            f"member_versions={self.populations.member_versions}, "
            f"status={self.overall_status.value!r})"
        )


def _public_integer_summary(summary: _IntegerSummary) -> AuditIntegerSummary:
    return AuditIntegerSummary(summary.distribution, summary.quantiles)


def _public_coverage_summary(summary: _CoverageSummary) -> AuditCoverageSummary:
    return AuditCoverageSummary(summary.coverage, summary.reasons)


def _public_source(source: _SourceAggregates) -> AuditSourceReport:
    return AuditSourceReport(
        files_seen=source.files_seen,
        classified_files=source.classified_files,
        traversal_failures=source.traversal_failures,
        formats=source.formats,
        size_summary=_public_integer_summary(source.size_summary),
        failure_counts=source.failure_counts,
        reconciles=source.reconciles,
    )


def _public_parse(parse: _ParseAggregates) -> AuditParseReport:
    return AuditParseReport(
        status_counts=parse.status_counts,
        parser_counts=parse.parser_counts,
        media_type_counts=parse.media_type_counts,
        failure_counts=parse.failure_counts,
        field_coverage=tuple(
            (name, _public_coverage_summary(summary))
            for name, summary in parse.field_coverage
        ),
        language_counts=parse.language_counts,
        numeric_summaries=tuple(
            (name, _public_integer_summary(summary))
            for name, summary in parse.numeric_summaries
        ),
        token_summaries=tuple(
            (name, _public_integer_summary(summary))
            for name, summary in parse.token_summaries
        ),
    )


def _public_acquisition(
    acquisition: _AcquisitionAggregates,
) -> AuditAcquisitionReport:
    return AuditAcquisitionReport(
        status_counts=acquisition.status_counts,
        coverage=_public_coverage_summary(acquisition.coverage),
        known_only_rate=acquisition.known_only_rate,
        population_rate=acquisition.population_rate,
    )


def _public_structure(structure: _StructureAggregate) -> AuditStructureReport:
    return AuditStructureReport(
        kind=structure.kind,
        coverage=_public_coverage_summary(structure.coverage),
        present_documents=structure.present_documents,
        absent_documents=structure.absent_documents,
        unit_summary=_public_integer_summary(structure.unit_summary),
    )


def _public_spans(spans: _SpanAggregates) -> AuditSpanReport:
    return AuditSpanReport(
        block_summary=_public_integer_summary(spans.block_summary),
        source_span_count=spans.source_span_count,
        page_number_count=spans.page_number_count,
        bounding_box_count=spans.bounding_box_count,
        document_coverage=_public_coverage_summary(spans.document_coverage),
    )


def _public_ocr(ocr: _OcrAggregates) -> AuditOcrReport:
    return AuditOcrReport(
        coverage=_public_coverage_summary(ocr.coverage),
        zero_text_page_count=ocr.zero_text_page_count,
        page_character_summary=_public_integer_summary(ocr.page_character_summary),
        mixed_text_documents=ocr.mixed_text_documents,
        pages_with_layout=ocr.pages_with_layout,
        pages_without_layout=ocr.pages_without_layout,
        pages_with_boxes=ocr.pages_with_boxes,
        pages_without_boxes=ocr.pages_without_boxes,
    )


def _public_relationships(
    relationships: _RelationshipAggregates,
) -> AuditRelationshipReport:
    return AuditRelationshipReport(
        duplicate_edge_count=relationships.duplicate_edge_count,
        duplicate_participating_members=relationships.duplicate_participating_members,
        duplicate_group_count=relationships.duplicate_group_count,
        duplicate_group_summary=_public_integer_summary(
            relationships.duplicate_group_summary
        ),
        duplicate_excess_copies=relationships.duplicate_excess_copies,
        duplicate_participating_rate=relationships.duplicate_participating_rate,
        duplicate_excess_rate=relationships.duplicate_excess_rate,
        family_count=relationships.family_count,
        family_document_count=relationships.family_document_count,
        family_version_count=relationships.family_version_count,
        unassigned_document_count=relationships.unassigned_document_count,
        family_size_summary=_public_integer_summary(relationships.family_size_summary),
        document_level_memberships=relationships.document_level_memberships,
        version_level_memberships=relationships.version_level_memberships,
        family_candidate_count=relationships.family_candidate_count,
        family_candidate_family_count=relationships.family_candidate_family_count,
    )


def _validate_nonnegative(value: object, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _validate_count_pairs(
    values: tuple[tuple[str, int], ...], *, field_name: str
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple.")
    keys = tuple(key for key, _ in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be ordered and unique.")
    for key, value in values:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        _validate_nonnegative(value, field_name=field_name)


def _validate_summary_pairs(
    values: tuple[tuple[str, AuditIntegerSummary], ...], *, field_name: str
) -> None:
    keys = tuple(key for key, _ in values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be unique.")
    for key, summary in values:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        if not isinstance(summary, AuditIntegerSummary):
            raise TypeError(f"{field_name} must contain audit summaries.")


def _overall_status(
    request: CorpusAuditRequest,
    aggregates: _CorpusAuditAggregates,
) -> AuditOverallStatus:
    if not request.scope.memberships and aggregates.source.files_seen == 0:
        return AuditOverallStatus.NO_OBSERVATIONS
    return (
        AuditOverallStatus.PARTIAL
        if aggregates.limitations
        else AuditOverallStatus.COMPLETE
    )


def report_content_digest(report: CorpusAuditReport) -> str:
    """Return the digest of canonical report content excluding its digest field."""
    if not isinstance(report, CorpusAuditReport):
        raise TypeError("report must be a CorpusAuditReport.")
    return _report_content_digest(report)


def _report_content_digest(report: CorpusAuditReport) -> str:
    payload = asdict(report)
    payload.pop("report_digest", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _semantic_digest(request: CorpusAuditRequest) -> str:
    payload = _semantic_input(request)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _report_identity(request: CorpusAuditRequest, input_digest: str) -> str:
    payload = {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "corpus_id": request.scope.corpus_id.as_str(),
        "corpus_version": request.scope.corpus_version.as_str(),
        "input_digest": input_digest,
        "implementation": AUDIT_IMPLEMENTATION,
        "implementation_version": AUDIT_IMPLEMENTATION_VERSION,
        "contract_version": AUDIT_CONTRACT_VERSION,
        "policy": _policy_payload(request.policy),
        "capability_profiles": [
            profile.value for profile in request.capability_profiles
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _semantic_input(request: CorpusAuditRequest) -> dict[str, Any]:
    """Build the complete payload-free semantic identity input."""
    return {
        "contract_version": AUDIT_CONTRACT_VERSION,
        "scope": {
            "corpus_id": request.scope.corpus_id.as_str(),
            "corpus_version": request.scope.corpus_version.as_str(),
            "memberships": [
                {
                    "corpus_id": item.corpus_id.as_str(),
                    "document_id": item.document_id.as_str(),
                    "document_version_id": item.document_version_id.as_str(),
                }
                for item in request.scope.memberships
            ],
        },
        "source_inventory": _source_inventory_payload(request),
        "parse_observations": [
            _parse_observation_payload(item) for item in request.parse_observations
        ],
        "acquisition_observations": [
            _acquisition_payload(item) for item in request.acquisition_observations
        ],
        "relationships": _relationship_payload(request),
        "capability_profiles": [
            _profile_payload(profile) for profile in request.capability_profiles
        ],
        "policy": _policy_payload(request.policy),
    }


def _source_inventory_payload(request: CorpusAuditRequest) -> dict[str, Any]:
    inventory = request.source_inventory
    return {
        "corpus_id": inventory.corpus_id.as_str(),
        "policy": _discovery_payload(inventory.policy),
        "format_counts": [
            {
                "media_type": item.media_type,
                "support": item.support.value,
                "file_count": item.file_count,
                "total_bytes": item.total_bytes,
            }
            for item in inventory.format_counts
        ],
        "format_coverage": [
            _format_coverage_payload(item) for item in inventory.format_coverage
        ],
        "discovery_counts": {
            field: getattr(inventory.discovery_counts, field)
            for field in (
                "files_seen",
                "supported",
                "unsupported",
                "hashed",
                "unchanged",
                "changed",
                "new",
                "duplicate",
                "ambiguous",
                "conflicting",
                "failed",
            )
        },
        "failures": [_source_failure_payload(item) for item in inventory.failures],
        "known_source_byte_counts": list(inventory.known_source_byte_counts),
    }


def _parse_observation_payload(observation: AuditParseObservation) -> dict[str, Any]:
    return {
        "membership": {
            "document_id": observation.membership.document_id.as_str(),
            "document_version_id": observation.membership.document_version_id.as_str(),
        },
        "parser_version": (
            observation.parser_version.value
            if observation.parser_version is not None
            else None
        ),
        "media_type": observation.media_type,
        "source_byte_count": observation.source_byte_count,
        "evidence": _evidence_payload(observation.evidence),
        "missing_reason": (
            observation.missing_reason.value
            if observation.missing_reason is not None
            else None
        ),
    }


def _evidence_payload(evidence: AuditParseEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "status": evidence.status,
        "document": _document_facts_payload(evidence.document),
        "failures": [
            {
                "category": item.category,
                "severity": item.severity,
                "stage": item.stage,
                "unit_kind": item.unit_kind,
                "unit_index": item.unit_index,
            }
            for item in evidence.failures
        ],
    }


def _document_facts_payload(
    document: AuditParsedDocumentFacts | None,
) -> dict[str, Any] | None:
    if document is None:
        return None
    return {
        "language": document.language,
        "page_count": document.page_count,
        "character_count": document.character_count,
        "token_count": document.token_count,
        "structure_counts": [
            {"kind": item.kind.value, "count": item.count}
            for item in document.structure_counts
        ],
        "available_structure_kinds": sorted(
            item.value for item in document.available_structure_kinds
        ),
        "source_span_count": document.source_span_count,
        "page_number_count": document.page_number_count,
        "page_character_counts": (
            list(document.page_character_counts)
            if document.page_character_counts is not None
            else None
        ),
        "page_layout_count": document.page_layout_count,
        "page_layout_page_count": document.page_layout_page_count,
        "bounding_box_count": document.bounding_box_count,
        "bounding_box_page_count": document.bounding_box_page_count,
    }


def _acquisition_payload(observation: AuditAcquisitionObservation) -> dict[str, Any]:
    return {
        "document_id": observation.membership.document_id.as_str(),
        "document_version_id": observation.membership.document_version_id.as_str(),
        "status": observation.status.value,
        "producer_name": observation.producer_name,
        "producer_version": observation.producer_version,
        "unknown_reason": (
            observation.unknown_reason.value
            if observation.unknown_reason is not None
            else None
        ),
    }


def _relationship_payload(request: CorpusAuditRequest) -> dict[str, Any]:
    relationships = request.relationships
    return {
        "duplicate_relationships": [
            {
                "first_version_id": item.first_version_id.as_str(),
                "second_version_id": item.second_version_id.as_str(),
                "duplicate_class": item.duplicate_class.value,
            }
            for item in relationships.duplicate_relationships
        ],
        "family_memberships": [
            {
                "family_id": item.family_id.as_str(),
                "document_id": item.document_id.as_str(),
                "document_version_id": (
                    item.document_version_id.as_str()
                    if item.document_version_id is not None
                    else None
                ),
            }
            for item in relationships.family_memberships
        ],
        "family_candidates": [
            {
                "family_id": item.document_family_id.as_str(),
                "evidence_kind": item.evidence_kind.value,
                "document_id": (
                    item.document_id.as_str() if item.document_id is not None else None
                ),
                "document_version_id": (
                    item.document_version_id.as_str()
                    if item.document_version_id is not None
                    else None
                ),
            }
            for item in relationships.family_candidates
        ],
    }


def _discovery_payload(policy: DiscoveryPolicy) -> dict[str, str]:
    return {
        "policy_name": policy.policy_name,
        "policy_version": policy.policy_version,
        "symlink_policy": policy.symlink_policy.value,
        "nested_mount_policy": policy.nested_mount_policy.value,
        "hidden_entry_policy": policy.hidden_entry_policy.value,
        "directory_snapshot_policy": policy.directory_snapshot_policy.value,
        "stable_read_policy": policy.stable_read_policy.value,
        "classifier_name": policy.classifier_name,
        "classifier_version": policy.classifier_version,
    }


def _policy_payload(policy: CorpusAuditPolicy) -> dict[str, Any]:
    return {
        "policy_name": policy.policy_name,
        "policy_version": policy.policy_version,
        "availability_version": policy.availability_version,
        "rate_decimal_places": policy.rate_decimal_places,
        "rate_rounding": policy.rate_rounding,
        "quantile_method": policy.quantile_method,
        "duplicate_grouping_algorithm": policy.duplicate_grouping_algorithm,
        "rendering_version": policy.rendering_version,
    }


def _profile_payload(profile: ParserCapabilityProfile) -> dict[str, Any]:
    return {
        "parser_version": profile.parser_version.value,
        "capability_version": profile.capability_version,
        "language": profile.language,
        "page_count": profile.page_count,
        "page_text": profile.page_text,
        "character_count": profile.character_count,
        "token_count": profile.token_count,
        "acquisition_statuses": sorted(profile.acquisition_statuses),
        "layout": profile.layout,
        "bounding_boxes": profile.bounding_boxes,
        "source_spans": profile.source_spans,
        "structure_kinds": sorted(profile.structure_kinds),
    }


def _format_coverage_payload(item: AuditSourceFormatCoverage) -> dict[str, Any]:
    return {
        "media_type": item.media_type,
        "support": item.support.value,
        "file_count": item.file_count,
        "total_bytes": item.total_bytes,
        "hashed_count": item.hashed_count,
        "identity_count": item.identity_count,
    }


def _source_failure_payload(item: AuditSourceFailure) -> dict[str, Any]:
    return {
        "ordinal": item.ordinal,
        "category": item.category.value,
        "scope": item.scope.value,
        "stage": item.stage.value,
        "retryable": item.retryable,
    }
