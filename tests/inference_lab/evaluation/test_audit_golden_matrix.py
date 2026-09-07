"""Golden mixed-capability fixtures for corpus-audit aggregation."""

from dataclasses import dataclass

from inference_lab.evaluation import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    AcquisitionStatus,
    AuditAcquisitionObservation,
    AuditCoverage,
    AuditDistribution,
    AuditMissingReason,
    AuditParsedDocumentFacts,
    AuditParseEvidence,
    AuditParseFailure,
    AuditParseObservation,
    AuditRate,
    AuditRelationshipInventory,
    AuditSourceFailure,
    AuditSourceFormatCoverage,
    AuditSourceInventory,
    AuditStructureCount,
    AuditStructureKind,
    CorpusAuditRequest,
    CorpusAuditScope,
    ParserCapabilityProfile,
    aggregate_corpus_audit,
)
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveryCounts,
    FamilyCandidate,
    FamilyEvidenceKind,
    ParserVersion,
    ParseStatus,
    SourceFormatCount,
)
from inference_lab.ingest.discovery import (
    SourceSupport,
)
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryFailureStage,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
)

_CORPUS_ID = CorpusId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_CORPUS_VERSION = CorpusVersionId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_DOCUMENT_IDS = tuple(
    DocumentId(value)
    for value in (
        "44444444-4444-4444-8444-444444444444",
        "55555555-5555-4555-8555-555555555555",
        "66666666-6666-4666-8666-666666666666",
        "77777777-7777-4777-8777-777777777777",
        "88888888-8888-4888-8888-888888888888",
        "99999999-9999-4999-8999-999999999999",
    )
)
_VERSION_IDS = tuple(
    DocumentVersionId(value)
    for value in (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
        "44444444-4444-4444-8444-444444444444",
        "55555555-5555-4555-8555-555555555555",
        "66666666-6666-4666-8666-666666666666",
    )
)
_MEMBERSHIPS = tuple(
    CorpusDocumentVersionMembership(_CORPUS_ID, document_id, version_id)
    for document_id, version_id in zip(_DOCUMENT_IDS, _VERSION_IDS, strict=True)
)
_FULL_PARSER_VERSION = ParserVersion(
    implementation="pdf-parser",
    contract_version="1.0.0",
    normalisation_version="1.0.0",
)
_FULL_PROFILE = ParserCapabilityProfile(
    parser_version=_FULL_PARSER_VERSION,
    capability_version="1.0.0",
    language=True,
    page_count=True,
    page_text=True,
    character_count=True,
    token_count=True,
    acquisition_statuses=frozenset(status.value for status in AcquisitionStatus),
    layout=True,
    bounding_boxes=True,
    source_spans=True,
    structure_kinds=frozenset(kind.value for kind in AuditStructureKind),
)
_STRUCTURE_KINDS = tuple(sorted(AuditStructureKind, key=lambda kind: kind.value))


@dataclass(frozen=True, slots=True)
class _GoldenDocument:
    """Describe one deterministic member row in the mixed-capability matrix."""

    parser_version: ParserVersion | None
    evidence: AuditParseEvidence | None
    missing_reason: AuditMissingReason | None
    media_type: str | None
    source_byte_count: int | None
    acquisition: AcquisitionStatus | None
    acquisition_reason: AuditMissingReason | None = None


def _facts(
    *,
    language: str | None,
    page_count: int | None,
    character_count: int,
    token_count: int | None,
    structure_counts: dict[AuditStructureKind, int],
    available_structure_kinds: frozenset[AuditStructureKind],
    source_span_count: int,
    page_number_count: int,
    page_character_counts: tuple[int, ...] | None,
    page_layout_count: int,
    page_layout_page_count: int,
    bounding_box_count: int,
    bounding_box_page_count: int,
):
    return AuditParsedDocumentFacts(
        language=language,
        page_count=page_count,
        character_count=character_count,
        token_count=token_count,
        structure_counts=tuple(
            AuditStructureCount(kind, structure_counts.get(kind, 0))
            for kind in _STRUCTURE_KINDS
        ),
        available_structure_kinds=available_structure_kinds,
        source_span_count=source_span_count,
        page_number_count=page_number_count,
        page_character_counts=page_character_counts,
        page_layout_count=page_layout_count,
        page_layout_page_count=page_layout_page_count,
        bounding_box_count=bounding_box_count,
        bounding_box_page_count=bounding_box_page_count,
    )


def _evidence(
    *,
    status: ParseStatus,
    document: AuditParsedDocumentFacts | None = None,
    failures: tuple[AuditParseFailure, ...] = (),
) -> AuditParseEvidence:
    return AuditParseEvidence(status, document, failures)


def _source_inventory() -> AuditSourceInventory:
    formats = (
        SourceFormatCount(
            "application/pdf",
            SourceSupport.RECOGNISED_UNSUPPORTED,
            1,
            20,
        ),
        SourceFormatCount(
            "application/x-unknown",
            SourceSupport.UNKNOWN,
            1,
            30,
        ),
        SourceFormatCount("text/plain", SourceSupport.SUPPORTED, 1, 10),
    )
    coverage = tuple(
        AuditSourceFormatCoverage(
            item.media_type,
            item.support,
            item.file_count,
            item.total_bytes,
            hashed_count=1 if item.support is SourceSupport.SUPPORTED else 0,
            identity_count=1 if item.support is SourceSupport.SUPPORTED else 0,
        )
        for item in formats
    )
    return AuditSourceInventory(
        corpus_id=_CORPUS_ID,
        policy=CURRENT_DISCOVERY_POLICY,
        format_counts=formats,
        discovery_counts=DiscoveryCounts(
            files_seen=4,
            supported=1,
            unsupported=2,
            hashed=1,
            unchanged=0,
            changed=0,
            new=1,
            duplicate=0,
            ambiguous=0,
            conflicting=0,
            failed=3,
        ),
        failures=(
            AuditSourceFailure(
                ordinal=0,
                category=DiscoveryFailureCategory.TRAVERSAL_FAILED,
                scope=DiscoveryFailureScope.SOURCE,
                stage=DiscoveryFailureStage.TRAVERSAL,
                retryable=True,
            ),
            AuditSourceFailure(
                ordinal=1,
                category=DiscoveryFailureCategory.RECOGNISED_UNSUPPORTED_FORMAT,
                scope=DiscoveryFailureScope.SOURCE,
                stage=DiscoveryFailureStage.CLASSIFICATION,
                retryable=False,
            ),
            AuditSourceFailure(
                ordinal=2,
                category=DiscoveryFailureCategory.UNKNOWN_FORMAT,
                scope=DiscoveryFailureScope.SOURCE,
                stage=DiscoveryFailureStage.CLASSIFICATION,
                retryable=False,
            ),
        ),
        format_coverage=coverage,
        known_source_byte_counts=(10, 20, 30),
    )


def _relationships() -> AuditRelationshipInventory:
    family_one = DocumentFamilyId("11111111-1111-4111-8111-111111111111")
    family_two = DocumentFamilyId("22222222-2222-4222-8222-222222222222")
    family_three = DocumentFamilyId("33333333-3333-4333-8333-333333333333")
    family_four = DocumentFamilyId("44444444-4444-4444-8444-444444444444")
    family_five = DocumentFamilyId("55555555-5555-4555-8555-555555555555")
    return AuditRelationshipInventory(
        duplicate_relationships=(
            DuplicateRelationship(
                _VERSION_IDS[0], _VERSION_IDS[1], DuplicateClass.BYTE_IDENTICAL
            ),
            DuplicateRelationship(
                _VERSION_IDS[1], _VERSION_IDS[2], DuplicateClass.BYTE_IDENTICAL
            ),
            DuplicateRelationship(
                _VERSION_IDS[3], _VERSION_IDS[4], DuplicateClass.BYTE_IDENTICAL
            ),
        ),
        family_memberships=(
            DocumentFamilyMembership(family_one, _DOCUMENT_IDS[0], _VERSION_IDS[0]),
            DocumentFamilyMembership(family_one, _DOCUMENT_IDS[1], _VERSION_IDS[1]),
            DocumentFamilyMembership(family_two, _DOCUMENT_IDS[2]),
            DocumentFamilyMembership(family_three, _DOCUMENT_IDS[3], _VERSION_IDS[3]),
        ),
        family_candidates=(
            FamilyCandidate(
                family_four,
                FamilyEvidenceKind.EXPLICIT_ASSOCIATION,
                document_id=_DOCUMENT_IDS[4],
            ),
            FamilyCandidate(family_five, FamilyEvidenceKind.EXPLICIT_ASSOCIATION),
        ),
    )


def _golden_rows() -> tuple[_GoldenDocument, ...]:
    full_available = frozenset(
        {
            AuditStructureKind.PAGE,
            AuditStructureKind.SECTION,
            AuditStructureKind.PARAGRAPH,
        }
    )
    full_structure = {
        AuditStructureKind.PAGE: 3,
        AuditStructureKind.SECTION: 2,
        AuditStructureKind.PARAGRAPH: 4,
        AuditStructureKind.TABLE: 1,
        AuditStructureKind.FIGURE_CAPTION: 1,
        AuditStructureKind.FIGURE: 1,
        AuditStructureKind.CLAIM: 1,
        AuditStructureKind.EXAMPLE: 1,
        AuditStructureKind.REFERENCE: 1,
        AuditStructureKind.OTHER: 1,
    }
    recoverable_structure = {
        AuditStructureKind.PAGE: 2,
        AuditStructureKind.PARAGRAPH: 2,
    }
    return (
        _GoldenDocument(
            _FULL_PARSER_VERSION,
            _evidence(
                status="succeeded",
                document=_facts(
                    language="en",
                    page_count=3,
                    character_count=120,
                    token_count=30,
                    structure_counts=full_structure,
                    available_structure_kinds=full_available,
                    source_span_count=10,
                    page_number_count=3,
                    page_character_counts=(10, 0, 20),
                    page_layout_count=2,
                    page_layout_page_count=2,
                    bounding_box_count=1,
                    bounding_box_page_count=1,
                ),
            ),
            None,
            "application/pdf",
            20,
            AcquisitionStatus.TEXT_NATIVE,
        ),
        _GoldenDocument(
            _FULL_PARSER_VERSION,
            _evidence(
                status="completed_with_recoverable_failures",
                document=_facts(
                    language=None,
                    page_count=2,
                    character_count=80,
                    token_count=None,
                    structure_counts=recoverable_structure,
                    available_structure_kinds=full_available,
                    source_span_count=4,
                    page_number_count=2,
                    page_character_counts=(0, 40),
                    page_layout_count=1,
                    page_layout_page_count=1,
                    bounding_box_count=0,
                    bounding_box_page_count=0,
                ),
                failures=(
                    AuditParseFailure(
                        "malformed_unit",
                        "recoverable",
                        "structure",
                        unit_kind="paragraph",
                        unit_index=0,
                    ),
                ),
            ),
            None,
            "application/pdf",
            20,
            AcquisitionStatus.SCANNED,
        ),
        _GoldenDocument(
            CURRENT_PARSER_CAPABILITY_PROFILE.parser_version,
            _evidence(
                status="succeeded",
                document=_facts(
                    language=None,
                    page_count=None,
                    character_count=5,
                    token_count=None,
                    structure_counts={AuditStructureKind.PARAGRAPH: 1},
                    available_structure_kinds=frozenset({AuditStructureKind.PARAGRAPH}),
                    source_span_count=1,
                    page_number_count=1,
                    page_character_counts=None,
                    page_layout_count=0,
                    page_layout_page_count=0,
                    bounding_box_count=0,
                    bounding_box_page_count=0,
                ),
            ),
            None,
            "text/plain",
            5,
            AcquisitionStatus.TEXT_NATIVE,
        ),
        _GoldenDocument(
            _FULL_PARSER_VERSION,
            _evidence(
                status="failed",
                failures=(AuditParseFailure("decode_failed", "fatal", "decode"),),
            ),
            None,
            "application/pdf",
            20,
            AcquisitionStatus.MIXED,
        ),
        _GoldenDocument(
            _FULL_PARSER_VERSION,
            _evidence(
                status="succeeded",
                document=_facts(
                    language="fr",
                    page_count=1,
                    character_count=0,
                    token_count=0,
                    structure_counts={},
                    available_structure_kinds=full_available,
                    source_span_count=0,
                    page_number_count=0,
                    page_character_counts=(0,),
                    page_layout_count=0,
                    page_layout_page_count=0,
                    bounding_box_count=0,
                    bounding_box_page_count=0,
                ),
            ),
            None,
            "application/pdf",
            20,
            AcquisitionStatus.UNKNOWN,
            AuditMissingReason.PARSER_FIELD_UNAVAILABLE,
        ),
        _GoldenDocument(
            None,
            None,
            AuditMissingReason.PARSE_NOT_ATTEMPTED,
            None,
            None,
            None,
        ),
    )


def _golden_request() -> CorpusAuditRequest:
    rows = _golden_rows()
    observations = tuple(
        AuditParseObservation(
            membership=membership,
            parser_version=row.parser_version,
            media_type=row.media_type,
            source_byte_count=row.source_byte_count,
            evidence=row.evidence,
            missing_reason=row.missing_reason,
        )
        for membership, row in zip(_MEMBERSHIPS, rows, strict=True)
    )
    acquisition = tuple(
        AuditAcquisitionObservation(
            membership=membership,
            status=row.acquisition,
            producer_name="audit-fixture",
            producer_version="1.0.0",
            unknown_reason=row.acquisition_reason,
        )
        for membership, row in zip(_MEMBERSHIPS, rows, strict=True)
        if row.acquisition is not None
    )
    return CorpusAuditRequest(
        scope=CorpusAuditScope(_CORPUS_ID, _CORPUS_VERSION, _MEMBERSHIPS),
        source_inventory=_source_inventory(),
        parse_observations=observations,
        acquisition_observations=acquisition,
        relationships=_relationships(),
        capability_profiles=(
            _FULL_PROFILE,
            CURRENT_PARSER_CAPABILITY_PROFILE,
        ),
    )


def _empty_request() -> CorpusAuditRequest:
    return CorpusAuditRequest(
        scope=CorpusAuditScope(_CORPUS_ID, _CORPUS_VERSION, ()),
        source_inventory=AuditSourceInventory(
            corpus_id=_CORPUS_ID,
            policy=CURRENT_DISCOVERY_POLICY,
            format_counts=(),
            discovery_counts=DiscoveryCounts.empty(),
            failures=(),
        ),
        parse_observations=(),
        capability_profiles=(),
    )


def _coverage_values(coverage: AuditCoverage) -> tuple[int, ...]:
    return (
        coverage.observed,
        coverage.observed_absent,
        coverage.unavailable,
        coverage.missing,
        coverage.failed,
        coverage.invalid,
        coverage.population,
    )


def _rate_values(rate: AuditRate) -> tuple[int, int]:
    return rate.numerator, rate.denominator


def test_mixed_capability_golden_matrix_has_exact_aggregate_totals() -> None:
    result = aggregate_corpus_audit(_golden_request())

    assert result.source.files_seen == 4
    assert result.source.classified_files == 3
    assert result.source.traversal_failures == 1
    assert result.source.reconciles is True
    assert result.source.size_summary.distribution == AuditDistribution(
        3, 60, 10, 30, 0
    )
    assert result.source.failure_counts == (
        ("recognised_unsupported_format|source|classification|false", 1),
        ("traversal_failed|source|traversal|true", 1),
        ("unknown_format|source|classification|false", 1),
    )

    assert result.parse.status_counts == (
        ("completed_with_recoverable_failures", 1),
        ("failed", 1),
        ("missing", 1),
        ("succeeded", 3),
    )
    assert result.parse.parser_counts == (
        (_FULL_PARSER_VERSION.value, 4),
        (CURRENT_PARSER_CAPABILITY_PROFILE.parser_version.value, 1),
    )
    assert result.parse.language_counts == (("en", 1), ("fr", 1))
    assert result.parse.failure_counts == (
        ("decode_failed|fatal|decode|none", 1),
        ("malformed_unit|recoverable|structure|paragraph", 1),
    )
    coverage = dict(result.parse.field_coverage)
    assert _coverage_values(coverage["language"].coverage) == (2, 1, 1, 1, 1, 0, 6)
    assert _coverage_values(coverage["page_count"].coverage) == (3, 0, 1, 1, 1, 0, 6)
    assert _coverage_values(coverage["character_count"].coverage) == (
        4,
        0,
        0,
        1,
        1,
        0,
        6,
    )
    assert _coverage_values(coverage["token_count"].coverage) == (2, 1, 1, 1, 1, 0, 6)

    numeric = dict(result.parse.numeric_summaries)
    assert numeric["page_count"].distribution == AuditDistribution(3, 6, 1, 3, 0)
    assert numeric["character_count"].distribution == AuditDistribution(
        4, 205, 0, 120, 1
    )
    assert numeric["token_count"].distribution == AuditDistribution(2, 30, 0, 30, 1)
    assert result.parse.token_summaries[0][0] == _FULL_PARSER_VERSION.value

    assert result.acquisition.status_counts == (
        ("missing", 1),
        ("mixed", 1),
        ("scanned", 1),
        ("text_native", 2),
        ("unknown", 1),
    )
    assert _coverage_values(result.acquisition.coverage.coverage) == (
        4,
        0,
        1,
        1,
        0,
        0,
        6,
    )
    assert result.acquisition.known_only_rate.numerator == 4
    assert result.acquisition.known_only_rate.denominator == 5
    assert result.acquisition.population_rate.numerator == 5
    assert result.acquisition.population_rate.denominator == 6

    structures = {item.kind: item for item in result.structures}
    assert _coverage_values(structures[AuditStructureKind.PAGE].coverage.coverage) == (
        2,
        1,
        1,
        1,
        1,
        0,
        6,
    )
    assert structures[AuditStructureKind.PAGE].present_documents == 2
    assert structures[AuditStructureKind.PAGE].absent_documents == 1
    assert structures[AuditStructureKind.TABLE].coverage.coverage.unavailable == 1
    assert structures[AuditStructureKind.TABLE].present_documents == 1
    assert structures[AuditStructureKind.PARAGRAPH].present_documents == 3

    assert result.spans.source_span_count == 15
    assert result.spans.page_number_count == 6
    assert result.spans.bounding_box_count == 1
    assert result.spans.document_coverage.coverage.observed == 4
    assert _coverage_values(result.ocr.coverage.coverage) == (3, 0, 1, 1, 1, 0, 6)
    assert result.ocr.zero_text_page_count == 3
    assert result.ocr.page_character_summary.distribution == AuditDistribution(
        6, 70, 0, 40, 3
    )
    assert result.ocr.mixed_text_documents == 2
    assert result.ocr.pages_with_layout == 3
    assert result.ocr.pages_without_layout == 3
    assert result.ocr.pages_with_boxes == 1
    assert result.ocr.pages_without_boxes == 5

    assert result.relationships.duplicate_edge_count == 3
    assert result.relationships.duplicate_participating_members == 5
    assert result.relationships.duplicate_group_count == 2
    assert result.relationships.duplicate_excess_copies == 3
    assert result.relationships.family_count == 3
    assert result.relationships.family_document_count == 4
    assert result.relationships.family_version_count == 3
    assert result.relationships.unassigned_document_count == 2
    assert result.relationships.document_level_memberships == 1
    assert result.relationships.version_level_memberships == 3
    assert result.relationships.family_candidate_count == 2
    assert result.relationships.family_candidate_family_count == 2
    assert set(result.limitations) == {
        "acquisition_status_incomplete",
        "family_candidates_are_advisory",
        "ocr_indicators_unavailable",
        "parse_field_coverage_incomplete",
        "source_format_not_supported",
        "source_upstream_failures",
        "structure_capability_unavailable",
        "token_semantics_are_parser_versioned",
    }


def test_empty_golden_matrix_preserves_zero_denominators_and_distributions() -> None:
    result = aggregate_corpus_audit(_empty_request())

    assert result.source.files_seen == 0
    assert result.source.size_summary.distribution == AuditDistribution(
        0, 0, None, None, 0
    )
    assert result.parse.status_counts == ()
    assert all(
        summary.coverage.population == 0 for _, summary in result.parse.field_coverage
    )
    assert result.acquisition.status_counts == ()
    assert result.acquisition.coverage.coverage.population == 0
    assert _rate_values(result.acquisition.known_only_rate) == (0, 0)
    assert _rate_values(result.acquisition.population_rate) == (0, 0)
    assert all(item.coverage.coverage.population == 0 for item in result.structures)
    assert result.spans.block_summary.distribution == AuditDistribution(
        0, 0, None, None, 0
    )
    assert result.ocr.page_character_summary.distribution == AuditDistribution(
        0, 0, None, None, 0
    )
    assert _rate_values(result.relationships.duplicate_participating_rate) == (0, 0)
    assert _rate_values(result.relationships.duplicate_excess_rate) == (0, 0)
