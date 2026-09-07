"""Hand-calculated tests for the pure corpus-audit aggregation engine."""

from inference_lab.evaluation import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    AcquisitionStatus,
    AuditAcquisitionObservation,
    AuditMissingReason,
    AuditOverallStatus,
    AuditParsedDocumentFacts,
    AuditParseEvidence,
    AuditParseObservation,
    AuditRelationshipInventory,
    AuditSourceFormatCoverage,
    AuditSourceInventory,
    AuditStructureCount,
    AuditStructureKind,
    CorpusAuditReport,
    CorpusAuditRequest,
    CorpusAuditScope,
    aggregate_corpus_audit,
)
from inference_lab.evaluation.audit_aggregation import _IntegerSummary
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassificationInventory,
    DiscoveryCounts,
    DiscoveryResult,
    DiscoveryStatus,
    SourceFormatCount,
    TraversalInventory,
)
from inference_lab.ingest.discovery import SourceSupport
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION
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

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_CORPUS_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_ONE = DocumentId("44444444-4444-4444-8444-444444444444")
_DOCUMENT_TWO = DocumentId("77777777-7777-4777-8777-777777777777")
_VERSION_ONE = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_VERSION_TWO = DocumentVersionId("88888888-8888-4888-8888-888888888888")


def _membership(
    document_id: DocumentId,
    version_id: DocumentVersionId,
) -> CorpusDocumentVersionMembership:
    return CorpusDocumentVersionMembership(_CORPUS_ID, document_id, version_id)


def _source_inventory() -> AuditSourceInventory:
    traversal = TraversalInventory(
        corpus_id=_CORPUS_ID,
        root_count=1,
        candidates=(),
        failures=(),
        complete_root_ordinals=(0,),
    )
    classification = ClassificationInventory(
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        traversal=traversal,
        classified_candidates=(),
        failures=(),
        format_counts=(),
    )
    discovery = DiscoveryResult(
        corpus_id=_CORPUS_ID,
        policy=classification.policy,
        status=DiscoveryStatus.COMPLETED,
        sources=(),
        failures=(),
        counts=DiscoveryCounts.empty(),
    )
    return AuditSourceInventory.from_ingest(classification, discovery)


def _scope(
    memberships: tuple[CorpusDocumentVersionMembership, ...],
) -> CorpusAuditScope:
    return CorpusAuditScope(_CORPUS_ID, _CORPUS_VERSION, memberships)


def _evidence(
    *,
    characters: int,
    paragraphs: int = 1,
    token_count: int | None = None,
) -> AuditParseEvidence:
    return AuditParseEvidence(
        status="succeeded",
        document=AuditParsedDocumentFacts(
            language=None,
            page_count=None,
            character_count=characters,
            token_count=token_count,
            structure_counts=(
                AuditStructureCount(AuditStructureKind.PARAGRAPH, paragraphs),
            ),
            available_structure_kinds=frozenset({AuditStructureKind.PARAGRAPH}),
            source_span_count=paragraphs,
            page_number_count=0,
            page_character_counts=None,
            page_layout_count=0,
            page_layout_page_count=0,
            bounding_box_count=0,
            bounding_box_page_count=0,
        ),
        failures=(),
    )


def _observation(
    membership: CorpusDocumentVersionMembership,
    *,
    characters: int = 1,
    token_count: int | None = None,
) -> AuditParseObservation:
    return AuditParseObservation(
        membership=membership,
        parser_version=CURRENT_PARSER_VERSION,
        media_type="text/plain",
        source_byte_count=characters,
        evidence=_evidence(
            characters=characters,
            token_count=token_count,
        ),
    )


def _request(
    memberships: tuple[CorpusDocumentVersionMembership, ...],
    observations: tuple[AuditParseObservation, ...],
    *,
    acquisition_observations: tuple[AuditAcquisitionObservation, ...] = (),
    relationships: AuditRelationshipInventory = AuditRelationshipInventory(),
    source_inventory: AuditSourceInventory | None = None,
) -> CorpusAuditRequest:
    return CorpusAuditRequest(
        scope=_scope(memberships),
        source_inventory=(
            _source_inventory() if source_inventory is None else source_inventory
        ),
        parse_observations=observations,
        acquisition_observations=acquisition_observations,
        relationships=relationships,
        capability_profiles=(CURRENT_PARSER_CAPABILITY_PROFILE,),
    )


def test_source_sizes_and_format_progress_are_reconciled() -> None:
    membership = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    source_inventory = AuditSourceInventory(
        corpus_id=_CORPUS_ID,
        policy=CURRENT_DISCOVERY_POLICY,
        format_counts=(
            SourceFormatCount("text/plain", SourceSupport.SUPPORTED, 2, 10),
        ),
        discovery_counts=DiscoveryCounts(
            files_seen=2,
            supported=2,
            unsupported=0,
            hashed=1,
            unchanged=1,
            changed=0,
            new=0,
            duplicate=0,
            ambiguous=0,
            conflicting=0,
            failed=0,
        ),
        failures=(),
        format_coverage=(
            AuditSourceFormatCoverage(
                media_type="text/plain",
                support=SourceSupport.SUPPORTED,
                file_count=2,
                total_bytes=10,
                hashed_count=1,
                identity_count=1,
            ),
        ),
        known_source_byte_counts=(3, 7),
    )

    result = aggregate_corpus_audit(
        _request(
            (membership,),
            (_observation(membership),),
            source_inventory=source_inventory,
        )
    )

    assert result.source.files_seen == 2
    assert result.source.classified_files == 2
    assert result.source.traversal_failures == 0
    assert result.source.reconciles is True
    assert result.source.size_summary.distribution.total == 10
    assert result.source.size_summary.quantiles[0] == ("p50", 3)


def test_numeric_summaries_use_nearest_rank_quantiles() -> None:
    first = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    second = _membership(_DOCUMENT_TWO, _VERSION_TWO)

    result = aggregate_corpus_audit(
        _request(
            (first, second),
            (_observation(first, characters=3), _observation(second, characters=7)),
        )
    )

    character_summary = dict(result.parse.numeric_summaries)["character_count"]
    assert isinstance(character_summary, _IntegerSummary)
    assert character_summary.distribution.total == 10
    assert character_summary.distribution.minimum == 3
    assert character_summary.distribution.maximum == 7
    assert character_summary.quantiles == (
        ("p50", 3),
        ("p90", 7),
        ("p95", 7),
        ("p99", 7),
    )


def test_coverage_keeps_unavailable_and_missing_distinct() -> None:
    first = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    second = _membership(_DOCUMENT_TWO, _VERSION_TWO)
    missing = AuditParseObservation(
        membership=second,
        missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
    )

    result = aggregate_corpus_audit(
        _request((first, second), (_observation(first), missing))
    )

    fields = dict(result.parse.field_coverage)
    assert fields["language"].coverage.unavailable == 1
    assert fields["language"].coverage.missing == 1
    assert fields["character_count"].coverage.observed == 1
    assert fields["character_count"].coverage.missing == 1


def test_acquisition_rates_use_assessed_and_complete_denominators() -> None:
    first = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    second = _membership(_DOCUMENT_TWO, _VERSION_TWO)
    acquisition = AuditAcquisitionObservation(
        membership=first,
        status=AcquisitionStatus.TEXT_NATIVE,
        producer_name="text-native",
        producer_version="3.0.0",
    )

    result = aggregate_corpus_audit(
        _request(
            (first, second),
            (
                _observation(first),
                AuditParseObservation(
                    membership=second,
                    missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
                ),
            ),
            acquisition_observations=(acquisition,),
        )
    )

    assert result.acquisition.known_only_rate.numerator == 1
    assert result.acquisition.known_only_rate.denominator == 1
    assert result.acquisition.population_rate.numerator == 1
    assert result.acquisition.population_rate.denominator == 2


def test_duplicate_components_and_family_counts_are_deterministic() -> None:
    first = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    second = _membership(_DOCUMENT_TWO, _VERSION_TWO)
    relationship = DuplicateRelationship(
        _VERSION_ONE,
        _VERSION_TWO,
        DuplicateClass.BYTE_IDENTICAL,
    )
    family = DocumentFamilyId("99999999-9999-4999-8999-999999999999")
    relationships = AuditRelationshipInventory(
        duplicate_relationships=(relationship,),
        family_memberships=(
            DocumentFamilyMembership(family, _DOCUMENT_ONE, _VERSION_ONE),
            DocumentFamilyMembership(family, _DOCUMENT_TWO, _VERSION_TWO),
        ),
    )

    result = aggregate_corpus_audit(
        _request(
            (first, second),
            (_observation(first), _observation(second)),
            relationships=relationships,
        )
    )

    assert result.relationships.duplicate_group_count == 1
    assert result.relationships.duplicate_participating_members == 2
    assert result.relationships.duplicate_excess_copies == 1
    assert result.relationships.family_count == 1
    assert result.relationships.family_document_count == 2
    assert result.relationships.unassigned_document_count == 0


def test_structure_and_capability_limitations_are_explicit() -> None:
    membership = _membership(_DOCUMENT_ONE, _VERSION_ONE)

    result = aggregate_corpus_audit(
        _request((membership,), (_observation(membership),))
    )

    structures = dict((item.kind, item) for item in result.structures)
    assert structures[AuditStructureKind.PARAGRAPH].present_documents == 1
    assert structures[AuditStructureKind.TABLE].coverage.coverage.unavailable == 1
    assert "structure_capability_unavailable" in result.limitations
    assert "ocr_indicators_unavailable" in result.limitations


def test_report_projects_aggregates_and_reconciles_both_populations() -> None:
    membership = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    report = CorpusAuditReport.from_request(
        _request((membership,), (_observation(membership),))
    )

    assert report.schema_version == "1.0.0"
    assert report.input_digest.startswith("sha256:")
    assert report.report_id.startswith("sha256:")
    assert report.populations.member_versions == 1
    assert report.populations.source_files_seen == 0
    assert report.reconciliation.member_versions_reconcile is True
    assert report.reconciliation.section_populations_reconcile is True
    assert report.overall_status is AuditOverallStatus.PARTIAL
    assert report.parse.field_coverage[2][1].coverage.observed == 1


def test_report_identity_changes_when_semantic_policy_changes() -> None:
    membership = _membership(_DOCUMENT_ONE, _VERSION_ONE)
    request = _request((membership,), (_observation(membership),))
    first = CorpusAuditReport.from_request(request)
    changed = CorpusAuditReport.from_request(
        request.__class__(
            scope=request.scope,
            source_inventory=request.source_inventory,
            parse_observations=request.parse_observations,
            acquisition_observations=request.acquisition_observations,
            relationships=request.relationships,
            capability_profiles=request.capability_profiles,
            policy=request.policy.__class__(rate_decimal_places=3),
        )
    )

    assert changed.input_digest != first.input_digest
    assert changed.report_id != first.report_id


def test_empty_report_is_explicitly_no_observations_and_payload_free() -> None:
    report = CorpusAuditReport.from_request(_request((), ()))

    assert report.overall_status is AuditOverallStatus.NO_OBSERVATIONS
    assert report.populations.member_versions == 0
    assert "private source text" not in repr(report)
