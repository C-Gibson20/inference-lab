"""Shared minimal corpus-audit records for acceptance and operator tests."""

from inference_lab.evaluation import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    AuditAcquisitionObservation,
    AuditParsedDocumentFacts,
    AuditParseEvidence,
    AuditParseObservation,
    AuditRelationshipInventory,
    AuditSourceInventory,
    AuditStructureCount,
    AuditStructureKind,
    CorpusAuditRequest,
    CorpusAuditScope,
)
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassificationInventory,
    DiscoveryCounts,
    DiscoveryResult,
    DiscoveryStatus,
    TraversalInventory,
)
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
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
        evidence=_evidence(characters=characters, token_count=token_count),
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
