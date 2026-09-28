"""Contract tests for the storage-neutral corpus-audit input boundary."""

from dataclasses import replace

import pytest

from inference_lab.evaluation import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    AcquisitionStatus,
    AuditAcquisitionObservation,
    AuditContractError,
    AuditErrorCategory,
    AuditMissingReason,
    AuditParseEvidence,
    AuditParseObservation,
    AuditRelationshipInventory,
    AuditSourceInventory,
    CorpusAuditPolicy,
    CorpusAuditRequest,
    CorpusAuditScope,
)
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassificationInventory,
    DiscoveryCounts,
    DiscoveryResult,
    DiscoveryStatus,
    ParseOutcome,
    ParserFailure,
    TraversalInventory,
)
from inference_lab.ingest.parser import NormalisedText, ParsedDocument
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_OTHER_CORPUS_ID = CorpusId("22222222-2222-4222-8222-222222222222")
_CORPUS_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_ID = DocumentId("44444444-4444-4444-8444-444444444444")
_VERSION_ID = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_OTHER_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")


def _membership(
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    document_id: DocumentId = _DOCUMENT_ID,
    document_version_id: DocumentVersionId = _VERSION_ID,
) -> CorpusDocumentVersionMembership:
    return CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def _source_inventory(corpus_id: CorpusId = _CORPUS_ID) -> AuditSourceInventory:
    traversal = TraversalInventory(
        corpus_id=corpus_id,
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
        corpus_id=corpus_id,
        policy=classification.policy,
        status=DiscoveryStatus.COMPLETED,
        sources=(),
        failures=(),
        counts=DiscoveryCounts.empty(),
    )
    return AuditSourceInventory.from_ingest(classification, discovery)


def _scope(membership: CorpusDocumentVersionMembership) -> CorpusAuditScope:
    return CorpusAuditScope(
        corpus_id=membership.corpus_id,
        corpus_version=_CORPUS_VERSION,
        memberships=(membership,),
    )


def _empty_document(membership: CorpusDocumentVersionMembership) -> ParsedDocument:
    return ParsedDocument(
        document_id=membership.document_id,
        document_version_id=membership.document_version_id,
        text=NormalisedText("private source text"),
        blocks=(),
        pages=None,
        sections=None,
        paragraphs=(),
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=19,
        token_count=None,
        parser_version=CURRENT_PARSER_VERSION,
    )


def _successful_observation(
    membership: CorpusDocumentVersionMembership,
) -> AuditParseObservation:
    document = _empty_document(membership)
    return AuditParseObservation(
        membership=membership,
        parser_version=CURRENT_PARSER_VERSION,
        media_type="text/plain",
        source_byte_count=0,
        evidence=AuditParseEvidence.from_ingest(
            ParseOutcome("succeeded", document, ())
        ),
    )


def _request(
    membership: CorpusDocumentVersionMembership,
    *,
    observation: AuditParseObservation | None = None,
) -> CorpusAuditRequest:
    return CorpusAuditRequest(
        scope=_scope(membership),
        source_inventory=_source_inventory(membership.corpus_id),
        parse_observations=(
            observation
            if observation is not None
            else AuditParseObservation(
                membership=membership,
                missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
            ),
        ),
        capability_profiles=(CURRENT_PARSER_CAPABILITY_PROFILE,)
        if observation is not None and observation.parser_version is not None
        else (),
    )


def test_scope_requires_one_version_per_logical_document() -> None:
    first = _membership()
    second = _membership(document_version_id=_OTHER_VERSION_ID)

    with pytest.raises(AuditContractError) as error:
        CorpusAuditScope(_CORPUS_ID, _CORPUS_VERSION, (first, second))

    assert error.value.category is AuditErrorCategory.MEMBERSHIP_MISMATCH


def test_request_requires_exactly_one_parse_slot_per_member() -> None:
    membership = _membership()
    missing = AuditParseObservation(
        membership=membership,
        missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
    )

    request = _request(membership, observation=missing)

    assert request.scope.document_version_ids == frozenset({_VERSION_ID})
    assert request.parse_observations == (missing,)

    with pytest.raises(AuditContractError) as error:
        CorpusAuditRequest(
            scope=_scope(membership),
            source_inventory=_source_inventory(),
            parse_observations=(),
        )
    assert error.value.category is AuditErrorCategory.MEMBERSHIP_MISMATCH


def test_successful_parse_is_checked_against_member_and_capability() -> None:
    membership = _membership()
    observation = _successful_observation(membership)

    request = _request(membership, observation=observation)

    assert request.capability_profiles == (CURRENT_PARSER_CAPABILITY_PROFILE,)

    invalid_document = replace(_empty_document(membership), token_count=3)
    invalid_observation = replace(
        observation,
        evidence=AuditParseEvidence.from_ingest(
            ParseOutcome("succeeded", invalid_document, ())
        ),
    )
    with pytest.raises(AuditContractError) as error:
        _request(membership, observation=invalid_observation)
    assert error.value.category is AuditErrorCategory.UNSUPPORTED_OBSERVATION


def test_fatal_parse_retains_wrapper_attribution_without_a_document() -> None:
    membership = _membership()
    observation = AuditParseObservation(
        membership=membership,
        parser_version=CURRENT_PARSER_VERSION,
        media_type="text/plain",
        evidence=AuditParseEvidence.from_ingest(
            ParseOutcome(
                "failed",
                None,
                (ParserFailure("decode_failed", "fatal", "decode"),),
            )
        ),
    )

    request = _request(membership, observation=observation)

    assert request.parse_observations[0].evidence is not None
    assert request.parse_observations[0].evidence.document is None


def test_parser_field_absence_keeps_capability_attribution() -> None:
    membership = _membership()
    observation = AuditParseObservation(
        membership=membership,
        parser_version=CURRENT_PARSER_VERSION,
        missing_reason=AuditMissingReason.PARSER_FIELD_UNAVAILABLE,
    )

    request = _request(membership, observation=observation)

    assert request.parse_observations[0].parser_version == CURRENT_PARSER_VERSION


def test_unregistered_parser_version_is_rejected() -> None:
    membership = _membership()
    observation = AuditParseObservation(
        membership=membership,
        parser_version=replace(CURRENT_PARSER_VERSION, implementation="future-parser"),
        missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
    )

    with pytest.raises(AuditContractError) as error:
        _request(membership, observation=observation)
    assert error.value.category is AuditErrorCategory.UNKNOWN_PARSER_CAPABILITY


def test_acquisition_unknown_requires_a_bounded_reason() -> None:
    membership = _membership()
    observation = AuditAcquisitionObservation(
        membership=membership,
        status=AcquisitionStatus.UNKNOWN,
        producer_name="pdf-baseline",
        producer_version="1.0.0",
        unknown_reason=AuditMissingReason.UNSUPPORTED_FORMAT,
    )

    assert observation.status is AcquisitionStatus.UNKNOWN

    with pytest.raises(AuditContractError):
        AuditAcquisitionObservation(
            membership=membership,
            status=AcquisitionStatus.UNKNOWN,
            producer_name="pdf-baseline",
            producer_version="1.0.0",
        )


def test_acquisition_status_must_be_possible_for_selected_parser() -> None:
    membership = _membership()
    parse_observation = _successful_observation(membership)
    acquisition = AuditAcquisitionObservation(
        membership=membership,
        status=AcquisitionStatus.SCANNED,
        producer_name="scan-detector",
        producer_version="1.0.0",
    )

    with pytest.raises(AuditContractError) as error:
        CorpusAuditRequest(
            scope=_scope(membership),
            source_inventory=_source_inventory(),
            parse_observations=(parse_observation,),
            acquisition_observations=(acquisition,),
            capability_profiles=(CURRENT_PARSER_CAPABILITY_PROFILE,),
        )
    assert error.value.category is AuditErrorCategory.UNSUPPORTED_OBSERVATION


def test_relationships_must_remain_inside_selected_population() -> None:
    membership = _membership()
    outside = _membership(
        document_id=DocumentId("77777777-7777-4777-8777-777777777777"),
        document_version_id=DocumentVersionId("88888888-8888-4888-8888-888888888888"),
    )
    from inference_lab.provenance import DuplicateClass, DuplicateRelationship

    relationship = DuplicateRelationship(
        membership.document_version_id,
        outside.document_version_id,
        DuplicateClass.BYTE_IDENTICAL,
    )
    relationships = AuditRelationshipInventory(
        duplicate_relationships=(relationship,),
    )

    with pytest.raises(AuditContractError) as error:
        CorpusAuditRequest(
            scope=_scope(membership),
            source_inventory=_source_inventory(),
            parse_observations=(
                AuditParseObservation(
                    membership=membership,
                    missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
                ),
            ),
            relationships=relationships,
        )
    assert error.value.category is AuditErrorCategory.RELATIONSHIP_OUTSIDE_SCOPE
