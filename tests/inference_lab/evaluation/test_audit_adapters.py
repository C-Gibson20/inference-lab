"""Tests for the privacy-safe ING-01 and ING-02 audit adapters."""

import pytest

from inference_lab.evaluation import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    AcquisitionStatus,
    AuditContractError,
    AuditErrorCategory,
    AuditMissingReason,
    AuditParseObservation,
    CorpusAuditScope,
    adapt_parse_persistence_request,
    adapt_reconstructed_parse_outcome,
    adapt_source_inventory,
    adapt_text_native_acquisition,
    complete_parse_observations,
)
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassificationInventory,
    DiscoveryCounts,
    DiscoveryResult,
    DiscoveryStatus,
    ParseOutcome,
    ParserFailure,
    ParserInput,
    ReconstructedParseOutcome,
    TraversalInventory,
)
from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.ingest.parser import NormalisedText, ParsedDocument
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_OTHER_CORPUS_ID = CorpusId("22222222-2222-4222-8222-222222222222")
_CORPUS_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_DOCUMENT_ID = DocumentId("44444444-4444-4444-8444-444444444444")
_OTHER_DOCUMENT_ID = DocumentId("77777777-7777-4777-8777-777777777777")
_VERSION_ID = DocumentVersionId("55555555-5555-4555-8555-555555555555")
_OTHER_VERSION_ID = DocumentVersionId("88888888-8888-4888-8888-888888888888")


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


def _scope() -> CorpusAuditScope:
    return CorpusAuditScope(
        corpus_id=_CORPUS_ID,
        corpus_version=_CORPUS_VERSION,
        memberships=(
            _membership(),
            _membership(
                document_id=_OTHER_DOCUMENT_ID,
                document_version_id=_OTHER_VERSION_ID,
            ),
        ),
    )


def _document(membership: CorpusDocumentVersionMembership) -> ParsedDocument:
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


def _successful_request(
    membership: CorpusDocumentVersionMembership,
) -> ParsePersistenceRequest:
    return ParsePersistenceRequest(
        corpus_id=membership.corpus_id,
        parser_input=ParserInput(
            document_id=membership.document_id,
            document_version_id=membership.document_version_id,
            content=b"private source text",
            media_type="text/plain",
            encoding="utf-8",
        ),
        parser_version=CURRENT_PARSER_VERSION,
        outcome=ParseOutcome("succeeded", _document(membership), ()),
        parse_id=RecordId("99999999-9999-4999-8999-999999999999"),
        parsed_record_id=RecordId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
    )


def test_source_inventory_adapter_delegates_only_safe_ingest_facts() -> None:
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

    inventory = adapt_source_inventory(classification, discovery)

    assert inventory.corpus_id == _CORPUS_ID
    assert inventory.discovery_counts == DiscoveryCounts.empty()
    assert "locator" not in repr(inventory)


def test_parse_request_adapter_retains_only_safe_aggregate_facts() -> None:
    membership = _membership()

    observation = adapt_parse_persistence_request(
        _successful_request(membership),
        membership=membership,
    )

    assert observation.membership == membership
    assert observation.parser_version == CURRENT_PARSER_VERSION
    assert observation.media_type == "text/plain"
    assert observation.source_byte_count == 19
    assert observation.evidence is not None
    assert observation.evidence.document is not None
    assert not hasattr(observation.evidence.document, "text")
    assert "private source text" not in repr(observation)
    assert "parser_input" not in repr(observation)


def test_parse_request_adapter_rejects_wrong_corpus() -> None:
    membership = _membership()
    request = _successful_request(membership)
    request = ParsePersistenceRequest(
        corpus_id=_OTHER_CORPUS_ID,
        parser_input=request.parser_input,
        parser_version=request.parser_version,
        outcome=request.outcome,
        parse_id=request.parse_id,
        parsed_record_id=request.parsed_record_id,
    )

    with pytest.raises(AuditContractError) as error:
        adapt_parse_persistence_request(request, membership=membership)

    assert error.value.category is AuditErrorCategory.CORPUS_MISMATCH


def test_reconstructed_failed_outcome_keeps_parser_attribution() -> None:
    membership = _membership()
    outcome = ParseOutcome(
        "failed",
        None,
        (ParserFailure("decode_failed", "fatal", "decode"),),
    )

    observation = adapt_reconstructed_parse_outcome(
        ReconstructedParseOutcome(CURRENT_PARSER_VERSION, outcome),
        membership=membership,
        media_type="text/plain",
        source_byte_count=7,
    )

    assert observation.parser_version == CURRENT_PARSER_VERSION
    assert observation.evidence is not None
    assert observation.evidence.status == outcome.status
    assert observation.evidence.document is None
    assert observation.missing_reason is None
    assert observation.source_byte_count == 7


def test_complete_parse_observations_fills_missing_members_in_scope_order() -> None:
    scope = _scope()
    observed = AuditParseObservation(
        membership=scope.memberships[1],
        missing_reason=AuditMissingReason.SOURCE_FAILURE,
    )

    completed = complete_parse_observations(scope, (observed,))

    assert tuple(item.membership.document_version_id for item in completed) == tuple(
        item.document_version_id for item in scope.memberships
    )
    assert completed[0].missing_reason is AuditMissingReason.PARSE_NOT_ATTEMPTED
    assert completed[1] == observed


def test_complete_parse_observations_rejects_duplicate_member_slots() -> None:
    scope = _scope()
    observation = AuditParseObservation(
        membership=scope.memberships[0],
        missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
    )

    with pytest.raises(AuditContractError) as error:
        complete_parse_observations(scope, (observation, observation))

    assert error.value.category is AuditErrorCategory.OBSERVATION_MISMATCH


def test_text_native_adapter_requires_positive_parser_evidence() -> None:
    membership = _membership()
    observation = adapt_parse_persistence_request(
        _successful_request(membership),
        membership=membership,
    )

    acquisition = adapt_text_native_acquisition(observation)

    assert acquisition is not None
    assert acquisition.status is AcquisitionStatus.TEXT_NATIVE
    assert acquisition.producer_name == CURRENT_PARSER_VERSION.implementation
    assert acquisition.producer_version == CURRENT_PARSER_VERSION.contract_version

    failed = adapt_reconstructed_parse_outcome(
        ReconstructedParseOutcome(
            CURRENT_PARSER_VERSION,
            ParseOutcome(
                "failed",
                None,
                (ParserFailure("decode_failed", "fatal", "decode"),),
            ),
        ),
        membership=membership,
        media_type="text/plain",
    )
    assert adapt_text_native_acquisition(failed) is None
    assert CURRENT_PARSER_CAPABILITY_PROFILE.acquisition_statuses == frozenset(
        {AcquisitionStatus.TEXT_NATIVE.value}
    )
