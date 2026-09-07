"""Acceptance tests for audit reconciliation, privacy, and corpus isolation."""

from dataclasses import replace

import pytest
from structlog.testing import capture_logs

import inference_lab.evaluation.audit_observability as audit_observability
from inference_lab.evaluation import (
    CURRENT_PARSER_CAPABILITY_PROFILE,
    AuditContractError,
    AuditMissingReason,
    AuditParseObservation,
    CorpusAuditReport,
    CorpusAuditRequest,
    CorpusAuditScope,
    adapt_parse_persistence_request,
    adapt_source_inventory,
    render_audit_report,
    run_corpus_audit,
    serialise_audit_report,
)
from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    ClassificationInventory,
    DiscoveryCounts,
    DiscoveryResult,
    DiscoveryStatus,
    ParseOutcome,
    ParserInput,
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

from .test_audit_golden_matrix import _golden_request  # noqa: TID252

_CORPUS_ONE = CorpusId("11111111-1111-4111-8111-111111111111")
_CORPUS_TWO = CorpusId("22222222-2222-4222-8222-222222222222")
_VERSION_ONE = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_VERSION_TWO = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_SHARED_DOCUMENT = DocumentId("55555555-5555-4555-8555-555555555555")
_SHARED_VERSION = DocumentVersionId("66666666-6666-4666-8666-666666666666")
_CORPUS_ONE_DOCUMENT = DocumentId("77777777-7777-4777-8777-777777777777")
_CORPUS_ONE_VERSION = DocumentVersionId("88888888-8888-4888-8888-888888888888")
_CORPUS_TWO_DOCUMENT = DocumentId("99999999-9999-4999-8999-999999999999")
_CORPUS_TWO_VERSION = DocumentVersionId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _membership(
    corpus_id: CorpusId,
    document_id: DocumentId,
    version_id: DocumentVersionId,
) -> CorpusDocumentVersionMembership:
    return CorpusDocumentVersionMembership(corpus_id, document_id, version_id)


def _source_inventory(corpus_id: CorpusId):
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
    return adapt_source_inventory(classification, discovery)


def _source_observation(
    membership: CorpusDocumentVersionMembership,
    *,
    character_count: int,
) -> AuditParseObservation:
    facts = adapt_parse_persistence_request(
        _parse_request(membership, text="synthetic source text"),
        membership=membership,
    )
    assert facts.evidence is not None
    assert facts.evidence.document is not None
    document = replace(facts.evidence.document, character_count=character_count)
    return replace(
        facts,
        evidence=replace(facts.evidence, document=document),
    )


def _parse_request(
    membership: CorpusDocumentVersionMembership,
    *,
    text: str,
) -> ParsePersistenceRequest:
    document = ParsedDocument(
        document_id=membership.document_id,
        document_version_id=membership.document_version_id,
        text=NormalisedText(text),
        blocks=(),
        pages=None,
        sections=None,
        paragraphs=(),
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(text),
        token_count=None,
        parser_version=CURRENT_PARSER_VERSION,
    )
    return ParsePersistenceRequest(
        corpus_id=membership.corpus_id,
        parser_input=ParserInput(
            document_id=membership.document_id,
            document_version_id=membership.document_version_id,
            content=text.encode("utf-8"),
            media_type="text/plain",
            encoding="utf-8",
        ),
        parser_version=CURRENT_PARSER_VERSION,
        outcome=ParseOutcome("succeeded", document, ()),
        parse_id=RecordId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        parsed_record_id=RecordId("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
    )


def _request(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    rows: tuple[tuple[DocumentId, DocumentVersionId, int], ...],
) -> CorpusAuditRequest:
    memberships = tuple(
        _membership(corpus_id, document_id, version_id)
        for document_id, version_id, _ in rows
    )
    observations = tuple(
        _source_observation(membership, character_count=character_count)
        for membership, (_, _, character_count) in zip(memberships, rows, strict=True)
    )
    return CorpusAuditRequest(
        scope=CorpusAuditScope(corpus_id, corpus_version, memberships),
        source_inventory=_source_inventory(corpus_id),
        parse_observations=observations,
        capability_profiles=(CURRENT_PARSER_CAPABILITY_PROFILE,),
    )


def test_report_reconciliation_keeps_source_and_member_denominators_distinct() -> None:
    request = _golden_request()
    report = CorpusAuditReport.from_request(request)

    assert report.populations.source_files_seen == 4
    assert report.populations.source_classified_files == 3
    assert report.populations.source_traversal_failures == 1
    assert report.populations.member_versions == 6
    assert report.source.files_seen == 4
    assert report.source.classified_files == 3
    assert report.source.traversal_failures == 1
    assert sum(count for _, count in report.parse.status_counts) == 6
    assert report.reconciliation.source_files_reconcile is True
    assert report.reconciliation.source_format_files == 3
    assert report.reconciliation.source_traversal_failures == 1
    assert report.reconciliation.member_versions_reconcile is True
    assert report.reconciliation.parse_observation_slots == 6
    assert report.reconciliation.acquisition_observation_slots == 5
    assert report.reconciliation.section_populations_reconcile is True


def test_adapted_report_and_observability_outputs_exclude_private_canaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = (
        "CANARY_SOURCE_TEXT /private/canary/source.pdf "
        "canary-filename.pdf 10.1000/canary "
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "canary-corpus-name"
    )
    membership = _membership(_CORPUS_ONE, _SHARED_DOCUMENT, _SHARED_VERSION)
    request = CorpusAuditRequest(
        scope=CorpusAuditScope(_CORPUS_ONE, _VERSION_ONE, (membership,)),
        source_inventory=_source_inventory(_CORPUS_ONE),
        parse_observations=(
            adapt_parse_persistence_request(
                _parse_request(membership, text=marker),
                membership=membership,
            ),
        ),
        capability_profiles=(CURRENT_PARSER_CAPABILITY_PROFILE,),
    )

    report = run_corpus_audit(request, clock=iter((1, 2)).__next__)
    outputs = "\n".join(
        (
            serialise_audit_report(report).decode("utf-8"),
            render_audit_report(report),
            repr(report),
            repr(request),
            repr(request.parse_observations[0]),
        )
    )
    assert marker not in outputs
    assert "CANARY_SOURCE_TEXT" not in outputs
    assert "/private/canary/source.pdf" not in outputs
    assert "canary-filename.pdf" not in outputs
    assert "10.1000/canary" not in outputs
    assert "canary-corpus-name" not in outputs

    private_error = RuntimeError(marker)

    def fail(_request: CorpusAuditRequest) -> CorpusAuditReport:
        raise private_error

    monkeypatch.setattr(audit_observability, "_build_report", fail)
    with capture_logs() as logs:
        with pytest.raises(RuntimeError) as captured:
            run_corpus_audit(request, clock=iter((1, 2)).__next__)

    assert captured.value is not private_error
    assert marker not in str(captured.value)
    assert marker not in repr(logs)


def test_two_corpora_isolate_overlapping_global_documents() -> None:
    corpus_one = _request(
        _CORPUS_ONE,
        _VERSION_ONE,
        (
            (_SHARED_DOCUMENT, _SHARED_VERSION, 10),
            (_CORPUS_ONE_DOCUMENT, _CORPUS_ONE_VERSION, 20),
        ),
    )
    corpus_two = _request(
        _CORPUS_TWO,
        _VERSION_TWO,
        (
            (_SHARED_DOCUMENT, _SHARED_VERSION, 10),
            (_CORPUS_TWO_DOCUMENT, _CORPUS_TWO_VERSION, 40),
        ),
    )

    first = CorpusAuditReport.from_request(corpus_one)
    second = CorpusAuditReport.from_request(corpus_two)

    first_characters = dict(first.parse.numeric_summaries)["character_count"]
    second_characters = dict(second.parse.numeric_summaries)["character_count"]
    assert first.populations.member_versions == 2
    assert second.populations.member_versions == 2
    assert first_characters.distribution.total == 30
    assert second_characters.distribution.total == 50
    assert first.corpus_id == _CORPUS_ONE.as_str()
    assert second.corpus_id == _CORPUS_TWO.as_str()
    assert first.input_digest != second.input_digest
    assert first.report_id != second.report_id
    assert serialise_audit_report(first) != serialise_audit_report(second)
    assert _CORPUS_TWO.as_str() not in serialise_audit_report(first).decode()
    assert _CORPUS_ONE.as_str() not in serialise_audit_report(second).decode()


def test_wrong_corpus_source_inventory_fails_before_report_creation() -> None:
    membership = _membership(_CORPUS_ONE, _SHARED_DOCUMENT, _SHARED_VERSION)
    with pytest.raises(AuditContractError):
        CorpusAuditRequest(
            scope=CorpusAuditScope(_CORPUS_ONE, _VERSION_ONE, (membership,)),
            source_inventory=_source_inventory(_CORPUS_TWO),
            parse_observations=(
                AuditParseObservation(
                    membership=membership,
                    missing_reason=AuditMissingReason.PARSE_NOT_ATTEMPTED,
                ),
            ),
        )
