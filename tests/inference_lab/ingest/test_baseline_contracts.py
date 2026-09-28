"""Compact ING-05 contracts; full stage behavior lives in ingest acceptance."""

from pathlib import Path

import pytest

from inference_lab.ingest.baseline_contracts import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineExecutionStatus,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineIngestRequest,
    BaselineIngestSummary,
    BaselinePublicationStatus,
    BaselineStage,
    BaselineStageCounts,
    BaselineStageDisposition,
)
from inference_lab.ingest.chunking import (
    CURRENT_CHUNKER_VERSION,
    ChunkingPolicy,
    TokeniserIdentity,
)
from inference_lab.ingest.discovery import (
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
)
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_OTHER_CORPUS_ID = CorpusId("22222222-2222-4222-8222-222222222222")
_PROPOSED_VERSION = CorpusVersionId("33333333-3333-4333-8333-333333333333")
_EXPECTED_VERSION = CorpusVersionId("44444444-4444-4444-8444-444444444444")
_DOCUMENT_ID = DocumentId("55555555-5555-4555-8555-555555555555")
_DOCUMENT_VERSION_ID = DocumentVersionId("66666666-6666-4666-8666-666666666666")


def _request(
    *,
    corpus_id: CorpusId = _CORPUS_ID,
    discovery_corpus_id: CorpusId = _CORPUS_ID,
    expected_corpus_version: CorpusVersionId | None = _EXPECTED_VERSION,
    max_concurrent_documents: int = 1,
) -> BaselineIngestRequest:
    discovery_request = DiscoveryRequest(
        corpus_id=discovery_corpus_id,
        inputs=(
            DiscoveryInput(
                root=Path("/private/input"),
                kind=DiscoveryInputKind.DIRECTORY,
            ),
        ),
    )
    return BaselineIngestRequest(
        corpus_id=corpus_id,
        discovery_request=discovery_request,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=expected_corpus_version,
        parser_version=CURRENT_PARSER_VERSION,
        chunker_version=CURRENT_CHUNKER_VERSION,
        tokeniser_identity=TokeniserIdentity(
            implementation="test-tokeniser",
            contract_version="1.0.0",
            artefact_identity="fixture",
            artefact_revision="1",
        ),
        chunking_policy=ChunkingPolicy(),
        max_concurrent_documents=max_concurrent_documents,
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest="0" * 64,
        ),
    )


def _unchanged_outcome() -> BaselineDocumentOutcome:
    return BaselineDocumentOutcome(
        ordinal=0,
        kind=BaselineDocumentOutcomeKind.UNCHANGED_REUSED,
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        parse_disposition=BaselineStageDisposition.REUSED,
        chunk_disposition=BaselineStageDisposition.REUSED,
    )


def test_ing05_request_and_stage_identity_are_corpus_scoped_and_closed() -> None:
    request = _request(expected_corpus_version=None)
    assert request.expected_corpus_version is None
    assert "/private/input" not in repr(request)
    assert tuple(stage.value for stage in BaselineStage) == (
        "discovery",
        "source_acquisition",
        "parse",
        "parse_persistence",
        "chunk",
        "chunk_persistence",
        "membership",
        "publication",
    )
    assert BaselineExecutionStatus.COMPLETED.value == "completed"
    assert BaselinePublicationStatus.NO_CHANGE.value == "no_change"
    assert BaselineDocumentOutcomeKind.FAILED.value == "failed"
    assert BaselineFailureScope.REQUEST.value == "request"
    assert BaselineFailureSeverity.FATAL.value == "fatal"
    assert BaselineFailureCategory.CORRUPT_STATE.value == "corrupt_state"

    with pytest.raises(ValueError):
        BaselineStage("not_a_stage")
    with pytest.raises(ValueError):
        BaselineExecutionStatus("not_a_status")
    with pytest.raises(ValueError, match="request corpus"):
        _request(discovery_corpus_id=_OTHER_CORPUS_ID)
    with pytest.raises(ValueError, match="must differ"):
        BaselineIngestRequest(
            corpus_id=request.corpus_id,
            discovery_request=request.discovery_request,
            proposed_corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=_PROPOSED_VERSION,
            parser_version=request.parser_version,
            chunker_version=request.chunker_version,
            tokeniser_identity=request.tokeniser_identity,
            chunking_policy=request.chunking_policy,
            max_concurrent_documents=request.max_concurrent_documents,
            configuration_identity=request.configuration_identity,
        )
    for concurrency, error in ((0, ValueError), (-1, ValueError), (True, TypeError)):
        with pytest.raises(error):
            _request(max_concurrent_documents=concurrency)  # type: ignore[arg-type]


def test_ing05_failures_keep_bounded_stage_scope_and_retryability() -> None:
    parse_failure = BaselineFailure(
        stage=BaselineStage.PARSE,
        category=BaselineFailureCategory.PARSE_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )
    request_failure = BaselineFailure(
        stage=BaselineStage.PUBLICATION,
        category=BaselineFailureCategory.PUBLICATION_CONFLICT,
        scope=BaselineFailureScope.REQUEST,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )
    assert parse_failure.scope is BaselineFailureScope.DOCUMENT
    assert parse_failure.key() == (
        "parse",
        "parse_failed",
        "document",
        "fatal",
        0,
    )
    assert request_failure.category is BaselineFailureCategory.PUBLICATION_CONFLICT
    assert "source text" not in repr(parse_failure)

    fields = {
        "stage": BaselineStage.PARSE,
        "category": BaselineFailureCategory.PARSE_FAILED,
        "scope": BaselineFailureScope.DOCUMENT,
        "severity": BaselineFailureSeverity.FATAL,
        "retryable": False,
    }
    for field in ("stage", "category", "scope", "severity"):
        invalid = dict(fields)
        invalid[field] = "invalid"
        with pytest.raises(TypeError):
            BaselineFailure(**invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        BaselineFailure(**(fields | {"retryable": 1}))  # type: ignore[arg-type]


def test_ing05_document_outcomes_distinguish_reuse_unsupported_and_failure() -> None:
    unchanged = _unchanged_outcome()
    unsupported = BaselineDocumentOutcome(
        ordinal=1,
        kind=BaselineDocumentOutcomeKind.UNSUPPORTED,
    )
    assert unchanged.parse_disposition is BaselineStageDisposition.REUSED
    assert unchanged.chunk_disposition is BaselineStageDisposition.REUSED
    assert unsupported.parse_disposition is BaselineStageDisposition.NOT_ATTEMPTED
    assert unsupported.chunk_disposition is BaselineStageDisposition.NOT_ATTEMPTED

    parse_failure = BaselineFailure(
        stage=BaselineStage.PARSE,
        category=BaselineFailureCategory.PARSE_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )
    chunk_write_failure = BaselineFailure(
        stage=BaselineStage.CHUNK_PERSISTENCE,
        category=BaselineFailureCategory.CHUNK_PERSISTENCE_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.FATAL,
        retryable=True,
    )
    for ordinal, failure, parse, chunk in (
        (
            2,
            parse_failure,
            BaselineStageDisposition.FAILED,
            BaselineStageDisposition.NOT_ATTEMPTED,
        ),
        (
            3,
            chunk_write_failure,
            BaselineStageDisposition.COMPLETED,
            BaselineStageDisposition.FAILED,
        ),
    ):
        failed = BaselineDocumentOutcome(
            ordinal=ordinal,
            kind=BaselineDocumentOutcomeKind.FAILED,
            document_id=_DOCUMENT_ID,
            document_version_id=_DOCUMENT_VERSION_ID,
            parse_disposition=parse,
            chunk_disposition=chunk,
            failures=(failure,),
        )
        assert failed.failures == (failure,)

    with pytest.raises(ValueError, match="at least one failure"):
        BaselineDocumentOutcome(ordinal=4, kind=BaselineDocumentOutcomeKind.FAILED)
    request_failure = BaselineFailure(
        stage=BaselineStage.PUBLICATION,
        category=BaselineFailureCategory.PUBLICATION_CONFLICT,
        scope=BaselineFailureScope.REQUEST,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )
    with pytest.raises(ValueError, match="document failures"):
        BaselineDocumentOutcome(
            ordinal=5,
            kind=BaselineDocumentOutcomeKind.FAILED,
            failures=(request_failure,),
        )


def test_ing05_summaries_reconcile_document_failures_and_publication_state() -> None:
    completed = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=_EXPECTED_VERSION,
        execution_status=BaselineExecutionStatus.COMPLETED,
        publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
        document_outcomes=(_unchanged_outcome(),),
        stage_counts=(
            BaselineStageCounts(stage=BaselineStage.PARSE, reused=1),
            BaselineStageCounts(stage=BaselineStage.CHUNK, reused=1),
        ),
    )
    assert completed.stage_counts[0].stage is BaselineStage.PARSE
    assert completed.document_outcomes[0].ordinal == 0

    request_failure = BaselineFailure(
        stage=BaselineStage.DISCOVERY,
        category=BaselineFailureCategory.DISCOVERY_FAILED,
        scope=BaselineFailureScope.REQUEST,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )
    failed = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        execution_status=BaselineExecutionStatus.FAILED,
        publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
        failures=(request_failure,),
    )
    assert failed.publication_status is BaselinePublicationStatus.NOT_PUBLISHED

    with pytest.raises(ValueError, match="failed document"):
        BaselineIngestSummary(
            corpus_id=_CORPUS_ID,
            proposed_corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=_EXPECTED_VERSION,
            execution_status=BaselineExecutionStatus.COMPLETED_WITH_FAILURES,
            publication_status=BaselinePublicationStatus.NO_CHANGE,
        )
    with pytest.raises(ValueError, match="canonical order"):
        BaselineIngestSummary(
            corpus_id=_CORPUS_ID,
            proposed_corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=_EXPECTED_VERSION,
            execution_status=BaselineExecutionStatus.COMPLETED,
            publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
            stage_counts=(
                BaselineStageCounts(stage=BaselineStage.CHUNK),
                BaselineStageCounts(stage=BaselineStage.DISCOVERY),
            ),
        )


def test_ing05_public_api_preserves_corpus_and_proposal_identity() -> None:
    from inference_lab import ingest

    public_names = (
        "BaselineCheckpoint",
        "BaselineCheckpointIdentity",
        "BaselineCheckpointMappingError",
        "BaselineCheckpointStatus",
        "BaselineDocumentOutcome",
        "BaselineDocumentOutcomeKind",
        "BaselineExecutionStatus",
        "BaselineFailure",
        "BaselineFailureCategory",
        "BaselineFailureScope",
        "BaselineFailureSeverity",
        "BaselineIngestManifest",
        "BaselineIngestRequest",
        "BaselineIngestSummary",
        "BaselineManifestState",
        "BaselinePublicationStatus",
        "BaselineStage",
        "BaselineStageCounts",
        "BaselineStageDisposition",
        "BaselineWorkKey",
        "baseline_manifest_from_record",
        "baseline_manifest_to_record",
        "baseline_ingest_summary_payload",
        "serialise_baseline_ingest_summary",
    )
    assert all(
        name in ingest.__all__ and hasattr(ingest, name) for name in public_names
    )

    request = _request()
    summary = ingest.BaselineIngestSummary(
        corpus_id=request.corpus_id,
        proposed_corpus_version=request.proposed_corpus_version,
        expected_corpus_version=request.expected_corpus_version,
        execution_status=ingest.BaselineExecutionStatus.COMPLETED,
        publication_status=ingest.BaselinePublicationStatus.NOT_PUBLISHED,
    )
    assert summary.corpus_id == request.corpus_id
    assert summary.proposed_corpus_version == request.proposed_corpus_version
    assert summary.expected_corpus_version == request.expected_corpus_version
