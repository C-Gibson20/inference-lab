"""Tests for the immutable baseline-ingest contract vocabulary."""

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


def test_baseline_stages_have_stable_order_and_values():
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


def test_baseline_statuses_are_closed() -> None:
    assert BaselineExecutionStatus.COMPLETED.value == "completed"
    assert BaselinePublicationStatus.NO_CHANGE.value == "no_change"
    assert BaselineDocumentOutcomeKind.FAILED.value == "failed"
    assert BaselineFailureScope.REQUEST.value == "request"
    assert BaselineFailureSeverity.FATAL.value == "fatal"
    assert BaselineFailureCategory.CORRUPT_STATE.value == "corrupt_state"


def test_unknown_baseline_stage_is_rejected() -> None:
    with pytest.raises(ValueError):
        BaselineStage("not_a_stage")


def test_request_preserves_explicit_empty_parent() -> None:
    request = _request(expected_corpus_version=None)

    assert request.expected_corpus_version is None
    assert "/private/input" not in repr(request)


def test_request_rejects_discovery_for_another_corpus() -> None:
    with pytest.raises(ValueError, match="request corpus"):
        _request(discovery_corpus_id=_OTHER_CORPUS_ID)


def test_request_rejects_proposed_version_equal_to_expected_version() -> None:
    request = _request()

    with pytest.raises(ValueError, match="must differ"):
        BaselineIngestRequest(
            corpus_id=request.corpus_id,
            discovery_request=request.discovery_request,
            proposed_corpus_version=request.proposed_corpus_version,
            expected_corpus_version=request.proposed_corpus_version,
            parser_version=request.parser_version,
            chunker_version=request.chunker_version,
            tokeniser_identity=request.tokeniser_identity,
            chunking_policy=request.chunking_policy,
            max_concurrent_documents=request.max_concurrent_documents,
            configuration_identity=request.configuration_identity,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        ("2", TypeError),
    ],
)
def test_request_rejects_invalid_concurrency(
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _request(max_concurrent_documents=value)  # type: ignore[arg-type]


def test_document_parse_failure_is_bounded_and_orderable() -> None:
    failure = BaselineFailure(
        stage=BaselineStage.PARSE,
        category=BaselineFailureCategory.PARSE_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )

    assert failure.key() == (
        "parse",
        "parse_failed",
        "document",
        "fatal",
        0,
    )
    assert "source text" not in repr(failure)


def test_request_publication_conflict_is_bounded() -> None:
    failure = BaselineFailure(
        stage=BaselineStage.PUBLICATION,
        category=BaselineFailureCategory.PUBLICATION_CONFLICT,
        scope=BaselineFailureScope.REQUEST,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )

    assert failure.scope is BaselineFailureScope.REQUEST
    assert failure.category is BaselineFailureCategory.PUBLICATION_CONFLICT


@pytest.mark.parametrize(
    "field_name",
    ("stage", "category", "scope", "severity"),
)
def test_failure_rejects_wrong_closed_value_type(field_name: str) -> None:
    values = {
        "stage": BaselineStage.PARSE,
        "category": BaselineFailureCategory.PARSE_FAILED,
        "scope": BaselineFailureScope.DOCUMENT,
        "severity": BaselineFailureSeverity.FATAL,
        "retryable": False,
    }
    values[field_name] = "invalid"  # type: ignore[assignment]

    with pytest.raises(TypeError):
        BaselineFailure(**values)  # type: ignore[arg-type]


def test_failure_rejects_non_boolean_retryability() -> None:
    with pytest.raises(TypeError):
        BaselineFailure(
            stage=BaselineStage.PARSE,
            category=BaselineFailureCategory.PARSE_FAILED,
            scope=BaselineFailureScope.DOCUMENT,
            severity=BaselineFailureSeverity.FATAL,
            retryable=1,  # type: ignore[arg-type]
        )


def test_unchanged_document_requires_reused_stages() -> None:
    outcome = BaselineDocumentOutcome(
        ordinal=3,
        kind=BaselineDocumentOutcomeKind.UNCHANGED_REUSED,
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        parse_disposition=BaselineStageDisposition.REUSED,
        chunk_disposition=BaselineStageDisposition.REUSED,
    )

    assert outcome.key() == (
        3,
        "unchanged_reused",
        _DOCUMENT_ID.as_str(),
        _DOCUMENT_VERSION_ID.as_str(),
    )


def test_unsupported_document_does_not_enter_parse_or_chunk() -> None:
    outcome = BaselineDocumentOutcome(
        ordinal=4,
        kind=BaselineDocumentOutcomeKind.UNSUPPORTED,
    )

    assert outcome.parse_disposition is BaselineStageDisposition.NOT_ATTEMPTED
    assert outcome.chunk_disposition is BaselineStageDisposition.NOT_ATTEMPTED


def test_parse_failure_is_attached_to_a_failed_document() -> None:
    failure = BaselineFailure(
        stage=BaselineStage.PARSE,
        category=BaselineFailureCategory.PARSE_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )
    outcome = BaselineDocumentOutcome(
        ordinal=5,
        kind=BaselineDocumentOutcomeKind.FAILED,
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        parse_disposition=BaselineStageDisposition.FAILED,
        failures=(failure,),
    )

    assert outcome.failures == (failure,)


def test_chunk_persistence_failure_remains_distinct_from_chunk_failure() -> None:
    failure = BaselineFailure(
        stage=BaselineStage.CHUNK_PERSISTENCE,
        category=BaselineFailureCategory.CHUNK_PERSISTENCE_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.FATAL,
        retryable=True,
    )
    outcome = BaselineDocumentOutcome(
        ordinal=6,
        kind=BaselineDocumentOutcomeKind.FAILED,
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        parse_disposition=BaselineStageDisposition.COMPLETED,
        chunk_disposition=BaselineStageDisposition.FAILED,
        failures=(failure,),
    )

    assert outcome.failures[0].stage is BaselineStage.CHUNK_PERSISTENCE
    assert outcome.failures[0].category is (
        BaselineFailureCategory.CHUNK_PERSISTENCE_FAILED
    )


def test_failed_document_requires_a_failure() -> None:
    with pytest.raises(ValueError, match="at least one failure"):
        BaselineDocumentOutcome(
            ordinal=7,
            kind=BaselineDocumentOutcomeKind.FAILED,
        )


def test_document_outcome_rejects_request_failure() -> None:
    failure = BaselineFailure(
        stage=BaselineStage.PUBLICATION,
        category=BaselineFailureCategory.PUBLICATION_CONFLICT,
        scope=BaselineFailureScope.REQUEST,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )

    with pytest.raises(ValueError, match="document failures"):
        BaselineDocumentOutcome(
            ordinal=8,
            kind=BaselineDocumentOutcomeKind.FAILED,
            failures=(failure,),
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


def test_completed_summary_is_canonically_ordered() -> None:
    summary = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=_EXPECTED_VERSION,
        execution_status=BaselineExecutionStatus.COMPLETED,
        publication_status=BaselinePublicationStatus.COMMITTED,
        document_outcomes=(_unchanged_outcome(),),
        stage_counts=(
            BaselineStageCounts(
                stage=BaselineStage.PARSE,
                reused=1,
            ),
            BaselineStageCounts(
                stage=BaselineStage.CHUNK,
                reused=1,
            ),
        ),
    )

    assert summary.stage_counts[0].stage is BaselineStage.PARSE
    assert summary.document_outcomes[0].ordinal == 0


def test_completed_with_failures_requires_failed_document() -> None:
    with pytest.raises(ValueError, match="failed document"):
        BaselineIngestSummary(
            corpus_id=_CORPUS_ID,
            proposed_corpus_version=_PROPOSED_VERSION,
            expected_corpus_version=_EXPECTED_VERSION,
            execution_status=(BaselineExecutionStatus.COMPLETED_WITH_FAILURES),
            publication_status=BaselinePublicationStatus.NO_CHANGE,
        )


def test_failed_summary_requires_request_failure_and_no_publication() -> None:
    failure = BaselineFailure(
        stage=BaselineStage.DISCOVERY,
        category=BaselineFailureCategory.DISCOVERY_FAILED,
        scope=BaselineFailureScope.REQUEST,
        severity=BaselineFailureSeverity.FATAL,
        retryable=False,
    )

    summary = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=None,
        execution_status=BaselineExecutionStatus.FAILED,
        publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
        failures=(failure,),
    )

    assert summary.publication_status is (BaselinePublicationStatus.NOT_PUBLISHED)


def test_summary_rejects_out_of_order_stage_counts() -> None:
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


def test_summary_allows_completed_execution_before_publication() -> None:
    summary = BaselineIngestSummary(
        corpus_id=_CORPUS_ID,
        proposed_corpus_version=_PROPOSED_VERSION,
        expected_corpus_version=_EXPECTED_VERSION,
        execution_status=BaselineExecutionStatus.COMPLETED,
        publication_status=BaselinePublicationStatus.NOT_PUBLISHED,
        document_outcomes=(_unchanged_outcome(),),
    )

    assert summary.execution_status is BaselineExecutionStatus.COMPLETED


def test_baseline_contracts_are_public_ingest_api() -> None:
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
    )

    for name in public_names:
        assert name in ingest.__all__
        assert hasattr(ingest, name)


def test_public_contracts_preserve_corpus_and_proposal_identity() -> None:
    from inference_lab import ingest

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
