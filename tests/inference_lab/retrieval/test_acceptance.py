"""Acceptance tests for channel-neutral retrieval contracts."""

import pytest
from structlog.testing import capture_logs

import inference_lab.retrieval as retrieval
from inference_lab.provenance import (
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    FailureCategory,
    InvalidRetrievalContractError,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
    RetrievalQuery,
    deserialise_evidence_result,
    deserialise_query,
    serialise_evidence_result,
    serialise_query,
)


def _source(number: int) -> EvidenceSource:
    """Return deterministic source identities for the acceptance fixture."""
    return EvidenceSource(
        DocumentId.from_str(f"00000000-0000-4000-8000-{number:012d}"),
        DocumentVersionId.from_str(f"00000000-0000-4000-8000-{number + 10:012d}"),
        SourceSpanId.from_str(f"00000000-0000-4000-8000-{number + 20:012d}"),
    )


def test_all_retrieval_channels_share_one_provenance_preserving_envelope() -> None:
    channels = (
        (RetrievalChannel.LEXICAL, CandidateKind.ORIGINAL_PASSAGE, "bm25"),
        (RetrievalChannel.DENSE, CandidateKind.ORIGINAL_PASSAGE, "cosine"),
        (RetrievalChannel.METADATA, CandidateKind.ORIGINAL_PASSAGE, None),
        (RetrievalChannel.STRUCTURED, CandidateKind.STRUCTURED_RECORD, "fact-match"),
        (RetrievalChannel.SUMMARY, CandidateKind.GENERATED_SUMMARY, "summary-rank"),
    )
    hits = tuple(
        EvidenceHit(
            RetrievalCandidate(
                RecordId.from_str(f"00000000-0000-4000-8000-{index + 100:012d}"),
                kind,
                f"candidate {index}",
                (_source(index + 1),),
            ),
            (
                ChannelScore(
                    channel,
                    index,
                    None if label is None else float(index),
                    label,
                ),
            ),
            index,
        )
        for index, (channel, kind, label) in enumerate(channels, start=1)
    )
    result = EvidenceResult(
        CorpusVersionId.from_str("00000000-0000-4000-8000-000000000200"),
        RetrievalLimit(5),
        hits,
        ResultStatus.COMPLETED,
        (),
    )

    restored = deserialise_evidence_result(serialise_evidence_result(result))
    assert restored == result
    assert tuple(hit.contributions[0].channel for hit in restored.hits) == tuple(
        channel for channel, _, _ in channels
    )
    assert tuple(hit.contributions[0].score_label for hit in restored.hits) == tuple(
        label for _, _, label in channels
    )
    assert tuple(hit.candidate.sources[0] for hit in restored.hits) == tuple(
        _source(index) for index in range(2, 7)
    )


def test_retrieval_public_api_contains_only_completed_contract_values() -> None:
    expected = {
        "CandidateKind",
        "ChannelScore",
        "EvidenceHit",
        "EvidenceResult",
        "EvidenceSource",
        "FailureCategory",
        "InvalidRetrievalContractError",
        "InvalidRetrievalResultError",
        "InvalidRetrievalSerialisationError",
        "ResultStatus",
        "RetrievalCandidate",
        "RetrievalChannel",
        "RetrievalFailure",
        "RetrievalLimit",
        "RetrievalQuery",
        "CURRENT_METADATA_FILTER_REGISTRY",
        "CURRENT_RETRIEVAL_SCHEMA_VERSION",
        "MAX_METADATA_FILTER_DEPTH",
        "MAX_METADATA_FILTER_NODES",
        "METADATA_FILTER_REGISTRY_VERSION",
        "SUPPORTED_RETRIEVAL_SCHEMA_VERSIONS",
        "UnsupportedRetrievalSchemaVersionError",
        "AllFilter",
        "AnyFilter",
        "ExactFilter",
        "ExistsFilter",
        "FilterExpression",
        "FilterScalar",
        "MissingFilter",
        "NotFilter",
        "OneOfFilter",
        "RangeFilter",
        "LEXICAL_COMPONENT_KIND",
        "LEXICAL_CONTRACT_VERSION",
        "LEXICAL_FORMAT_VERSION",
        "LEXICAL_IDENTIFIER_SCORE_LABEL",
        "LEXICAL_IMPLEMENTATION",
        "LEXICAL_MANIFEST_SCHEMA_VERSION",
        "LEXICAL_SCORE_LABEL",
        "InvalidLexicalContractError",
        "InvalidLexicalSerialisationError",
        "InvalidMetadataFilterError",
        "LexicalError",
        "LexicalFailureCategory",
        "LexicalIndexBuildOutcome",
        "LexicalIndexBuildStatistics",
        "LexicalIndexConfiguration",
        "LexicalIndexManifest",
        "LexicalGeneration",
        "LexicalGenerationParticipant",
        "LexicalGenerationValidator",
        "LexicalIndexedChunk",
        "LexicalMemberSelection",
        "LexicalMatchMode",
        "LexicalQuery",
        "LexicalRetrievalRequest",
        "LexicalSourceLink",
        "MetadataCardinality",
        "MetadataFieldDescriptor",
        "MetadataFieldType",
        "MetadataFilterErrorCategory",
        "MetadataFilterOperator",
        "MetadataFilterRegistry",
        "MetadataFilterValidation",
        "MetadataMissingPolicy",
        "MetadataValue",
        "MetadataValues",
        "analyse_terms",
        "build_fts5_query",
        "build_lexical_generation",
        "build_lexical_generation_for_corpus_update",
        "derive_lexical_generation_id",
        "deserialise_lexical_manifest",
        "deserialise_evidence_result",
        "deserialise_query",
        "evaluate_metadata_filter",
        "escape_fts5_literal",
        "execute_lexical_retrieval",
        "expose_bm25_score",
        "extract_identifiers",
        "identifier_score",
        "lexical_order_key",
        "lexical_score_label",
        "open_lexical_generation",
        "serialise_evidence_result",
        "serialise_lexical_manifest",
        "serialise_query",
        "validate_metadata_filter",
        "select_lexical_parent_reference",
    }
    assert set(retrieval.__all__) == expected
    assert not hasattr(retrieval, "_filter_payload")
    assert not hasattr(retrieval, "_candidate_payload")


def test_validation_and_serialisation_are_payload_private_and_log_free() -> None:
    marker = "private-retrieval-payload-marker"
    query = RetrievalQuery(marker, CorpusVersionId.generate(), RetrievalLimit(1))
    with capture_logs() as logs:
        assert deserialise_query(serialise_query(query)) == query
        with pytest.raises(InvalidRetrievalContractError) as error:
            RetrievalCandidate(
                RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, marker, ()
            )
    assert marker not in str(error.value)
    assert logs == []


def test_failure_values_remain_typed_and_do_not_change_empty_result_semantics() -> None:
    version = CorpusVersionId.generate()
    empty = EvidenceResult(version, RetrievalLimit(1), (), ResultStatus.COMPLETED, ())
    failed = EvidenceResult(
        version,
        RetrievalLimit(1),
        (),
        ResultStatus.FAILED,
        (RetrievalFailure(FailureCategory.CHANNEL_EXECUTION_FAILURE, True),),
    )
    assert empty.status is ResultStatus.COMPLETED
    assert failed.status is ResultStatus.FAILED
    assert failed.failures[0].category is FailureCategory.CHANNEL_EXECUTION_FAILURE
