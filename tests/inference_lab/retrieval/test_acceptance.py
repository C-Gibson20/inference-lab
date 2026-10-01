"""Acceptance tests for channel-neutral retrieval contracts."""

from typing import cast

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
    AllFilter,
    AnyFilter,
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    ExactFilter,
    ExistsFilter,
    FailureCategory,
    InvalidRetrievalContractError,
    InvalidRetrievalResultError,
    MissingFilter,
    NotFilter,
    OneOfFilter,
    RangeFilter,
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
        "FUSION_POLICY_VERSION",
        "FusionPolicy",
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
        "DENSE_COMPONENT_KIND",
        "DENSE_CONTRACT_VERSION",
        "DENSE_FORMAT_VERSION",
        "DENSE_IMPLEMENTATION",
        "DENSE_SCORE_LABEL",
        "DenseDistanceMetric",
        "DenseError",
        "DenseFailureCategory",
        "DenseGeneration",
        "DenseGenerationParticipant",
        "DenseGenerationValidator",
        "DenseIndexBuildOutcome",
        "DenseIndexBuildRequest",
        "DenseIndexBuildStatistics",
        "DenseIndexCompatibility",
        "DenseIndexConfiguration",
        "DenseIndexEntry",
        "DenseIndexManifest",
        "DenseIndexMatch",
        "DenseIndexQuery",
        "DenseMemberSelection",
        "DenseVector",
        "DenseVectorIndex",
        "ExactDenseVectorIndex",
        "InvalidDenseContractError",
        "build_dense_generation",
        "build_dense_generation_for_corpus_update",
        "open_dense_generation",
        "rank_dense_entries",
        "select_dense_parent_reference",
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
        "LexicalCandidateMetadata",
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
        "fuse_evidence_results",
        "escape_fts5_literal",
        "execute_lexical_retrieval",
        "execute_dense_retrieval",
        "expose_bm25_score",
        "extract_identifiers",
        "identifier_score",
        "lexical_order_key",
        "lexical_score_label",
        "open_lexical_generation",
        "serialise_evidence_result",
        "serialise_lexical_manifest",
        "serialise_query",
        "resolve_lexical_candidate_metadata",
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


def test_ret01_query_candidate_and_score_boundaries_reject_invalid_values() -> None:
    source = _source(31)
    candidate = RetrievalCandidate(
        RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, "passage", (source,)
    )
    version = CorpusVersionId.generate()

    with pytest.raises(InvalidRetrievalContractError):
        RetrievalQuery(" \n", version, RetrievalLimit(1))
    with pytest.raises(TypeError):
        RetrievalQuery("question", str(version), RetrievalLimit(1))  # type: ignore[arg-type]
    with pytest.raises(InvalidRetrievalContractError):
        RetrievalLimit(0)
    with pytest.raises(InvalidRetrievalContractError):
        RetrievalCandidate(
            RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, " ", (source,)
        )
    with pytest.raises(TypeError):
        EvidenceSource(
            cast(DocumentId, source.document_version_id),
            source.document_version_id,
            source.source_span_id,
        )  # type: ignore[arg-type]
    with pytest.raises(InvalidRetrievalContractError):
        ChannelScore(RetrievalChannel.LEXICAL, 1, float("nan"), "score")

    lexical = ChannelScore(RetrievalChannel.LEXICAL, 1, 1.0, "bm25")
    with pytest.raises(InvalidRetrievalContractError):
        EvidenceHit(candidate, (lexical, lexical), 1)


def test_ret01_order_limit_and_corpus_version_mismatch_fail_closed() -> None:
    version = CorpusVersionId.generate()
    other_version = CorpusVersionId.generate()
    first = RetrievalCandidate(
        RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, "first", (_source(41),)
    )
    second = RetrievalCandidate(
        RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, "second", (_source(42),)
    )
    first_hit = EvidenceHit(first, (ChannelScore(RetrievalChannel.LEXICAL, 1),), 1)
    second_hit = EvidenceHit(second, (ChannelScore(RetrievalChannel.LEXICAL, 2),), 2)
    query = RetrievalQuery("query", version, RetrievalLimit(2))

    result = EvidenceResult.from_query(
        query,
        observed_corpus_version=other_version,
        hits=(first_hit,),
        status=ResultStatus.COMPLETED,
        failures=(),
    )
    assert result.status is ResultStatus.FAILED
    assert result.hits == ()
    assert result.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH

    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(
            version,
            RetrievalLimit(2),
            (second_hit, first_hit),
            ResultStatus.COMPLETED,
            (),
        )
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult.from_query(
            RetrievalQuery("query", version, RetrievalLimit(1)),
            observed_corpus_version=version,
            hits=(first_hit, second_hit),
            status=ResultStatus.COMPLETED,
            failures=(),
        )


def test_ret01_query_serialisation_round_trips_nested_filter_shapes() -> None:
    expression = AllFilter(
        (
            AnyFilter(
                (
                    ExactFilter("language", "en"),
                    OneOfFilter("document-type", ("text/plain", "application/pdf")),
                )
            ),
            NotFilter(MissingFilter("page-count")),
            RangeFilter("token-count", 1, 100),
            ExistsFilter("source-byte-count"),
        )
    )
    query = RetrievalQuery(
        "filtered query",
        CorpusVersionId.generate(),
        RetrievalLimit(5),
        expression,
    )

    encoded = serialise_query(query)
    assert deserialise_query(encoded) == query
