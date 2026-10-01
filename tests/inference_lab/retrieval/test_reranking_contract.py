"""RET-06 query and fused-result boundary validation."""

import pytest

from inference_lab.backends.runtime import (
    BackendMetadata,
    BackendOperation,
    FakeBackend,
    FakeBackendConfig,
    FakeFailureMode,
    RerankResult,
    ScoredCandidate,
)
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
    EvidenceCount,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    FailureCategory,
    InvalidRetrievalContractError,
    RerankCandidateLimit,
    RerankingRequest,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
    ScoredEvidence,
    deduplicate_evidence_hits,
    execute_reranking,
    map_rerank_result,
    prepare_rerank_request,
    rank_evidence_hits,
    rerank_evidence,
)


def _version(number: int) -> CorpusVersionId:
    """Return one stable corpus version for a contract fixture."""
    return CorpusVersionId.from_str(f"00000000-0000-4000-8000-{number:012d}")


def _query(*, version: CorpusVersionId | None = None, limit: int = 8) -> RetrievalQuery:
    """Build a query whose limit is retained from the fused result."""
    return RetrievalQuery("question", version or _version(1), RetrievalLimit(limit))


def _result(
    query: RetrievalQuery,
    *,
    version: CorpusVersionId | None = None,
    limit: int | None = None,
    with_hit: bool = True,
) -> EvidenceResult:
    """Build a source-linked fused result for the selected query."""
    hits: tuple[EvidenceHit, ...] = ()
    if with_hit:
        candidate = RetrievalCandidate(
            record_id=RecordId.from_str("00000000-0000-4000-8000-000000000010"),
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content="source passage",
            sources=(
                EvidenceSource(
                    document_id=DocumentId.from_str(
                        "00000000-0000-4000-8000-000000000020"
                    ),
                    document_version_id=DocumentVersionId.from_str(
                        "00000000-0000-4000-8000-000000000030"
                    ),
                    source_span_id=SourceSpanId.from_str(
                        "00000000-0000-4000-8000-000000000040"
                    ),
                ),
            ),
        )
        hits = (
            EvidenceHit(
                candidate=candidate,
                contributions=(ChannelScore(RetrievalChannel.LEXICAL, 1),),
                rank=1,
            ),
        )
    selected_limit = query.limit if limit is None else RetrievalLimit(limit)
    return EvidenceResult(
        corpus_version=version or query.corpus_version,
        requested_limit=selected_limit,
        hits=hits,
        status=ResultStatus.COMPLETED,
        failures=(),
    )


def _many_candidate_result(query: RetrievalQuery, count: int) -> EvidenceResult:
    """Build a fused pool whose IDs and source links are easy to inspect."""
    hits = tuple(
        EvidenceHit(
            candidate=RetrievalCandidate(
                record_id=RecordId.from_str(
                    f"00000000-0000-4000-8000-{number + 10:012d}"
                ),
                kind=CandidateKind.ORIGINAL_PASSAGE,
                content=f"source passage {number}",
                sources=(
                    EvidenceSource(
                        document_id=DocumentId.from_str(
                            f"00000000-0000-4000-8000-{number + 100:012d}"
                        ),
                        document_version_id=DocumentVersionId.from_str(
                            f"00000000-0000-4000-8000-{number + 200:012d}"
                        ),
                        source_span_id=SourceSpanId.from_str(
                            f"00000000-0000-4000-8000-{number + 300:012d}"
                        ),
                    ),
                ),
            ),
            contributions=(ChannelScore(RetrievalChannel.LEXICAL, number + 1),),
            rank=number + 1,
        )
        for number in range(count)
    )
    return EvidenceResult(
        corpus_version=query.corpus_version,
        requested_limit=query.limit,
        hits=hits,
        status=ResultStatus.COMPLETED,
        failures=(),
    )


def test_reranking_request_accepts_matching_source_linked_result() -> None:
    query = _query()
    result = _result(query)

    request = RerankingRequest(query, result)

    assert request.query is query
    assert request.fused_result is result
    assert request.fused_result.hits[0].candidate.sources[0].source_span_id == (
        SourceSpanId.from_str("00000000-0000-4000-8000-000000000040")
    )


def test_reranking_request_accepts_empty_fused_result() -> None:
    query = _query()

    request = RerankingRequest(query, _result(query, with_hit=False))

    assert request.fused_result.hits == ()


def test_reranking_request_rejects_other_corpus_version() -> None:
    query = _query()

    with pytest.raises(InvalidRetrievalContractError):
        RerankingRequest(query, _result(query, version=_version(2)))


def test_reranking_request_rejects_different_result_limit() -> None:
    query = _query(limit=8)

    with pytest.raises(InvalidRetrievalContractError):
        RerankingRequest(query, _result(query, limit=7))


@pytest.mark.parametrize("value", (5, 8, 10))
def test_evidence_count_accepts_values_in_the_baseline_range(value: int) -> None:
    assert EvidenceCount(value).value == value


@pytest.mark.parametrize("value", (0, 4, 11))
def test_evidence_count_rejects_values_outside_the_baseline_range(value: int) -> None:
    with pytest.raises(InvalidRetrievalContractError):
        EvidenceCount(value)


@pytest.mark.parametrize("value", (1, 100))
def test_rerank_candidate_limit_accepts_values_within_backend_cap(value: int) -> None:
    assert RerankCandidateLimit(value).value == value


def test_rerank_candidate_limit_rejects_values_above_backend_cap() -> None:
    with pytest.raises(InvalidRetrievalContractError):
        RerankCandidateLimit(101)


def test_request_builder_applies_candidate_cap_and_preserves_candidate_payloads() -> (
    None
):
    query = _query(limit=12)
    original = _many_candidate_result(query, 12)
    reranking_request = RerankingRequest(query, original)

    prepared = prepare_rerank_request(
        reranking_request,
        evidence_count=EvidenceCount(8),
        candidate_limit=RerankCandidateLimit(5),
    )

    assert prepared.backend_request is not None
    assert prepared.backend_request.query == query.text
    assert prepared.backend_request.top_k == 8
    assert len(prepared.backend_request.candidates) == 5
    assert prepared.available_candidate_count == 12
    assert prepared.truncated_candidate_count == 7
    assert tuple(
        candidate.candidate_id for candidate in prepared.backend_request.candidates
    ) == tuple(hit.candidate.record_id.as_str() for hit in original.hits[:5])
    assert tuple(
        candidate.candidate_text for candidate in prepared.backend_request.candidates
    ) == tuple(hit.candidate.content for hit in original.hits[:5])
    assert original.hits == reranking_request.fused_result.hits


def test_fake_backend_response_maps_back_to_original_evidence() -> None:
    query = _query(limit=3)
    original = _many_candidate_result(query, 3)
    request = RerankingRequest(query, original)
    backend = FakeBackend(
        FakeBackendConfig(
            rerank_scores=(
                (original.hits[0].candidate.record_id.as_str(), 0.1),
                (original.hits[1].candidate.record_id.as_str(), 0.9),
                (original.hits[2].candidate.record_id.as_str(), 0.5),
            )
        )
    )
    backend.load()

    prepared, scored = rerank_evidence(
        backend,
        request,
        evidence_count=EvidenceCount(5),
        candidate_limit=RerankCandidateLimit(3),
    )

    assert prepared.truncated_candidate_count == 0
    assert tuple(item.hit for item in scored) == (
        original.hits[1],
        original.hits[2],
        original.hits[0],
    )
    assert tuple(item.score for item in scored) == (0.9, 0.5, 0.1)
    backend.close()


def test_response_mapping_rejects_a_candidate_id_outside_the_pool() -> None:
    query = _query(limit=2)
    request = RerankingRequest(query, _many_candidate_result(query, 2))
    prepared = prepare_rerank_request(request)
    assert prepared.backend_request is not None
    backend_request = prepared.backend_request
    response = RerankResult(
        scored_candidates=(
            ScoredCandidate("foreign-record", 0.9, 1),
            ScoredCandidate(
                backend_request.candidates[0].candidate_id,
                0.8,
                2,
            ),
        ),
        model_metadata=BackendMetadata(
            backend_name="fixture",
            model_name="fixture-reranker",
            model_revision="1",
            supported_operations=(BackendOperation.RERANK,),
            capabilities=(),
        ),
    )

    with pytest.raises(InvalidRetrievalContractError):
        map_rerank_result(prepared, response)


def test_equal_scores_use_record_id_and_rebuild_contiguous_provenance_hits() -> None:
    query = _query(limit=3)
    original = _many_candidate_result(query, 3)
    prepared = prepare_rerank_request(RerankingRequest(query, original))
    assert prepared.backend_request is not None
    response = RerankResult(
        scored_candidates=tuple(
            ScoredCandidate(candidate.candidate_id, 0.5, rank)
            for rank, candidate in enumerate(
                reversed(prepared.backend_request.candidates),
                start=1,
            )
        ),
        model_metadata=BackendMetadata(
            backend_name="fixture",
            model_name="fixture-reranker",
            model_revision="1",
            supported_operations=(BackendOperation.RERANK,),
            capabilities=(),
        ),
    )

    mapped = map_rerank_result(prepared, response)
    ranked = rank_evidence_hits(mapped)

    assert tuple(hit.candidate.record_id for hit in ranked) == tuple(
        hit.candidate.record_id for hit in original.hits
    )
    assert tuple(hit.rank for hit in ranked) == (1, 2, 3)
    assert tuple(hit.candidate for hit in ranked) == tuple(
        hit.candidate for hit in original.hits
    )
    assert tuple(hit.contributions for hit in ranked) == tuple(
        hit.contributions for hit in original.hits
    )
    assert tuple(hit.candidate.sources[0].source_span_id for hit in ranked) == tuple(
        hit.candidate.sources[0].source_span_id for hit in original.hits
    )


def test_deduplication_removes_only_same_kind_and_exact_source_spans() -> None:
    query = _query(limit=4)
    original = _many_candidate_result(query, 4)
    first = original.hits[0]
    same_span_candidate = RetrievalCandidate(
        record_id=RecordId.from_str("00000000-0000-4000-8000-000000000099"),
        kind=CandidateKind.ORIGINAL_PASSAGE,
        content=first.candidate.content,
        sources=first.candidate.sources,
    )
    same_span_different_kind = RetrievalCandidate(
        record_id=RecordId.from_str("00000000-0000-4000-8000-000000000098"),
        kind=CandidateKind.STRUCTURED_RECORD,
        content="structured value from same span",
        sources=first.candidate.sources,
    )
    ranked = (
        first,
        EvidenceHit(same_span_candidate, first.contributions, 2),
        EvidenceHit(
            original.hits[1].candidate,
            original.hits[1].contributions,
            3,
        ),
        EvidenceHit(same_span_different_kind, first.contributions, 4),
    )

    selected = deduplicate_evidence_hits(ranked)

    assert tuple(hit.candidate.record_id for hit in selected) == (
        first.candidate.record_id,
        original.hits[1].candidate.record_id,
        same_span_different_kind.record_id,
    )
    assert tuple(hit.rank for hit in selected) == (1, 2, 3)
    assert selected[0].candidate.sources == first.candidate.sources
    assert selected[1].candidate.sources == original.hits[1].candidate.sources


def test_deduplication_rejects_non_contiguous_input_ranks() -> None:
    query = _query(limit=1)
    hit = _many_candidate_result(query, 1).hits[0]

    with pytest.raises(InvalidRetrievalContractError):
        deduplicate_evidence_hits((EvidenceHit(hit.candidate, hit.contributions, 2),))


def test_ranking_pipeline_keeps_the_highest_scored_exact_duplicate() -> None:
    query = _query(limit=3)
    original = _many_candidate_result(query, 3)
    first, second, _ = original.hits
    duplicate = RetrievalCandidate(
        record_id=RecordId.from_str("00000000-0000-4000-8000-000000000099"),
        kind=first.candidate.kind,
        content=first.candidate.content,
        sources=first.candidate.sources,
    )
    scored = (
        ScoredEvidence(first, 0.2),
        ScoredEvidence(
            EvidenceHit(duplicate, first.contributions, 2),
            0.9,
        ),
        ScoredEvidence(second, 0.7),
    )

    selected = rank_evidence_hits(scored)

    assert tuple(hit.candidate.record_id for hit in selected) == (
        duplicate.record_id,
        second.candidate.record_id,
    )
    assert tuple(hit.rank for hit in selected) == (1, 2)


def test_empty_fused_result_does_not_call_backend() -> None:
    query = _query()
    request = RerankingRequest(query, _result(query, with_hit=False))

    class Backend:
        def rerank(self, request: object) -> RerankResult:
            pytest.fail("an empty candidate pool must skip backend execution")

    prepared, scored = rerank_evidence(Backend(), request)  # type: ignore[arg-type]

    assert prepared.backend_request is None
    assert scored == ()


def test_retryable_resource_failure_returns_degraded_fused_order() -> None:
    query = _query(limit=3)
    fused = _many_candidate_result(query, 3)
    backend = FakeBackend(
        FakeBackendConfig(
            failure_mode=FakeFailureMode.RESOURCE,
            fail_operation=BackendOperation.RERANK,
        )
    )
    backend.load()

    outcome = execute_reranking(
        backend,
        query,
        fused,
        evidence_count=EvidenceCount(5),
        candidate_limit=RerankCandidateLimit(3),
    )

    assert outcome.degraded is True
    assert outcome.result.status is ResultStatus.PARTIAL
    assert (
        outcome.result.failures[0].category is FailureCategory.CHANNEL_EXECUTION_FAILURE
    )
    assert outcome.result.failures[0].retryable is True
    assert tuple(hit.candidate for hit in outcome.result.hits) == tuple(
        hit.candidate for hit in fused.hits
    )
    assert tuple(hit.rank for hit in outcome.result.hits) == (1, 2, 3)
    backend.close()


def test_non_retryable_backend_failure_fails_closed() -> None:
    query = _query(limit=2)
    fused = _many_candidate_result(query, 2)
    backend = FakeBackend(
        FakeBackendConfig(
            failure_mode=FakeFailureMode.EXECUTION,
            fail_operation=BackendOperation.RERANK,
        )
    )
    backend.load()

    outcome = execute_reranking(backend, query, fused)

    assert outcome.degraded is False
    assert outcome.result.status is ResultStatus.FAILED
    assert outcome.result.hits == ()
    assert (
        outcome.result.failures[0].category is FailureCategory.CHANNEL_EXECUTION_FAILURE
    )
    assert outcome.result.failures[0].retryable is False
    backend.close()


def test_malformed_backend_response_fails_closed() -> None:
    query = _query(limit=2)
    fused = _many_candidate_result(query, 2)

    class MalformedBackend:
        def rerank(self, request: object) -> object:
            return object()

    outcome = execute_reranking(MalformedBackend(), query, fused)  # type: ignore[arg-type]

    assert outcome.degraded is False
    assert outcome.result.status is ResultStatus.FAILED
    assert outcome.result.hits == ()
    assert (
        outcome.result.failures[0].category is FailureCategory.INVALID_CHANNEL_RESPONSE
    )


def test_corpus_version_mismatch_fails_closed_without_backend_call() -> None:
    query = _query(version=_version(1), limit=2)
    fused = _many_candidate_result(_query(version=_version(2), limit=2), 2)

    class Backend:
        def rerank(self, request: object) -> RerankResult:
            pytest.fail("mismatched corpus evidence must not reach the backend")

    outcome = execute_reranking(Backend(), query, fused)  # type: ignore[arg-type]

    assert outcome.degraded is False
    assert outcome.result.status is ResultStatus.FAILED
    assert outcome.result.hits == ()
    assert (
        outcome.result.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH
    )
