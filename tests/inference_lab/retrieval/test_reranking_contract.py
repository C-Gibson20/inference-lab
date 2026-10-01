"""RET-06 query and fused-result boundary validation."""

import pytest

from inference_lab.backends.runtime import (
    BackendMetadata,
    BackendOperation,
    FakeBackend,
    FakeBackendConfig,
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
    InvalidRetrievalContractError,
    RerankCandidateLimit,
    RerankingRequest,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
    map_rerank_result,
    prepare_rerank_request,
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


def test_empty_fused_result_does_not_call_backend() -> None:
    query = _query()
    request = RerankingRequest(query, _result(query, with_hit=False))

    class Backend:
        def rerank(self, request: object) -> RerankResult:
            pytest.fail("an empty candidate pool must skip backend execution")

    prepared, scored = rerank_evidence(Backend(), request)  # type: ignore[arg-type]

    assert prepared.backend_request is None
    assert scored == ()
