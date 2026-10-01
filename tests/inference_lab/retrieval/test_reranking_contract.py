"""RET-06 query and fused-result boundary validation."""

import pytest

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
    InvalidRetrievalContractError,
    RerankingRequest,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
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
