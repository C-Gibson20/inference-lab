"""Contract fixtures for deterministic lexical and dense result fusion."""

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
    FailureCategory,
    FusionPolicy,
    InvalidRetrievalContractError,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
    RetrievalQuery,
    fuse_evidence_results,
)


def _query(
    *, version: CorpusVersionId | None = None, limit: int = 10
) -> RetrievalQuery:
    """Create a query with deterministic scope and a bounded result limit."""
    return RetrievalQuery("question", version or _version(900), RetrievalLimit(limit))


def _version(number: int) -> CorpusVersionId:
    """Return a stable corpus-version identity for fixtures."""
    return CorpusVersionId.from_str(f"00000000-0000-4000-8000-{number:012d}")


def _candidate(number: int, *source_numbers: int) -> RetrievalCandidate:
    """Build one candidate with an explicit set of source links."""
    return RetrievalCandidate(
        record_id=RecordId.from_str(f"00000000-0000-4000-8000-{number:012d}"),
        kind=CandidateKind.ORIGINAL_PASSAGE,
        content=f"passage {number}",
        sources=tuple(_source(source_number) for source_number in source_numbers),
    )


def _source(number: int) -> EvidenceSource:
    """Return deterministic source identities."""
    return EvidenceSource(
        document_id=DocumentId.from_str(
            f"00000000-0000-4000-8000-{number + 1000:012d}"
        ),
        document_version_id=DocumentVersionId.from_str(
            f"00000000-0000-4000-8000-{number + 2000:012d}"
        ),
        source_span_id=SourceSpanId.from_str(
            f"00000000-0000-4000-8000-{number + 3000:012d}"
        ),
    )


def _result(
    channel: RetrievalChannel,
    candidates: tuple[tuple[RetrievalCandidate, int], ...],
    *,
    version: CorpusVersionId | None = None,
    limit: int = 10,
    failures: tuple[RetrievalFailure, ...] = (),
) -> EvidenceResult:
    """Build one valid channel-local result envelope."""
    hits = tuple(
        EvidenceHit(
            candidate=candidate,
            contributions=(ChannelScore(channel=channel, rank=channel_rank),),
            rank=index,
        )
        for index, (candidate, channel_rank) in enumerate(candidates, start=1)
    )
    if failures:
        status = ResultStatus.PARTIAL if hits else ResultStatus.FAILED
    else:
        status = ResultStatus.COMPLETED
    return EvidenceResult(
        corpus_version=version or _version(900),
        requested_limit=RetrievalLimit(limit),
        hits=hits,
        status=status,
        failures=failures,
    )


def test_rrf_uses_hand_calculated_weighted_channel_ranks() -> None:
    query = _query()
    lexical = _result(
        RetrievalChannel.LEXICAL,
        ((_candidate(10, 1), 1), (_candidate(20, 2), 2)),
    )
    dense = _result(
        RetrievalChannel.DENSE,
        ((_candidate(20, 2), 1), (_candidate(30, 3), 2)),
    )

    fused = fuse_evidence_results(
        query,
        {RetrievalChannel.LEXICAL: lexical, RetrievalChannel.DENSE: dense},
        policy=FusionPolicy(rank_constant=1, lexical_weight=2, dense_weight=1),
    )

    # RRF: record 10 = 2/(1+1); record 20 = 2/(1+2) + 1/(1+1);
    # record 30 = 1/(1+2). The raw channel scores are not consulted.
    assert tuple(hit.candidate.record_id.as_str() for hit in fused.hits) == (
        _candidate(20, 2).record_id.as_str(),
        _candidate(10, 1).record_id.as_str(),
        _candidate(30, 3).record_id.as_str(),
    )
    assert fused.status is ResultStatus.COMPLETED
    assert tuple(hit.rank for hit in fused.hits) == (1, 2, 3)


def test_same_record_merges_contributions_and_all_distinct_source_links() -> None:
    query = _query()
    shared_lexical = _candidate(10, 1)
    shared_dense = _candidate(10, 2)

    fused = fuse_evidence_results(
        query,
        {
            RetrievalChannel.DENSE: _result(
                RetrievalChannel.DENSE, ((shared_dense, 1),)
            ),
            RetrievalChannel.LEXICAL: _result(
                RetrievalChannel.LEXICAL, ((shared_lexical, 1),)
            ),
        },
    )

    assert len(fused.hits) == 1
    hit = fused.hits[0]
    assert tuple(score.channel for score in hit.contributions) == (
        RetrievalChannel.DENSE,
        RetrievalChannel.LEXICAL,
    )
    assert hit.candidate.sources == (_source(1), _source(2))


def test_equal_fused_scores_use_record_id_and_mapping_order_does_not_matter() -> None:
    query = _query()
    lexical = _result(
        RetrievalChannel.LEXICAL,
        ((_candidate(10, 1), 1), (_candidate(20, 2), 2)),
    )
    dense = _result(
        RetrievalChannel.DENSE,
        ((_candidate(20, 2), 1), (_candidate(10, 1), 2)),
    )
    results = {
        RetrievalChannel.LEXICAL: lexical,
        RetrievalChannel.DENSE: dense,
    }

    forward = fuse_evidence_results(query, results)
    reverse = fuse_evidence_results(query, dict(reversed(tuple(results.items()))))

    assert forward == reverse
    assert tuple(hit.candidate.record_id.as_str() for hit in forward.hits) == (
        _candidate(10, 1).record_id.as_str(),
        _candidate(20, 2).record_id.as_str(),
    )


def test_missing_and_successful_empty_channels_do_not_create_failures() -> None:
    query = _query()
    missing_dense = fuse_evidence_results(
        query,
        {RetrievalChannel.LEXICAL: _result(RetrievalChannel.LEXICAL, ())},
    )
    both_empty = fuse_evidence_results(
        query,
        {
            RetrievalChannel.LEXICAL: _result(RetrievalChannel.LEXICAL, ()),
            RetrievalChannel.DENSE: _result(RetrievalChannel.DENSE, ()),
        },
    )

    assert missing_dense.status is ResultStatus.COMPLETED
    assert missing_dense.hits == ()
    assert both_empty.status is ResultStatus.COMPLETED
    assert both_empty.failures == ()


def test_query_limit_is_a_hard_maximum_after_fusion_ordering() -> None:
    query = _query(limit=1)
    lexical = _result(
        RetrievalChannel.LEXICAL,
        ((_candidate(10, 1), 1),),
        limit=1,
    )
    dense = _result(
        RetrievalChannel.DENSE,
        ((_candidate(20, 2), 1),),
        limit=1,
    )

    fused = fuse_evidence_results(
        query,
        {RetrievalChannel.LEXICAL: lexical, RetrievalChannel.DENSE: dense},
    )

    assert fused.requested_limit == query.limit
    assert len(fused.hits) == 1
    assert fused.hits[0].candidate.record_id == _candidate(10, 1).record_id


def test_degraded_channel_returns_partial_or_failed_using_surviving_hits() -> None:
    query = _query()
    failure = RetrievalFailure(
        FailureCategory.CHANNEL_UNAVAILABLE,
        retryable=True,
        channel=RetrievalChannel.DENSE,
    )
    partial = fuse_evidence_results(
        query,
        {
            RetrievalChannel.LEXICAL: _result(
                RetrievalChannel.LEXICAL, ((_candidate(10, 1), 1),)
            ),
            RetrievalChannel.DENSE: _result(
                RetrievalChannel.DENSE, (), failures=(failure,)
            ),
        },
    )
    failed = fuse_evidence_results(
        query,
        {
            RetrievalChannel.LEXICAL: _result(RetrievalChannel.LEXICAL, ()),
            RetrievalChannel.DENSE: _result(
                RetrievalChannel.DENSE, (), failures=(failure,)
            ),
        },
    )

    assert partial.status is ResultStatus.PARTIAL
    assert partial.hits
    assert partial.failures == (failure,)
    assert failed.status is ResultStatus.FAILED
    assert failed.hits == ()


def test_version_mismatch_fails_closed_without_returning_candidates() -> None:
    query = _query()
    result = _result(
        RetrievalChannel.LEXICAL,
        ((_candidate(10, 1), 1),),
        version=_version(901),
    )

    fused = fuse_evidence_results(query, {RetrievalChannel.LEXICAL: result})

    assert fused.status is ResultStatus.FAILED
    assert fused.hits == ()
    assert fused.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH
    assert fused.failures[0].channel is None


def test_policy_identity_includes_algorithm_version_and_every_parameter() -> None:
    baseline = FusionPolicy()
    changed = FusionPolicy(rank_constant=61, lexical_weight=2.0, dense_weight=3.0)

    assert baseline.algorithm == "reciprocal_rank_fusion"
    assert baseline.version == "1.0.0"
    assert baseline.identity == "reciprocal_rank_fusion/1.0.0/k=60/dense=1/lexical=1"
    assert changed.identity != baseline.identity
    assert baseline.weight_for(RetrievalChannel.LEXICAL) == 1.0
    assert baseline.weight_for(RetrievalChannel.DENSE) == 1.0


def test_fusion_rejects_metadata_as_a_ranked_channel_and_invalid_policy_values() -> (
    None
):
    query = _query()
    with pytest.raises(InvalidRetrievalContractError):
        fuse_evidence_results(
            query,
            {RetrievalChannel.METADATA: _result(RetrievalChannel.METADATA, ())},
        )
    with pytest.raises(InvalidRetrievalContractError):
        FusionPolicy(rank_constant=0)
    with pytest.raises(InvalidRetrievalContractError):
        FusionPolicy(lexical_weight=float("inf"))
