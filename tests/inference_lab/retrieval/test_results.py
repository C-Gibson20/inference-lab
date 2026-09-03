"""Tests for the results module."""

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
    InvalidRetrievalResultError,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
    RetrievalQuery,
)


@pytest.fixture
def source() -> EvidenceSource:
    """Return one valid source link for candidate fixtures."""
    return EvidenceSource(
        DocumentId.generate(), DocumentVersionId.generate(), SourceSpanId.generate()
    )


@pytest.fixture
def candidate(source: EvidenceSource) -> RetrievalCandidate:
    """Return one valid original-passage candidate."""
    return RetrievalCandidate(
        RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, "source passage", (source,)
    )


def test_result_validates_order_identity_and_statuses(
    candidate: RetrievalCandidate,
) -> None:
    version = CorpusVersionId.generate()
    hit = EvidenceHit(candidate, (ChannelScore(RetrievalChannel.LEXICAL, 1),), 1)
    failure = RetrievalFailure(
        FailureCategory.CHANNEL_UNAVAILABLE,
        True,
        RetrievalChannel.DENSE,
    )
    limit = RetrievalLimit(1)
    assert EvidenceResult(version, limit, (hit,), ResultStatus.COMPLETED, ()).hits == (
        hit,
    )
    assert EvidenceResult(version, limit, (), ResultStatus.COMPLETED, ()).hits == ()
    assert (
        EvidenceResult(version, limit, (hit,), ResultStatus.PARTIAL, (failure,)).status
        is ResultStatus.PARTIAL
    )
    assert (
        EvidenceResult(version, limit, (), ResultStatus.FAILED, (failure,)).status
        is ResultStatus.FAILED
    )

    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, limit, (hit,), ResultStatus.COMPLETED, (failure,))
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, limit, (), ResultStatus.PARTIAL, (failure,))
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, limit, (hit,), ResultStatus.FAILED, (failure,))


def test_failure_and_result_reject_wrong_contract_types(
    candidate: RetrievalCandidate,
) -> None:
    version = CorpusVersionId.generate()
    limit = RetrievalLimit(1)
    hit = EvidenceHit(candidate, (ChannelScore(RetrievalChannel.LEXICAL, 1),), 1)
    failure = RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, True)

    with pytest.raises(TypeError, match="category"):
        RetrievalFailure("channel_unavailable", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="channel"):
        RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, True, "dense")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retryable"):
        RetrievalFailure(FailureCategory.CHANNEL_UNAVAILABLE, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="corpus_version"):
        EvidenceResult(limit, limit, (), ResultStatus.COMPLETED, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requested_limit"):
        EvidenceResult(version, 1, (), ResultStatus.COMPLETED, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hits must be a tuple"):
        EvidenceResult(version, limit, [hit], ResultStatus.COMPLETED, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EvidenceHit"):
        EvidenceResult(version, limit, (candidate,), ResultStatus.COMPLETED, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="status"):
        EvidenceResult(version, limit, (), "completed", ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="failures must be a tuple"):
        EvidenceResult(version, limit, (), ResultStatus.COMPLETED, [failure])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RetrievalFailure"):
        EvidenceResult(version, limit, (), ResultStatus.COMPLETED, (hit,))  # type: ignore[arg-type]


def test_result_ranks_are_contiguous_and_candidate_ids_unique(
    source: EvidenceSource,
) -> None:
    version = CorpusVersionId.generate()
    first = RetrievalCandidate(
        RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, "one", (source,)
    )
    second = RetrievalCandidate(
        RecordId.generate(),
        CandidateKind.ORIGINAL_PASSAGE,
        "two",
        (
            EvidenceSource(
                DocumentId.generate(),
                DocumentVersionId.generate(),
                SourceSpanId.generate(),
            ),
        ),
    )
    first_hit = EvidenceHit(first, (ChannelScore(RetrievalChannel.LEXICAL, 1),), 1)
    second_hit = EvidenceHit(second, (ChannelScore(RetrievalChannel.LEXICAL, 2),), 2)
    result = EvidenceResult(
        version, RetrievalLimit(2), (first_hit, second_hit), ResultStatus.COMPLETED, ()
    )
    assert result.hits == (first_hit, second_hit)
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(
            version,
            RetrievalLimit(2),
            (second_hit, first_hit),
            ResultStatus.COMPLETED,
            (),
        )

    duplicate_identity = EvidenceHit(
        first,
        (ChannelScore(RetrievalChannel.DENSE, 2),),
        2,
    )
    with pytest.raises(InvalidRetrievalResultError, match="duplicate candidate"):
        EvidenceResult(
            version,
            RetrievalLimit(2),
            (first_hit, duplicate_identity),
            ResultStatus.COMPLETED,
            (),
        )


def test_from_query_fails_closed_on_corpus_version_mismatch(
    candidate: RetrievalCandidate,
) -> None:
    requested = CorpusVersionId.generate()
    observed = CorpusVersionId.generate()
    query = RetrievalQuery("question", requested, RetrievalLimit(5))
    hit = EvidenceHit(candidate, (ChannelScore(RetrievalChannel.LEXICAL, 1),), 1)

    result = EvidenceResult.from_query(
        query,
        observed_corpus_version=observed,
        hits=(hit,),
        status=ResultStatus.COMPLETED,
        failures=(),
    )
    assert result.corpus_version is requested
    assert result.requested_limit == query.limit
    assert result.hits == ()
    assert result.status is ResultStatus.FAILED
    assert result.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH
    assert result.failures[0].channel is None


def test_from_query_rejects_wrong_boundary_types() -> None:
    version = CorpusVersionId.generate()
    query = RetrievalQuery("question", version, RetrievalLimit(1))

    with pytest.raises(TypeError, match="query"):
        EvidenceResult.from_query(
            "question",  # type: ignore[arg-type]
            observed_corpus_version=version,
            hits=(),
            status=ResultStatus.COMPLETED,
            failures=(),
        )
    with pytest.raises(TypeError, match="observed_corpus_version"):
        EvidenceResult.from_query(
            query,
            observed_corpus_version="version",  # type: ignore[arg-type]
            hits=(),
            status=ResultStatus.COMPLETED,
            failures=(),
        )


def test_result_enforces_requested_limit_at_query_and_value_boundaries(
    candidate: RetrievalCandidate,
    source: EvidenceSource,
) -> None:
    version = CorpusVersionId.generate()
    query = RetrievalQuery("question", version, RetrievalLimit(1))
    first = EvidenceHit(
        candidate,
        (ChannelScore(RetrievalChannel.LEXICAL, 1),),
        1,
    )
    second_candidate = RetrievalCandidate(
        RecordId.generate(),
        CandidateKind.ORIGINAL_PASSAGE,
        "second passage",
        (source,),
    )
    second = EvidenceHit(
        second_candidate,
        (ChannelScore(RetrievalChannel.LEXICAL, 2),),
        2,
    )

    exact = EvidenceResult.from_query(
        query,
        observed_corpus_version=version,
        hits=(first,),
        status=ResultStatus.COMPLETED,
        failures=(),
    )
    assert exact.requested_limit == query.limit

    with pytest.raises(InvalidRetrievalResultError, match="requested retrieval limit"):
        EvidenceResult.from_query(
            query,
            observed_corpus_version=version,
            hits=(first, second),
            status=ResultStatus.COMPLETED,
            failures=(),
        )


def test_corpus_mismatch_failures_exclude_only_affected_channel_hits(
    candidate: RetrievalCandidate,
) -> None:
    version = CorpusVersionId.generate()
    hit = EvidenceHit(
        candidate,
        (ChannelScore(RetrievalChannel.LEXICAL, 1),),
        1,
    )
    dense_mismatch = RetrievalFailure(
        FailureCategory.CORPUS_VERSION_MISMATCH,
        False,
        RetrievalChannel.DENSE,
    )

    partial = EvidenceResult(
        version,
        RetrievalLimit(1),
        (hit,),
        ResultStatus.PARTIAL,
        (dense_mismatch,),
    )
    assert partial.hits == (hit,)

    lexical_mismatch = RetrievalFailure(
        FailureCategory.CORPUS_VERSION_MISMATCH,
        False,
        RetrievalChannel.LEXICAL,
    )
    with pytest.raises(InvalidRetrievalResultError, match="mismatched channel"):
        EvidenceResult(
            version,
            RetrievalLimit(1),
            (hit,),
            ResultStatus.PARTIAL,
            (lexical_mismatch,),
        )
    with pytest.raises(InvalidRetrievalResultError, match="unscoped corpus mismatch"):
        EvidenceResult(
            version,
            RetrievalLimit(1),
            (hit,),
            ResultStatus.PARTIAL,
            (RetrievalFailure(FailureCategory.CORPUS_VERSION_MISMATCH, False),),
        )
