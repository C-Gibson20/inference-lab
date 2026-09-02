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
    """Accept complete, empty, partial, and failed result states precisely."""
    version = CorpusVersionId.generate()
    hit = EvidenceHit(candidate, (ChannelScore(RetrievalChannel.LEXICAL, 1),), 1)
    failure = RetrievalFailure(
        FailureCategory.CHANNEL_UNAVAILABLE,
        True,
        RetrievalChannel.DENSE,
    )
    assert EvidenceResult(version, (hit,), ResultStatus.COMPLETED, ()).hits == (hit,)
    assert EvidenceResult(version, (), ResultStatus.COMPLETED, ()).hits == ()
    assert (
        EvidenceResult(version, (hit,), ResultStatus.PARTIAL, (failure,)).status
        is ResultStatus.PARTIAL
    )
    assert (
        EvidenceResult(version, (), ResultStatus.FAILED, (failure,)).status
        is ResultStatus.FAILED
    )

    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, (hit,), ResultStatus.COMPLETED, (failure,))
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, (), ResultStatus.PARTIAL, (failure,))
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, (hit,), ResultStatus.FAILED, (failure,))


def test_result_ranks_are_contiguous_and_candidate_ids_unique(
    source: EvidenceSource,
) -> None:
    """Prevent ambiguous result order and duplicate candidate identities."""
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
        version, (first_hit, second_hit), ResultStatus.COMPLETED, ()
    )
    assert result.hits == (first_hit, second_hit)
    with pytest.raises(InvalidRetrievalResultError):
        EvidenceResult(version, (second_hit, first_hit), ResultStatus.COMPLETED, ())


def test_from_query_fails_closed_on_corpus_version_mismatch(
    candidate: RetrievalCandidate,
) -> None:
    """Never mix hits observed from a different immutable corpus snapshot."""
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
    assert result.hits == ()
    assert result.status is ResultStatus.FAILED
    assert result.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH
    assert result.failures[0].channel is None
