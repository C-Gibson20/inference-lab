"""Tests for the evidence module."""

import pytest

from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceSource,
    InvalidRetrievalContractError,
    RetrievalCandidate,
    RetrievalChannel,
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


def test_source_and_candidate_require_typed_provenance_and_sources(
    source: EvidenceSource,
) -> None:
    """Retain all identity domains and reject missing or duplicate spans."""
    assert source.document_id != DocumentId.generate()
    second = EvidenceSource(
        source.document_id, source.document_version_id, SourceSpanId.generate()
    )
    candidate = RetrievalCandidate(
        RecordId.generate(), CandidateKind.STRUCTURED_RECORD, "fact", (source, second)
    )
    assert candidate.sources == (source, second)
    with pytest.raises(InvalidRetrievalContractError):
        RetrievalCandidate(
            RecordId.generate(), CandidateKind.GENERATED_SUMMARY, "x", ()
        )
    with pytest.raises(InvalidRetrievalContractError):
        RetrievalCandidate(
            RecordId.generate(), CandidateKind.ORIGINAL_PASSAGE, "x", (source, source)
        )


@pytest.mark.parametrize("score", [None, 1, 1.5])
def test_channel_scores_allow_ranked_only_or_labelled_finite_scores(
    score: int | float | None,
) -> None:
    """Keep score semantics explicit while always requiring channel rank."""
    label = None if score is None else "engine-score"
    value = ChannelScore(RetrievalChannel.LEXICAL, 1, score, label)
    assert value.raw_score == score
    with pytest.raises(InvalidRetrievalContractError):
        ChannelScore(RetrievalChannel.LEXICAL, 1, None, "unlabelled")
    with pytest.raises(InvalidRetrievalContractError):
        ChannelScore(RetrievalChannel.LEXICAL, 1, float("nan"), "score")


def test_hit_requires_unique_channel_contributions(
    candidate: RetrievalCandidate,
) -> None:
    """Require one contribution per channel and preserve distinct labels."""
    lexical = ChannelScore(RetrievalChannel.LEXICAL, 1, 3.2, "bm25")
    dense = ChannelScore(RetrievalChannel.DENSE, 4, 0.8, "cosine-similarity")
    hit = EvidenceHit(candidate, (lexical, dense), 1)
    assert hit.contributions == (lexical, dense)
    with pytest.raises(InvalidRetrievalContractError):
        EvidenceHit(candidate, (lexical, lexical), 1)
