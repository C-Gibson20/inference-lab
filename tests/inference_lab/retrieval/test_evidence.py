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


def test_source_rejects_wrong_provenance_identifier_types() -> None:
    document_id = DocumentId.generate()
    document_version_id = DocumentVersionId.generate()
    source_span_id = SourceSpanId.generate()

    with pytest.raises(TypeError, match="document_id"):
        EvidenceSource(document_version_id, document_version_id, source_span_id)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document_version_id"):
        EvidenceSource(document_id, document_id, source_span_id)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source_span_id"):
        EvidenceSource(document_id, document_version_id, document_id)  # type: ignore[arg-type]


def test_candidate_rejects_wrong_types_and_blank_content(
    source: EvidenceSource,
) -> None:
    record_id = RecordId.generate()

    with pytest.raises(TypeError, match="record_id"):
        RetrievalCandidate(
            source.source_span_id,  # type: ignore[arg-type]
            CandidateKind.ORIGINAL_PASSAGE,
            "x",
            (source,),
        )
    with pytest.raises(TypeError, match="kind"):
        RetrievalCandidate(record_id, "original_passage", "x", (source,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="content"):
        RetrievalCandidate(record_id, CandidateKind.ORIGINAL_PASSAGE, 1, (source,))  # type: ignore[arg-type]
    with pytest.raises(InvalidRetrievalContractError, match="blank"):
        RetrievalCandidate(record_id, CandidateKind.ORIGINAL_PASSAGE, " \t", (source,))
    with pytest.raises(TypeError, match="sources must be a tuple"):
        RetrievalCandidate(record_id, CandidateKind.ORIGINAL_PASSAGE, "x", [source])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EvidenceSource"):
        RetrievalCandidate(record_id, CandidateKind.ORIGINAL_PASSAGE, "x", (record_id,))  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [None, 1, 1.5])
def test_channel_scores_allow_ranked_only_or_labelled_finite_scores(
    score: int | float | None,
) -> None:
    label = None if score is None else "engine-score"
    value = ChannelScore(RetrievalChannel.LEXICAL, 1, score, label)
    assert value.raw_score == score
    with pytest.raises(InvalidRetrievalContractError):
        ChannelScore(RetrievalChannel.LEXICAL, 1, None, "unlabelled")
    with pytest.raises(InvalidRetrievalContractError):
        ChannelScore(RetrievalChannel.LEXICAL, 1, float("nan"), "score")


def test_channel_score_rejects_wrong_types_and_invalid_values() -> None:
    with pytest.raises(TypeError, match="channel"):
        ChannelScore("lexical", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rank"):
        ChannelScore(RetrievalChannel.LEXICAL, True)
    with pytest.raises(InvalidRetrievalContractError, match="positive"):
        ChannelScore(RetrievalChannel.LEXICAL, 0)
    with pytest.raises(TypeError, match="raw_score"):
        ChannelScore(RetrievalChannel.LEXICAL, 1, True, "score")
    with pytest.raises(TypeError, match="raw_score"):
        ChannelScore(RetrievalChannel.LEXICAL, 1, "1", "score")  # type: ignore[arg-type]
    with pytest.raises(InvalidRetrievalContractError, match="finite"):
        ChannelScore(RetrievalChannel.LEXICAL, 1, float("inf"), "score")
    with pytest.raises(TypeError, match="score_label"):
        ChannelScore(RetrievalChannel.LEXICAL, 1, 1.0, None)
    with pytest.raises(TypeError, match="score_label"):
        ChannelScore(RetrievalChannel.LEXICAL, 1, 1.0, 1)  # type: ignore[arg-type]
    with pytest.raises(InvalidRetrievalContractError, match="blank"):
        ChannelScore(RetrievalChannel.LEXICAL, 1, 1.0, " ")


def test_hit_requires_unique_channel_contributions(
    candidate: RetrievalCandidate,
) -> None:
    lexical = ChannelScore(RetrievalChannel.LEXICAL, 1, 3.2, "bm25")
    dense = ChannelScore(RetrievalChannel.DENSE, 4, 0.8, "cosine-similarity")
    hit = EvidenceHit(candidate, (lexical, dense), 1)
    assert hit.contributions == (lexical, dense)
    with pytest.raises(InvalidRetrievalContractError):
        EvidenceHit(candidate, (lexical, lexical), 1)


def test_hit_rejects_wrong_types_empty_contributions_and_invalid_rank(
    candidate: RetrievalCandidate,
) -> None:
    contribution = ChannelScore(RetrievalChannel.LEXICAL, 1)

    with pytest.raises(TypeError, match="candidate"):
        EvidenceHit(candidate.record_id, (contribution,), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contributions must be a tuple"):
        EvidenceHit(candidate, [contribution], 1)  # type: ignore[arg-type]
    with pytest.raises(InvalidRetrievalContractError, match="empty"):
        EvidenceHit(candidate, (), 1)
    with pytest.raises(TypeError, match="ChannelScore"):
        EvidenceHit(candidate, (candidate,), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rank"):
        EvidenceHit(candidate, (contribution,), True)
    with pytest.raises(InvalidRetrievalContractError, match="positive"):
        EvidenceHit(candidate, (contribution,), 0)
