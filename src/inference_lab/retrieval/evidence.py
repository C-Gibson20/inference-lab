"""Immutable provenance-bearing candidates and channel contributions."""

import math
from dataclasses import dataclass
from enum import StrEnum

from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.values import RetrievalChannel


class CandidateKind(StrEnum):
    """Authority and provenance kind of a retrieval candidate."""

    ORIGINAL_PASSAGE = "original_passage"
    STRUCTURED_RECORD = "structured_record"
    GENERATED_SUMMARY = "generated_summary"


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """The smallest source link needed to reopen retrieved evidence."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the source's fields."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One uniquely identifiable, source-linked retrieval candidate."""

    record_id: RecordId
    kind: CandidateKind
    content: str
    sources: tuple[EvidenceSource, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the candidate's fields."""
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not isinstance(self.kind, CandidateKind):
            raise TypeError("kind must be a CandidateKind.")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string.")
        if not self.content.strip():
            raise InvalidRetrievalContractError("content must not be blank.")
        if not isinstance(self.sources, tuple):
            raise TypeError("sources must be a tuple.")
        if not self.sources:
            raise InvalidRetrievalContractError("sources must not be empty.")
        if any(not isinstance(source, EvidenceSource) for source in self.sources):
            raise TypeError("sources must contain only EvidenceSource values.")
        span_ids = tuple(source.source_span_id for source in self.sources)
        if len(span_ids) != len(set(span_ids)):
            raise InvalidRetrievalContractError(
                "sources must not contain duplicate source spans."
            )


@dataclass(frozen=True, slots=True)
class ChannelScore:
    """A channel-local rank and optional labelled raw score."""

    channel: RetrievalChannel
    rank: int
    raw_score: int | float | None = None
    score_label: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the channel score's fields."""
        if not isinstance(self.channel, RetrievalChannel):
            raise TypeError("channel must be a RetrievalChannel.")
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer.")
        if self.rank <= 0:
            raise InvalidRetrievalContractError("rank must be positive.")
        if self.raw_score is None:
            if self.score_label is not None:
                raise InvalidRetrievalContractError(
                    "a ranked-only score must not have a score label."
                )
            return
        if isinstance(self.raw_score, bool) or type(self.raw_score) not in (int, float):
            raise TypeError("raw_score must be a finite number or None.")
        if not math.isfinite(self.raw_score):
            raise InvalidRetrievalContractError("raw_score must be finite.")
        if not isinstance(self.score_label, str):
            raise TypeError("score_label must be a string when raw_score is present.")
        if not self.score_label.strip():
            raise InvalidRetrievalContractError("score_label must not be blank.")


@dataclass(frozen=True, slots=True)
class EvidenceHit:
    """One result-ranked candidate with unique channel contributions."""

    candidate: RetrievalCandidate
    contributions: tuple[ChannelScore, ...]
    rank: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.candidate, RetrievalCandidate):
            raise TypeError("candidate must be a RetrievalCandidate.")
        if not isinstance(self.contributions, tuple):
            raise TypeError("contributions must be a tuple.")
        if not self.contributions:
            raise InvalidRetrievalContractError("contributions must not be empty.")
        if any(
            not isinstance(contribution, ChannelScore)
            for contribution in self.contributions
        ):
            raise TypeError("contributions must contain only ChannelScore values.")
        channels = tuple(contribution.channel for contribution in self.contributions)
        if len(channels) != len(set(channels)):
            raise InvalidRetrievalContractError(
                "contributions must not contain duplicate channels."
            )
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer.")
        if self.rank <= 0:
            raise InvalidRetrievalContractError("rank must be positive.")
