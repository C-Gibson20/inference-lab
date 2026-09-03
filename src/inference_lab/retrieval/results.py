"""Immutable retrieval outcomes and typed operational failures."""

from dataclasses import dataclass
from enum import StrEnum

from inference_lab.provenance import CorpusVersionId
from inference_lab.retrieval.errors import (
    InvalidRetrievalResultError,
)
from inference_lab.retrieval.evidence import EvidenceHit
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.values import RetrievalChannel, RetrievalLimit


class ResultStatus(StrEnum):
    """Completion state of a retrieval result."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class FailureCategory(StrEnum):
    """Bounded failure categories exposed by retrieval channels."""

    CHANNEL_UNAVAILABLE = "channel_unavailable"
    CHANNEL_EXECUTION_FAILURE = "channel_execution_failure"
    INVALID_CHANNEL_RESPONSE = "invalid_channel_response"
    CORPUS_VERSION_MISMATCH = "corpus_version_mismatch"


@dataclass(frozen=True, slots=True)
class RetrievalFailure:
    """A privacy-safe, optionally channel-specific retrieval failure."""

    category: FailureCategory
    retryable: bool
    channel: RetrievalChannel | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the failure's fields."""
        if not isinstance(self.category, FailureCategory):
            raise TypeError("category must be a FailureCategory.")
        if self.channel is not None and not isinstance(self.channel, RetrievalChannel):
            raise TypeError("channel must be a RetrievalChannel or None.")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a boolean.")


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """A version-scoped ordered result, including successful empty output."""

    corpus_version: CorpusVersionId
    requested_limit: RetrievalLimit
    hits: tuple[EvidenceHit, ...]
    status: ResultStatus
    failures: tuple[RetrievalFailure, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the result's fields."""
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.requested_limit, RetrievalLimit):
            raise TypeError("requested_limit must be a RetrievalLimit.")
        if not isinstance(self.hits, tuple):
            raise TypeError("hits must be a tuple.")
        if any(not isinstance(hit, EvidenceHit) for hit in self.hits):
            raise TypeError("hits must contain only EvidenceHit values.")
        if not isinstance(self.status, ResultStatus):
            raise TypeError("status must be a ResultStatus.")
        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(failure, RetrievalFailure) for failure in self.failures):
            raise TypeError("failures must contain only RetrievalFailure values.")

        ranks = tuple(hit.rank for hit in self.hits)
        if ranks != tuple(range(1, len(self.hits) + 1)):
            raise InvalidRetrievalResultError(
                "hit ranks must be contiguous and match tuple order."
            )
        candidate_ids = tuple(hit.candidate.record_id for hit in self.hits)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise InvalidRetrievalResultError(
                "hits must not contain duplicate candidate identities."
            )
        if len(self.hits) > self.requested_limit.value:
            raise InvalidRetrievalResultError(
                "hits must not exceed the requested retrieval limit."
            )

        if self.status is ResultStatus.COMPLETED and self.failures:
            raise InvalidRetrievalResultError(
                "a completed result must not contain failures."
            )
        if self.status is ResultStatus.PARTIAL and (not self.hits or not self.failures):
            raise InvalidRetrievalResultError(
                "a partial result requires hits and failures."
            )
        if self.status is ResultStatus.FAILED and (self.hits or not self.failures):
            raise InvalidRetrievalResultError(
                "a failed result requires failures and no hits."
            )

        mismatch_failures = tuple(
            failure
            for failure in self.failures
            if failure.category is FailureCategory.CORPUS_VERSION_MISMATCH
        )
        if any(failure.channel is None for failure in mismatch_failures):
            if self.status is not ResultStatus.FAILED or self.hits:
                raise InvalidRetrievalResultError(
                    "an unscoped corpus mismatch requires a failed result with no hits."
                )
        mismatched_channels = {
            failure.channel
            for failure in mismatch_failures
            if failure.channel is not None
        }
        if any(
            contribution.channel in mismatched_channels
            for hit in self.hits
            for contribution in hit.contributions
        ):
            raise InvalidRetrievalResultError(
                "hits must not contain contributions from a mismatched channel."
            )

    @classmethod
    def from_query(
        cls,
        query: RetrievalQuery,
        *,
        observed_corpus_version: CorpusVersionId,
        hits: tuple[EvidenceHit, ...],
        status: ResultStatus,
        failures: tuple[RetrievalFailure, ...],
    ) -> "EvidenceResult":
        """Build a result and fail closed when a channel serves another version."""
        if not isinstance(query, RetrievalQuery):
            raise TypeError("query must be a RetrievalQuery.")
        if not isinstance(observed_corpus_version, CorpusVersionId):
            raise TypeError("observed_corpus_version must be a CorpusVersionId.")
        if observed_corpus_version != query.corpus_version:
            return cls(
                corpus_version=query.corpus_version,
                requested_limit=query.limit,
                hits=(),
                status=ResultStatus.FAILED,
                failures=(
                    RetrievalFailure(
                        FailureCategory.CORPUS_VERSION_MISMATCH,
                        False,
                    ),
                ),
            )
        return cls(
            corpus_version=query.corpus_version,
            requested_limit=query.limit,
            hits=hits,
            status=status,
            failures=failures,
        )
