"""Validated request boundary for RET-06 evidence reranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.backends.runtime import (
    BackendClosedError,
    BackendError,
    BackendNotReadyError,
    RerankCandidate,
    RerankerBackend,
    RerankRequest,
    RerankResult,
)
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.evidence import EvidenceHit
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)

MAX_RERANK_CANDIDATES = 100
RERANK_DEDUPLICATION_VERSION = "1.0.0"
RET06_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class EvidenceCount:
    """Configured final evidence count, constrained to the 5–10 baseline."""

    value: int = 8

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError("value must be an integer.")
        if not 5 <= self.value <= 10:
            raise InvalidRetrievalContractError(
                "evidence count must be between 5 and 10."
            )


@dataclass(frozen=True, slots=True)
class RerankCandidateLimit:
    """Configured reranker request size, bounded by RUN-05's hard cap."""

    value: int = MAX_RERANK_CANDIDATES

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError("value must be an integer.")
        if not 1 <= self.value <= MAX_RERANK_CANDIDATES:
            raise InvalidRetrievalContractError(
                "reranker candidate limit must be between 1 and 100."
            )


@dataclass(frozen=True, slots=True)
class RerankingRequest:
    """Pair one query with the fused evidence result from the same snapshot.

    The request retains RET-01 candidates and their provenance unchanged.
    Ranking scores are supplied only by the backend at execution time and are
    not added to this input contract.
    """

    query: RetrievalQuery
    fused_result: EvidenceResult

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject a result that cannot safely be reranked for this query."""
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("query must be a RetrievalQuery.")
        if not isinstance(self.fused_result, EvidenceResult):
            raise TypeError("fused_result must be an EvidenceResult.")
        if self.fused_result.corpus_version != self.query.corpus_version:
            raise InvalidRetrievalContractError(
                "fused result must target the query corpus version."
            )
        if self.fused_result.requested_limit != self.query.limit:
            raise InvalidRetrievalContractError(
                "fused result limit must match the query limit."
            )
        record_ids = tuple(hit.candidate.record_id for hit in self.fused_result.hits)
        if len(record_ids) != len(set(record_ids)):
            raise InvalidRetrievalContractError(
                "fused candidates must have unique record identities."
            )
        if any(not hit.candidate.sources for hit in self.fused_result.hits):
            raise InvalidRetrievalContractError(
                "fused candidates must retain source provenance."
            )


@dataclass(frozen=True, slots=True)
class PreparedReranking:
    """Bounded backend request and the original hits it identifies."""

    backend_request: RerankRequest | None
    candidate_hits: tuple[EvidenceHit, ...]
    available_candidate_count: int
    evidence_count: EvidenceCount
    candidate_limit: RerankCandidateLimit

    @property
    def truncated_candidate_count(self) -> int:
        """Return how many fused candidates the declared pool cap excluded."""
        return self.available_candidate_count - len(self.candidate_hits)


@dataclass(frozen=True, slots=True)
class ScoredEvidence:
    """An original evidence hit paired with its ordering-only reranker score."""

    hit: EvidenceHit
    score: float


@dataclass(frozen=True, slots=True)
class RerankingOutcome:
    """RET-01 result plus an explicit marker for fused-order degradation."""

    result: EvidenceResult
    degraded: bool
    identity: RerankingIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.result, EvidenceResult):
            raise TypeError("result must be an EvidenceResult.")
        if type(self.degraded) is not bool:
            raise TypeError("degraded must be a boolean.")
        if not isinstance(self.identity, RerankingIdentity):
            raise TypeError("identity must be a RerankingIdentity.")


@dataclass(frozen=True, slots=True)
class RerankingIdentity:
    """Stable, path-free configuration identity for one RET-06 run."""

    backend_name: str
    model_name: str
    model_revision: str
    preprocessing_profile_version: str
    evidence_count: int
    candidate_limit: int
    deduplication_version: str = RERANK_DEDUPLICATION_VERSION
    policy_version: str = RET06_POLICY_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("backend_name", self.backend_name),
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
            ("preprocessing_profile_version", self.preprocessing_profile_version),
            ("deduplication_version", self.deduplication_version),
            ("policy_version", self.policy_version),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        for name, value in (
            ("backend_name", self.backend_name),
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
        ):
            if (
                value.startswith(("/", "\\", "~"))
                or "://" in value
                or (len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\"))
            ):
                raise InvalidRetrievalContractError(
                    f"{name} must not contain a deployment path."
                )
        for name, value in (
            ("preprocessing_profile_version", self.preprocessing_profile_version),
            ("deduplication_version", self.deduplication_version),
            ("policy_version", self.policy_version),
        ):
            if not is_canonical_semantic_version(value):
                raise InvalidRetrievalContractError(
                    f"{name} must be a semantic version."
                )
        if type(self.evidence_count) is not int or not 5 <= self.evidence_count <= 10:
            raise InvalidRetrievalContractError(
                "identity evidence count must be between 5 and 10."
            )
        if (
            type(self.candidate_limit) is not int
            or not 1 <= self.candidate_limit <= MAX_RERANK_CANDIDATES
        ):
            raise InvalidRetrievalContractError(
                "identity candidate limit must be between 1 and 100."
            )

    @property
    def digest(self) -> str:
        """Return a stable digest over every result-affecting RET-06 choice."""
        encoded = json.dumps(
            {
                "backend_name": self.backend_name,
                "candidate_limit": self.candidate_limit,
                "deduplication_version": self.deduplication_version,
                "evidence_count": self.evidence_count,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "policy_version": self.policy_version,
                "preprocessing_profile_version": self.preprocessing_profile_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def prepare_rerank_request(
    request: RerankingRequest,
    *,
    evidence_count: EvidenceCount = EvidenceCount(),
    candidate_limit: RerankCandidateLimit = RerankCandidateLimit(),
) -> PreparedReranking:
    """Build a bounded RUN-05 request while retaining original source evidence.

    The pool policy retains the leading RET-05 fused candidates in their
    existing deterministic order. An empty pool requires no backend request.
    """
    if not isinstance(request, RerankingRequest):
        raise TypeError("request must be a RerankingRequest.")
    if not isinstance(evidence_count, EvidenceCount):
        raise TypeError("evidence_count must be an EvidenceCount.")
    if not isinstance(candidate_limit, RerankCandidateLimit):
        raise TypeError("candidate_limit must be a RerankCandidateLimit.")

    candidate_hits = request.fused_result.hits[: candidate_limit.value]
    if not candidate_hits:
        return PreparedReranking(
            backend_request=None,
            candidate_hits=(),
            available_candidate_count=0,
            evidence_count=evidence_count,
            candidate_limit=candidate_limit,
        )

    backend_request = RerankRequest(
        query=request.query.text,
        candidates=tuple(
            RerankCandidate(
                candidate_id=hit.candidate.record_id.as_str(),
                candidate_text=hit.candidate.content,
            )
            for hit in candidate_hits
        ),
        top_k=evidence_count.value,
    )
    return PreparedReranking(
        backend_request=backend_request,
        candidate_hits=candidate_hits,
        available_candidate_count=len(request.fused_result.hits),
        evidence_count=evidence_count,
        candidate_limit=candidate_limit,
    )


def map_rerank_result(
    prepared: PreparedReranking,
    response: RerankResult,
) -> tuple[ScoredEvidence, ...]:
    """Validate a backend response and restore original evidence identities."""
    if not isinstance(prepared, PreparedReranking):
        raise TypeError("prepared must be a PreparedReranking.")
    if not isinstance(response, RerankResult):
        raise InvalidRetrievalContractError(
            "reranker returned an invalid response type."
        )
    if prepared.backend_request is None:
        raise InvalidRetrievalContractError(
            "an empty candidate pool must not have a backend response."
        )

    hit_by_id = {
        hit.candidate.record_id.as_str(): hit for hit in prepared.candidate_hits
    }
    scored = response.scored_candidates
    expected_count = min(
        prepared.evidence_count.value,
        len(prepared.candidate_hits),
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in scored)
    ranks = tuple(candidate.rank for candidate in scored)
    if (
        len(scored) != expected_count
        or len(candidate_ids) != len(set(candidate_ids))
        or any(candidate_id not in hit_by_id for candidate_id in candidate_ids)
        or ranks != tuple(range(1, len(scored) + 1))
        or any(
            left.score < right.score
            for left, right in zip(scored, scored[1:], strict=False)
        )
    ):
        raise InvalidRetrievalContractError(
            "reranker response does not match the requested candidate pool."
        )
    return tuple(
        ScoredEvidence(hit=hit_by_id[item.candidate_id], score=item.score)
        for item in scored
    )


def rank_evidence_hits(
    scored_evidence: tuple[ScoredEvidence, ...],
) -> tuple[EvidenceHit, ...]:
    """Return score-ordered evidence with deterministic ties and fresh ranks.

    Scores affect ordering only. The returned hits are rebuilt from the
    original candidates and channel contributions, so source provenance and
    RET-05 diagnostics survive unchanged.
    """
    if not isinstance(scored_evidence, tuple):
        raise TypeError("scored_evidence must be a tuple.")
    if any(not isinstance(item, ScoredEvidence) for item in scored_evidence):
        raise TypeError("scored_evidence must contain ScoredEvidence values.")
    ordered = sorted(
        scored_evidence,
        key=lambda item: (-item.score, item.hit.candidate.record_id.as_str()),
    )
    ranked_hits = tuple(
        EvidenceHit(
            candidate=item.hit.candidate,
            contributions=item.hit.contributions,
            rank=rank,
        )
        for rank, item in enumerate(ordered, start=1)
    )
    return deduplicate_evidence_hits(ranked_hits)


def deduplicate_evidence_hits(
    ranked_hits: tuple[EvidenceHit, ...],
) -> tuple[EvidenceHit, ...]:
    """Drop exact repeated source evidence, retaining the highest-ranked hit.

    The key uses candidate kind and the complete ordered-independent set of
    source-span identities with their owning document versions. Document- or
    family-level similarity and span overlap do not make candidates duplicates.
    """
    if not isinstance(ranked_hits, tuple):
        raise TypeError("ranked_hits must be a tuple.")
    if any(not isinstance(hit, EvidenceHit) for hit in ranked_hits):
        raise TypeError("ranked_hits must contain EvidenceHit values.")
    ranks = tuple(hit.rank for hit in ranked_hits)
    if ranks != tuple(range(1, len(ranked_hits) + 1)):
        raise InvalidRetrievalContractError(
            "ranked hits must have contiguous ranks matching tuple order."
        )
    record_ids = tuple(hit.candidate.record_id for hit in ranked_hits)
    if len(record_ids) != len(set(record_ids)):
        raise InvalidRetrievalContractError(
            "ranked hits must have unique candidate identities."
        )

    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    selected: list[EvidenceHit] = []
    for hit in ranked_hits:
        key = (
            hit.candidate.kind.value,
            tuple(
                sorted(
                    (
                        source.document_version_id.as_str(),
                        source.source_span_id.as_str(),
                    )
                    for source in hit.candidate.sources
                )
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)
    return tuple(
        EvidenceHit(
            candidate=hit.candidate,
            contributions=hit.contributions,
            rank=rank,
        )
        for rank, hit in enumerate(selected, start=1)
    )


def rerank_evidence(
    backend: RerankerBackend,
    request: RerankingRequest,
    *,
    evidence_count: EvidenceCount = EvidenceCount(),
    candidate_limit: RerankCandidateLimit = RerankCandidateLimit(),
) -> tuple[PreparedReranking, tuple[ScoredEvidence, ...]]:
    """Call RUN-05 with the bounded request and map scores to source evidence."""
    if not callable(getattr(backend, "rerank", None)):
        raise TypeError("backend must implement the reranker contract.")
    prepared = prepare_rerank_request(
        request,
        evidence_count=evidence_count,
        candidate_limit=candidate_limit,
    )
    if prepared.backend_request is None:
        return prepared, ()
    response = backend.rerank(prepared.backend_request)
    return prepared, map_rerank_result(prepared, response)


def execute_reranking(
    backend: RerankerBackend,
    query: RetrievalQuery,
    fused_result: EvidenceResult,
    *,
    evidence_count: EvidenceCount = EvidenceCount(),
    candidate_limit: RerankCandidateLimit = RerankCandidateLimit(),
) -> RerankingOutcome:
    """Select evidence and convert backend failures to bounded outcomes.

    Retryable backend failures may return the deterministic fused order as a
    partial result. Invalid inputs, malformed responses, and non-retryable
    failures return no hits. Error messages and candidate payloads never cross
    into the result contract.
    """
    if not isinstance(query, RetrievalQuery):
        raise TypeError("query must be a RetrievalQuery.")
    if not isinstance(fused_result, EvidenceResult):
        raise TypeError("fused_result must be an EvidenceResult.")
    if not isinstance(evidence_count, EvidenceCount):
        raise TypeError("evidence_count must be an EvidenceCount.")
    if not isinstance(candidate_limit, RerankCandidateLimit):
        raise TypeError("candidate_limit must be a RerankCandidateLimit.")
    if not callable(getattr(backend, "rerank", None)):
        raise TypeError("backend must implement the reranker contract.")
    identity = _reranking_identity(backend, evidence_count, candidate_limit)

    if fused_result.corpus_version != query.corpus_version:
        return _failed_reranking_outcome(
            query,
            RetrievalFailure(
                FailureCategory.CORPUS_VERSION_MISMATCH,
                retryable=False,
            ),
            identity=identity,
        )
    if fused_result.requested_limit != query.limit:
        return _failed_reranking_outcome(
            query,
            RetrievalFailure(
                FailureCategory.INVALID_CHANNEL_RESPONSE,
                retryable=False,
            ),
            identity=identity,
        )
    if fused_result.status is ResultStatus.FAILED or not fused_result.hits:
        return RerankingOutcome(fused_result, degraded=False, identity=identity)

    request = RerankingRequest(query, fused_result)
    try:
        _, scored = rerank_evidence(
            backend,
            request,
            evidence_count=evidence_count,
            candidate_limit=candidate_limit,
        )
    except BackendError as error:
        failure = RetrievalFailure(
            category=(
                FailureCategory.CHANNEL_UNAVAILABLE
                if isinstance(error, (BackendNotReadyError, BackendClosedError))
                else FailureCategory.CHANNEL_EXECUTION_FAILURE
            ),
            retryable=error.retryable,
        )
        if error.retryable:
            fallback = deduplicate_evidence_hits(fused_result.hits)[
                : evidence_count.value
            ]
            return RerankingOutcome(
                _result_with_outcome(
                    query,
                    fallback,
                    (*fused_result.failures, failure),
                ),
                degraded=bool(fallback),
                identity=identity,
            )
        return _failed_reranking_outcome(
            query,
            failure,
            prior_failures=fused_result.failures,
            identity=identity,
        )
    except InvalidRetrievalContractError:
        return _failed_reranking_outcome(
            query,
            RetrievalFailure(FailureCategory.INVALID_CHANNEL_RESPONSE, False),
            prior_failures=fused_result.failures,
            identity=identity,
        )
    except Exception:
        return _failed_reranking_outcome(
            query,
            RetrievalFailure(FailureCategory.CHANNEL_EXECUTION_FAILURE, False),
            prior_failures=fused_result.failures,
            identity=identity,
        )

    hits = rank_evidence_hits(scored)[: evidence_count.value]
    return RerankingOutcome(
        _result_with_outcome(query, hits, fused_result.failures),
        degraded=False,
        identity=identity,
    )


def _reranking_identity(
    backend: RerankerBackend,
    evidence_count: EvidenceCount,
    candidate_limit: RerankCandidateLimit,
) -> RerankingIdentity:
    """Capture backend and policy identity without exposing deployment paths."""
    metadata = backend.metadata
    if metadata.preprocessing_profile_version is None:
        raise InvalidRetrievalContractError(
            "reranker metadata must declare its preprocessing profile version."
        )
    return RerankingIdentity(
        backend_name=metadata.backend_name,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        preprocessing_profile_version=metadata.preprocessing_profile_version,
        evidence_count=evidence_count.value,
        candidate_limit=candidate_limit.value,
    )


def _result_with_outcome(
    query: RetrievalQuery,
    hits: tuple[EvidenceHit, ...],
    failures: tuple[RetrievalFailure, ...],
) -> EvidenceResult:
    """Construct a valid final RET-01 result with contiguous ranks."""
    ranked_hits = tuple(
        EvidenceHit(hit.candidate, hit.contributions, rank)
        for rank, hit in enumerate(hits, start=1)
    )
    if failures:
        status = ResultStatus.PARTIAL if ranked_hits else ResultStatus.FAILED
    else:
        status = ResultStatus.COMPLETED
    return EvidenceResult(
        corpus_version=query.corpus_version,
        requested_limit=query.limit,
        hits=ranked_hits,
        status=status,
        failures=failures,
    )


def _failed_reranking_outcome(
    query: RetrievalQuery,
    failure: RetrievalFailure,
    *,
    identity: RerankingIdentity,
    prior_failures: tuple[RetrievalFailure, ...] = (),
) -> RerankingOutcome:
    """Build a hit-free failure result without retaining private error data."""
    return RerankingOutcome(
        _result_with_outcome(query, (), (*prior_failures, failure)),
        degraded=False,
        identity=identity,
    )
