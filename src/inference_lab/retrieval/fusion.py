"""Deterministic rank fusion for channel-scoped retrieval results."""

import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field

from inference_lab.provenance import RecordId
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.evidence import (
    ChannelScore,
    EvidenceHit,
    RetrievalCandidate,
)
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import (
    EvidenceResult,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.values import RetrievalChannel

FUSION_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    """Versioned reciprocal-rank-fusion parameters for lexical and dense search."""

    rank_constant: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate all values that can affect the fused ranking."""
        if type(self.rank_constant) is not int:
            raise TypeError("rank_constant must be an integer.")
        if self.rank_constant <= 0:
            raise InvalidRetrievalContractError("rank_constant must be positive.")
        for name, weight in (
            ("lexical_weight", self.lexical_weight),
            ("dense_weight", self.dense_weight),
        ):
            if isinstance(weight, bool) or type(weight) not in (int, float):
                raise TypeError(f"{name} must be a finite positive number.")
            if not math.isfinite(weight) or weight <= 0:
                raise InvalidRetrievalContractError(
                    f"{name} must be finite and positive."
                )

    @property
    def algorithm(self) -> str:
        """Return the named baseline fusion algorithm."""
        return "reciprocal_rank_fusion"

    @property
    def version(self) -> str:
        """Return the semantic version of the fusion policy contract."""
        return FUSION_POLICY_VERSION

    @property
    def identity(self) -> str:
        """Return a stable identity containing every ranking parameter."""
        return (
            f"{self.algorithm}/{self.version}/k={self.rank_constant}"
            f"/dense={_canonical_weight(self.dense_weight)}"
            f"/lexical={_canonical_weight(self.lexical_weight)}"
        )

    def weight_for(self, channel: RetrievalChannel) -> float:
        """Return the configured weight for a supported ranked channel."""
        if not isinstance(channel, RetrievalChannel):
            raise TypeError("channel must be a RetrievalChannel.")
        if channel is RetrievalChannel.LEXICAL:
            return self.lexical_weight
        if channel is RetrievalChannel.DENSE:
            return self.dense_weight
        raise InvalidRetrievalContractError(
            "fusion supports lexical and dense ranked channels only."
        )


def fuse_evidence_results(
    query: RetrievalQuery,
    channel_results: Mapping[RetrievalChannel, EvidenceResult],
    *,
    policy: FusionPolicy | None = None,
    candidate_deduplication_key: Callable[[RetrievalCandidate], Hashable] | None = None,
) -> EvidenceResult:
    """Fuse lexical and dense envelopes using weighted reciprocal ranks."""
    if not isinstance(query, RetrievalQuery):
        raise TypeError("query must be a RetrievalQuery.")
    if not isinstance(channel_results, Mapping):
        raise TypeError("channel_results must be a mapping.")
    if not channel_results:
        raise InvalidRetrievalContractError(
            "at least one ranked channel result is required."
        )
    selected_policy = FusionPolicy() if policy is None else policy
    if not isinstance(selected_policy, FusionPolicy):
        raise TypeError("policy must be a FusionPolicy or None.")
    if candidate_deduplication_key is not None and not callable(
        candidate_deduplication_key
    ):
        raise TypeError("candidate_deduplication_key must be callable or None.")

    results = _validate_channel_results(query, channel_results, selected_policy)
    mismatch = next(
        (
            result
            for _, result in results
            if result.corpus_version != query.corpus_version
        ),
        None,
    )
    if mismatch is not None:
        return EvidenceResult.from_query(
            query,
            observed_corpus_version=mismatch.corpus_version,
            hits=(),
            status=ResultStatus.COMPLETED,
            failures=(),
        )

    candidates: dict[RecordId, _CandidateContribution] = {}
    failures: list[RetrievalFailure] = []
    for channel, result in results:
        failures.extend(result.failures)
        weight = selected_policy.weight_for(channel)
        for hit in result.hits:
            contribution = hit.contributions[0]
            entry = candidates.get(hit.candidate.record_id)
            if entry is None:
                entry = _CandidateContribution(hit.candidate)
                candidates[hit.candidate.record_id] = entry
            else:
                entry.merge_candidate(hit.candidate)
            entry.add(channel, contribution, weight, selected_policy.rank_constant)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (-item.fused_score, item.candidate.record_id.as_str()),
    )
    ordered: list[_CandidateContribution] = []
    seen_keys: set[Hashable] = set()
    for item in ranked:
        key = (
            item.candidate.record_id
            if candidate_deduplication_key is None
            else candidate_deduplication_key(item.candidate)
        )
        try:
            duplicate = key in seen_keys
            seen_keys.add(key)
        except TypeError:
            raise InvalidRetrievalContractError(
                "candidate deduplication keys must be hashable."
            ) from None
        if duplicate:
            continue
        ordered.append(item)
        if len(ordered) == query.limit.value:
            break
    hits = tuple(
        EvidenceHit(
            candidate=item.candidate,
            contributions=tuple(
                item.contributions[channel]
                for channel in sorted(item.contributions, key=lambda value: value.value)
            ),
            rank=index,
        )
        for index, item in enumerate(ordered, start=1)
    )
    selected_failures = tuple(failures)
    if selected_failures:
        status = ResultStatus.PARTIAL if hits else ResultStatus.FAILED
    else:
        status = ResultStatus.COMPLETED
    return EvidenceResult(
        corpus_version=query.corpus_version,
        requested_limit=query.limit,
        hits=hits,
        status=status,
        failures=selected_failures,
    )


def _require_ranked_channel(policy: FusionPolicy, channel: RetrievalChannel) -> None:
    """Reject a channel that the fusion policy cannot weight."""
    policy.weight_for(channel)


def _validate_channel_results(
    query: RetrievalQuery,
    channel_results: Mapping[RetrievalChannel, EvidenceResult],
    policy: FusionPolicy,
) -> tuple[tuple[RetrievalChannel, EvidenceResult], ...]:
    """Validate the channel envelope and its channel-local rank contract."""
    results: list[tuple[RetrievalChannel, EvidenceResult]] = []
    for channel, result in channel_results.items():
        if not isinstance(channel, RetrievalChannel):
            raise TypeError("channel result keys must be RetrievalChannel values.")
        _require_ranked_channel(policy, channel)
        if not isinstance(result, EvidenceResult):
            raise TypeError("channel results must contain EvidenceResult values.")
        if result.requested_limit != query.limit:
            raise InvalidRetrievalContractError(
                "channel result limit must match the query limit."
            )
        if any(failure.channel not in (None, channel) for failure in result.failures):
            raise InvalidRetrievalContractError(
                "channel failures must match their result channel."
            )
        for hit in result.hits:
            if len(hit.contributions) != 1:
                raise InvalidRetrievalContractError(
                    "each channel hit must have exactly one contribution."
                )
            if hit.contributions[0].channel is not channel:
                raise InvalidRetrievalContractError(
                    "channel hit contribution must match its result channel."
                )
            if hit.contributions[0].rank != hit.rank:
                raise InvalidRetrievalContractError(
                    "channel contribution rank must match its result-local rank."
                )
        results.append((channel, result))
    return tuple(sorted(results, key=lambda item: item[0].value))


def _canonical_weight(value: float) -> str:
    """Format a validated numeric weight without losing identity precision."""
    return format(value, ".17g")


@dataclass(slots=True)
class _CandidateContribution:
    """Accumulate one record's candidate payload and ranked contributions."""

    candidate: RetrievalCandidate
    contributions: dict[RetrievalChannel, ChannelScore] = field(default_factory=dict)
    fused_score: float = 0.0

    def merge_candidate(self, candidate: RetrievalCandidate) -> None:
        """Merge exact record payloads and distinct provenance links."""
        if (
            candidate.kind is not self.candidate.kind
            or candidate.content != self.candidate.content
        ):
            raise InvalidRetrievalContractError(
                "one record identity must have consistent candidate content."
            )
        sources = {source.source_span_id: source for source in self.candidate.sources}
        for source in candidate.sources:
            existing = sources.get(source.source_span_id)
            if existing is not None and existing != source:
                raise InvalidRetrievalContractError(
                    "one source span identity must have consistent ownership."
                )
            sources[source.source_span_id] = source
        self.candidate = RetrievalCandidate(
            record_id=self.candidate.record_id,
            kind=self.candidate.kind,
            content=self.candidate.content,
            sources=tuple(
                sorted(
                    sources.values(),
                    key=lambda source: (
                        source.document_id.as_str(),
                        source.document_version_id.as_str(),
                        source.source_span_id.as_str(),
                    ),
                )
            ),
        )

    def add(
        self,
        channel: RetrievalChannel,
        contribution: ChannelScore,
        weight: float,
        rank_constant: int,
    ) -> None:
        """Record one channel and add its weighted reciprocal-rank score."""
        if channel in self.contributions:
            raise InvalidRetrievalContractError(
                "a record may contribute at most once per channel."
            )
        self.contributions[channel] = contribution
        self.fused_score += weight / (rank_constant + contribution.rank)
