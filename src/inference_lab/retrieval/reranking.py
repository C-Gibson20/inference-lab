"""Validated request boundary for RET-06 evidence reranking."""

from dataclasses import dataclass

from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import EvidenceResult


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
