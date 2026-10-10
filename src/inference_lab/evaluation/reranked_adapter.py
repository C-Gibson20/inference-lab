"""Deployment-local adapter for pinned fused retrieval with reranking."""

import hashlib

from inference_lab._json import canonical_json_bytes
from inference_lab.backends.runtime import RerankerBackend
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.fused_adapter import LocalFusedEvaluationAdapter
from inference_lab.evaluation.retrieval_contracts import RetrievalSystemIdentity
from inference_lab.evaluation.retrieval_execution import (
    RetrievalAdapterResponse,
    RetrievalEvaluationExecutionError,
)
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import (
    FusionPolicy,
    InvalidRetrievalContractError,
    RetrievalLimit,
    RetrievalQuery,
)
from inference_lab.retrieval.reranking import (
    EvidenceCount,
    RerankCandidateLimit,
    RerankingIdentity,
    execute_reranking,
    reranking_identity_for_backend,
)


class LocalRerankedEvaluationAdapter:
    """Execute one question through pinned fusion and configured reranking."""

    def __init__(
        self,
        fused_adapter: LocalFusedEvaluationAdapter,
        reranker: RerankerBackend,
        *,
        evidence_count: EvidenceCount = EvidenceCount(),
        candidate_limit: RerankCandidateLimit = RerankCandidateLimit(),
    ) -> None:
        if not isinstance(fused_adapter, LocalFusedEvaluationAdapter):
            raise TypeError("fused_adapter must be a LocalFusedEvaluationAdapter.")
        if not callable(getattr(reranker, "rerank", None)):
            raise TypeError("reranker must implement RerankerBackend.")
        if not isinstance(evidence_count, EvidenceCount):
            raise TypeError("evidence_count must be an EvidenceCount.")
        if not isinstance(candidate_limit, RerankCandidateLimit):
            raise TypeError("candidate_limit must be a RerankCandidateLimit.")
        self._fused_adapter = fused_adapter
        self._reranker = reranker
        self._evidence_count = evidence_count
        self._candidate_limit = candidate_limit
        try:
            self._reranking_identity = reranking_identity_for_backend(
                reranker, evidence_count, candidate_limit
            )
        except InvalidRetrievalContractError:
            raise RetrievalEvaluationExecutionError(
                "invalid_reranker_identity"
            ) from None
        fused_identity = fused_adapter.system_identity
        identity_payload = {
            "fused_system_identity": fused_identity.as_dict(),
            "reranking_identity_digest": self._reranking_identity.digest,
        }
        self._identity = RetrievalSystemIdentity(
            adapter_implementation="local-reranked-adapter",
            adapter_contract_version="1.0.0",
            query_construction_algorithm="verbatim-question",
            query_construction_version="1.0.0",
            retrieval_algorithm="fused-retrieval-reranking",
            retrieval_algorithm_version=self._reranking_identity.policy_version,
            configuration_digest=(
                "sha256:"
                + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
            ),
            components=fused_identity.components,
        )

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the complete path-free fused and reranking identity."""
        return self._identity

    @property
    def fusion_policy(self) -> FusionPolicy:
        """Return the fusion policy shared with the pre-rerank channel."""
        return self._fused_adapter.fusion_policy

    @property
    def reranking_identity(self) -> RerankingIdentity:
        """Return the exact backend, limit, and policy identity."""
        return self._reranking_identity

    def execute(
        self,
        case: EvaluationCase,
        *,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        candidate_depth: RetrievalLimit,
        profiling: ProfilingContext,
        profiling_identity: str,
    ) -> RetrievalAdapterResponse:
        """Run fused retrieval then rerank its original provenance-bearing hits."""
        fused_response = self._fused_adapter.execute(
            case,
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            candidate_depth=candidate_depth,
            profiling=profiling,
            profiling_identity=f"{profiling_identity}-fused",
        )
        query = RetrievalQuery(case.question, corpus_version, candidate_depth)
        outcome = execute_reranking(
            self._reranker,
            query,
            fused_response.result,
            corpus_id=corpus_id,
            evidence_count=self._evidence_count,
            candidate_limit=self._candidate_limit,
        )
        if outcome.identity != self._reranking_identity:
            raise RetrievalEvaluationExecutionError("system_identity_changed")
        return RetrievalAdapterResponse(outcome.result, self._identity)
