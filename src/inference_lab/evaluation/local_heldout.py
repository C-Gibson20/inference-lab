"""Compose local retrieval adapters for a frozen channel baseline."""

import hashlib
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from inference_lab._json import canonical_json_bytes
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.heldout_gate import (
    HeldOutChannel,
    HeldOutRunContract,
)
from inference_lab.evaluation.heldout_runner import HeldOutChannelRun
from inference_lab.evaluation.retrieval_contracts import (
    RetrievalEvaluationContractError,
    RetrievalSystemIdentity,
)
from inference_lab.evaluation.retrieval_execution import (
    RetrievalAdapterResponse,
    RetrievalEvaluationAdapter,
    RetrievalEvaluationRequest,
    RetrievalEvaluationRunResult,
    run_retrieval_evaluation,
)
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
    RetrievalQuery,
)


def _unavailable_metadata_identity() -> RetrievalSystemIdentity:
    """Identify the explicit unavailable metadata-filter baseline."""
    configuration = canonical_json_bytes(
        {"channel": "metadata", "status": "unavailable"}
    )
    return RetrievalSystemIdentity(
        adapter_implementation="unavailable-metadata-adapter",
        adapter_contract_version="1.0.0",
        query_construction_algorithm="verbatim-question",
        query_construction_version="1.0.0",
        retrieval_algorithm="metadata-filter-unavailable",
        retrieval_algorithm_version="1.0.0",
        configuration_digest=f"sha256:{hashlib.sha256(configuration).hexdigest()}",
        components=(),
    )


class UnavailableMetadataEvaluationAdapter:
    """Return a typed unavailable result when no metadata-only path exists."""

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the fixed identity of this unavailable comparison path."""
        return _unavailable_metadata_identity()

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
        """Preserve metadata unavailability as a bounded retrieval failure."""
        del corpus_id, profiling, profiling_identity
        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase.")
        if not isinstance(candidate_depth, RetrievalLimit):
            raise TypeError("candidate_depth must be a RetrievalLimit.")
        query = RetrievalQuery(case.question, corpus_version, candidate_depth)
        result = EvidenceResult.from_query(
            query,
            observed_corpus_version=corpus_version,
            hits=(),
            status=ResultStatus.FAILED,
            failures=(
                RetrievalFailure(
                    FailureCategory.CHANNEL_UNAVAILABLE,
                    retryable=False,
                    channel=RetrievalChannel.METADATA,
                ),
            ),
        )
        return RetrievalAdapterResponse(result, self.system_identity)


class LocalHeldOutChannelRunner:
    """Run the declared local adapters under one frozen evaluation scope."""

    def __init__(
        self,
        contract: HeldOutRunContract,
        request: RetrievalEvaluationRequest,
        adapters: Mapping[HeldOutChannel, RetrievalEvaluationAdapter],
    ) -> None:
        if not isinstance(contract, HeldOutRunContract):
            raise TypeError("contract must be a HeldOutRunContract.")
        if not isinstance(request, RetrievalEvaluationRequest):
            raise TypeError("request must be a RetrievalEvaluationRequest.")
        if not isinstance(adapters, Mapping):
            raise TypeError("adapters must map each HeldOutChannel to an adapter.")
        selected_adapters = dict(adapters)
        if set(selected_adapters) != set(HeldOutChannel):
            raise RetrievalEvaluationContractError(
                "adapters must provide every declared baseline channel exactly once."
            )
        self._contract = contract
        self._request = request
        self._adapters = MappingProxyType(selected_adapters)
        self._identity_by_channel = {
            identity.channel: identity for identity in contract.channel_identities
        }
        self._validate_scope()
        self._validate_adapters()

    def _validate_scope(self) -> None:
        """Require the shared request to match the frozen baseline identity."""
        expected = self._contract.evaluation_identity
        request = self._request
        manifest = request.dataset.manifest
        checksum = manifest.checksums.get(manifest.case_file)
        snapshot = request.relationship_snapshot
        if (
            manifest.dataset_id != expected.dataset_id
            or manifest.dataset_version != expected.dataset_version
            or manifest.schema_version != expected.dataset_schema_version
            or checksum != expected.case_file_checksum
            or request.corpus_id != expected.corpus_id
            or request.corpus_version != expected.corpus_version
            or request.selected_split != "held_out"
            or request.selected_split != expected.selected_split
            or request.selected_case_ids != expected.selected_case_ids
            or request.policy != expected.policy
            or snapshot.corpus_id != expected.corpus_id
            or snapshot.corpus_version != expected.corpus_version
            or snapshot.content_digest != expected.relationship_snapshot_digest
        ):
            raise RetrievalEvaluationContractError(
                "request must match the frozen dataset, case set, and corpus scope."
            )
        cases_by_id = {case.case_id: case for case in request.dataset.cases}
        if len(cases_by_id) != len(request.dataset.cases):
            raise RetrievalEvaluationContractError("dataset case IDs must be unique.")
        if any(
            case_id not in cases_by_id or cases_by_id[case_id].split != "held_out"
            for case_id in expected.selected_case_ids
        ):
            raise RetrievalEvaluationContractError(
                "request cases must be the frozen held-out cases."
            )

    def _validate_adapters(self) -> None:
        """Pin every adapter identity and its specialized retrieval policy."""
        for channel, adapter in self._adapters.items():
            if not isinstance(channel, HeldOutChannel):
                raise TypeError("adapter keys must be HeldOutChannel values.")
            if not isinstance(adapter, RetrievalEvaluationAdapter):
                raise TypeError("each adapter must implement the evaluation contract.")
            declared = self._identity_by_channel[channel]
            if adapter.system_identity != declared.system_identity:
                raise RetrievalEvaluationContractError(
                    "adapter identity differs from the frozen channel identity."
                )
            if getattr(adapter, "fusion_policy", None) != declared.fusion_policy:
                raise RetrievalEvaluationContractError(
                    "adapter fusion policy differs from the frozen channel identity."
                )
            if getattr(adapter, "reranking_identity", None) != (
                declared.reranking_identity
            ):
                raise RetrievalEvaluationContractError(
                    "adapter reranking identity differs from the frozen channel identity."
                )

    def system_identity(self, channel: HeldOutChannel) -> RetrievalSystemIdentity:
        """Return the predeclared identity for one comparison path."""
        if not isinstance(channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        return self._adapters[channel].system_identity

    def __call__(self, channel: HeldOutChannel) -> HeldOutChannelRun:
        """Run one path through the shared retrieval measurement contract."""
        if not isinstance(channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        channel_request = replace(self._request, adapter=self._adapters[channel])
        result: RetrievalEvaluationRunResult = run_retrieval_evaluation(channel_request)
        degraded_case_ids = tuple(
            case.case_id
            for case in result.case_results
            if case.status is ResultStatus.PARTIAL
        )
        return HeldOutChannelRun(
            channel=channel,
            result=result,
            degraded_case_ids=degraded_case_ids,
        )
