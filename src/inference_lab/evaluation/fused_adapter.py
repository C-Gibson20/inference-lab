"""Deployment-local adapter for pinned lexical+dense retrieval fusion."""

from __future__ import annotations

import hashlib
from pathlib import Path

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.backends.runtime import EmbeddingBackend
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.retrieval_contracts import (
    RetrievalComponentIdentity,
    RetrievalSystemIdentity,
)
from inference_lab.evaluation.retrieval_execution import (
    RetrievalAdapterResponse,
    RetrievalEvaluationExecutionError,
)
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import (
    DENSE_COMPONENT_KIND,
    FUSION_DEDUPLICATION_VERSION,
    LEXICAL_COMPONENT_KIND,
    DenseError,
    DenseGeneration,
    FusionPolicy,
    LexicalError,
    LexicalGeneration,
    RetrievalLimit,
    RetrievalQuery,
    execute_hybrid_retrieval,
    open_dense_generation,
    open_lexical_generation,
)
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    CorpusSnapshot,
    MetadataRepository,
)


def _component_reference(
    snapshot: CorpusSnapshot,
    component_kind: str,
) -> ComponentGenerationReference:
    """Select one ready component generation from a pinned snapshot."""
    references = tuple(
        reference
        for reference in snapshot.manifest.component_generations
        if reference.component_kind.as_str() == component_kind
    )
    if len(references) != 1:
        raise RetrievalEvaluationExecutionError("invalid_hybrid_generation")
    reference = references[0]
    if (
        reference.state is not ComponentGenerationState.READY
        or reference.corpus_id != snapshot.corpus_id
        or reference.corpus_version != snapshot.version.corpus_version
    ):
        raise RetrievalEvaluationExecutionError("invalid_hybrid_generation")
    return reference


def _validate_lexical_generation(
    reference: ComponentGenerationReference,
    generation: LexicalGeneration,
) -> None:
    """Require the lexical manifest to match its committed reference."""
    manifest = generation.manifest
    actual = (
        manifest.corpus_id,
        manifest.corpus_version,
        manifest.generation_id,
        manifest.implementation,
        manifest.format_version,
        manifest.contract_version,
        manifest.configuration_digest,
        manifest.source_manifest_digest,
    )
    expected = (
        reference.corpus_id,
        reference.corpus_version,
        reference.generation_id,
        reference.implementation,
        reference.format_version,
        reference.contract_version,
        reference.configuration_digest,
        reference.source_manifest_digest,
    )
    if actual != expected:
        raise RetrievalEvaluationExecutionError("invalid_hybrid_generation")


def _validate_dense_generation(
    reference: ComponentGenerationReference,
    generation: DenseGeneration,
) -> None:
    """Require the dense manifest to match its committed reference."""
    manifest = generation.manifest
    configuration = manifest.compatibility.configuration
    actual = (
        manifest.corpus_id,
        manifest.corpus_version,
        manifest.generation_id,
        configuration.implementation,
        configuration.format_version,
        configuration.contract_version,
        manifest.configuration_digest,
        manifest.source_manifest_digest,
    )
    expected = (
        reference.corpus_id,
        reference.corpus_version,
        reference.generation_id,
        reference.implementation,
        reference.format_version,
        reference.contract_version,
        reference.configuration_digest,
        reference.source_manifest_digest,
    )
    if actual != expected:
        raise RetrievalEvaluationExecutionError("invalid_hybrid_generation")


class LocalFusedEvaluationAdapter:
    """Execute verbatim questions against one pinned lexical+dense fusion."""

    def __init__(
        self,
        repository: MetadataRepository,
        index_dir: Path,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        embedder: EmbeddingBackend,
        *,
        policy: FusionPolicy | None = None,
    ) -> None:
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(embedder, EmbeddingBackend):
            raise TypeError("embedder must implement EmbeddingBackend.")
        if policy is not None and not isinstance(policy, FusionPolicy):
            raise TypeError("policy must be a FusionPolicy or None.")
        self._repository = repository
        self._index_dir = index_dir
        self._corpus_id = corpus_id
        self._corpus_version = corpus_version
        self._embedder = embedder
        self._policy = FusionPolicy() if policy is None else policy
        lexical_reference, dense_reference, dense_compatibility = (
            self._pin_generations()
        )
        self._lexical_reference = lexical_reference
        self._dense_reference = dense_reference
        self._identity = self._build_identity(
            lexical_reference,
            dense_reference,
            dense_compatibility,
        )

    def _build_identity(
        self,
        lexical_reference: ComponentGenerationReference,
        dense_reference: ComponentGenerationReference,
        dense_compatibility: JsonObject,
    ) -> RetrievalSystemIdentity:
        """Build the complete path-free identity for this fused retriever."""
        policy_payload = {
            "algorithm": self._policy.algorithm,
            "version": self._policy.version,
            "rank_constant": self._policy.rank_constant,
            "lexical_weight": self._policy.lexical_weight,
            "dense_weight": self._policy.dense_weight,
            "identity": self._policy.identity,
        }
        identity_payload = {
            "dense_compatibility": dense_compatibility,
            "dense_generation_id": dense_reference.generation_id,
            "fusion_deduplication_version": FUSION_DEDUPLICATION_VERSION,
            "fusion_policy": policy_payload,
            "lexical_generation_id": lexical_reference.generation_id,
        }
        dense_model = self._embedder.metadata
        return RetrievalSystemIdentity(
            adapter_implementation="local-fused-adapter",
            adapter_contract_version="1.0.0",
            query_construction_algorithm="verbatim-question",
            query_construction_version="1.0.0",
            retrieval_algorithm="weighted-reciprocal-rank-fusion",
            retrieval_algorithm_version=self._policy.version,
            configuration_digest=(
                "sha256:"
                + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
            ),
            components=(
                RetrievalComponentIdentity(
                    component="dense-index",
                    implementation=dense_reference.implementation,
                    format_version=dense_reference.format_version,
                    contract_version=dense_reference.contract_version,
                    configuration_digest=(
                        "sha256:" + dense_reference.configuration_digest
                    ),
                    generation_id=dense_reference.generation_id,
                    source_manifest_digest=(
                        "sha256:" + dense_reference.source_manifest_digest
                    ),
                    model_identity=dense_model.backend_name,
                    model_revision=dense_model.model_revision,
                ),
                RetrievalComponentIdentity(
                    component="lexical-index",
                    implementation=lexical_reference.implementation,
                    format_version=lexical_reference.format_version,
                    contract_version=lexical_reference.contract_version,
                    configuration_digest=(
                        "sha256:" + lexical_reference.configuration_digest
                    ),
                    generation_id=lexical_reference.generation_id,
                    source_manifest_digest=(
                        "sha256:" + lexical_reference.source_manifest_digest
                    ),
                ),
            ),
        )

    def _pin_generations(
        self,
    ) -> tuple[ComponentGenerationReference, ComponentGenerationReference, JsonObject]:
        """Validate and return both index generations from one snapshot."""
        with self._repository.open_corpus_snapshot(
            self._corpus_id, self._corpus_version
        ) as snapshot:
            lexical_reference = _component_reference(snapshot, LEXICAL_COMPONENT_KIND)
            dense_reference = _component_reference(
                snapshot, DENSE_COMPONENT_KIND.as_str()
            )
            try:
                with (
                    open_lexical_generation(
                        self._index_dir, lexical_reference.generation_id
                    ) as lexical_generation,
                    open_dense_generation(
                        self._index_dir, dense_reference.generation_id
                    ) as dense_generation,
                ):
                    _validate_lexical_generation(lexical_reference, lexical_generation)
                    _validate_dense_generation(dense_reference, dense_generation)
                    compatibility = dense_generation.manifest.compatibility
                    if compatibility.model_metadata != self._embedder.metadata:
                        raise RetrievalEvaluationExecutionError(
                            "embedding_model_mismatch"
                        )
            except (DenseError, LexicalError):
                raise RetrievalEvaluationExecutionError(
                    "invalid_hybrid_generation"
                ) from None
            return (
                lexical_reference,
                dense_reference,
                compatibility.canonical_payload(),
            )
        raise RetrievalEvaluationExecutionError("invalid_hybrid_generation")

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the pinned indexes, embedding, and fusion identity."""
        return self._identity

    @property
    def fusion_policy(self) -> FusionPolicy:
        """Return the immutable policy used by the fused channel."""
        return self._policy

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
        """Run one question through both channels under one corpus snapshot."""
        if corpus_id != self._corpus_id or corpus_version != self._corpus_version:
            raise RetrievalEvaluationExecutionError("adapter_scope_mismatch")
        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase.")
        query = RetrievalQuery(case.question, corpus_version, candidate_depth)
        response: RetrievalAdapterResponse | None = None
        with self._repository.open_corpus_snapshot(
            corpus_id, corpus_version
        ) as snapshot:
            if (
                _component_reference(snapshot, LEXICAL_COMPONENT_KIND)
                != self._lexical_reference
                or _component_reference(snapshot, DENSE_COMPONENT_KIND.as_str())
                != self._dense_reference
            ):
                raise RetrievalEvaluationExecutionError("system_identity_changed")
            try:
                with (
                    open_lexical_generation(
                        self._index_dir, self._lexical_reference.generation_id
                    ) as lexical_generation,
                    open_dense_generation(
                        self._index_dir, self._dense_reference.generation_id
                    ) as dense_generation,
                ):
                    _validate_lexical_generation(
                        self._lexical_reference, lexical_generation
                    )
                    _validate_dense_generation(self._dense_reference, dense_generation)
                    result = execute_hybrid_retrieval(
                        self._repository,
                        self._index_dir,
                        corpus_id,
                        query,
                        self._embedder,
                        policy=self._policy,
                        profiling=profiling,
                        profiling_identity=profiling_identity,
                        snapshot=snapshot,
                        lexical_generation=lexical_generation,
                        dense_generation=dense_generation,
                    )
            except (DenseError, LexicalError):
                raise RetrievalEvaluationExecutionError(
                    "invalid_hybrid_generation"
                ) from None
            response = RetrievalAdapterResponse(result, self._identity)
        if response is None:
            raise RetrievalEvaluationExecutionError("invalid_hybrid_generation")
        return response
