"""Deployment-local adapter for the pinned dense retriever."""

from __future__ import annotations

import hashlib
from pathlib import Path

from inference_lab._json import canonical_json_bytes
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
    DenseError,
    DenseGeneration,
    RetrievalLimit,
    RetrievalQuery,
    execute_dense_retrieval,
    open_dense_generation,
)
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    CorpusSnapshot,
    MetadataRepository,
)


def _dense_reference(snapshot: CorpusSnapshot) -> ComponentGenerationReference:
    """Select the ready dense generation that belongs to one pinned version."""
    references = tuple(
        reference
        for reference in snapshot.manifest.component_generations
        if reference.component_kind == DENSE_COMPONENT_KIND
    )
    if len(references) != 1:
        raise RetrievalEvaluationExecutionError("invalid_dense_generation")
    reference = references[0]
    if (
        reference.state is not ComponentGenerationState.READY
        or reference.corpus_id != snapshot.corpus_id
        or reference.corpus_version != snapshot.version.corpus_version
    ):
        raise RetrievalEvaluationExecutionError("invalid_dense_generation")
    return reference


def _validate_dense_generation(
    reference: ComponentGenerationReference,
    generation: DenseGeneration,
) -> None:
    """Require the durable dense manifest to match its committed reference."""
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
        raise RetrievalEvaluationExecutionError("invalid_dense_generation")


class LocalDenseEvaluationAdapter:
    """Execute verbatim evaluation questions against one pinned dense index."""

    def __init__(
        self,
        repository: MetadataRepository,
        index_dir: Path,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        embedder: EmbeddingBackend,
    ) -> None:
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(embedder, EmbeddingBackend):
            raise TypeError("embedder must implement EmbeddingBackend.")
        self._repository = repository
        self._index_dir = index_dir
        self._corpus_id = corpus_id
        self._corpus_version = corpus_version
        self._embedder = embedder
        reference = self._pin_generation()
        self._reference = reference
        with open_dense_generation(index_dir, reference.generation_id) as generation:
            _validate_dense_generation(reference, generation)
            compatibility = generation.manifest.compatibility
            if compatibility.model_metadata != embedder.metadata:
                raise RetrievalEvaluationExecutionError("embedding_model_mismatch")
            identity_payload = {
                "compatibility": compatibility.canonical_payload(),
                "generation_id": reference.generation_id,
                "query_construction": "verbatim-question",
            }
            identity = RetrievalSystemIdentity(
                adapter_implementation="local-dense-adapter",
                adapter_contract_version="1.0.0",
                query_construction_algorithm="verbatim-question",
                query_construction_version="1.0.0",
                retrieval_algorithm="dense-vector-ranking",
                retrieval_algorithm_version=compatibility.configuration.contract_version,
                configuration_digest=(
                    "sha256:"
                    + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
                ),
                components=(
                    RetrievalComponentIdentity(
                        component=DENSE_COMPONENT_KIND.as_str(),
                        implementation=compatibility.configuration.implementation,
                        format_version=compatibility.configuration.format_version,
                        contract_version=compatibility.configuration.contract_version,
                        configuration_digest=(
                            "sha256:" + reference.configuration_digest
                        ),
                        generation_id=reference.generation_id,
                        source_manifest_digest=(
                            "sha256:" + reference.source_manifest_digest
                        ),
                        model_identity=compatibility.model_metadata.backend_name,
                        model_revision=compatibility.model_metadata.model_revision,
                    ),
                ),
            )
        self._identity = identity

    def _pin_generation(self) -> ComponentGenerationReference:
        """Validate and return the exact dense generation for this version."""
        with self._repository.open_corpus_snapshot(
            self._corpus_id, self._corpus_version
        ) as snapshot:
            reference = _dense_reference(snapshot)
            try:
                with open_dense_generation(
                    self._index_dir, reference.generation_id
                ) as generation:
                    _validate_dense_generation(reference, generation)
            except DenseError:
                raise RetrievalEvaluationExecutionError(
                    "invalid_dense_generation"
                ) from None
            return reference
        raise RetrievalEvaluationExecutionError("invalid_dense_generation")

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the path-free model, configuration, and generation identity."""
        return self._identity

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
        """Run one private question against the exact pinned dense generation."""
        if corpus_id != self._corpus_id or corpus_version != self._corpus_version:
            raise RetrievalEvaluationExecutionError("adapter_scope_mismatch")
        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase.")
        query = RetrievalQuery(case.question, corpus_version, candidate_depth)
        response: RetrievalAdapterResponse | None = None
        with self._repository.open_corpus_snapshot(
            corpus_id, corpus_version
        ) as snapshot:
            if _dense_reference(snapshot) != self._reference:
                raise RetrievalEvaluationExecutionError("system_identity_changed")
            try:
                with open_dense_generation(
                    self._index_dir, self._reference.generation_id
                ) as generation:
                    _validate_dense_generation(self._reference, generation)
                    result = execute_dense_retrieval(
                        self._repository,
                        self._index_dir,
                        corpus_id,
                        query,
                        self._embedder,
                        profiling=profiling,
                        profiling_identity=profiling_identity,
                        snapshot=snapshot,
                        generation=generation,
                    )
            except DenseError:
                raise RetrievalEvaluationExecutionError(
                    "invalid_dense_generation"
                ) from None
            response = RetrievalAdapterResponse(result, self._identity)
        if response is None:
            raise RetrievalEvaluationExecutionError("invalid_dense_generation")
        return response
