"""Deployment-local EVAL-03 adapter for the shipped lexical retriever."""

from __future__ import annotations

import hashlib
from pathlib import Path

from inference_lab._json import canonical_json_bytes
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.retrieval_contracts import (
    DuplicateRelationshipSnapshot,
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
    LexicalMatchMode,
    RetrievalLimit,
    RetrievalQuery,
    execute_lexical_retrieval,
    open_lexical_generation,
)
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    CorpusSnapshot,
    MetadataRepository,
)


def _lexical_reference(snapshot: CorpusSnapshot) -> ComponentGenerationReference:
    """Select the one ready lexical generation of this immutable snapshot."""
    references = tuple(
        reference
        for reference in snapshot.manifest.component_generations
        if reference.component_kind.as_str() == "lexical-index"
    )
    if len(references) != 1:
        raise RetrievalEvaluationExecutionError("invalid_lexical_generation")
    reference = references[0]
    if (
        reference.state is not ComponentGenerationState.READY
        or reference.corpus_id != snapshot.corpus_id
        or reference.corpus_version != snapshot.version.corpus_version
    ):
        raise RetrievalEvaluationExecutionError("invalid_lexical_generation")
    return reference


def _validate_generation(
    reference: ComponentGenerationReference, manifest: object
) -> None:
    """Require the opened generation to agree with its committed reference."""
    for field in (
        "corpus_id",
        "corpus_version",
        "generation_id",
        "implementation",
        "format_version",
        "contract_version",
        "configuration_digest",
        "source_manifest_digest",
    ):
        if getattr(reference, field) != getattr(manifest, field, None):
            raise RetrievalEvaluationExecutionError("invalid_lexical_generation")


def lexical_relationship_snapshot(
    repository: MetadataRepository,
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
) -> DuplicateRelationshipSnapshot:
    """Capture visible duplicate and family facts for one committed version."""
    with repository.open_corpus_snapshot(corpus_id, corpus_version) as snapshot:
        memberships = snapshot.manifest.document_memberships
        duplicates = {
            relationship.key(): relationship
            for membership in memberships
            for relationship in snapshot.list_duplicate_relationships(
                membership.document_version_id
            )
        }
        families = {
            family.key(): family
            for membership in memberships
            for family in snapshot.list_family_memberships_by_document(
                membership.document_id
            )
        }
        return DuplicateRelationshipSnapshot(
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            memberships=memberships,
            duplicate_relationships=tuple(duplicates.values()),
            family_memberships=tuple(families.values()),
        )
    raise RetrievalEvaluationExecutionError("invalid_lexical_generation")


class LocalLexicalEvaluationAdapter:
    """Execute verbatim evaluation questions against one pinned lexical index."""

    def __init__(
        self,
        repository: MetadataRepository,
        index_dir: Path,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        *,
        mode: LexicalMatchMode = LexicalMatchMode.TERMS,
    ) -> None:
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(mode, LexicalMatchMode):
            raise TypeError("mode must be a LexicalMatchMode.")
        self._repository = repository
        self._index_dir = index_dir
        self._corpus_id = corpus_id
        self._corpus_version = corpus_version
        self._mode = mode
        reference, algorithm = self._pin_generation()
        self._reference = reference
        configuration = canonical_json_bytes(
            {
                "lexical_configuration_digest": reference.configuration_digest,
                "mode": mode.value,
            }
        )
        self._identity = RetrievalSystemIdentity(
            adapter_implementation="local-lexical-adapter",
            adapter_contract_version="1.0.0",
            query_construction_algorithm="verbatim-question",
            query_construction_version="1.0.0",
            retrieval_algorithm=algorithm,
            retrieval_algorithm_version=reference.contract_version,
            configuration_digest=f"sha256:{hashlib.sha256(configuration).hexdigest()}",
            components=(
                RetrievalComponentIdentity(
                    component="lexical-index",
                    implementation=reference.implementation,
                    format_version=reference.format_version,
                    contract_version=reference.contract_version,
                    configuration_digest=f"sha256:{reference.configuration_digest}",
                    generation_id=reference.generation_id,
                    source_manifest_digest=f"sha256:{reference.source_manifest_digest}",
                ),
            ),
        )

    def _pin_generation(self) -> tuple[ComponentGenerationReference, str]:
        """Validate and identify the exact generation used by this adapter."""
        with self._repository.open_corpus_snapshot(
            self._corpus_id, self._corpus_version
        ) as snapshot:
            reference = _lexical_reference(snapshot)
            with open_lexical_generation(
                self._index_dir, reference.generation_id
            ) as generation:
                _validate_generation(reference, generation.manifest)
                return reference, generation.manifest.configuration.engine_revision
        raise RetrievalEvaluationExecutionError("invalid_lexical_generation")

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the immutable, path-free configuration and generation identity."""
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
        """Run one private question with the evaluator's profiling context."""
        if corpus_id != self._corpus_id or corpus_version != self._corpus_version:
            raise RetrievalEvaluationExecutionError("adapter_scope_mismatch")
        with self._repository.open_corpus_snapshot(
            corpus_id, corpus_version
        ) as snapshot:
            if _lexical_reference(snapshot) != self._reference:
                raise RetrievalEvaluationExecutionError("system_identity_changed")
            with open_lexical_generation(
                self._index_dir, self._reference.generation_id
            ) as generation:
                _validate_generation(self._reference, generation.manifest)
                result = execute_lexical_retrieval(
                    self._repository,
                    self._index_dir,
                    corpus_id,
                    RetrievalQuery(case.question, corpus_version, candidate_depth),
                    mode=self._mode,
                    profiling=profiling,
                    profiling_identity=profiling_identity,
                    snapshot=snapshot,
                    generation=generation,
                )
                return RetrievalAdapterResponse(result, self._identity)
        raise RetrievalEvaluationExecutionError("invalid_lexical_generation")
