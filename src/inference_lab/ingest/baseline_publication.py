"""Register, seal, and publish baseline component generations."""

from dataclasses import replace
from typing import Protocol, TypeAlias

from inference_lab.ingest.baseline_contracts import BaselineFailureCategory
from inference_lab.ingest.baseline_generation import (
    BASELINE_CHUNK_COMPONENT_KIND,
    BaselineChunkGenerationManifest,
    BaselineGenerationValidationError,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import (
    BeginCorpusUpdateOutcome,
    ChunkSetReader,
    CommitCorpusUpdateOutcome,
    ComponentGenerationValidator,
    MetadataTransaction,
    NoCorpusChangeOutcome,
)
from inference_lab.storage.versioning import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
)

BaselinePublicationOutcome: TypeAlias = (
    CommitCorpusUpdateOutcome | NoCorpusChangeOutcome
)
BaselineChunkPublicationOutcome: TypeAlias = BaselinePublicationOutcome


class BaselineGenerationParticipant(Protocol):
    """Provide one component generation and its corpus-scoped validator."""

    def component_generation(self) -> ComponentGenerationReference:
        """Return the preparing generation reference to register."""
        ...

    def build_validator(
        self,
        reader: ChunkSetReader,
    ) -> ComponentGenerationValidator:
        """Return a validator for the exact durable component outputs."""
        ...


def begin_baseline_corpus_update(
    transaction: MetadataTransaction,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    *,
    expected_corpus_version: CorpusVersionId | None,
    required_component_kinds: tuple[ComponentKind, ...] = (
        BASELINE_CHUNK_COMPONENT_KIND,
    ),
) -> BeginCorpusUpdateOutcome:
    """Begin an exact-parent update requiring selected components."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
    if expected_corpus_version is not None and not isinstance(
        expected_corpus_version,
        CorpusVersionId,
    ):
        raise TypeError("expected_corpus_version must be a CorpusVersionId or None.")
    if not isinstance(required_component_kinds, tuple):
        raise TypeError("required_component_kinds must be a tuple.")
    if not required_component_kinds or any(
        not isinstance(kind, ComponentKind) for kind in required_component_kinds
    ):
        raise ValueError("required_component_kinds must contain component kinds.")
    return transaction.begin_corpus_update(
        corpus_id,
        proposed_corpus_version,
        expected_corpus_version=expected_corpus_version,
        required_component_kinds=required_component_kinds,
    )


def publish_baseline_generations(
    transaction: MetadataTransaction,
    reader: ChunkSetReader,
    participants: tuple[BaselineGenerationParticipant, ...],
    *,
    profiling: BaselineProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> BaselinePublicationOutcome:
    """Register, seal, and commit a complete set of component generations.

    The caller owns the transaction context. Storage conflicts and concurrency
    errors intentionally cross this boundary unchanged so their retry and
    reconciliation semantics remain available to orchestration.
    """
    if not isinstance(participants, tuple):
        raise TypeError("participants must be a tuple.")
    if not participants:
        raise ValueError("participants must not be empty.")
    if not callable(getattr(reader, "get_chunk_set_selection", None)):
        raise TypeError("reader must provide corpus-scoped chunk selections.")
    if profiling is not None and not isinstance(profiling, BaselineProfilingContext):
        raise TypeError("profiling must be a BaselineProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")

    prepared: list[
        tuple[ComponentGenerationReference, ComponentGenerationValidator]
    ] = []
    for participant in participants:
        generation = participant.component_generation()
        validator = participant.build_validator(reader)
        if not isinstance(generation, ComponentGenerationReference):
            raise TypeError(
                "participants must return ComponentGenerationReference values."
            )
        if not isinstance(validator, ComponentGenerationValidator):
            raise TypeError("participants must return generation validators.")
        prepared.append((generation, validator))

    first_generation = prepared[0][0]
    selected_profiling = (
        profiling if profiling is not None else BaselineProfilingContext.create()
    )
    published: BaselinePublicationOutcome | None = None
    with selected_profiling.observed_stage(
        stage="publication",
        stage_id=selected_profiling.stage_id("publication", profiling_identity),
        document_id=None,
        corpus_id=first_generation.corpus_id.as_str(),
        corpus_version=first_generation.corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
    ) as stage:
        stage.set_counts(items_processed=len(prepared), records_emitted=0)
        component_kinds = tuple(item[0].component_kind for item in prepared)
        if any(
            generation.corpus_id != first_generation.corpus_id
            or generation.corpus_version != first_generation.corpus_version
            for generation, _ in prepared
        ) or len(component_kinds) != len(set(component_kinds)):
            raise BaselineGenerationValidationError(
                BaselineFailureCategory.GENERATION_VALIDATION_FAILED,
            )

        for generation, validator in prepared:
            registered = transaction.register_corpus_update_component_generation(
                first_generation.corpus_id,
                first_generation.corpus_version,
                generation,
                validator=validator,
            )
            expected_ready = replace(generation, state=ComponentGenerationState.READY)
            if registered != expected_ready:
                raise BaselineGenerationValidationError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )

        sealed = transaction.seal_corpus_update(
            first_generation.corpus_id,
            first_generation.corpus_version,
        )
        if isinstance(sealed, NoCorpusChangeOutcome):
            published = sealed
        else:
            published = transaction.commit_corpus_update(
                first_generation.corpus_id,
                first_generation.corpus_version,
            )
            stage.set_counts(items_processed=len(prepared), records_emitted=1)
    assert published is not None
    return published


def publish_baseline_chunk_generation(
    transaction: MetadataTransaction,
    reader: ChunkSetReader,
    manifest: BaselineChunkGenerationManifest,
    *,
    profiling: BaselineProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> BaselineChunkPublicationOutcome:
    """Register, seal, and commit one complete baseline chunks generation."""
    if not isinstance(manifest, BaselineChunkGenerationManifest):
        raise TypeError("manifest must be a BaselineChunkGenerationManifest.")
    return publish_baseline_generations(
        transaction,
        reader,
        (manifest,),
        profiling=profiling,
        parent_stage_id=parent_stage_id,
        profiling_identity=profiling_identity,
    )
