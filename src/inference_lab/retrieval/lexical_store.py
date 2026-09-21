"""Immutable SQLite storage for one corpus-scoped lexical generation."""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
)
from inference_lab.profiling import FailureMapping, ProfilingContext
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    RecordId,
)
from inference_lab.retrieval.lexical import (
    LEXICAL_ENGINE_REVISION,
    LexicalError,
    LexicalFailureCategory,
    LexicalIndexBuildOutcome,
    LexicalIndexConfiguration,
    LexicalIndexManifest,
    derive_lexical_generation_id,
)
from inference_lab.retrieval.lexical_serialisation import serialise_lexical_manifest
from inference_lab.retrieval.lexical_store_builder import (
    _build_from_parent,
    _build_full_database,
    _build_manifest,
    _ensure_directory,
    _prepare_chunks,
    _reuse_existing_generation,
)
from inference_lab.retrieval.lexical_store_models import (
    LEXICAL_COMPONENT_KIND as _LEXICAL_COMPONENT_KIND,
)
from inference_lab.retrieval.lexical_store_models import (
    LEXICAL_DATABASE_NAME,
    LEXICAL_DIRECTORY_NAME,
    LEXICAL_MANIFEST_NAME,
    LexicalIndexedChunk,  # noqa: F401 - retained package re-export
    LexicalMemberSelection,
    LexicalSourceLink,  # noqa: F401 - retained package re-export
)
from inference_lab.retrieval.lexical_store_reader import (
    LexicalGeneration,  # noqa: F401 - retained package re-export
    _open_generation_path,
    open_lexical_generation,
)
from inference_lab.storage import (
    ChunkSetReader,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentGenerationValidator,
    MetadataRepository,
    StorageError,
)


def map_lexical_failure(error: BaseException) -> FailureMapping:
    """Map lexical build failures to the closed profiling vocabulary."""
    if isinstance(error, asyncio.CancelledError):
        return FailureMapping(
            failure_category="cancelled",
            failure_code="stage_cancelled",
            status="cancelled",
        )
    if isinstance(error, LexicalError):
        if error.category is LexicalFailureCategory.INVALID_REQUEST:
            failure_category = "input"
        else:
            failure_category = "dependency"
        if error.category is LexicalFailureCategory.CANCELLED:
            return FailureMapping(
                failure_category="cancelled",
                failure_code=error.category.value,
                status="cancelled",
            )
        return FailureMapping(
            failure_category=failure_category,
            failure_code=error.category.value,
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="stage_failed",
    )


def _component_generation_reference(
    manifest: LexicalIndexManifest,
) -> ComponentGenerationReference:
    """Return the storage reference represented by one lexical manifest."""
    if not isinstance(manifest, LexicalIndexManifest):
        raise TypeError("manifest must be a LexicalIndexManifest.")
    return ComponentGenerationReference(
        component_kind=_LEXICAL_COMPONENT_KIND,
        generation_id=manifest.generation_id,
        implementation=manifest.implementation,
        format_version=manifest.format_version,
        contract_version=manifest.contract_version,
        configuration_digest=manifest.configuration_digest,
        source_manifest_digest=manifest.source_manifest_digest,
        corpus_id=manifest.corpus_id,
        corpus_version=manifest.corpus_version,
        state=ComponentGenerationState.PREPARING,
    )


class LexicalGenerationParticipant:
    """Expose a prepared lexical generation to the generic publication seam."""

    def __init__(
        self,
        index_dir: Path,
        outcome: LexicalIndexBuildOutcome,
        source_manifest: BaselineChunkGenerationManifest,
        expected_parent_reference: ComponentGenerationReference | None = None,
    ) -> None:
        """Bind one built lexical generation to its exact staged source."""
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(outcome, LexicalIndexBuildOutcome):
            raise TypeError("outcome must be a LexicalIndexBuildOutcome.")
        if not isinstance(source_manifest, BaselineChunkGenerationManifest):
            raise TypeError(
                "source_manifest must be a BaselineChunkGenerationManifest."
            )
        if expected_parent_reference is not None and not isinstance(
            expected_parent_reference,
            ComponentGenerationReference,
        ):
            raise TypeError(
                "expected_parent_reference must be a ComponentGenerationReference "
                "or None."
            )
        if expected_parent_reference is not None and (
            expected_parent_reference.component_kind != _LEXICAL_COMPONENT_KIND
            or expected_parent_reference.state is not ComponentGenerationState.READY
            or expected_parent_reference.corpus_id != source_manifest.corpus_id
            or expected_parent_reference.corpus_version
            == source_manifest.corpus_version
        ):
            raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
        if (
            outcome.manifest.corpus_id != source_manifest.corpus_id
            or outcome.manifest.corpus_version != source_manifest.corpus_version
            or outcome.manifest.source_manifest_digest
            != source_manifest.source_manifest_digest
        ):
            raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
        self._index_dir = index_dir
        self._outcome = outcome
        self._source_manifest = source_manifest
        self._expected_parent_reference = expected_parent_reference

    def component_generation(self) -> ComponentGenerationReference:
        """Return the preparing reference for generic registration."""
        return _component_generation_reference(self._outcome.manifest)

    def build_validator(
        self,
        reader: ChunkSetReader,
    ) -> ComponentGenerationValidator:
        """Build a validator for this exact staged lexical generation."""
        return LexicalGenerationValidator(
            self._index_dir,
            self._outcome.manifest,
            self._source_manifest,
        )


def select_lexical_parent_reference(
    repository: MetadataRepository,
    corpus_id: CorpusId,
    expected_corpus_version: CorpusVersionId | None,
) -> ComponentGenerationReference | None:
    """Select the committed lexical parent from the expected active snapshot."""
    if not callable(getattr(repository, "open_active_corpus_snapshot", None)):
        raise TypeError("repository must provide active corpus snapshots.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if expected_corpus_version is not None and not isinstance(
        expected_corpus_version,
        CorpusVersionId,
    ):
        raise TypeError("expected_corpus_version must be a CorpusVersionId or None.")
    if expected_corpus_version is None:
        return None

    try:
        with repository.open_active_corpus_snapshot(corpus_id) as snapshot:
            if snapshot.version.corpus_version != expected_corpus_version:
                raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
            lexical_references = tuple(
                generation
                for generation in snapshot.manifest.component_generations
                if generation.component_kind == _LEXICAL_COMPONENT_KIND
            )
            if len(lexical_references) != 1:
                raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
            reference = lexical_references[0]
            if (
                reference.state is not ComponentGenerationState.READY
                or reference.corpus_id != corpus_id
                or reference.corpus_version != expected_corpus_version
            ):
                raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
            return reference
    except LexicalError:
        raise
    except StorageError:
        raise LexicalError(LexicalFailureCategory.GENERATION_UNAVAILABLE) from None
    except (OSError, TypeError, ValueError, KeyError, IndexError):
        raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None


def build_lexical_generation_for_corpus_update(
    index_dir: Path,
    source_manifest: BaselineChunkGenerationManifest,
    repository: MetadataRepository,
    *,
    expected_corpus_version: CorpusVersionId | None,
    configuration: LexicalIndexConfiguration | None = None,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> LexicalIndexBuildOutcome:
    """Build a lexical generation using only the expected committed parent."""
    if not isinstance(source_manifest, BaselineChunkGenerationManifest):
        raise TypeError("source_manifest must be a BaselineChunkGenerationManifest.")
    if expected_corpus_version == source_manifest.corpus_version:
        raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
    parent_reference = select_lexical_parent_reference(
        repository,
        source_manifest.corpus_id,
        expected_corpus_version,
    )
    return build_lexical_generation(
        index_dir,
        source_manifest,
        configuration=configuration,
        parent_generation_id=(
            None if parent_reference is None else parent_reference.generation_id
        ),
        profiling=profiling,
        parent_stage_id=parent_stage_id,
        profiling_identity=profiling_identity,
    )


class LexicalGenerationValidator:
    """Validate one lexical generation before component registration."""

    def __init__(
        self,
        index_dir: Path,
        manifest: LexicalIndexManifest,
        source_manifest: BaselineChunkGenerationManifest,
    ) -> None:
        """Bind one lexical manifest to its complete staged source manifest."""
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(manifest, LexicalIndexManifest):
            raise TypeError("manifest must be a LexicalIndexManifest.")
        if not isinstance(source_manifest, BaselineChunkGenerationManifest):
            raise TypeError(
                "source_manifest must be a BaselineChunkGenerationManifest."
            )
        self._index_dir = index_dir
        self._manifest = manifest
        self._source_manifest = source_manifest

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        """Return whether the durable generation is exact and complete."""
        if not isinstance(generation, ComponentGenerationReference):
            raise TypeError("generation must be a ComponentGenerationReference.")
        if not isinstance(document_memberships, tuple):
            raise TypeError("document_memberships must be a tuple.")

        try:
            if generation != _component_generation_reference(self._manifest):
                return False
            if (
                self._manifest.corpus_id != self._source_manifest.corpus_id
                or self._manifest.corpus_version != self._source_manifest.corpus_version
                or self._manifest.source_manifest_digest
                != self._source_manifest.source_manifest_digest
            ):
                return False

            expected_memberships = tuple(
                entry.membership for entry in self._source_manifest.entries
            )
            if document_memberships != expected_memberships:
                return False

            expected_chunks, expected_members = _prepare_chunks(self._source_manifest)
            expected_selections = tuple(
                LexicalMemberSelection(
                    document_id=member.document_id,
                    document_version_id=member.document_version_id,
                    selection_digest=member.selection_digest,
                    chunk_count=member.chunk_count,
                )
                for member in expected_members
            )
            expected_by_record_id = {
                chunk.record_id: chunk for chunk in expected_chunks
            }

            with open_lexical_generation(
                self._index_dir,
                generation.generation_id,
            ) as opened:
                if opened.manifest != self._manifest:
                    return False
                if opened.member_selections() != expected_selections:
                    return False

                actual_chunks = opened.iter_chunks()
                if len(actual_chunks) != len(expected_chunks):
                    return False
                seen_record_ids: set[RecordId] = set()
                for actual in actual_chunks:
                    if actual.record_id in seen_record_ids:
                        return False
                    seen_record_ids.add(actual.record_id)
                    expected = expected_by_record_id.get(actual.record_id)
                    if expected is None:
                        return False
                    if (
                        actual.document_id != expected.document_id
                        or actual.document_version_id != expected.document_version_id
                        or actual.ordinal != expected.ordinal
                        or actual.content != expected.content
                        or actual.token_count != expected.token_count
                        or actual.source_links != expected.source_links
                    ):
                        return False
                return seen_record_ids == set(expected_by_record_id)
        except (
            LexicalError,
            OSError,
            sqlite3.Error,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
        ):
            return False


def _build_lexical_generation(
    index_dir: Path,
    source_manifest: BaselineChunkGenerationManifest,
    *,
    configuration: LexicalIndexConfiguration | None = None,
    parent_generation_id: str | None = None,
) -> LexicalIndexBuildOutcome:
    """Build or exactly reuse one immutable full lexical generation."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(source_manifest, BaselineChunkGenerationManifest):
        raise TypeError("source_manifest must be a BaselineChunkGenerationManifest.")
    if configuration is None:
        configuration = LexicalIndexConfiguration()
    if not isinstance(configuration, LexicalIndexConfiguration):
        raise TypeError("configuration must be a LexicalIndexConfiguration.")
    if configuration.engine_revision != LEXICAL_ENGINE_REVISION:
        raise LexicalError(LexicalFailureCategory.INVALID_REQUEST)
    if parent_generation_id is not None and not isinstance(parent_generation_id, str):
        raise TypeError("parent_generation_id must be a string or None.")

    generation_id = derive_lexical_generation_id(
        source_manifest.corpus_id,
        source_manifest.corpus_version,
        configuration,
        source_manifest.source_manifest_digest,
    )
    lexical_dir = index_dir / LEXICAL_DIRECTORY_NAME
    target_dir = lexical_dir / generation_id
    if target_dir.exists():
        return _reuse_existing_generation(
            target_dir,
            source_manifest=source_manifest,
            configuration=configuration,
            generation_id=generation_id,
        )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".lexical-staging-", dir=str(_ensure_directory(lexical_dir))
        )
    )
    try:
        database_path = staging_dir / LEXICAL_DATABASE_NAME
        if parent_generation_id is None:
            statistics = _build_full_database(
                database_path,
                source_manifest=source_manifest,
                configuration=configuration,
            )
        else:
            statistics = _build_from_parent(
                database_path,
                index_dir=index_dir,
                parent_generation_id=parent_generation_id,
                source_manifest=source_manifest,
                configuration=configuration,
            )
        manifest = _build_manifest(
            database_path,
            source_manifest=source_manifest,
            configuration=configuration,
            generation_id=generation_id,
        )
        (staging_dir / LEXICAL_MANIFEST_NAME).write_bytes(
            serialise_lexical_manifest(manifest)
        )
        staged_generation = _open_generation_path(
            staging_dir,
            expected_generation_id=generation_id,
        )
        staged_generation.close()
        try:
            os.rename(staging_dir, target_dir)
        except FileExistsError:
            return _reuse_existing_generation(
                target_dir,
                source_manifest=source_manifest,
                configuration=configuration,
                generation_id=generation_id,
            )
        return LexicalIndexBuildOutcome(
            manifest=manifest,
            statistics=statistics,
        )
    except LexicalError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def build_lexical_generation(
    index_dir: Path,
    source_manifest: BaselineChunkGenerationManifest,
    *,
    configuration: LexicalIndexConfiguration | None = None,
    parent_generation_id: str | None = None,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> LexicalIndexBuildOutcome:
    """Build or reuse one lexical generation with lifecycle profiling."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(source_manifest, BaselineChunkGenerationManifest):
        raise TypeError("source_manifest must be a BaselineChunkGenerationManifest.")
    if configuration is None:
        configuration = LexicalIndexConfiguration()
    if not isinstance(configuration, LexicalIndexConfiguration):
        raise TypeError("configuration must be a LexicalIndexConfiguration.")
    if parent_generation_id is not None and not isinstance(parent_generation_id, str):
        raise TypeError("parent_generation_id must be a string or None.")
    if profiling is not None and not isinstance(profiling, ProfilingContext):
        raise TypeError("profiling must be a ProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")

    selected_profiling = (
        profiling if profiling is not None else ProfilingContext.create()
    )
    stage_id = selected_profiling.stage_id("lexical_index", profiling_identity)
    outcome: LexicalIndexBuildOutcome | None = None
    with selected_profiling.observed_stage(
        stage="lexical_index",
        stage_id=stage_id,
        document_id=None,
        corpus_id=source_manifest.corpus_id.as_str(),
        corpus_version=source_manifest.corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
        module="inference_lab.retrieval",
        failure_mapper=map_lexical_failure,
    ) as stage:
        stage.set_counts(items_processed=0, records_emitted=0)
        outcome = _build_lexical_generation(
            index_dir,
            source_manifest,
            configuration=configuration,
            parent_generation_id=parent_generation_id,
        )
        stage.set_counts(
            items_processed=outcome.statistics.chunks_processed,
            records_emitted=outcome.manifest.chunk_count,
        )
    assert outcome is not None
    return outcome
