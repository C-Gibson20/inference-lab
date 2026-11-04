"""Immutable SQLite generations for corpus-scoped dense retrieval."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path

from inference_lab.backends.runtime import EmbeddingBackend
from inference_lab.ingest import BaselineChunkGenerationManifest
from inference_lab.profiling import FailureMapping, ProfilingContext
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
)
from inference_lab.retrieval.dense import DENSE_COMPONENT_KIND, DenseIndexEntry
from inference_lab.retrieval.dense_store_builder import (
    DenseBuildRequest,
    _build_statistics,
    _embed_pending,
    _generation_matches_source,
    _open_validated_parent,
    _publish_generation,
    _resolve_generation_target,
    _resolve_reusable_vectors,
    _reuse_published_generation,
)
from inference_lab.retrieval.dense_store_models import (
    DenseError,
    DenseFailureCategory,
    DenseIndexBuildOutcome,
    DenseIndexBuildStatistics,  # noqa: F401 - retained package re-export
    DenseIndexManifest,
    DenseMemberSelection,  # noqa: F401 - retained package re-export
)
from inference_lab.retrieval.dense_store_reader import (
    DenseGeneration,  # noqa: F401 - retained package re-export
    open_dense_generation,
)
from inference_lab.storage import (
    ChunkSetReader,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentGenerationValidator,
    MetadataRepository,
    StorageError,
    select_ready_component_generation,
)


def map_dense_build_failure(error: BaseException) -> FailureMapping:
    """Map dense build failures to the closed profiling vocabulary."""
    if isinstance(error, asyncio.CancelledError):
        return FailureMapping(
            failure_category="cancelled",
            failure_code="stage_cancelled",
            status="cancelled",
        )
    if isinstance(error, DenseError):
        return FailureMapping(
            failure_category=(
                "input"
                if error.category is DenseFailureCategory.INVALID_REQUEST
                else "dependency"
            ),
            failure_code=error.category.value,
        )
    if isinstance(error, StorageError):
        return FailureMapping(
            failure_category="dependency",
            failure_code=error.category.value,
        )
    return FailureMapping(
        failure_category="internal",
        failure_code="stage_failed",
    )


def build_dense_generation(
    index_dir: Path,
    request: DenseBuildRequest,
    embedder: EmbeddingBackend,
    *,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> DenseIndexBuildOutcome:
    """Build or reuse one dense generation with lifecycle profiling."""
    if not isinstance(request, DenseBuildRequest):
        raise TypeError("request must be a DenseBuildRequest.")
    if profiling is not None and not isinstance(profiling, ProfilingContext):
        raise TypeError("profiling must be a ProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if not isinstance(profiling_identity, str) or not profiling_identity:
        raise ValueError("profiling_identity must be a non-empty string.")

    selected_profiling = (
        profiling if profiling is not None else ProfilingContext.create()
    )
    stage_id = selected_profiling.stage_id("dense_index", profiling_identity)
    outcome: DenseIndexBuildOutcome | None = None
    with selected_profiling.observed_stage(
        stage="dense_index",
        stage_id=stage_id,
        document_id=None,
        corpus_id=request.source_manifest.corpus_id.as_str(),
        corpus_version=request.source_manifest.corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
        module="inference_lab.retrieval",
        failure_mapper=map_dense_build_failure,
    ) as stage:
        stage.set_counts(items_processed=0, records_emitted=0)
        outcome = _build_dense_generation(index_dir, request, embedder)
        stage.set_counts(
            items_processed=outcome.statistics.chunks_processed,
            records_emitted=outcome.manifest.entry_count,
        )
    return outcome


def _build_dense_generation(
    index_dir: Path,
    request: DenseBuildRequest,
    embedder: EmbeddingBackend,
) -> DenseIndexBuildOutcome:
    """Build a complete immutable generation from staged canonical chunks."""
    if not callable(getattr(embedder, "embed", None)):
        raise TypeError("embedder must implement EmbeddingBackend.")
    parent_reference = request.parent_reference
    if parent_reference is not None and (
        parent_reference.component_kind != DENSE_COMPONENT_KIND
        or parent_reference.state is not ComponentGenerationState.READY
        or parent_reference.corpus_id != request.source_manifest.corpus_id
        or parent_reference.corpus_version == request.source_manifest.corpus_version
    ):
        raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)

    location = _resolve_generation_target(index_dir, request)
    published = _reuse_published_generation(location, request)
    if published is not None:
        return published

    parent = _open_validated_parent(location, request)
    try:
        reusable = _resolve_reusable_vectors(
            parent, location.prepared, request.compatibility
        )
        embedded = _embed_pending(
            embedder,
            tuple(
                item
                for item in location.prepared
                if item.candidate.record_id not in reusable
            ),
            request.compatibility,
            batch_size=request.batch_size,
            source_manifest=request.source_manifest,
            reused=len(reusable),
        )
        entries = tuple(
            reusable[item.candidate.record_id]
            if item.candidate.record_id in reusable
            else DenseIndexEntry(item.candidate, embedded[item.candidate.record_id])
            for item in location.prepared
        )
        manifest = _publish_generation(location, request, entries)
        return DenseIndexBuildOutcome(
            manifest=manifest,
            statistics=_build_statistics(location, parent, reusable, len(embedded)),
        )
    except DenseError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, OverflowError):
        raise DenseError(DenseFailureCategory.EXECUTION_FAILURE) from None
    finally:
        if parent is not None:
            parent.close()


def select_dense_parent_reference(
    repository: MetadataRepository,
    corpus_id: CorpusId,
    expected_corpus_version: CorpusVersionId | None,
) -> ComponentGenerationReference | None:
    """Select the dense parent only from the exact expected active snapshot."""
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
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
            reference = select_ready_component_generation(
                snapshot.manifest,
                DENSE_COMPONENT_KIND,
                corpus_id=corpus_id,
                corpus_version=expected_corpus_version,
            )
            if reference is None:
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
            return reference
    except DenseError:
        raise
    except StorageError:
        raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE) from None
    except (OSError, TypeError, ValueError, KeyError, IndexError):
        raise DenseError(DenseFailureCategory.EXECUTION_FAILURE) from None


def build_dense_generation_for_corpus_update(
    index_dir: Path,
    request: DenseBuildRequest,
    embedder: EmbeddingBackend,
    repository: MetadataRepository,
    *,
    expected_corpus_version: CorpusVersionId | None,
    profiling: ProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str = "request",
) -> DenseIndexBuildOutcome:
    """Build against the exact expected active parent, if one exists."""
    if not isinstance(request, DenseBuildRequest):
        raise TypeError("request must be a DenseBuildRequest.")
    source_manifest = request.source_manifest
    if expected_corpus_version == source_manifest.corpus_version:
        raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
    parent = select_dense_parent_reference(
        repository,
        source_manifest.corpus_id,
        expected_corpus_version,
    )
    return build_dense_generation(
        index_dir,
        replace(request, parent_reference=parent),
        embedder,
        profiling=profiling,
        parent_stage_id=parent_stage_id,
        profiling_identity=profiling_identity,
    )


class DenseGenerationParticipant:
    """Expose a prepared dense generation to the storage publication seam."""

    def __init__(
        self,
        index_dir: Path,
        outcome: DenseIndexBuildOutcome,
        source_manifest: BaselineChunkGenerationManifest,
    ) -> None:
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(outcome, DenseIndexBuildOutcome):
            raise TypeError("outcome must be a DenseIndexBuildOutcome.")
        if not isinstance(source_manifest, BaselineChunkGenerationManifest):
            raise TypeError(
                "source_manifest must be a BaselineChunkGenerationManifest."
            )
        if (
            outcome.manifest.corpus_id != source_manifest.corpus_id
            or outcome.manifest.corpus_version != source_manifest.corpus_version
            or outcome.manifest.source_manifest_digest
            != source_manifest.source_manifest_digest
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        self._index_dir = index_dir
        self._outcome = outcome
        self._source_manifest = source_manifest

    def component_generation(self) -> ComponentGenerationReference:
        """Return the preparing reference for corpus-version registration."""
        return self._outcome.manifest.component_generation()

    def build_validator(
        self,
        reader: ChunkSetReader,
    ) -> ComponentGenerationValidator:
        """Bind the storage reader to this exact staged dense generation."""
        return DenseGenerationValidator(
            self._index_dir,
            self._outcome.manifest,
            self._source_manifest,
            reader,
        )


class DenseGenerationValidator:
    """Validate completeness and durable source ownership before registration."""

    def __init__(
        self,
        index_dir: Path,
        manifest: DenseIndexManifest,
        source_manifest: BaselineChunkGenerationManifest,
        reader: ChunkSetReader,
    ) -> None:
        if not isinstance(index_dir, Path):
            raise TypeError("index_dir must be a Path.")
        if not isinstance(manifest, DenseIndexManifest):
            raise TypeError("manifest must be a DenseIndexManifest.")
        if not isinstance(source_manifest, BaselineChunkGenerationManifest):
            raise TypeError(
                "source_manifest must be a BaselineChunkGenerationManifest."
            )
        if not callable(getattr(reader, "get_chunk_set_selection", None)):
            raise TypeError("reader must implement ChunkSetReader.")
        self._index_dir = index_dir
        self._manifest = manifest
        self._source_manifest = source_manifest
        self._reader = reader

    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        """Return whether the durable generation exactly covers staged inputs."""
        if not isinstance(generation, ComponentGenerationReference):
            raise TypeError("generation must be a ComponentGenerationReference.")
        if not isinstance(document_memberships, tuple):
            raise TypeError("document_memberships must be a tuple.")
        try:
            if generation != self._manifest.component_generation():
                return False
            expected_memberships = tuple(
                entry.membership for entry in self._source_manifest.entries
            )
            if document_memberships != expected_memberships:
                return False
            for source_entry in self._source_manifest.entries:
                selected = self._reader.get_chunk_set_selection(
                    self._source_manifest.corpus_id,
                    source_entry.membership.document_version_id,
                    self._source_manifest.configuration_identity,
                )
                if selected != source_entry.selection:
                    return False
            with open_dense_generation(
                self._index_dir,
                self._manifest.generation_id,
            ) as opened:
                if opened.manifest != self._manifest:
                    return False
                return _generation_matches_source(
                    opened.manifest,
                    self._source_manifest,
                    self._manifest.compatibility,
                )
        except DenseError as error:
            # An availability failure is not a mismatch verdict, so it must
            # reach the adapter rather than being reported as a conflict.
            if error.category is DenseFailureCategory.GENERATION_UNAVAILABLE:
                raise
            return False
        except (OSError, TypeError, ValueError):
            return False
