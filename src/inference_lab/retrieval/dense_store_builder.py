"""SQLite build and publication mechanics for immutable dense generations."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import structlog

from inference_lab._fs import (
    fsync_directory,
    rename_directory_without_replacement,
)
from inference_lab._json import (
    JsonObject,
    JsonValue,
    canonical_json_bytes,
)
from inference_lab._validation import paths_overlap
from inference_lab.backends.runtime import (
    EmbeddingBackend,
    EmbeddingOptions,
    EmbeddingRequest,
)
from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
    deserialise_chunk_set_selection,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
)
from inference_lab.retrieval.dense import (
    DENSE_CONTRACT_VERSION,
    DENSE_FORMAT_VERSION,
    DENSE_IMPLEMENTATION,
    DenseError,
    DenseFailureCategory,
    DenseIndexBuildOutcome,
    DenseIndexBuildStatistics,
    DenseIndexCompatibility,
    DenseIndexEntry,
    DenseIndexManifest,
    DenseMemberSelection,
    DenseVector,
    InvalidDenseContractError,
    l2_normalised,
)
from inference_lab.retrieval.dense_serialisation import _manifest_json
from inference_lab.retrieval.dense_store_models import (
    DENSE_DATABASE_NAME,
    DENSE_DATABASE_USER_VERSION,
    DENSE_DIRECTORY_NAME,
    DENSE_MANIFEST_NAME,
)
from inference_lab.retrieval.dense_store_reader import (
    DenseGeneration,
    _open_dense_path,
    _sha256_file,
    open_dense_generation,
)
from inference_lab.retrieval.evidence import (
    CandidateKind,
    EvidenceSource,
    RetrievalCandidate,
)
from inference_lab.storage import ComponentGenerationReference

_LOGGER = structlog.get_logger("inference_lab.retrieval")
_EMBED_BATCH_SIZE = 32


# Embedding a corpus is the longest single step in a build and writes nothing
# until it finishes, so it reports progress at this cadence. The interval
# bounds the record count for a large corpus while still showing movement.
_EMBED_PROGRESS_CHUNKS: Final = 250


@dataclass(frozen=True, slots=True)
class _PreparedChunk:
    candidate: RetrievalCandidate
    document_id: DocumentId
    document_version_id: DocumentVersionId
    ordinal: int


def _resolve_local_paths(
    index_dir: Path,
    metadata_db_path: Path,
    source_dir: Path,
) -> Path:
    paths = (index_dir, metadata_db_path, source_dir)
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("dense storage paths must be Path values.")
    try:
        resolved = tuple(path.expanduser().resolve(strict=False) for path in paths)
    except (OSError, RuntimeError):
        raise DenseError(DenseFailureCategory.INVALID_REQUEST) from None
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if paths_overlap(left, right):
                raise DenseError(DenseFailureCategory.INVALID_REQUEST)
    return resolved[0]


def _prepare_source(
    source_manifest: BaselineChunkGenerationManifest,
) -> tuple[tuple[_PreparedChunk, ...], tuple[DenseMemberSelection, ...]]:
    prepared: list[_PreparedChunk] = []
    members: list[DenseMemberSelection] = []
    try:
        for entry in source_manifest.entries:
            chunk_set = deserialise_chunk_set_selection(
                entry.selection.canonical_payload,
                corpus_id=source_manifest.corpus_id,
            )
            if (
                chunk_set.document_id != entry.membership.document_id
                or chunk_set.document_version_id != entry.membership.document_version_id
                or chunk_set.configuration_identity
                != source_manifest.configuration_identity
            ):
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
            members.append(
                DenseMemberSelection(
                    document_id=entry.membership.document_id,
                    document_version_id=entry.membership.document_version_id,
                    selection_digest=entry.selection.payload_digest.digest,
                    chunk_count=len(chunk_set.chunks),
                )
            )
            for chunk in chunk_set.chunks:
                sources = tuple(
                    EvidenceSource(
                        document_id=chunk.document_id,
                        document_version_id=reference.document_version_id,
                        source_span_id=reference.source_span_id,
                    )
                    for reference in chunk.source_references
                )
                prepared.append(
                    _PreparedChunk(
                        candidate=RetrievalCandidate(
                            record_id=chunk.record_id,
                            kind=CandidateKind.ORIGINAL_PASSAGE,
                            content=chunk.text,
                            sources=sources,
                        ),
                        document_id=chunk.document_id,
                        document_version_id=chunk.document_version_id,
                        ordinal=chunk.ordinal,
                    )
                )
    except DenseError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None
    return tuple(prepared), tuple(members)


def _derive_generation_id(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    source_manifest_digest: str,
    compatibility: DenseIndexCompatibility,
) -> str:
    identity: JsonObject = {
        "compatibility_digest": compatibility.configuration_digest,
        "corpus_id": corpus_id.as_str(),
        "corpus_version": corpus_version.as_str(),
        "source_manifest_digest": source_manifest_digest,
    }
    return "dense-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _write_generation_database(
    path: Path,
    *,
    compatibility: DenseIndexCompatibility,
    members: tuple[DenseMemberSelection, ...],
    entries: tuple[DenseIndexEntry, ...],
    prepared: tuple[_PreparedChunk, ...],
) -> None:
    prepared_by_id = {item.candidate.record_id: item for item in prepared}
    if set(prepared_by_id) != {entry.candidate.record_id for entry in entries}:
        raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
    try:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE members (
                    document_id TEXT PRIMARY KEY NOT NULL,
                    document_version_id TEXT NOT NULL,
                    selection_digest TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0)
                ) WITHOUT ROWID;
                CREATE TABLE entries (
                    record_id TEXT PRIMARY KEY NOT NULL,
                    document_id TEXT NOT NULL REFERENCES members(document_id),
                    document_version_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                    content TEXT NOT NULL,
                    source_links BLOB NOT NULL,
                    vector BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX entries_by_document
                    ON entries(document_id, ordinal, record_id);
                """
            )
            with connection:
                connection.execute(
                    f"PRAGMA user_version = {DENSE_DATABASE_USER_VERSION}"
                )
                connection.executemany(
                    "INSERT INTO members VALUES (?, ?, ?, ?)",
                    (
                        (
                            member.document_id.as_str(),
                            member.document_version_id.as_str(),
                            member.selection_digest,
                            member.chunk_count,
                        )
                        for member in members
                    ),
                )
                for entry in entries:
                    item = prepared_by_id[entry.candidate.record_id]
                    source_payload: list[JsonValue] = [
                        cast(
                            JsonObject,
                            {
                                "document_id": source.document_id.as_str(),
                                "document_version_id": (
                                    source.document_version_id.as_str()
                                ),
                                "source_span_id": source.source_span_id.as_str(),
                            },
                        )
                        for source in entry.candidate.sources
                    ]
                    vector = struct.pack(
                        f"<{compatibility.dimensions}f",
                        *l2_normalised(entry.vector).values,
                    )
                    connection.execute(
                        "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            entry.candidate.record_id.as_str(),
                            item.document_id.as_str(),
                            item.document_version_id.as_str(),
                            item.ordinal,
                            entry.candidate.content,
                            canonical_json_bytes(
                                cast(JsonObject, {"sources": source_payload})
                            ),
                            vector,
                        ),
                    )
            connection.execute("PRAGMA optimize")
        finally:
            connection.close()
    except (sqlite3.Error, struct.error, OverflowError):
        raise DenseError(DenseFailureCategory.EXECUTION_FAILURE) from None


def _log_embedding_progress(
    source_manifest: BaselineChunkGenerationManifest,
    *,
    embedded: int,
    pending: int,
    reused: int,
) -> None:
    """Report embedding progress with counts only and no candidate content."""
    _LOGGER.info(
        "dense_index_embedding_progress",
        module="inference_lab.retrieval",
        pipeline_stage="dense_index",
        document_id=None,
        corpus_id=source_manifest.corpus_id.as_str(),
        corpus_version=source_manifest.corpus_version.as_str(),
        embedded_count=embedded,
        pending_count=pending,
        reused_count=reused,
    )


def _embed_pending(
    embedder: EmbeddingBackend,
    pending: tuple[_PreparedChunk, ...],
    compatibility: DenseIndexCompatibility,
    *,
    batch_size: int,
    source_manifest: BaselineChunkGenerationManifest | None = None,
    reused: int = 0,
) -> dict[RecordId, DenseVector]:
    vectors: dict[RecordId, DenseVector] = {}
    reported = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            response = embedder.embed(
                EmbeddingRequest(
                    input_texts=tuple(item.candidate.content for item in batch),
                    options=EmbeddingOptions(
                        dimensions=compatibility.dimensions,
                        normalise=compatibility.embedding_profile.l2_normalised,
                    ),
                    purpose="document",
                )
            )
        except Exception:
            raise DenseError(DenseFailureCategory.EMBEDDING_FAILURE) from None
        if (
            response.model_metadata != compatibility.model_metadata
            or response.embedding_profile != compatibility.embedding_profile
            or len(response.vectors) != len(batch)
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        for item, values in zip(batch, response.vectors, strict=True):
            try:
                vector = DenseVector(tuple(values))
            except (TypeError, ValueError, InvalidDenseContractError):
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION) from None
            if len(vector.values) != compatibility.dimensions:
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
            vectors[item.candidate.record_id] = vector
        if (
            source_manifest is not None
            and len(vectors) - reported >= _EMBED_PROGRESS_CHUNKS
        ):
            reported = len(vectors)
            _log_embedding_progress(
                source_manifest,
                embedded=reported,
                pending=len(pending),
                reused=reused,
            )
    if source_manifest is not None and pending and reported != len(vectors):
        _log_embedding_progress(
            source_manifest,
            embedded=len(vectors),
            pending=len(pending),
            reused=reused,
        )
    return vectors


@dataclass(frozen=True, slots=True)
class DenseBuildRequest:
    """Validated inputs describing one dense generation build."""

    source_manifest: BaselineChunkGenerationManifest
    compatibility: DenseIndexCompatibility
    metadata_db_path: Path
    source_dir: Path
    parent_reference: ComponentGenerationReference | None = None
    batch_size: int = _EMBED_BATCH_SIZE

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the request can produce a compatible generation."""
        if not isinstance(self.source_manifest, BaselineChunkGenerationManifest):
            raise TypeError(
                "source_manifest must be a BaselineChunkGenerationManifest."
            )
        if not isinstance(self.compatibility, DenseIndexCompatibility):
            raise TypeError("compatibility must be a DenseIndexCompatibility.")
        for name in ("metadata_db_path", "source_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path.")
        if type(self.batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if self.batch_size <= 0:
            raise DenseError(DenseFailureCategory.INVALID_REQUEST)
        configuration = self.compatibility.configuration
        if (
            configuration.implementation != DENSE_IMPLEMENTATION
            or configuration.format_version != DENSE_FORMAT_VERSION
            or configuration.contract_version != DENSE_CONTRACT_VERSION
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if self.parent_reference is not None and not isinstance(
            self.parent_reference, ComponentGenerationReference
        ):
            raise TypeError(
                "parent_reference must be a ComponentGenerationReference or None."
            )


@dataclass(frozen=True, slots=True)
class _GenerationTarget:
    """Resolved durable locations and derived identity for one build."""

    resolved_index_dir: Path
    dense_root: Path
    target: Path
    generation_id: str
    prepared: tuple[_PreparedChunk, ...]
    members: tuple[DenseMemberSelection, ...]


def _resolve_generation_target(
    index_dir: Path,
    request: DenseBuildRequest,
) -> _GenerationTarget:
    """Resolve where this build writes and what identity it will carry."""
    source_manifest = request.source_manifest
    resolved_index_dir = _resolve_local_paths(
        index_dir,
        request.metadata_db_path,
        request.source_dir,
    )
    prepared, members = _prepare_source(source_manifest)
    generation_id = _derive_generation_id(
        source_manifest.corpus_id,
        source_manifest.corpus_version,
        source_manifest.source_manifest_digest,
        request.compatibility,
    )
    dense_root = resolved_index_dir / DENSE_DIRECTORY_NAME
    if dense_root.exists() and dense_root.is_symlink():
        raise DenseError(DenseFailureCategory.INVALID_REQUEST)
    return _GenerationTarget(
        resolved_index_dir=resolved_index_dir,
        dense_root=dense_root,
        target=dense_root / generation_id,
        generation_id=generation_id,
        prepared=prepared,
        members=members,
    )


def _reuse_published_generation(
    location: _GenerationTarget,
    request: DenseBuildRequest,
) -> DenseIndexBuildOutcome | None:
    """Return the already-published generation when this build would repeat it."""
    if not location.target.exists():
        return None
    if location.target.is_symlink():
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
    with open_dense_generation(
        location.resolved_index_dir, location.generation_id
    ) as generation:
        if not _generation_matches_selection(
            generation.manifest,
            request.source_manifest,
            request.compatibility,
            location.prepared,
            location.members,
        ):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        return DenseIndexBuildOutcome(
            manifest=generation.manifest,
            statistics=DenseIndexBuildStatistics(
                members_processed=len(location.members),
                chunks_processed=len(location.prepared),
                reused_vectors=len(location.prepared),
                embedded_vectors=0,
                replaced_vectors=0,
                removed_vectors=0,
                generation_reused=True,
            ),
        )


def _open_validated_parent(
    location: _GenerationTarget,
    request: DenseBuildRequest,
) -> DenseGeneration | None:
    """Open the declared parent, rejecting one that is not the named identity."""
    parent_reference = request.parent_reference
    if parent_reference is None:
        return None
    parent: DenseGeneration | None = None
    try:
        parent = open_dense_generation(
            location.resolved_index_dir,
            parent_reference.generation_id,
        )
    except DenseError as error:
        if error.category not in {
            DenseFailureCategory.CORRUPT_GENERATION,
            DenseFailureCategory.GENERATION_UNAVAILABLE,
        }:
            raise
    if parent is not None and (
        parent.manifest.corpus_id != request.source_manifest.corpus_id
        or parent.manifest.corpus_version != parent_reference.corpus_version
        or parent.manifest.compatibility.configuration.implementation
        != parent_reference.implementation
        or parent.manifest.configuration_digest != parent_reference.configuration_digest
    ):
        parent.close()
        raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
    return parent


def _resolve_reusable_vectors(
    parent: DenseGeneration | None,
    prepared: tuple[_PreparedChunk, ...],
    compatibility: DenseIndexCompatibility,
) -> dict[RecordId, DenseIndexEntry]:
    """Return the parent vectors whose candidate is byte-for-byte unchanged."""
    if (
        parent is None
        or parent.manifest.configuration_digest != compatibility.configuration_digest
    ):
        return {}
    proposed_by_id = {item.candidate.record_id: item for item in prepared}
    reusable: dict[RecordId, DenseIndexEntry] = {}
    for old_entry in parent.read_entries():
        new_item = proposed_by_id.get(old_entry.candidate.record_id)
        if new_item is not None and new_item.candidate == old_entry.candidate:
            reusable[old_entry.candidate.record_id] = old_entry
    return reusable


def _publish_generation(
    location: _GenerationTarget,
    request: DenseBuildRequest,
    entries: tuple[DenseIndexEntry, ...],
) -> DenseIndexManifest:
    """Write, verify, and atomically publish one immutable generation."""
    location.dense_root.mkdir(parents=True, exist_ok=True)
    fsync_directory(location.resolved_index_dir)
    staging = Path(tempfile.mkdtemp(prefix=".dense-staging-", dir=location.dense_root))
    try:
        database_path = staging / DENSE_DATABASE_NAME
        _write_generation_database(
            database_path,
            compatibility=request.compatibility,
            members=location.members,
            entries=entries,
            prepared=location.prepared,
        )
        manifest = DenseIndexManifest(
            generation_id=location.generation_id,
            corpus_id=request.source_manifest.corpus_id,
            corpus_version=request.source_manifest.corpus_version,
            source_manifest_digest=request.source_manifest.source_manifest_digest,
            compatibility=request.compatibility,
            members=location.members,
            entry_count=len(entries),
            database_digest=_sha256_file(database_path),
            library_version=sqlite3.sqlite_version,
        )
        _write_durable(staging / DENSE_MANIFEST_NAME, _manifest_json(manifest))
        with database_path.open("rb") as database_stream:
            os.fsync(database_stream.fileno())
        fsync_directory(staging)
        with _open_dense_path(staging, location.generation_id) as opened:
            if opened.manifest != manifest:
                raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
            if not _generation_matches_selection(
                opened.manifest,
                request.source_manifest,
                request.compatibility,
                location.prepared,
                location.members,
            ) or not _generation_contents_match(
                opened, location.prepared, request.compatibility
            ):
                raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        try:
            rename_directory_without_replacement(staging, location.target)
        except FileExistsError:
            with open_dense_generation(
                location.resolved_index_dir, location.generation_id
            ) as opened:
                if opened.manifest != manifest:
                    raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        fsync_directory(location.dense_root)
        return manifest
    finally:
        if staging.exists():
            _remove_staging_directory(staging)


def _build_statistics(
    location: _GenerationTarget,
    parent: DenseGeneration | None,
    reusable: dict[RecordId, DenseIndexEntry],
    embedded_count: int,
) -> DenseIndexBuildStatistics:
    """Account for the vectors this build reused, replaced, and dropped."""
    old_entries = () if parent is None else parent.read_entries()
    replaced_ids: set[RecordId] = set()
    removed_document_ids: set[DocumentId] = set()
    if parent is not None:
        old_by_document = {
            member.document_id: member.document_version_id
            for member in parent.manifest.members
        }
        new_by_document = {
            member.document_id: member.document_version_id
            for member in location.members
        }
        replaced_documents = {
            document_id
            for document_id, version in old_by_document.items()
            if document_id in new_by_document
            and new_by_document[document_id] != version
        }
        removed_document_ids = set(old_by_document) - set(new_by_document)
        replaced_ids = {
            entry.candidate.record_id
            for entry in old_entries
            if entry.candidate.sources[0].document_id in replaced_documents
        }
    return DenseIndexBuildStatistics(
        members_processed=len(location.members),
        chunks_processed=len(location.prepared),
        reused_vectors=len(reusable),
        embedded_vectors=embedded_count,
        replaced_vectors=len(replaced_ids),
        removed_vectors=sum(
            1
            for entry in old_entries
            if entry.candidate.sources[0].document_id in removed_document_ids
        ),
    )


def _write_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_staging_directory(path: Path) -> None:
    """Best-effort cleanup that cannot replace the primary build outcome."""
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def _generation_matches_source(
    manifest: DenseIndexManifest,
    source_manifest: BaselineChunkGenerationManifest,
    compatibility: DenseIndexCompatibility,
) -> bool:
    """Compare recorded identities without re-deriving the source selections."""
    return (
        manifest.corpus_id == source_manifest.corpus_id
        and manifest.corpus_version == source_manifest.corpus_version
        and manifest.source_manifest_digest == source_manifest.source_manifest_digest
        and manifest.configuration_digest == compatibility.configuration_digest
    )


def _generation_matches_selection(
    manifest: DenseIndexManifest,
    source_manifest: BaselineChunkGenerationManifest,
    compatibility: DenseIndexCompatibility,
    prepared: tuple[_PreparedChunk, ...],
    members: tuple[DenseMemberSelection, ...],
) -> bool:
    """Compare identities and the already-derived membership of one build."""
    return (
        _generation_matches_source(manifest, source_manifest, compatibility)
        and manifest.members == members
        and manifest.entry_count == len(prepared)
    )


def _generation_contents_match(
    generation: DenseGeneration,
    prepared: tuple[_PreparedChunk, ...],
    compatibility: DenseIndexCompatibility,
) -> bool:
    """Verify every written candidate, used where a fresh write is proved."""
    actual = {entry.candidate.record_id: entry for entry in generation.read_entries()}
    return all(
        (entry := actual.get(item.candidate.record_id)) is not None
        and entry.candidate == item.candidate
        and len(entry.vector.values) == compatibility.dimensions
        for item in prepared
    ) and len(actual) == len(prepared)
