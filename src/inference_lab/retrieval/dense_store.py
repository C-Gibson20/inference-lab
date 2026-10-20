"""Immutable SQLite generations for corpus-scoped dense retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass, replace
from enum import StrEnum
from operator import mul
from pathlib import Path
from typing import Final, Iterator, Self, cast
from urllib.parse import quote

import structlog

from inference_lab._fs import (
    fsync_directory,
    rename_directory_without_replacement,
)
from inference_lab._json import (
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_boolean,
    require_json_integer,
    require_json_string,
    require_optional_json_integer,
    require_optional_json_string,
)
from inference_lab._validation import is_path_free_identifier, paths_overlap
from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingBackend,
    EmbeddingOptions,
    EmbeddingProfile,
    EmbeddingRequest,
)
from inference_lab.backends.runtime.types import EmbeddingOverLimitPolicy
from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
    deserialise_chunk_set_selection,
)
from inference_lab.profiling import FailureMapping, ProfilingContext
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.dense import (
    DENSE_COMPONENT_KIND,
    DENSE_CONTRACT_VERSION,
    DENSE_FORMAT_VERSION,
    DENSE_IMPLEMENTATION,
    DenseDistanceMetric,
    DenseIndexCompatibility,
    DenseIndexConfiguration,
    DenseIndexEntry,
    DenseIndexMatch,
    DenseVector,
    InvalidDenseContractError,
    l2_normalised,
)
from inference_lab.retrieval.errors import RetrievalOperationalError
from inference_lab.retrieval.evidence import (
    CandidateKind,
    EvidenceSource,
    RetrievalCandidate,
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

DENSE_DIRECTORY_NAME = "dense"
DENSE_DATABASE_NAME = "vectors.sqlite3"
DENSE_MANIFEST_NAME = "manifest.json"
DENSE_DATABASE_USER_VERSION = 1
_LOGGER = structlog.get_logger("inference_lab.retrieval")
_EMBED_BATCH_SIZE = 32
_VECTOR_VALUE_BYTES = 4
# Embedding a corpus is the longest single step in a build and writes nothing
# until it finishes, so it reports progress at this cadence. The interval
# bounds the record count for a large corpus while still showing movement.
_EMBED_PROGRESS_CHUNKS: Final = 250


class DenseFailureCategory(StrEnum):
    """Bounded failure categories for dense generation operations."""

    INVALID_REQUEST = "invalid_request"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    CORRUPT_GENERATION = "corrupt_generation"
    INCOMPATIBLE_GENERATION = "incompatible_generation"
    EMBEDDING_FAILURE = "embedding_failure"
    EXECUTION_FAILURE = "execution_failure"


class DenseError(RetrievalOperationalError[DenseFailureCategory]):
    """A privacy-safe failure from dense generation operations."""

    category_type = DenseFailureCategory


@dataclass(frozen=True, slots=True)
class DenseMemberSelection:
    """Identity and chunk count for one selected document version."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    selection_digest: str
    chunk_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the member selection is consistent with the dense contract."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.selection_digest, str):
            raise TypeError("selection_digest must be a string.")
        if len(self.selection_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.selection_digest
        ):
            raise ValueError("selection_digest must be a lowercase SHA-256 digest.")
        if type(self.chunk_count) is not int:
            raise TypeError("chunk_count must be an integer.")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must not be negative.")


@dataclass(frozen=True, slots=True)
class DenseIndexManifest:
    """Durable ownership and compatibility identity for one generation."""

    generation_id: str
    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    source_manifest_digest: str
    compatibility: DenseIndexCompatibility
    members: tuple[DenseMemberSelection, ...]
    entry_count: int
    database_digest: str
    library_version: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the manifest is consistent with the dense contract."""
        if (
            not isinstance(self.generation_id, str)
            or len(self.generation_id) != 70
            or not self.generation_id.startswith("dense-")
            or any(
                character not in "0123456789abcdef"
                for character in self.generation_id[6:]
            )
        ):
            raise ValueError("generation_id must be a dense generation identifier.")
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        _validate_sha256(self.source_manifest_digest, "source_manifest_digest")
        if not isinstance(self.compatibility, DenseIndexCompatibility):
            raise TypeError("compatibility must be a DenseIndexCompatibility.")
        if not isinstance(self.members, tuple):
            raise TypeError("members must be a tuple.")
        if any(not isinstance(item, DenseMemberSelection) for item in self.members):
            raise TypeError("members must contain DenseMemberSelection values.")
        keys = tuple(item.document_id.as_str() for item in self.members)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("members must be unique and canonically ordered.")
        if type(self.entry_count) is not int or self.entry_count < 0:
            raise ValueError("entry_count must be a non-negative integer.")
        _validate_sha256(self.database_digest, "database_digest")
        # Recorded for restore decisions only; deliberately outside the
        # configuration digest so a host library upgrade cannot invalidate
        # an otherwise reusable generation.
        if not isinstance(self.library_version, str) or not self.library_version:
            raise ValueError("library_version must be a non-empty string.")
        if not is_path_free_identifier(self.library_version):
            raise ValueError("library_version must not be a local path.")

    @property
    def configuration_digest(self) -> str:
        """Return the canonical embedding and index compatibility digest."""
        return self.compatibility.configuration_digest

    @property
    def source_chunk_count(self) -> int:
        """Return the total number of chunks declared by membership."""
        return sum(member.chunk_count for member in self.members)

    def component_generation(self) -> ComponentGenerationReference:
        """Return the preparing reference used by corpus-version registration."""
        return ComponentGenerationReference(
            component_kind=DENSE_COMPONENT_KIND,
            generation_id=self.generation_id,
            implementation=self.compatibility.configuration.implementation,
            format_version=self.compatibility.configuration.format_version,
            contract_version=self.compatibility.configuration.contract_version,
            configuration_digest=self.configuration_digest,
            source_manifest_digest=self.source_manifest_digest,
            corpus_id=self.corpus_id,
            corpus_version=self.corpus_version,
            state=ComponentGenerationState.PREPARING,
        )

    def as_payload(self) -> JsonObject:
        """Return the exact path-free manifest payload for durable storage."""
        return {
            "compatibility": self.compatibility.canonical_payload(),
            "corpus_id": self.corpus_id.as_str(),
            "corpus_version": self.corpus_version.as_str(),
            "database_digest": self.database_digest,
            "entry_count": self.entry_count,
            "format_version": DENSE_FORMAT_VERSION,
            "generation_id": self.generation_id,
            "library_version": self.library_version,
            "members": [
                {
                    "chunk_count": item.chunk_count,
                    "document_id": item.document_id.as_str(),
                    "document_version_id": item.document_version_id.as_str(),
                    "selection_digest": item.selection_digest,
                }
                for item in self.members
            ],
            "source_manifest_digest": self.source_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class DenseIndexBuildStatistics:
    """Bounded build counts for one complete proposed generation."""

    members_processed: int
    chunks_processed: int
    reused_vectors: int
    embedded_vectors: int
    replaced_vectors: int
    removed_vectors: int
    generation_reused: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the statistics are consistent with the dense contract."""
        for field_name in (
            "members_processed",
            "chunks_processed",
            "reused_vectors",
            "embedded_vectors",
            "replaced_vectors",
            "removed_vectors",
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative.")
        if type(self.generation_reused) is not bool:
            raise TypeError("generation_reused must be a boolean.")


@dataclass(frozen=True, slots=True)
class DenseIndexBuildOutcome:
    """A validated immutable generation and its build statistics."""

    manifest: DenseIndexManifest
    statistics: DenseIndexBuildStatistics

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the outcome is consistent with the dense contract."""
        if not isinstance(self.manifest, DenseIndexManifest):
            raise TypeError("manifest must be a DenseIndexManifest.")
        if not isinstance(self.statistics, DenseIndexBuildStatistics):
            raise TypeError("statistics must be DenseIndexBuildStatistics.")


@dataclass(frozen=True, slots=True)
class _PreparedChunk:
    candidate: RetrievalCandidate
    document_id: DocumentId
    document_version_id: DocumentVersionId
    ordinal: int


def _validate_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _manifest_json(manifest: DenseIndexManifest) -> bytes:
    return canonical_json_bytes(manifest.as_payload())


def _read_manifest(path: Path) -> DenseIndexManifest:
    try:
        payload = load_json_object(path.read_bytes())
        require_exact_json_fields(
            payload,
            frozenset(
                {
                    "compatibility",
                    "corpus_id",
                    "corpus_version",
                    "database_digest",
                    "entry_count",
                    "format_version",
                    "generation_id",
                    "library_version",
                    "members",
                    "source_manifest_digest",
                }
            ),
        )
        if payload.get("format_version") != DENSE_FORMAT_VERSION:
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        raw_compatibility = payload["compatibility"]
        if not isinstance(raw_compatibility, dict):
            raise ValueError("manifest compatibility must be an object.")
        compatibility = _compatibility_from_payload(raw_compatibility)
        raw_members = payload["members"]
        if not isinstance(raw_members, list):
            raise ValueError("manifest members must be an array.")
        members = tuple(
            _member_from_payload(item) for item in raw_members if isinstance(item, dict)
        )
        if len(members) != len(raw_members):
            raise ValueError("every manifest member must be an object.")
        return DenseIndexManifest(
            generation_id=require_json_string(payload["generation_id"]),
            corpus_id=CorpusId.from_str(require_json_string(payload["corpus_id"])),
            corpus_version=CorpusVersionId.from_str(
                require_json_string(payload["corpus_version"])
            ),
            source_manifest_digest=require_json_string(
                payload["source_manifest_digest"]
            ),
            compatibility=compatibility,
            members=members,
            entry_count=require_json_integer(payload["entry_count"]),
            library_version=require_json_string(payload["library_version"]),
            database_digest=require_json_string(payload["database_digest"]),
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


def _member_from_payload(payload: JsonObject) -> DenseMemberSelection:
    require_exact_json_fields(
        payload,
        frozenset(
            {"chunk_count", "document_id", "document_version_id", "selection_digest"}
        ),
    )
    return DenseMemberSelection(
        document_id=DocumentId.from_str(require_json_string(payload["document_id"])),
        document_version_id=DocumentVersionId.from_str(
            require_json_string(payload["document_version_id"])
        ),
        selection_digest=require_json_string(payload["selection_digest"]),
        chunk_count=require_json_integer(payload["chunk_count"]),
    )


_COMPATIBILITY_FIELDS: Final = frozenset(
    {"configuration", "dimensions", "embedding_profile", "model"}
)
_MODEL_FIELDS: Final = frozenset(
    {
        "backend_name",
        "capabilities",
        "model_name",
        "model_revision",
        "preprocessing_profile_version",
        "supported_operations",
    }
)
_PROFILE_FIELDS: Final = frozenset(
    {
        "device_type",
        "document_prefix",
        "inference_dtype",
        "input_normalisation",
        "l2_normalised",
        "max_input_tokens",
        "over_limit_policy",
        "pooling_strategy",
        "profile_version",
        "query_prefix",
        "tokenizer_id",
        "tokenizer_revision",
    }
)
_CONFIGURATION_FIELDS: Final = frozenset(
    {
        "contract_version",
        "distance_metric",
        "format_version",
        "implementation",
        "tie_break_policy",
    }
)


def _decode_backend_metadata(model: JsonObject, dimensions: int) -> BackendMetadata:
    """Decode the backend identity that produced these vectors."""
    require_exact_json_fields(model, _MODEL_FIELDS)
    return BackendMetadata(
        backend_name=require_json_string(model["backend_name"]),
        model_name=require_json_string(model["model_name"]),
        model_revision=require_json_string(model["model_revision"]),
        preprocessing_profile_version=require_optional_json_string(
            model["preprocessing_profile_version"]
        ),
        supported_operations=tuple(
            BackendOperation(require_json_string(value))
            for value in require_json_array(model["supported_operations"])
        ),
        capabilities=tuple(
            BackendCapability(require_json_string(value))
            for value in require_json_array(model["capabilities"])
        ),
        embedding_dimensions=dimensions,
    )


def _decode_embedding_profile(profile: JsonObject) -> EmbeddingProfile:
    """Decode the runtime preprocessing identity recorded for these vectors."""
    require_exact_json_fields(profile, _PROFILE_FIELDS)
    policy = require_optional_json_string(profile["over_limit_policy"])
    if policy not in (None, "truncate_tail", "reject"):
        raise ValueError("over_limit_policy names an unsupported policy.")
    return EmbeddingProfile(
        profile_version=require_json_string(profile["profile_version"]),
        tokenizer_id=require_json_string(profile["tokenizer_id"]),
        tokenizer_revision=require_json_string(profile["tokenizer_revision"]),
        query_prefix=require_json_string(profile["query_prefix"]),
        document_prefix=require_json_string(profile["document_prefix"]),
        pooling_strategy=require_json_string(profile["pooling_strategy"]),
        input_normalisation=require_json_string(profile["input_normalisation"]),
        max_input_tokens=require_optional_json_integer(profile["max_input_tokens"]),
        over_limit_policy=cast(EmbeddingOverLimitPolicy | None, policy),
        l2_normalised=require_json_boolean(profile["l2_normalised"]),
        inference_dtype=require_json_string(profile["inference_dtype"]),
        device_type=require_json_string(profile["device_type"]),
    )


def _decode_index_configuration(config: JsonObject) -> DenseIndexConfiguration:
    """Decode the versioned settings that govern index interpretation."""
    require_exact_json_fields(config, _CONFIGURATION_FIELDS)
    return DenseIndexConfiguration(
        implementation=require_json_string(config["implementation"]),
        format_version=require_json_string(config["format_version"]),
        contract_version=require_json_string(config["contract_version"]),
        distance_metric=DenseDistanceMetric(
            require_json_string(config["distance_metric"])
        ),
        tie_break_policy=require_json_string(config["tie_break_policy"]),
    )


def _compatibility_from_payload(payload: JsonObject) -> DenseIndexCompatibility:
    try:
        model = payload["model"]
        profile = payload["embedding_profile"]
        config = payload["configuration"]
        if (
            not isinstance(model, dict)
            or not isinstance(profile, dict)
            or not isinstance(config, dict)
        ):
            raise ValueError(
                "compatibility must hold model, profile, and configuration objects."
            )
        require_exact_json_fields(payload, _COMPATIBILITY_FIELDS)
        dimensions = require_json_integer(payload["dimensions"])
        return DenseIndexCompatibility(
            model_metadata=_decode_backend_metadata(model, dimensions),
            dimensions=dimensions,
            embedding_profile=_decode_embedding_profile(profile),
            configuration=_decode_index_configuration(config),
        )
    except (KeyError, TypeError, ValueError):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


class DenseGeneration:
    """Read-only handle for a validated immutable dense generation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        manifest: DenseIndexManifest,
    ) -> None:
        self._connection = connection
        self._manifest = manifest
        self._closed = False

    @property
    def manifest(self) -> DenseIndexManifest:
        """Return the immutable manifest this generation was opened by."""
        return self._manifest

    def __enter__(self) -> Self:
        if self._closed:
            raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the read-only SQLite connection."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def read_entries(self) -> tuple[DenseIndexEntry, ...]:
        """Return all candidates and vectors in deterministic source order."""
        self._require_open()
        try:
            rows = self._connection.execute(
                """
                SELECT record_id, document_id, document_version_id, ordinal,
                       content, source_links, vector
                FROM entries ORDER BY document_id, ordinal, record_id
                """
            ).fetchall()
            return tuple(
                _entry_from_row(row, self.manifest.compatibility.dimensions)
                for row in rows
            )
        except DenseError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, struct.error):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def search(
        self,
        query_vector: DenseVector,
        *,
        limit: int,
    ) -> tuple[DenseIndexMatch, ...]:
        """Return the best matches without materialising the whole index."""
        if not isinstance(query_vector, DenseVector):
            raise TypeError("query_vector must be a DenseVector.")
        if type(limit) is not int:
            raise TypeError("limit must be an integer.")
        if limit <= 0:
            raise DenseError(DenseFailureCategory.INVALID_REQUEST)
        self._require_open()
        dimensions = self.manifest.compatibility.dimensions
        if len(query_vector.values) != dimensions:
            raise DenseError(DenseFailureCategory.INVALID_REQUEST)
        try:
            unit_query = l2_normalised(query_vector).values
            ranked = heapq.nsmallest(
                limit,
                self._stream_similarities(unit_query, dimensions),
                key=lambda scored: (-scored[0], scored[1]),
            )
            return tuple(
                DenseIndexMatch(
                    candidate=self._require_entry(record_id).candidate,
                    cosine_similarity=max(-1.0, min(1.0, similarity)),
                    rank=rank,
                )
                for rank, (similarity, record_id) in enumerate(ranked, start=1)
            )
        except DenseError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, struct.error):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def _stream_similarities(
        self,
        unit_query: tuple[float, ...],
        dimensions: int,
    ) -> Iterator[tuple[float, str]]:
        """Score every stored vector without holding more than one row."""
        expected_bytes = dimensions * _VECTOR_VALUE_BYTES
        for record_id, vector_bytes in self._connection.execute(
            "SELECT record_id, vector FROM entries"
        ):
            stored = bytes(vector_bytes)
            if len(stored) != expected_bytes:
                raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
            # Stored vectors are unit-length, so cosine similarity is the
            # dot product with the unit query.
            values = struct.unpack(f"<{dimensions}f", stored)
            yield sum(map(mul, unit_query, values)), record_id

    def _require_entry(self, record_id: str) -> DenseIndexEntry:
        """Hydrate one ranked record, rejecting a generation that lost it."""
        entry = self.get_entry(RecordId.from_str(record_id))
        if entry is None:
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        return entry

    def get_entry(self, record_id: RecordId) -> DenseIndexEntry | None:
        """Return one vector record by its immutable identity."""
        if not isinstance(record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        self._require_open()
        try:
            row = self._connection.execute(
                """
                SELECT record_id, document_id, document_version_id, ordinal,
                       content, source_links, vector
                FROM entries WHERE record_id = ?
                """,
                (record_id.as_str(),),
            ).fetchone()
            return (
                None
                if row is None
                else _entry_from_row(row, self.manifest.compatibility.dimensions)
            )
        except DenseError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, struct.error):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def member_selections(self) -> tuple[DenseMemberSelection, ...]:
        """Return complete staged membership represented by this generation."""
        self._require_open()
        try:
            return tuple(
                DenseMemberSelection(
                    document_id=DocumentId.from_str(row[0]),
                    document_version_id=DocumentVersionId.from_str(row[1]),
                    selection_digest=row[2],
                    chunk_count=row[3],
                )
                for row in self._connection.execute(
                    """
                    SELECT document_id, document_version_id, selection_digest,
                           chunk_count FROM members ORDER BY document_id
                    """
                ).fetchall()
            )
        except (sqlite3.Error, TypeError, ValueError, IndexError):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def _require_open(self) -> None:
        if self._closed:
            raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE)


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


def _entry_from_row(row: sqlite3.Row, dimensions: int) -> DenseIndexEntry:
    # The messages below describe the defect for a maintainer reading a
    # traceback; every one is converted to a bounded corrupt-generation
    # failure before it can cross the boundary.
    try:
        source_object = load_json_object(bytes(row[5]))
        if not isinstance(source_object, dict) or set(source_object) != {"sources"}:
            raise ValueError("stored source links must hold exactly one sources key.")
        source_payload = source_object["sources"]
        if not isinstance(source_payload, list):
            raise ValueError("stored source links must be an array.")
        sources = tuple(
            EvidenceSource(
                document_id=DocumentId.from_str(
                    require_json_string(item["document_id"])
                ),
                document_version_id=DocumentVersionId.from_str(
                    require_json_string(item["document_version_id"])
                ),
                source_span_id=SourceSpanId.from_str(
                    require_json_string(item["source_span_id"])
                ),
            )
            for item in source_payload
            if isinstance(item, dict)
        )
        if len(sources) != len(source_payload):
            raise ValueError("every stored source link must be an object.")
        candidate = RetrievalCandidate(
            record_id=RecordId.from_str(row[0]),
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content=row[4],
            sources=sources,
        )
        vector_bytes = bytes(row[6])
        if len(vector_bytes) != dimensions * _VECTOR_VALUE_BYTES:
            raise ValueError("stored vector width does not match the index.")
        vector = DenseVector(struct.unpack(f"<{dimensions}f", vector_bytes))
        document_id = DocumentId.from_str(row[1])
        document_version_id = DocumentVersionId.from_str(row[2])
        if (
            document_id != sources[0].document_id
            or document_version_id != sources[0].document_version_id
            or type(row[3]) is not int
            or row[3] < 0
            or any(
                source.document_id != document_id
                or source.document_version_id != document_version_id
                for source in sources
            )
        ):
            raise ValueError("stored row lineage disagrees with its source links.")
        return DenseIndexEntry(candidate, vector)
    except (TypeError, ValueError, IndexError, KeyError, struct.error):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


def open_dense_generation(index_dir: Path, generation_id: str) -> DenseGeneration:
    """Open and validate one generation by its immutable identity."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(generation_id, str):
        raise TypeError("generation_id must be a string.")
    if len(generation_id) != 70 or not generation_id.startswith("dense-"):
        raise DenseError(DenseFailureCategory.INVALID_REQUEST)
    root = index_dir / DENSE_DIRECTORY_NAME / generation_id
    return _open_dense_path(root, generation_id)


def _open_dense_path(path: Path, expected_generation_id: str) -> DenseGeneration:
    connection: sqlite3.Connection | None = None
    try:
        if path.is_symlink() or not path.is_dir():
            raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE)
        manifest_path = path / DENSE_MANIFEST_NAME
        database_path = path / DENSE_DATABASE_NAME
        if manifest_path.is_symlink() or database_path.is_symlink():
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        manifest = _read_manifest(manifest_path)
        if manifest.generation_id != expected_generation_id:
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if (
            manifest.compatibility.configuration.implementation != DENSE_IMPLEMENTATION
            or manifest.compatibility.configuration.format_version
            != DENSE_FORMAT_VERSION
            or manifest.compatibility.configuration.contract_version
            != DENSE_CONTRACT_VERSION
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if _sha256_file(database_path) != manifest.database_digest:
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        database_uri = f"file:{quote(str(database_path))}?mode=ro&immutable=1"
        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        if connection.execute("PRAGMA user_version").fetchone()[0] != (
            DENSE_DATABASE_USER_VERSION
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        generation = DenseGeneration(connection, manifest)
        members = generation.member_selections()
        entries = generation.read_entries()
        if members != manifest.members or len(entries) != manifest.entry_count:
            generation.close()
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        if sum(member.chunk_count for member in members) != len(entries):
            generation.close()
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        member_counts = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                """
                SELECT m.document_id, m.document_version_id, COUNT(e.record_id)
                FROM members AS m LEFT JOIN entries AS e USING (document_id)
                GROUP BY m.document_id, m.document_version_id
                """
            ).fetchall()
        }
        if any(
            member_counts.get(member.document_id.as_str())
            != (member.document_version_id.as_str(), member.chunk_count)
            for member in members
        ):
            generation.close()
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        return generation
    except DenseError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
        if connection is not None:
            connection.close()
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


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
