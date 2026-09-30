"""Immutable SQLite generations for corpus-scoped dense retrieval."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast
from urllib.parse import quote

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
from inference_lab._validation import paths_overlap
from inference_lab.backends.runtime import (
    BackendCapability,
    BackendError,
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
    DenseVector,
    InvalidDenseContractError,
)
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
)

DENSE_DIRECTORY_NAME = "dense"
DENSE_DATABASE_NAME = "vectors.sqlite3"
DENSE_MANIFEST_NAME = "manifest.json"
DENSE_DATABASE_USER_VERSION = 1
_EMBED_BATCH_SIZE = 32


class DenseFailureCategory(StrEnum):
    """Bounded failure categories for dense generation operations."""

    INVALID_REQUEST = "invalid_request"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    CORRUPT_GENERATION = "corrupt_generation"
    INCOMPATIBLE_GENERATION = "incompatible_generation"
    EMBEDDING_FAILURE = "embedding_failure"
    EXECUTION_FAILURE = "execution_failure"


class DenseError(RuntimeError):
    """A privacy-safe failure from dense generation operations."""

    def __init__(self, category: str, *, retryable: bool = False) -> None:
        if category not in {
            DenseFailureCategory.INVALID_REQUEST,
            DenseFailureCategory.GENERATION_UNAVAILABLE,
            DenseFailureCategory.CORRUPT_GENERATION,
            DenseFailureCategory.INCOMPATIBLE_GENERATION,
            DenseFailureCategory.EMBEDDING_FAILURE,
            DenseFailureCategory.EXECUTION_FAILURE,
        }:
            raise ValueError("category must be a supported dense failure category.")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean.")
        super().__init__(category)
        self.category = category
        self.retryable = retryable


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

    @property
    def configuration_digest(self) -> str:
        """Return the canonical embedding and index compatibility digest."""
        return self.compatibility.configuration_digest

    @property
    def source_chunk_count(self) -> int:
        """Return the total number of chunks declared by membership."""
        return sum(member.chunk_count for member in self.members)

    def component_generation(self) -> ComponentGenerationReference:
        """Return the preparing reference used by STOR-03 registration."""
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
                    "members",
                    "source_manifest_digest",
                }
            ),
        )
        if payload.get("format_version") != DENSE_FORMAT_VERSION:
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        raw_compatibility = payload["compatibility"]
        if not isinstance(raw_compatibility, dict):
            raise ValueError
        compatibility = _compatibility_from_payload(raw_compatibility)
        raw_members = payload["members"]
        if not isinstance(raw_members, list):
            raise ValueError
        members = tuple(
            _member_from_payload(item) for item in raw_members if isinstance(item, dict)
        )
        if len(members) != len(raw_members):
            raise ValueError
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
            raise ValueError
        require_exact_json_fields(
            payload,
            frozenset({"configuration", "dimensions", "embedding_profile", "model"}),
        )
        require_exact_json_fields(
            model,
            frozenset(
                {
                    "backend_name",
                    "capabilities",
                    "model_name",
                    "model_revision",
                    "supported_operations",
                }
            ),
        )
        require_exact_json_fields(
            profile,
            frozenset(
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
            ),
        )
        require_exact_json_fields(
            config,
            frozenset(
                {
                    "contract_version",
                    "distance_metric",
                    "format_version",
                    "implementation",
                    "library_version",
                    "tie_break_policy",
                }
            ),
        )
        dimensions = require_json_integer(payload["dimensions"])
        raw_operations = tuple(
            BackendOperation(require_json_string(value))
            for value in require_json_array(model["supported_operations"])
        )
        raw_capabilities = tuple(
            BackendCapability(require_json_string(value))
            for value in require_json_array(model["capabilities"])
        )
        metadata = BackendMetadata(
            backend_name=require_json_string(model["backend_name"]),
            model_name=require_json_string(model["model_name"]),
            model_revision=require_json_string(model["model_revision"]),
            supported_operations=raw_operations,
            capabilities=raw_capabilities,
            embedding_dimensions=dimensions,
        )
        policy = require_optional_json_string(profile["over_limit_policy"])
        if policy not in (None, "truncate_tail", "reject"):
            raise ValueError
        embedding_profile = EmbeddingProfile(
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
        configuration = DenseIndexConfiguration(
            implementation=require_json_string(config["implementation"]),
            library_version=require_json_string(config["library_version"]),
            format_version=require_json_string(config["format_version"]),
            contract_version=require_json_string(config["contract_version"]),
            distance_metric=DenseDistanceMetric(
                require_json_string(config["distance_metric"])
            ),
            tie_break_policy=require_json_string(config["tie_break_policy"]),
        )
        return DenseIndexCompatibility(
            model_metadata=metadata,
            dimensions=dimensions,
            embedding_profile=embedding_profile,
            configuration=configuration,
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
                        f"<{compatibility.dimensions}d",
                        *entry.vector.values,
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


def _embed_pending(
    embedder: EmbeddingBackend,
    pending: tuple[_PreparedChunk, ...],
    compatibility: DenseIndexCompatibility,
    *,
    batch_size: int,
) -> dict[RecordId, DenseVector]:
    vectors: dict[RecordId, DenseVector] = {}
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
        except BackendError:
            raise DenseError(DenseFailureCategory.EMBEDDING_FAILURE) from None
        except Exception:
            raise DenseError(DenseFailureCategory.EMBEDDING_FAILURE) from None
        if (
            response.model_metadata != compatibility.model_metadata
            or response.embedding_profile != compatibility.embedding_profile
            or len(response.vectors) != len(batch)
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        for item, values in zip(batch, response.vectors):
            try:
                vector = DenseVector(tuple(values))
            except (TypeError, ValueError, InvalidDenseContractError):
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION) from None
            if len(vector.values) != compatibility.dimensions:
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
            vectors[item.candidate.record_id] = vector
    return vectors


def build_dense_generation(
    index_dir: Path,
    source_manifest: BaselineChunkGenerationManifest,
    embedder: EmbeddingBackend,
    *,
    metadata_db_path: Path,
    source_dir: Path,
    compatibility: DenseIndexCompatibility,
    parent_reference: ComponentGenerationReference | None = None,
    batch_size: int = _EMBED_BATCH_SIZE,
) -> DenseIndexBuildOutcome:
    """Build a complete immutable generation from staged canonical chunks."""
    if not isinstance(source_manifest, BaselineChunkGenerationManifest):
        raise TypeError("source_manifest must be a BaselineChunkGenerationManifest.")
    if not isinstance(compatibility, DenseIndexCompatibility):
        raise TypeError("compatibility must be a DenseIndexCompatibility.")
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise DenseError(DenseFailureCategory.INVALID_REQUEST)
    if not callable(getattr(embedder, "embed", None)):
        raise TypeError("embedder must implement EmbeddingBackend.")
    configuration = compatibility.configuration
    if (
        configuration.implementation != DENSE_IMPLEMENTATION
        or configuration.format_version != DENSE_FORMAT_VERSION
        or configuration.contract_version != DENSE_CONTRACT_VERSION
        or configuration.library_version != sqlite3.sqlite_version
    ):
        raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
    if parent_reference is not None:
        if not isinstance(parent_reference, ComponentGenerationReference):
            raise TypeError("parent_reference must be a ComponentGenerationReference.")
        if (
            parent_reference.component_kind != DENSE_COMPONENT_KIND
            or parent_reference.state is not ComponentGenerationState.READY
            or parent_reference.corpus_id != source_manifest.corpus_id
            or parent_reference.corpus_version == source_manifest.corpus_version
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)

    resolved_index_dir = _resolve_local_paths(
        index_dir,
        metadata_db_path,
        source_dir,
    )
    prepared, members = _prepare_source(source_manifest)
    generation_id = _derive_generation_id(
        source_manifest.corpus_id,
        source_manifest.corpus_version,
        source_manifest.source_manifest_digest,
        compatibility,
    )
    dense_root = resolved_index_dir / DENSE_DIRECTORY_NAME
    target = dense_root / generation_id
    if dense_root.exists() and dense_root.is_symlink():
        raise DenseError(DenseFailureCategory.INVALID_REQUEST)
    if target.exists():
        if target.is_symlink():
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        with open_dense_generation(resolved_index_dir, generation_id) as generation:
            if not _generation_matches(generation, source_manifest, compatibility):
                raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
            return DenseIndexBuildOutcome(
                manifest=generation.manifest,
                statistics=DenseIndexBuildStatistics(
                    members_processed=len(members),
                    chunks_processed=len(prepared),
                    reused_vectors=len(prepared),
                    embedded_vectors=0,
                    replaced_vectors=0,
                    removed_vectors=0,
                    generation_reused=True,
                ),
            )

    parent: DenseGeneration | None = None
    if parent_reference is not None:
        try:
            parent = open_dense_generation(
                resolved_index_dir,
                parent_reference.generation_id,
            )
        except DenseError as error:
            if error.category not in {
                DenseFailureCategory.CORRUPT_GENERATION,
                DenseFailureCategory.GENERATION_UNAVAILABLE,
            }:
                raise
        if parent is not None and (
            parent.manifest.corpus_id != source_manifest.corpus_id
            or parent.manifest.corpus_version != parent_reference.corpus_version
            or parent.manifest.compatibility.configuration.implementation
            != parent_reference.implementation
            or parent.manifest.configuration_digest
            != parent_reference.configuration_digest
        ):
            parent.close()
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)

    try:
        reusable: dict[RecordId, DenseIndexEntry] = {}
        if (
            parent is not None
            and parent.manifest.configuration_digest
            == compatibility.configuration_digest
        ):
            proposed_by_id = {item.candidate.record_id: item for item in prepared}
            for old_entry in parent.iter_entries():
                new_item = proposed_by_id.get(old_entry.candidate.record_id)
                if new_item is not None and new_item.candidate == old_entry.candidate:
                    reusable[old_entry.candidate.record_id] = old_entry
        pending = tuple(
            item for item in prepared if item.candidate.record_id not in reusable
        )
        embedded = _embed_pending(
            embedder,
            pending,
            compatibility,
            batch_size=batch_size,
        )
        entries = tuple(
            reusable[item.candidate.record_id]
            if item.candidate.record_id in reusable
            else DenseIndexEntry(item.candidate, embedded[item.candidate.record_id])
            for item in prepared
        )

        dense_root.mkdir(parents=True, exist_ok=True)
        _fsync_directory(resolved_index_dir)
        staging = Path(tempfile.mkdtemp(prefix=".dense-staging-", dir=dense_root))
        try:
            database_path = staging / DENSE_DATABASE_NAME
            _write_generation_database(
                database_path,
                compatibility=compatibility,
                members=members,
                entries=entries,
                prepared=prepared,
            )
            manifest = DenseIndexManifest(
                generation_id=generation_id,
                corpus_id=source_manifest.corpus_id,
                corpus_version=source_manifest.corpus_version,
                source_manifest_digest=source_manifest.source_manifest_digest,
                compatibility=compatibility,
                members=members,
                entry_count=len(entries),
                database_digest=_sha256_file(database_path),
            )
            manifest_path = staging / DENSE_MANIFEST_NAME
            _write_durable(manifest_path, _manifest_json(manifest))
            with database_path.open("rb") as database_stream:
                os.fsync(database_stream.fileno())
            _fsync_directory(staging)
            with _open_dense_path(staging, generation_id) as opened:
                if opened.manifest != manifest:
                    raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
                if not _generation_matches(opened, source_manifest, compatibility):
                    raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
            try:
                os.rename(staging, target)
            except FileExistsError:
                with open_dense_generation(resolved_index_dir, generation_id) as opened:
                    if opened.manifest != manifest:
                        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
            _fsync_directory(dense_root)
        finally:
            if staging.exists():
                _remove_staging_directory(staging)

        old_entries = () if parent is None else parent.iter_entries()
        reused_ids = set(reusable)
        replaced_ids: set[RecordId] = set()
        removed_document_ids: set[DocumentId] = set()
        if parent is not None:
            old_by_document = {
                member.document_id: member.document_version_id
                for member in parent.manifest.members
            }
            new_by_document = {
                member.document_id: member.document_version_id for member in members
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
        return DenseIndexBuildOutcome(
            manifest=manifest,
            statistics=DenseIndexBuildStatistics(
                members_processed=len(members),
                chunks_processed=len(prepared),
                reused_vectors=len(reused_ids),
                embedded_vectors=len(embedded),
                replaced_vectors=len(replaced_ids),
                removed_vectors=sum(
                    1
                    for entry in old_entries
                    if entry.candidate.sources[0].document_id in removed_document_ids
                ),
            ),
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
            references = tuple(
                item
                for item in snapshot.manifest.component_generations
                if item.component_kind == DENSE_COMPONENT_KIND
            )
            if len(references) != 1:
                raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
            reference = references[0]
            if (
                reference.state is not ComponentGenerationState.READY
                or reference.corpus_id != corpus_id
                or reference.corpus_version != expected_corpus_version
            ):
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
    source_manifest: BaselineChunkGenerationManifest,
    embedder: EmbeddingBackend,
    repository: MetadataRepository,
    *,
    expected_corpus_version: CorpusVersionId | None,
    metadata_db_path: Path,
    source_dir: Path,
    compatibility: DenseIndexCompatibility,
    batch_size: int = _EMBED_BATCH_SIZE,
) -> DenseIndexBuildOutcome:
    """Build against the exact expected active parent, if one exists."""
    if not isinstance(source_manifest, BaselineChunkGenerationManifest):
        raise TypeError("source_manifest must be a BaselineChunkGenerationManifest.")
    if expected_corpus_version == source_manifest.corpus_version:
        raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
    parent = select_dense_parent_reference(
        repository,
        source_manifest.corpus_id,
        expected_corpus_version,
    )
    return build_dense_generation(
        index_dir,
        source_manifest,
        embedder,
        metadata_db_path=metadata_db_path,
        source_dir=source_dir,
        compatibility=compatibility,
        parent_reference=parent,
        batch_size=batch_size,
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _generation_matches(
    generation: DenseGeneration,
    source_manifest: BaselineChunkGenerationManifest,
    compatibility: DenseIndexCompatibility,
) -> bool:
    if (
        generation.manifest.corpus_id != source_manifest.corpus_id
        or generation.manifest.corpus_version != source_manifest.corpus_version
        or generation.manifest.source_manifest_digest
        != source_manifest.source_manifest_digest
        or generation.manifest.configuration_digest
        != compatibility.configuration_digest
    ):
        return False
    prepared, members = _prepare_source(source_manifest)
    if generation.manifest.members != members or generation.manifest.entry_count != len(
        prepared
    ):
        return False
    actual = {entry.candidate.record_id: entry for entry in generation.iter_entries()}
    return all(
        (entry := actual.get(item.candidate.record_id)) is not None
        and entry.candidate == item.candidate
        and len(entry.vector.values) == compatibility.dimensions
        for item in prepared
    ) and len(actual) == len(prepared)


class DenseGeneration:
    """Read-only handle for a validated immutable dense generation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        manifest: DenseIndexManifest,
    ) -> None:
        self._connection = connection
        self.manifest = manifest
        self._closed = False

    def __enter__(self) -> DenseGeneration:
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

    def iter_entries(self) -> tuple[DenseIndexEntry, ...]:
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


def _entry_from_row(row: sqlite3.Row, dimensions: int) -> DenseIndexEntry:
    try:
        source_object = load_json_object(bytes(row[5]))
        if not isinstance(source_object, dict) or set(source_object) != {"sources"}:
            raise ValueError
        source_payload = source_object["sources"]
        if not isinstance(source_payload, list):
            raise ValueError
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
            raise ValueError
        candidate = RetrievalCandidate(
            record_id=RecordId.from_str(row[0]),
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content=row[4],
            sources=sources,
        )
        vector_bytes = bytes(row[6])
        if len(vector_bytes) != dimensions * 8:
            raise ValueError
        vector = DenseVector(struct.unpack(f"<{dimensions}d", vector_bytes))
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
            raise ValueError
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
        entries = generation.iter_entries()
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
        """Return the preparing reference for STOR-03 registration."""
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
                return _generation_matches(
                    opened,
                    self._source_manifest,
                    self._manifest.compatibility,
                )
        except (DenseError, StorageError, OSError, TypeError, ValueError):
            return False
