"""Durable identity, failure categories, and on-disk layout for dense generations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from inference_lab._json import JsonObject
from inference_lab._validation import is_path_free_identifier
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.retrieval.dense import (
    DENSE_COMPONENT_KIND,
    DENSE_FORMAT_VERSION,
    DenseIndexCompatibility,
)
from inference_lab.retrieval.errors import RetrievalOperationalError
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
)

DENSE_DIRECTORY_NAME = "dense"
DENSE_DATABASE_NAME = "vectors.sqlite3"
DENSE_MANIFEST_NAME = "manifest.json"
DENSE_DATABASE_USER_VERSION = 1


_VECTOR_VALUE_BYTES = 4


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


def _validate_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value
