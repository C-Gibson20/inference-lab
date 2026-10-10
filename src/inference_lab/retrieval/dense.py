"""Versioned contracts and deterministic ranking for dense retrieval."""

import hashlib
import math
import re
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab._validation import is_canonical_semantic_version, is_lowercase_slug
from inference_lab.backends.runtime import BackendMetadata, EmbeddingProfile
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval.evidence import RetrievalCandidate
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.storage import ComponentKind

DENSE_COMPONENT_KIND: Final[ComponentKind] = ComponentKind("dense-index")
DENSE_IMPLEMENTATION: Final[str] = "sqlite-flat"
DENSE_FORMAT_VERSION: Final[str] = "1.0.0"
DENSE_CONTRACT_VERSION: Final[str] = "1.0.0"
DENSE_SCORE_LABEL: Final[str] = "cosine_similarity"
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class DenseDistanceMetric(StrEnum):
    """Distance semantics supported by the initial dense contract."""

    COSINE = "cosine"


class InvalidDenseContractError(ValueError):
    """Raised when dense retrieval values violate their contract."""


def _validate_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise InvalidDenseContractError(f"{field_name} must not be empty.")
    return value


def _validate_semantic_version(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise InvalidDenseContractError(
            f"{field_name} must be a canonical semantic version."
        )
    return value


def _validate_slug(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_lowercase_slug(value):
        raise InvalidDenseContractError(
            f"{field_name} must be a lowercase project-owned slug."
        )
    return value


def _path_free_identifier(value: str, field_name: str) -> str:
    if (
        value.startswith(("/", "\\", "~"))
        or "://" in value
        or (len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\"))
    ):
        raise InvalidDenseContractError(f"{field_name} must not be a local path.")
    return value


@dataclass(frozen=True, slots=True)
class DenseIndexConfiguration:
    """Versioned, path-free settings that affect index interpretation."""

    implementation: str = DENSE_IMPLEMENTATION
    library_version: str = field(default_factory=lambda: sqlite3.sqlite_version)
    format_version: str = DENSE_FORMAT_VERSION
    contract_version: str = DENSE_CONTRACT_VERSION
    distance_metric: DenseDistanceMetric = DenseDistanceMetric.COSINE
    tie_break_policy: str = "record-id-ascending"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the configuration is consistent with the dense contract."""
        _validate_slug(self.implementation, "implementation")
        _path_free_identifier(
            _validate_non_empty_string(self.library_version, "library_version"),
            "library_version",
        )
        _validate_semantic_version(self.format_version, "format_version")
        _validate_semantic_version(self.contract_version, "contract_version")
        if not isinstance(self.distance_metric, DenseDistanceMetric):
            raise TypeError("distance_metric must be a DenseDistanceMetric.")
        if not isinstance(self.tie_break_policy, str):
            raise TypeError("tie_break_policy must be a string.")
        if self.tie_break_policy != "record-id-ascending":
            raise InvalidDenseContractError("unsupported tie_break_policy.")

    def canonical_payload(self) -> JsonObject:
        """Return the complete deterministic index configuration payload."""
        return {
            "contract_version": self.contract_version,
            "distance_metric": self.distance_metric.value,
            "format_version": self.format_version,
            "implementation": self.implementation,
            "library_version": self.library_version,
            "tie_break_policy": self.tie_break_policy,
        }


@dataclass(frozen=True, slots=True)
class DenseIndexCompatibility:
    """Identify the embedding and index behaviour required to reuse vectors."""

    model_metadata: BackendMetadata
    dimensions: int
    embedding_profile: EmbeddingProfile
    configuration: DenseIndexConfiguration = field(
        default_factory=DenseIndexConfiguration
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the compatibility is consistent with the dense contract."""
        if not isinstance(self.model_metadata, BackendMetadata):
            raise TypeError("model_metadata must be BackendMetadata.")
        for name, value in (
            ("backend_name", self.model_metadata.backend_name),
            ("model_name", self.model_metadata.model_name),
            ("model_revision", self.model_metadata.model_revision),
        ):
            _path_free_identifier(
                _validate_non_empty_string(value, name),
                name,
            )
        if type(self.dimensions) is not int:
            raise TypeError("dimensions must be an integer.")
        if self.dimensions <= 0:
            raise InvalidDenseContractError("dimensions must be positive.")
        if self.model_metadata.embedding_dimensions not in (None, self.dimensions):
            raise InvalidDenseContractError(
                "dimensions must match embedding model metadata."
            )
        if not isinstance(self.embedding_profile, EmbeddingProfile):
            raise TypeError("embedding_profile must be an EmbeddingProfile.")
        if not isinstance(self.configuration, DenseIndexConfiguration):
            raise TypeError("configuration must be a DenseIndexConfiguration.")

    def canonical_payload(self) -> JsonObject:
        """Return the path-free identity used for compatibility checks."""
        profile = self.embedding_profile
        return {
            "configuration": self.configuration.canonical_payload(),
            "dimensions": self.dimensions,
            "embedding_profile": {
                "device_type": profile.device_type,
                "document_prefix": profile.document_prefix,
                "inference_dtype": profile.inference_dtype,
                "input_normalisation": profile.input_normalisation,
                "l2_normalised": profile.l2_normalised,
                "max_input_tokens": profile.max_input_tokens,
                "over_limit_policy": profile.over_limit_policy,
                "pooling_strategy": profile.pooling_strategy,
                "profile_version": profile.profile_version,
                "query_prefix": profile.query_prefix,
                "tokenizer_id": profile.tokenizer_id,
                "tokenizer_revision": profile.tokenizer_revision,
            },
            "model": {
                "backend_name": self.model_metadata.backend_name,
                "capabilities": [
                    item.value for item in self.model_metadata.capabilities
                ],
                "model_name": self.model_metadata.model_name,
                "model_revision": self.model_metadata.model_revision,
                "supported_operations": [
                    item.value for item in self.model_metadata.supported_operations
                ],
            },
        }

    @property
    def configuration_digest(self) -> str:
        """Return SHA-256 over all vector and index interpretation settings."""
        return hashlib.sha256(
            canonical_json_bytes(self.canonical_payload())
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class DenseVector:
    """One finite, non-zero vector with immutable numeric storage."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the vector is consistent with the dense contract."""
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple.")
        if not self.values:
            raise InvalidDenseContractError("vectors must not be empty.")
        for value in self.values:
            if isinstance(value, bool) or type(value) not in (int, float):
                raise TypeError("vector values must be numbers.")
            if not math.isfinite(value):
                raise InvalidDenseContractError("vector values must be finite.")
        if not any(value != 0 for value in self.values):
            raise InvalidDenseContractError("cosine vectors must not be zero vectors.")


@dataclass(frozen=True, slots=True)
class DenseIndexEntry:
    """One candidate and its vector in a dense generation."""

    candidate: RetrievalCandidate
    vector: DenseVector

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the entry is consistent with the dense contract."""
        if not isinstance(self.candidate, RetrievalCandidate):
            raise TypeError("candidate must be a RetrievalCandidate.")
        if not isinstance(self.vector, DenseVector):
            raise TypeError("vector must be a DenseVector.")


@dataclass(frozen=True, slots=True)
class DenseIndexBuildRequest:
    """Bind entries and compatibility to one proposed corpus generation."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    source_manifest_digest: str
    compatibility: DenseIndexCompatibility
    entries: tuple[DenseIndexEntry, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the build request is consistent with the dense contract."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.source_manifest_digest, str):
            raise TypeError("source_manifest_digest must be a string.")
        if _SHA256_PATTERN.fullmatch(self.source_manifest_digest) is None:
            raise InvalidDenseContractError(
                "source_manifest_digest must be a lowercase SHA-256 digest."
            )
        if not isinstance(self.compatibility, DenseIndexCompatibility):
            raise TypeError("compatibility must be a DenseIndexCompatibility.")
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple.")
        if any(not isinstance(entry, DenseIndexEntry) for entry in self.entries):
            raise TypeError("entries must contain DenseIndexEntry values.")
        record_ids = tuple(entry.candidate.record_id for entry in self.entries)
        if len(record_ids) != len(set(record_ids)):
            raise InvalidDenseContractError("entries must have unique RecordIds.")
        if any(
            len(entry.vector.values) != self.compatibility.dimensions
            for entry in self.entries
        ):
            raise InvalidDenseContractError(
                "entry vectors must match compatibility dimensions."
            )


@dataclass(frozen=True, slots=True)
class DenseIndexQuery:
    """Pair a retrieval query with its query-purpose vector and compatibility."""

    query: RetrievalQuery
    query_vector: DenseVector
    compatibility: DenseIndexCompatibility

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the query is consistent with the dense contract."""
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("query must be a RetrievalQuery.")
        if not isinstance(self.query_vector, DenseVector):
            raise TypeError("query_vector must be a DenseVector.")
        if not isinstance(self.compatibility, DenseIndexCompatibility):
            raise TypeError("compatibility must be a DenseIndexCompatibility.")
        if len(self.query_vector.values) != self.compatibility.dimensions:
            raise InvalidDenseContractError(
                "query vector must match compatibility dimensions."
            )


@dataclass(frozen=True, slots=True)
class DenseIndexMatch:
    """A candidate ranked by cosine similarity within the dense channel."""

    candidate: RetrievalCandidate
    cosine_similarity: float
    rank: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the match is consistent with the dense contract."""
        if not isinstance(self.candidate, RetrievalCandidate):
            raise TypeError("candidate must be a RetrievalCandidate.")
        if isinstance(self.cosine_similarity, bool) or type(
            self.cosine_similarity
        ) not in (int, float):
            raise TypeError("cosine_similarity must be a number.")
        if not math.isfinite(self.cosine_similarity):
            raise InvalidDenseContractError("cosine_similarity must be finite.")
        if not -1.0 <= self.cosine_similarity <= 1.0:
            raise InvalidDenseContractError(
                "cosine_similarity must be between -1 and 1."
            )
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer.")
        if self.rank <= 0:
            raise InvalidDenseContractError("rank must be positive.")


class DenseVectorIndex(Protocol):
    """Backend seam for exact or approximate local dense indexes."""

    def search(
        self,
        request: DenseIndexQuery,
    ) -> tuple[DenseIndexMatch, ...]:
        """Return at most the query limit in deterministic rank order."""
        ...


def rank_dense_entries(
    entries: tuple[DenseIndexEntry, ...],
    query_vector: DenseVector,
    *,
    limit: int,
) -> tuple[DenseIndexMatch, ...]:
    """Rank entries by cosine similarity with stable RecordId tie breaking."""
    if not isinstance(entries, tuple):
        raise TypeError("entries must be a tuple.")
    if any(not isinstance(entry, DenseIndexEntry) for entry in entries):
        raise TypeError("entries must contain DenseIndexEntry values.")
    if not isinstance(query_vector, DenseVector):
        raise TypeError("query_vector must be a DenseVector.")
    if type(limit) is not int:
        raise TypeError("limit must be an integer.")
    if limit <= 0:
        raise InvalidDenseContractError("limit must be positive.")
    record_ids = tuple(entry.candidate.record_id for entry in entries)
    if len(record_ids) != len(set(record_ids)):
        raise InvalidDenseContractError("entries must have unique RecordIds.")
    if any(len(entry.vector.values) != len(query_vector.values) for entry in entries):
        raise InvalidDenseContractError("all vectors must have matching dimensions.")

    query_norm = math.sqrt(sum(value * value for value in query_vector.values))
    scored = tuple(
        (
            entry,
            sum(
                left * right
                for left, right in zip(query_vector.values, entry.vector.values)
            )
            / (
                query_norm
                * math.sqrt(sum(value * value for value in entry.vector.values))
            ),
        )
        for entry in entries
    )
    ordered = sorted(
        scored,
        key=lambda item: (
            -item[1],
            item[0].candidate.record_id.as_str(),
        ),
    )[:limit]
    return tuple(
        DenseIndexMatch(
            candidate=entry.candidate,
            cosine_similarity=max(-1.0, min(1.0, similarity)),
            rank=rank,
        )
        for rank, (entry, similarity) in enumerate(ordered, start=1)
    )


class ExactDenseVectorIndex:
    """Deterministic in-memory adapter used for contract fixtures."""

    def __init__(self, entries: tuple[DenseIndexEntry, ...]) -> None:
        """Create an exact index over one immutable entry tuple."""
        if not isinstance(entries, tuple):
            raise TypeError("entries must be a tuple.")
        self._entries = entries

    def search(
        self,
        request: DenseIndexQuery,
    ) -> tuple[DenseIndexMatch, ...]:
        """Return exact cosine neighbours in deterministic order."""
        if not isinstance(request, DenseIndexQuery):
            raise TypeError("request must be a DenseIndexQuery.")
        return rank_dense_entries(
            self._entries,
            request.query_vector,
            limit=request.query.limit.value,
        )
