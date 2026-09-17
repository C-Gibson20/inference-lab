"""Storage-owned immutable records for canonical ingest chunks."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import (
    is_canonical_semantic_version,
    is_lowercase_slug,
    is_strict_integer,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    SourceSpanId,
)
from inference_lab.storage.source_locations import StructuralBlockKind

CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION: Final[str] = "1.0.0"


class ChunkStructuralBoundaryPolicy(StrEnum):
    """Supported structure-boundary policies for persisted chunking."""

    GREEDY = "structure-aware-greedy"


class ChunkOversizeUnitPolicy(StrEnum):
    """Supported policies for structures exceeding the token ceiling."""

    TOKEN_WINDOW = "token-window"


class ChunkMinimumExceptionRecord(StrEnum):
    """Audited reasons a persisted chunk may be below the policy minimum."""

    SHORT_DOCUMENT = "short_document"
    PROTECTED_STRUCTURE_NEIGHBOUR = "protected_structure_neighbour"
    FINAL_REMAINDER = "final_remainder_minimum_exception"


class ChunkSplitReasonRecord(StrEnum):
    """Audited reasons a persisted chunk used a non-natural split."""

    OVERSIZE_STRUCTURE = "oversize_structure"


def _validate_implementation(value: object, field_name: str) -> str:
    """Validate one lowercase implementation slug."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_lowercase_slug(value):
        raise ValueError(f"{field_name} must be a lowercase slug.")
    return value


def _validate_semantic_version(value: object, field_name: str) -> str:
    """Validate one canonical semantic version."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise ValueError(f"{field_name} must be a semantic version.")
    return value


def _validate_non_path_identifier(value: object, field_name: str) -> str:
    """Validate one non-empty identifier that cannot contain path syntax."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} must be a non-path identifier.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkSourceReferenceRecord:
    """Locate one parser span clipped to a persisted chunk."""

    source_span_id: SourceSpanId
    document_version_id: DocumentVersionId
    block_ordinal: int
    block_kind: StructuralBlockKind
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate identity, structural kind, and non-empty source range."""
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not is_strict_integer(self.block_ordinal):
            raise TypeError("block_ordinal must be an integer.")
        if self.block_ordinal < 0:
            raise ValueError("block_ordinal must not be negative.")
        if not isinstance(self.block_kind, StructuralBlockKind):
            raise TypeError("block_kind must be a StructuralBlockKind.")
        if not is_strict_integer(self.source_start):
            raise TypeError("source_start must be an integer.")
        if not is_strict_integer(self.source_end):
            raise TypeError("source_end must be an integer.")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("source range must be non-empty and ordered.")

    @property
    def source_range(self) -> tuple[int, int]:
        """Return the half-open source range."""
        return self.source_start, self.source_end

    def key(self) -> tuple[int, str, str, int, int]:
        """Return the canonical ordering key for source references."""
        return (
            self.block_ordinal,
            self.block_kind.value,
            self.source_span_id.as_str(),
            self.source_start,
            self.source_end,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkParserVersionRecord:
    """Identify the parser and normalisation contract used by a chunk set."""

    implementation: str
    contract_version: str
    normalisation_version: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate all parser identity components."""
        _validate_implementation(self.implementation, "implementation")
        _validate_semantic_version(self.contract_version, "contract_version")
        _validate_semantic_version(self.normalisation_version, "normalisation_version")

    @property
    def value(self) -> str:
        """Return the canonical parser identity."""
        return (
            f"{self.implementation}/"
            f"contract-{self.contract_version}/"
            f"normalisation-{self.normalisation_version}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkerVersionRecord:
    """Identify the chunker implementation and contract used by a chunk
    set."""

    implementation: str
    contract_version: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the chunker identity components."""
        _validate_implementation(self.implementation, "implementation")
        _validate_semantic_version(self.contract_version, "contract_version")

    @property
    def value(self) -> str:
        """Return the canonical chunker identity."""
        return f"{self.implementation}/contract-{self.contract_version}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkTokeniserIdentityRecord:
    """Identify every versioned input that affects tokenisation."""

    implementation: str
    contract_version: str
    artefact_identity: str
    artefact_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate implementation, artefact, and canonical options."""
        _validate_implementation(self.implementation, "implementation")
        _validate_semantic_version(self.contract_version, "contract_version")
        _validate_non_path_identifier(
            self.artefact_identity,
            "artefact_identity",
        )
        _validate_non_path_identifier(
            self.artefact_revision,
            "artefact_revision",
        )

        if not isinstance(self.options, tuple):
            raise TypeError("options must be a tuple.")

        option_keys: list[str] = []
        for option in self.options:
            if (
                not isinstance(option, tuple)
                or len(option) != 2
                or not isinstance(option[0], str)
                or not isinstance(option[1], str)
            ):
                raise TypeError("options must contain string pairs.")

            key, value = option
            if not key or key.lower() != key or "\x00" in key or "=" in key:
                raise ValueError("option keys must be lowercase names.")
            if not value or "\x00" in value or "/" in value or "\\" in value:
                raise ValueError("option values must be path-free.")
            option_keys.append(key)

        if tuple(option_keys) != tuple(sorted(set(option_keys))):
            raise ValueError("options must be sorted and unique.")

    def as_str(self) -> str:
        """Return the canonical path-free tokeniser identity."""
        options = ",".join(f"{key}={value}" for key, value in self.options)
        return (
            f"{self.implementation}/contract-{self.contract_version}/"
            f"artefact-{self.artefact_identity}@{self.artefact_revision}/"
            f"options-{options}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingPolicyRecord:
    """Persist the complete policy that determines chunk boundaries."""

    min_tokens: int
    max_tokens: int
    overlap_numerator: int
    overlap_denominator: int
    structural_boundary_policy: ChunkStructuralBoundaryPolicy
    oversize_unit_policy: ChunkOversizeUnitPolicy

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate token limits, overlap ratio, and closed policy values."""
        for field_name in (
            "min_tokens",
            "max_tokens",
            "overlap_numerator",
            "overlap_denominator",
        ):
            value = getattr(self, field_name)
            if not is_strict_integer(value):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative.")

        if self.min_tokens == 0:
            raise ValueError("min_tokens must be positive.")
        if self.max_tokens == 0:
            raise ValueError("max_tokens must be positive.")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens.")
        if self.overlap_denominator == 0:
            raise ValueError("overlap_denominator must be positive.")
        if self.overlap_numerator > self.overlap_denominator:
            raise ValueError("overlap_numerator must not exceed overlap_denominator.")
        if not isinstance(
            self.structural_boundary_policy,
            ChunkStructuralBoundaryPolicy,
        ):
            raise TypeError(
                "structural_boundary_policy must be a ChunkStructuralBoundaryPolicy."
            )
        if not isinstance(
            self.oversize_unit_policy,
            ChunkOversizeUnitPolicy,
        ):
            raise TypeError("oversize_unit_policy must be a ChunkOversizeUnitPolicy.")

    def key(self) -> tuple[int, int, int, int, str, str]:
        """Return the canonical policy identity key."""
        return (
            self.min_tokens,
            self.max_tokens,
            self.overlap_numerator,
            self.overlap_denominator,
            self.structural_boundary_policy.value,
            self.oversize_unit_policy.value,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkSetSelectionRecord:
    """Represent one corpus-authorised exact chunk-set selection."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    parser_version: ChunkParserVersionRecord
    chunker_version: ChunkerVersionRecord
    tokeniser_identity: ChunkTokeniserIdentityRecord
    policy: ChunkingPolicyRecord
    configuration_identity: ConfigurationIdentity
    payload_digest: OutputDigest
    canonical_payload: bytes
    payload_schema_version: str = CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the exact selection metadata and detached payload."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.parser_version, ChunkParserVersionRecord):
            raise TypeError("parser_version must be a ChunkParserVersionRecord.")
        if not isinstance(self.chunker_version, ChunkerVersionRecord):
            raise TypeError("chunker_version must be a ChunkerVersionRecord.")
        if not isinstance(
            self.tokeniser_identity,
            ChunkTokeniserIdentityRecord,
        ):
            raise TypeError(
                "tokeniser_identity must be a ChunkTokeniserIdentityRecord."
            )
        if not isinstance(self.policy, ChunkingPolicyRecord):
            raise TypeError("policy must be a ChunkingPolicyRecord.")
        if not isinstance(self.configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if not isinstance(self.payload_digest, OutputDigest):
            raise TypeError("payload_digest must be an OutputDigest.")
        if not isinstance(self.canonical_payload, bytes):
            raise TypeError("canonical_payload must be bytes.")
        if not self.canonical_payload:
            raise ValueError("canonical_payload must not be empty.")
        if hashlib.sha256(self.canonical_payload).hexdigest() != (
            self.payload_digest.digest
        ):
            raise ValueError("payload_digest does not match canonical_payload.")
        if self.payload_schema_version != CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported chunk payload schema version.")
