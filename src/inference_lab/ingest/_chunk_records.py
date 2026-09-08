"""Immutable public records produced by the ingest chunking engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import is_strict_integer
from inference_lab.ingest._chunk_identity import (
    _CHUNK_RECORD_KIND,
    CHUNK_PAYLOAD_SCHEMA_VERSION,
    _chunk_output_digest,
    _configuration_identity_from_values,
    _record_id,
    _source_inputs,
    _tokeniser_component,
)
from inference_lab.ingest.chunking import (
    ChunkerVersion,
    ChunkingPolicy,
    TokeniserIdentity,
)
from inference_lab.ingest.sources import StructuralBlockKind
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    CURRENT_LINEAGE_SCHEMA_VERSION,
    ConfigurationIdentity,
    CorpusId,
    DerivedRecordMetadata,
    DocumentId,
    DocumentVersionId,
    InvalidLineageError,
    KnownSourceSpan,
    LineageBundle,
    OperationKind,
    RecordId,
    SourceSpanId,
    build_derivation_fingerprint,
    validate_lineage_bundle,
)

_STRUCTURAL_KINDS: Final[frozenset[str]] = frozenset(
    {"page", "section", "paragraph", "table", "figure_caption", "other"}
)


class ChunkMinimumException(StrEnum):
    """Audited reasons why one chunk may be shorter than the policy minimum."""

    SHORT_DOCUMENT = "short_document"
    PROTECTED_STRUCTURE_NEIGHBOUR = "protected_structure_neighbour"
    FINAL_REMAINDER = "final_remainder_minimum_exception"


class ChunkSplitReason(StrEnum):
    """Audited reasons why a chunk used a non-natural split."""

    OVERSIZE_STRUCTURE = "oversize_structure"


class ChunkingFailureCategory(StrEnum):
    """Bounded, privacy-safe failure categories for one chunking operation."""

    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_STRUCTURE = "unsupported_structure"
    TOKENISER_UNAVAILABLE = "tokeniser_unavailable"
    INVALID_TOKENISER = "invalid_tokeniser"
    UNSAFE_NORMALISATION_BOUNDARY = "unsafe_normalisation_boundary"
    TOKEN_BUDGET_FAILURE = "token_budget_failure"
    UNCOVERED_CONTENT = "uncovered_content"
    INVALID_LINEAGE = "invalid_lineage"
    INVALID_OUTPUT = "invalid_output"


class ChunkingContractError(ValueError):
    """Report a payload-free failure in the public chunking contract."""

    def __init__(self, category: ChunkingFailureCategory) -> None:
        """Create a contract failure without retaining source payloads."""
        super().__init__(category.value)
        self.category = category


@dataclass(frozen=True, slots=True)
class ChunkSourceReference:
    """Reference one clipped parser span intersecting a chunk."""

    source_span_id: SourceSpanId
    document_version_id: DocumentVersionId
    block_ordinal: int
    block_kind: StructuralBlockKind
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the source reference's identity, kind, and range."""
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not is_strict_integer(self.block_ordinal) or self.block_ordinal < 0:
            raise ValueError("block_ordinal must be a non-negative integer.")
        if self.block_kind not in _STRUCTURAL_KINDS:
            raise ValueError("block_kind must be a supported structural kind.")
        if not is_strict_integer(self.source_start) or not is_strict_integer(
            self.source_end
        ):
            raise TypeError("source reference offsets must be integers.")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("source reference must be a non-empty range.")

    @property
    def source_range(self) -> tuple[int, int]:
        """Return the clipped half-open decoded-source range."""
        return self.source_start, self.source_end

    def key(self) -> tuple[int, str, str, int, int]:
        """Return the semantic order used for source references."""
        return (
            self.block_ordinal,
            self.block_kind,
            self.source_span_id.as_str(),
            self.source_start,
            self.source_end,
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One immutable, source-navigable derived retrieval chunk."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    ordinal: int
    text: str
    token_count: int
    source_start: int
    source_end: int
    target_overlap_tokens: int
    overlap_source_start: int
    overlap_source_end: int
    overlap_token_count: int
    source_references: tuple[ChunkSourceReference, ...]
    structural_kinds: tuple[str, ...]
    structural_block_ordinals: tuple[int, ...]
    minimum_exception: ChunkMinimumException | None
    split_reason: ChunkSplitReason | None
    chunker_version: ChunkerVersion
    tokeniser_identity: TokeniserIdentity
    configuration_identity: ConfigurationIdentity
    metadata: DerivedRecordMetadata
    payload_schema_version: str = CHUNK_PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the chunk payload and its provenance metadata together."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not is_strict_integer(self.ordinal) or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("chunk text must not be blank.")
        for field_name, value in (
            ("token_count", self.token_count),
            ("target_overlap_tokens", self.target_overlap_tokens),
            ("overlap_token_count", self.overlap_token_count),
        ):
            if not is_strict_integer(value) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if self.token_count == 0:
            raise ValueError("chunk token_count must be positive.")
        self._validate_range(self.source_start, self.source_end, "source range")
        self._validate_range(
            self.overlap_source_start,
            self.overlap_source_end,
            "overlap source range",
            allow_empty=True,
        )
        if self.overlap_token_count > self.target_overlap_tokens:
            raise ValueError("overlap_token_count must not exceed its target.")
        if not isinstance(self.source_references, tuple):
            raise TypeError("source_references must be a tuple.")
        if any(
            not isinstance(reference, ChunkSourceReference)
            for reference in self.source_references
        ):
            raise TypeError(
                "source_references must contain ChunkSourceReference values."
            )
        reference_keys = tuple(reference.key() for reference in self.source_references)
        if reference_keys != tuple(sorted(reference_keys)):
            raise ValueError("source_references must be in block order.")
        if len(
            {reference.source_span_id for reference in self.source_references}
        ) != len(self.source_references):
            raise ValueError("source_references must not contain duplicates.")
        for reference in self.source_references:
            if reference.document_version_id != self.document_version_id:
                raise ValueError("source reference document version must match chunk.")
            if not (
                self.source_start <= reference.source_start
                and reference.source_end <= self.source_end
            ):
                raise ValueError("source reference must be clipped inside the chunk.")
        self._validate_annotations()
        self._validate_metadata()

    @staticmethod
    def _validate_range(
        start: int,
        end: int,
        field_name: str,
        *,
        allow_empty: bool = False,
    ) -> None:
        """Validate one non-negative half-open range."""
        if not is_strict_integer(start) or not is_strict_integer(end):
            raise TypeError(f"{field_name} offsets must be integers.")
        if start < 0 or (end < start if allow_empty else end <= start):
            raise ValueError(f"{field_name} must be a valid half-open range.")

    def _validate_annotations(self) -> None:
        """Validate structural and minimum/split annotations."""
        if not isinstance(self.structural_kinds, tuple) or any(
            kind not in _STRUCTURAL_KINDS for kind in self.structural_kinds
        ):
            raise ValueError("structural_kinds must contain supported kinds.")
        if not isinstance(self.structural_block_ordinals, tuple):
            raise TypeError("structural_block_ordinals must be a tuple.")
        if any(
            not is_strict_integer(ordinal) or ordinal < 0
            for ordinal in self.structural_block_ordinals
        ):
            raise ValueError("structural block ordinals must be non-negative integers.")
        if len(self.structural_block_ordinals) != len(self.source_references):
            raise ValueError("structural annotations must match source references.")
        if self.structural_block_ordinals != tuple(
            reference.block_ordinal for reference in self.source_references
        ):
            raise ValueError("structural block ordinals must match source references.")
        if self.minimum_exception is not None and not isinstance(
            self.minimum_exception, ChunkMinimumException
        ):
            raise TypeError(
                "minimum_exception must be a ChunkMinimumException or None."
            )
        if self.split_reason is not None and not isinstance(
            self.split_reason, ChunkSplitReason
        ):
            raise TypeError("split_reason must be a ChunkSplitReason or None.")
        if self.payload_schema_version != CHUNK_PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported chunk payload schema version.")

    def _validate_metadata(self) -> None:
        """Require the derived metadata to describe this exact chunk."""
        if not isinstance(self.chunker_version, ChunkerVersion):
            raise TypeError("chunker_version must be a ChunkerVersion.")
        if not isinstance(self.tokeniser_identity, TokeniserIdentity):
            raise TypeError("tokeniser_identity must be a TokeniserIdentity.")
        if not isinstance(self.configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if not isinstance(self.metadata, DerivedRecordMetadata):
            raise TypeError("metadata must be a DerivedRecordMetadata.")
        if self.metadata.record_kind != _CHUNK_RECORD_KIND:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        if self.metadata.lineage_schema_version != CURRENT_LINEAGE_SCHEMA_VERSION:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        operation = self.metadata.operation
        if (
            operation.kind != OperationKind("chunk")
            or operation.implementation != self.chunker_version.implementation
            or operation.contract_version != self.chunker_version.contract_version
            or operation.configuration_identity != self.configuration_identity
        ):
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        expected_component = _tokeniser_component(self.tokeniser_identity)
        if operation.components != (expected_component,):
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        expected_inputs = _source_inputs(
            self.document_version_id, self.source_references
        )
        if self.metadata.inputs != expected_inputs:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        expected_output = _chunk_output_digest(self)
        if self.metadata.output_digest != expected_output:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        expected_fingerprint = build_derivation_fingerprint(
            record_kind=_CHUNK_RECORD_KIND,
            lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
            inputs=expected_inputs,
            operation=operation,
            output_digest=expected_output,
        )
        if self.metadata.derivation_fingerprint != expected_fingerprint:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        expected_record_id = _record_id(expected_fingerprint)
        if self.metadata.record_id != expected_record_id:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)

    @property
    def record_id(self) -> RecordId:
        """Return the deterministic derived-record identifier."""
        return self.metadata.record_id

    @property
    def source_range(self) -> tuple[int, int]:
        """Return the half-open decoded-source range covered by the chunk."""
        return self.source_start, self.source_end

    @property
    def overlap_source_range(self) -> tuple[int, int]:
        """Return the half-open decoded-source range overlapping its predecessor."""
        return self.overlap_source_start, self.overlap_source_end


@dataclass(frozen=True, slots=True)
class ChunkSet:
    """Complete immutable chunk output for one document and corpus operation."""

    corpus_id: CorpusId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    parser_version: ParserVersion
    chunker_version: ChunkerVersion
    tokeniser_identity: TokeniserIdentity
    policy: ChunkingPolicy
    configuration_identity: ConfigurationIdentity
    chunks: tuple[Chunk, ...]
    lineage: LineageBundle
    payload_schema_version: str = CHUNK_PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate collection-level identity, packing, and lineage invariants."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.chunker_version, ChunkerVersion):
            raise TypeError("chunker_version must be a ChunkerVersion.")
        if not isinstance(self.tokeniser_identity, TokeniserIdentity):
            raise TypeError("tokeniser_identity must be a TokeniserIdentity.")
        if not isinstance(self.policy, ChunkingPolicy):
            raise TypeError("policy must be a ChunkingPolicy.")
        if not isinstance(self.configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        if self.payload_schema_version != CHUNK_PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported chunk payload schema version.")
        if self.configuration_identity != _configuration_identity_from_values(
            parser_version=self.parser_version,
            chunker_version=self.chunker_version,
            tokeniser_identity=self.tokeniser_identity,
            policy=self.policy,
        ):
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        if not isinstance(self.chunks, tuple):
            raise TypeError("chunks must be a tuple.")
        if any(not isinstance(chunk, Chunk) for chunk in self.chunks):
            raise TypeError("chunks must contain only Chunk values.")
        if not isinstance(self.lineage, LineageBundle):
            raise TypeError("lineage must be a LineageBundle.")
        self._validate_chunks()
        self._validate_lineage()

    def _validate_chunks(self) -> None:
        """Validate ordered chunk identity, limits, progress, and overlap."""
        previous: Chunk | None = None
        record_ids: set[RecordId] = set()
        for expected_ordinal, chunk in enumerate(self.chunks):
            if chunk.ordinal != expected_ordinal:
                raise ValueError("chunks must have contiguous ordinals.")
            if chunk.document_id != self.document_id:
                raise ValueError("chunk document ID does not match the chunk set.")
            if chunk.document_version_id != self.document_version_id:
                raise ValueError("chunk document version does not match the chunk set.")
            if chunk.chunker_version != self.chunker_version:
                raise ValueError("chunker version does not match the chunk set.")
            if chunk.tokeniser_identity != self.tokeniser_identity:
                raise ValueError("tokeniser identity does not match the chunk set.")
            if chunk.configuration_identity != self.configuration_identity:
                raise ValueError("configuration identity does not match the chunk set.")
            if chunk.token_count > self.policy.max_tokens:
                raise ValueError("chunk exceeds the policy token maximum.")
            if chunk.token_count < self.policy.min_tokens and (
                chunk.minimum_exception is None
            ):
                raise ValueError("undersized chunk lacks a minimum exception.")
            if chunk.token_count >= self.policy.min_tokens and (
                chunk.minimum_exception is not None
            ):
                raise ValueError("minimum exception is invalid for this chunk.")
            if chunk.record_id in record_ids:
                raise ValueError("chunk record IDs must be unique.")
            record_ids.add(chunk.record_id)
            if previous is None:
                if chunk.overlap_token_count != 0 or chunk.overlap_source_range != (
                    0,
                    0,
                ):
                    raise ValueError("the first chunk must not have overlap.")
            else:
                if chunk.source_start > previous.source_end:
                    raise ValueError("chunks must not leave a source gap.")
                if chunk.source_end <= previous.source_end:
                    raise ValueError("chunks must make forward source progress.")
                if chunk.overlap_source_start != chunk.source_start:
                    raise ValueError("overlap must begin at the current chunk start.")
                if chunk.overlap_source_end != previous.source_end:
                    raise ValueError("overlap must end at the previous chunk end.")
                if chunk.overlap_token_count > chunk.target_overlap_tokens:
                    raise ValueError("overlap exceeds its target.")
            previous = chunk

    def _validate_lineage(self) -> None:
        """Require the lineage bundle to exactly cover this chunk set."""
        try:
            validate_lineage_bundle(self.lineage)
        except InvalidLineageError:
            raise ChunkingContractError(
                ChunkingFailureCategory.INVALID_LINEAGE
            ) from None
        expected_records = tuple(
            sorted(
                (chunk.metadata for chunk in self.chunks),
                key=lambda record: record.record_id.as_str(),
            )
        )
        if self.lineage.records != expected_records:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
        expected_sources = tuple(
            sorted(
                {
                    KnownSourceSpan(
                        document_version_id=reference.document_version_id,
                        source_span_id=reference.source_span_id,
                    )
                    for chunk in self.chunks
                    for reference in chunk.source_references
                },
                key=lambda source: source.key(),
            )
        )
        if self.lineage.source_spans != expected_sources:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)


@dataclass(frozen=True, slots=True)
class ChunkingFailure:
    """A bounded failure without partial chunks or sensitive payloads."""

    category: ChunkingFailureCategory

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the failure category is a supported enum value."""
        if not isinstance(self.category, ChunkingFailureCategory):
            raise TypeError("category must be a ChunkingFailureCategory.")


@dataclass(frozen=True, slots=True)
class ChunkingOutcome:
    """All-or-nothing result for one pure chunking request."""

    corpus_id: CorpusId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    chunk_set: ChunkSet | None = None
    failure: ChunkingFailure | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the outcome's identity and exactly one success or failure."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if (self.chunk_set is None) == (self.failure is None):
            raise ValueError("outcome must contain exactly one success or failure.")
        if self.chunk_set is not None:
            if self.chunk_set.corpus_id != self.corpus_id:
                raise ValueError("chunk-set corpus ID does not match the outcome.")
            if self.chunk_set.document_id != self.document_id:
                raise ValueError("chunk-set document ID does not match the outcome.")
            if self.chunk_set.document_version_id != self.document_version_id:
                raise ValueError("chunk-set version does not match the outcome.")

    @property
    def succeeded(self) -> bool:
        """Return whether the operation produced a complete chunk set."""
        return self.chunk_set is not None
