"""Canonical payload and provenance identity construction for chunks."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Final

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab.ingest.chunking import (
    CURRENT_CHUNKER_VERSION,
    ChunkerVersion,
    ChunkingPolicy,
    ChunkingRequest,
    TokeniserIdentity,
)
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    ComponentRole,
    ConfigurationIdentity,
    DerivationFingerprint,
    DerivedRecordKind,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    InputRole,
    LineageInputs,
    OperationKind,
    OutputDigest,
    ProducingComponent,
    ProducingOperation,
    RecordId,
    SourceAnchor,
)

if TYPE_CHECKING:
    from inference_lab.ingest._chunk_records import (
        Chunk,
        ChunkMinimumException,
        ChunkSourceReference,
        ChunkSplitReason,
    )

CHUNK_PAYLOAD_SCHEMA_VERSION: Final[str] = "1.0.0"
_CHUNK_RECORD_NAMESPACE: Final = uuid.UUID("7ca16d7e-1c95-5ab6-9bb1-5cfce1d68f1a")
_CHUNK_RECORD_KIND: Final = DerivedRecordKind("chunk")
_SOURCE_TEXT_ROLE: Final = InputRole("source-text")


def _source_inputs(
    document_version_id: DocumentVersionId,
    references: tuple[ChunkSourceReference, ...],
) -> LineageInputs:
    """Build direct source anchors with stable parser-block ordinals."""
    return LineageInputs(
        source_anchors=tuple(
            SourceAnchor(
                document_version_id=document_version_id,
                source_span_id=reference.source_span_id,
                input_role=_SOURCE_TEXT_ROLE,
                input_ordinal=reference.block_ordinal,
            )
            for reference in references
        )
    )


def _tokeniser_component(identity: TokeniserIdentity) -> ProducingComponent:
    """Translate the complete tokeniser identity into a PROV component."""
    return ProducingComponent(
        role=ComponentRole("tokeniser"),
        implementation=identity.implementation,
        contract_version=identity.contract_version,
        artefact_identity=identity.artefact_identity,
        artefact_revision=identity.artefact_revision,
    )


def _chunk_operation(
    *,
    chunker_version: ChunkerVersion,
    tokeniser_identity: TokeniserIdentity,
    configuration_identity: ConfigurationIdentity,
) -> ProducingOperation:
    """Build the immutable operation metadata shared by every chunk."""
    return ProducingOperation(
        kind=OperationKind("chunk"),
        implementation=chunker_version.implementation,
        contract_version=chunker_version.contract_version,
        configuration_identity=configuration_identity,
        components=(_tokeniser_component(tokeniser_identity),),
    )


def _configuration_identity(
    request: ChunkingRequest,
    tokeniser_identity: TokeniserIdentity,
) -> ConfigurationIdentity:
    """Digest every invalidation-relevant chunking choice without raw payloads."""
    return _configuration_identity_from_values(
        parser_version=request.parsed_document.parser_version,
        chunker_version=CURRENT_CHUNKER_VERSION,
        tokeniser_identity=tokeniser_identity,
        policy=request.policy,
    )


def _configuration_identity_from_values(
    *,
    parser_version: ParserVersion,
    chunker_version: ChunkerVersion,
    tokeniser_identity: TokeniserIdentity,
    policy: ChunkingPolicy,
) -> ConfigurationIdentity:
    """Digest the explicit values that determine one chunk generation."""
    payload: JsonObject = {
        "chunker": {
            "implementation": chunker_version.implementation,
            "contract_version": chunker_version.contract_version,
        },
        "parser": {
            "implementation": parser_version.implementation,
            "contract_version": parser_version.contract_version,
            "normalisation_version": parser_version.normalisation_version,
        },
        "policy": {
            "min_tokens": policy.min_tokens,
            "max_tokens": policy.max_tokens,
            "overlap_numerator": policy.overlap_numerator,
            "overlap_denominator": policy.overlap_denominator,
            "structural_boundary_policy": policy.structural_boundary_policy.value,
            "oversize_unit_policy": policy.oversize_unit_policy.value,
        },
        "tokeniser": {
            "implementation": tokeniser_identity.implementation,
            "contract_version": tokeniser_identity.contract_version,
            "artefact_identity": tokeniser_identity.artefact_identity,
            "artefact_revision": tokeniser_identity.artefact_revision,
            "options": [list(option) for option in tokeniser_identity.options],
        },
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return ConfigurationIdentity(algorithm=DigestAlgorithm.SHA256, digest=digest)


def derive_chunk_configuration_identity(
    *,
    parser_version: ParserVersion,
    chunker_version: ChunkerVersion,
    tokeniser_identity: TokeniserIdentity,
    policy: ChunkingPolicy,
) -> ConfigurationIdentity:
    """Derive the public, replay-stable identity for an ingest chunk configuration."""
    if not isinstance(parser_version, ParserVersion):
        raise TypeError("parser_version must be a ParserVersion.")
    if not isinstance(chunker_version, ChunkerVersion):
        raise TypeError("chunker_version must be a ChunkerVersion.")
    if not isinstance(tokeniser_identity, TokeniserIdentity):
        raise TypeError("tokeniser_identity must be a TokeniserIdentity.")
    if not isinstance(policy, ChunkingPolicy):
        raise TypeError("policy must be a ChunkingPolicy.")
    return _configuration_identity_from_values(
        parser_version=parser_version,
        chunker_version=chunker_version,
        tokeniser_identity=tokeniser_identity,
        policy=policy,
    )


def _chunk_output_digest(chunk: Chunk) -> OutputDigest:
    """Digest the canonical domain-owned chunk payload, excluding metadata."""
    return _output_digest_for_payload(_chunk_payload(chunk))


def _chunk_output_digest_for_values(
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    ordinal: int,
    text: str,
    token_count: int,
    source_start: int,
    source_end: int,
    target_overlap_tokens: int,
    overlap_source_start: int,
    overlap_source_end: int,
    overlap_token_count: int,
    source_references: tuple[ChunkSourceReference, ...],
    structural_kinds: tuple[str, ...],
    structural_block_ordinals: tuple[int, ...],
    minimum_exception: ChunkMinimumException | None,
    split_reason: ChunkSplitReason | None,
    payload_schema_version: str = CHUNK_PAYLOAD_SCHEMA_VERSION,
) -> OutputDigest:
    """Digest chunk values before derived metadata exists."""
    return _output_digest_for_payload(
        _chunk_payload_values(
            document_id=document_id,
            document_version_id=document_version_id,
            ordinal=ordinal,
            text=text,
            token_count=token_count,
            source_start=source_start,
            source_end=source_end,
            target_overlap_tokens=target_overlap_tokens,
            overlap_source_start=overlap_source_start,
            overlap_source_end=overlap_source_end,
            overlap_token_count=overlap_token_count,
            source_references=source_references,
            structural_kinds=structural_kinds,
            structural_block_ordinals=structural_block_ordinals,
            minimum_exception=minimum_exception,
            split_reason=split_reason,
            payload_schema_version=payload_schema_version,
        )
    )


def _chunk_payload(chunk: Chunk) -> JsonObject:
    """Return the canonical domain-owned payload for one chunk."""
    return _chunk_payload_values(
        document_id=chunk.document_id,
        document_version_id=chunk.document_version_id,
        ordinal=chunk.ordinal,
        text=chunk.text,
        token_count=chunk.token_count,
        source_start=chunk.source_start,
        source_end=chunk.source_end,
        target_overlap_tokens=chunk.target_overlap_tokens,
        overlap_source_start=chunk.overlap_source_start,
        overlap_source_end=chunk.overlap_source_end,
        overlap_token_count=chunk.overlap_token_count,
        source_references=chunk.source_references,
        structural_kinds=chunk.structural_kinds,
        structural_block_ordinals=chunk.structural_block_ordinals,
        minimum_exception=chunk.minimum_exception,
        split_reason=chunk.split_reason,
        payload_schema_version=chunk.payload_schema_version,
    )


def _chunk_payload_values(
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    ordinal: int,
    text: str,
    token_count: int,
    source_start: int,
    source_end: int,
    target_overlap_tokens: int,
    overlap_source_start: int,
    overlap_source_end: int,
    overlap_token_count: int,
    source_references: tuple[ChunkSourceReference, ...],
    structural_kinds: tuple[str, ...],
    structural_block_ordinals: tuple[int, ...],
    minimum_exception: ChunkMinimumException | None,
    split_reason: ChunkSplitReason | None,
    payload_schema_version: str,
) -> JsonObject:
    """Build the canonical JSON-compatible chunk payload."""
    return {
        "document_id": document_id.as_str(),
        "document_version_id": document_version_id.as_str(),
        "ordinal": ordinal,
        "text": text,
        "token_count": token_count,
        "source_range": {"start": source_start, "end": source_end},
        "overlap": {
            "target_tokens": target_overlap_tokens,
            "source_start": overlap_source_start,
            "source_end": overlap_source_end,
            "token_count": overlap_token_count,
        },
        "source_references": [
            {
                "source_span_id": reference.source_span_id.as_str(),
                "document_version_id": reference.document_version_id.as_str(),
                "block_ordinal": reference.block_ordinal,
                "block_kind": reference.block_kind,
                "source_start": reference.source_start,
                "source_end": reference.source_end,
            }
            for reference in source_references
        ],
        "structural_kinds": list(structural_kinds),
        "structural_block_ordinals": list(structural_block_ordinals),
        "minimum_exception": (
            minimum_exception.value if minimum_exception is not None else None
        ),
        "split_reason": split_reason.value if split_reason is not None else None,
        "payload_schema_version": payload_schema_version,
    }


def _output_digest_for_payload(payload: JsonObject) -> OutputDigest:
    """Hash one canonical chunk payload."""
    return OutputDigest(
        algorithm=DigestAlgorithm.SHA256,
        digest=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )


def _record_id(fingerprint: DerivationFingerprint) -> RecordId:
    """Derive the stable chunk record ID from its complete fingerprint."""
    name = f"{_CHUNK_RECORD_KIND.as_str()}:{fingerprint.as_str()}"
    return RecordId.from_str(str(uuid.uuid5(_CHUNK_RECORD_NAMESPACE, name)))
