"""Explicit mappings from ING-04 chunk values to storage records."""

import hashlib
from typing import Final

from inference_lab._json import canonical_json_bytes
from inference_lab.ingest._chunk_records import ChunkSet, ChunkSourceReference
from inference_lab.ingest.chunk_serialisation import chunk_set_payload
from inference_lab.ingest.chunking import (
    ChunkerVersion,
    ChunkingPolicy,
    OversizeUnitPolicy,
    StructuralBoundaryPolicy,
    TokeniserIdentity,
)
from inference_lab.ingest.sources import StructuralBlockKind
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import DigestAlgorithm, OutputDigest
from inference_lab.storage.chunk_records import (
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSetSelectionRecord,
    ChunkSourceReferenceRecord,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
)
from inference_lab.storage.source_locations import (
    StructuralBlockKind as StorageStructuralBlockKind,
)


class ChunkMappingError(ValueError):
    """Report a payload-free failure at the ING-04 storage boundary."""


_STRUCTURAL_BLOCK_KIND_MAP: Final[
    dict[StructuralBlockKind, StorageStructuralBlockKind]
] = {
    "page": StorageStructuralBlockKind.PAGE,
    "section": StorageStructuralBlockKind.SECTION,
    "paragraph": StorageStructuralBlockKind.PARAGRAPH,
    "table": StorageStructuralBlockKind.TABLE,
    "figure_caption": StorageStructuralBlockKind.FIGURE_CAPTION,
    "other": StorageStructuralBlockKind.OTHER,
}

_STRUCTURAL_BOUNDARY_POLICY_MAP: Final[
    dict[StructuralBoundaryPolicy, ChunkStructuralBoundaryPolicy]
] = {
    StructuralBoundaryPolicy.GREEDY: ChunkStructuralBoundaryPolicy.GREEDY,
}

_OVERSIZE_UNIT_POLICY_MAP: Final[dict[OversizeUnitPolicy, ChunkOversizeUnitPolicy]] = {
    OversizeUnitPolicy.TOKEN_WINDOW: ChunkOversizeUnitPolicy.TOKEN_WINDOW,
}


def map_chunk_source_reference(
    reference: ChunkSourceReference,
) -> ChunkSourceReferenceRecord:
    """Map one validated ING-04 source reference to a storage record."""
    if not isinstance(reference, ChunkSourceReference):
        raise TypeError("reference must be a ChunkSourceReference.")
    try:
        block_kind = _STRUCTURAL_BLOCK_KIND_MAP[reference.block_kind]
    except KeyError as error:
        raise ChunkMappingError(
            "the chunk source reference has an unsupported block kind."
        ) from error

    try:
        return ChunkSourceReferenceRecord(
            source_span_id=reference.source_span_id,
            document_version_id=reference.document_version_id,
            block_ordinal=reference.block_ordinal,
            block_kind=block_kind,
            source_start=reference.source_start,
            source_end=reference.source_end,
        )
    except (TypeError, ValueError) as error:
        raise ChunkMappingError(
            "the chunk source reference cannot be represented by storage."
        ) from error


def map_chunk_set(chunk_set: ChunkSet) -> ChunkSetSelectionRecord:
    """Map one validated ING-04 chunk set to its corpus-free storage record."""
    if not isinstance(chunk_set, ChunkSet):
        raise TypeError("chunk_set must be a ChunkSet.")

    try:
        payload = dict(chunk_set_payload(chunk_set))
        del payload["corpus_id"]
        canonical_payload = canonical_json_bytes(payload)
        return ChunkSetSelectionRecord(
            document_id=chunk_set.document_id,
            document_version_id=chunk_set.document_version_id,
            parser_version=_map_parser_version(chunk_set.parser_version),
            chunker_version=_map_chunker_version(chunk_set.chunker_version),
            tokeniser_identity=_map_tokeniser_identity(chunk_set.tokeniser_identity),
            policy=_map_policy(chunk_set.policy),
            configuration_identity=chunk_set.configuration_identity,
            payload_digest=OutputDigest(
                algorithm=DigestAlgorithm.SHA256,
                digest=hashlib.sha256(canonical_payload).hexdigest(),
            ),
            canonical_payload=canonical_payload,
            payload_schema_version=chunk_set.payload_schema_version,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ChunkMappingError(
            "the chunk set cannot be represented by storage."
        ) from error


def _map_parser_version(version: ParserVersion) -> ChunkParserVersionRecord:
    """Map the parser implementation and both parser contract versions."""
    return ChunkParserVersionRecord(
        implementation=version.implementation,
        contract_version=version.contract_version,
        normalisation_version=version.normalisation_version,
    )


def _map_chunker_version(version: ChunkerVersion) -> ChunkerVersionRecord:
    """Map the chunker implementation and contract version."""
    return ChunkerVersionRecord(
        implementation=version.implementation,
        contract_version=version.contract_version,
    )


def _map_tokeniser_identity(
    identity: TokeniserIdentity,
) -> ChunkTokeniserIdentityRecord:
    """Map all tokeniser identity fields, including canonical options."""
    return ChunkTokeniserIdentityRecord(
        implementation=identity.implementation,
        contract_version=identity.contract_version,
        artefact_identity=identity.artefact_identity,
        artefact_revision=identity.artefact_revision,
        options=identity.options,
    )


def _map_policy(policy: ChunkingPolicy) -> ChunkingPolicyRecord:
    """Map the complete chunking policy without widening its closed values."""
    return ChunkingPolicyRecord(
        min_tokens=policy.min_tokens,
        max_tokens=policy.max_tokens,
        overlap_numerator=policy.overlap_numerator,
        overlap_denominator=policy.overlap_denominator,
        structural_boundary_policy=_STRUCTURAL_BOUNDARY_POLICY_MAP[
            policy.structural_boundary_policy
        ],
        oversize_unit_policy=_OVERSIZE_UNIT_POLICY_MAP[policy.oversize_unit_policy],
    )
