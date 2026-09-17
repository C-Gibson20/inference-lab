"""Explicit mappings from ING-04 chunk values to storage records."""

from typing import Final

from inference_lab.ingest._chunk_records import ChunkSourceReference
from inference_lab.ingest.sources import StructuralBlockKind
from inference_lab.storage.chunk_records import ChunkSourceReferenceRecord
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
