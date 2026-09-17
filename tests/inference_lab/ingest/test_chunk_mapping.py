"""Tests for the explicit ING-04-to-storage chunk mapping boundary."""

import pytest

from inference_lab.ingest import (
    ChunkSourceReference,
    map_chunk_source_reference,
)
from inference_lab.provenance import DocumentVersionId, SourceSpanId
from inference_lab.storage import StructuralBlockKind as StorageStructuralBlockKind

_DOCUMENT_VERSION_ID = DocumentVersionId("11111111-1111-4111-8111-111111111111")
_SOURCE_SPAN_ID = SourceSpanId("22222222-2222-4222-8222-222222222222")


def _reference() -> ChunkSourceReference:
    return ChunkSourceReference(
        source_span_id=_SOURCE_SPAN_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        block_ordinal=2,
        block_kind="paragraph",
        source_start=10,
        source_end=20,
    )


def test_map_chunk_source_reference_preserves_identity_and_range() -> None:
    mapped = map_chunk_source_reference(_reference())

    assert mapped.source_span_id == _SOURCE_SPAN_ID
    assert mapped.document_version_id == _DOCUMENT_VERSION_ID
    assert mapped.block_ordinal == 2
    assert mapped.block_kind is StorageStructuralBlockKind.PARAGRAPH
    assert mapped.source_range == (10, 20)


def test_map_chunk_source_reference_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="reference"):
        map_chunk_source_reference(object())  # type: ignore[arg-type]
