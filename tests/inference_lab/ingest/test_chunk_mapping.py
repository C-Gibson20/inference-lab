"""Tests for the explicit ING-04-to-storage chunk mapping boundary."""

import hashlib

import pytest

from inference_lab._json import canonical_json_bytes
from inference_lab.ingest import (
    ChunkingRequest,
    ChunkMappingError,
    ChunkSet,
    ChunkSourceReference,
    FakeChunkTokeniser,
    build_chunk_set,
    chunk_set_payload,
    map_chunk_set,
    map_chunk_source_reference,
)
from inference_lab.provenance import CorpusId, DocumentVersionId, SourceSpanId
from inference_lab.storage import StructuralBlockKind as StorageStructuralBlockKind

from .fixtures import fixture_parsed_document  # noqa: TID252

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


def _chunk_set() -> ChunkSet:
    """Build a complete deterministic ING-04 chunk set."""
    return build_chunk_set(
        ChunkingRequest(
            corpus_id=_corpus_id("301"),
            parsed_document=fixture_parsed_document(),
            tokeniser=FakeChunkTokeniser(),
        )
    )


def _corpus_id(suffix: str) -> CorpusId:
    """Return a stable test corpus identifier."""
    return CorpusId.from_str(f"00000000-0000-4000-8000-000000000{suffix}")


def test_map_chunk_set_preserves_selection_and_excludes_corpus_identity() -> None:
    chunk_set = _chunk_set()

    mapped = map_chunk_set(chunk_set)
    expected_payload = dict(chunk_set_payload(chunk_set))
    del expected_payload["corpus_id"]
    expected_bytes = canonical_json_bytes(expected_payload)

    assert mapped.document_id == chunk_set.document_id
    assert mapped.document_version_id == chunk_set.document_version_id
    assert (
        mapped.parser_version.implementation == chunk_set.parser_version.implementation
    )
    assert mapped.chunker_version.contract_version == (
        chunk_set.chunker_version.contract_version
    )
    assert mapped.tokeniser_identity.options == chunk_set.tokeniser_identity.options
    assert mapped.policy.min_tokens == chunk_set.policy.min_tokens
    assert mapped.configuration_identity == chunk_set.configuration_identity
    assert mapped.canonical_payload == expected_bytes
    assert b"corpus_id" not in mapped.canonical_payload
    assert mapped.payload_digest.digest == hashlib.sha256(expected_bytes).hexdigest()


def test_map_chunk_set_rejects_wrong_input_type_without_payload_leakage() -> None:
    with pytest.raises(TypeError, match="chunk_set"):
        map_chunk_set(object())  # type: ignore[arg-type]


def test_map_chunk_set_wraps_unrepresentable_storage_values(monkeypatch) -> None:
    chunk_set = _chunk_set()

    monkeypatch.setattr(
        "inference_lab.ingest.chunk_mapping.ChunkSetSelectionRecord",
        lambda **_: (_ for _ in ()).throw(ValueError("payload detail")),
    )

    with pytest.raises(ChunkMappingError, match="cannot be represented") as error:
        map_chunk_set(chunk_set)
    assert "payload detail" not in str(error.value)
