"""Tests for immutable provenance-backed chunk records."""

from collections.abc import Iterable
from dataclasses import replace

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ChunkingPolicy,
    ChunkingRequest,
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    build_chunk_set,
    generate_source_span_id,
)
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.tokenisation import FakeChunkTokeniser
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    LineageBundle,
)


def _document(
    source_text: str,
    spans: Iterable[tuple[str, int, int]],
) -> ParsedDocument:
    """Build a valid parser document with deterministic structural spans."""
    document_id = DocumentId.from_str("00000000-0000-4000-8000-000000000201")
    document_version_id = DocumentVersionId.from_str(
        "00000000-0000-4000-8000-000000000202"
    )
    blocks = tuple(
        StructuralBlock(
            kind=kind,  # type: ignore[arg-type]
            text=normalise_source_text(source_text[start:end]),
            source_span=SourceSpan(
                source_span_id=generate_source_span_id(
                    document_version_id,
                    parser_version=CURRENT_PARSER_VERSION,
                    ordinal=ordinal,
                    kind=kind,
                    start_offset=start,
                    end_offset=end,
                ),
                document_version_id=document_version_id,
                start_offset=start,
                end_offset=end,
            ),
        )
        for ordinal, (kind, start, end) in enumerate(spans)
    )
    return ParsedDocument(
        document_id=document_id,
        document_version_id=document_version_id,
        text=NormalisedText(source_text),
        blocks=blocks,
        pages=tuple(block for block in blocks if block.kind == "page") or None,
        sections=tuple(block for block in blocks if block.kind == "section") or None,
        paragraphs=tuple(block for block in blocks if block.kind == "paragraph")
        or None,
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(normalise_source_text(source_text)),
        token_count=None,
    )


def _request(
    document: ParsedDocument,
    *,
    corpus_value: str = "00000000-0000-4000-8000-000000000203",
    policy: ChunkingPolicy | None = None,
) -> ChunkingRequest:
    """Build a corpus-scoped fake-tokeniser request."""
    return ChunkingRequest(
        corpus_id=CorpusId.from_str(corpus_value),
        parsed_document=document,
        tokeniser=FakeChunkTokeniser(),
        policy=policy or ChunkingPolicy(),
    )


def _words(count: int) -> str:
    """Build deterministic whitespace-tokenised source text."""
    return " ".join(f"word{index}" for index in range(count))


def test_policy_change_invalidates_configuration_and_chunk_records() -> None:
    source = _words(600)
    document = _document(source, [("paragraph", 0, len(source))])

    baseline = build_chunk_set(_request(document))
    changed = build_chunk_set(_request(document, policy=ChunkingPolicy(min_tokens=128)))

    assert baseline.configuration_identity != changed.configuration_identity
    assert tuple(chunk.record_id for chunk in baseline.chunks) != tuple(
        chunk.record_id for chunk in changed.chunks
    )


def test_chunk_set_rejects_missing_lineage_records() -> None:
    source = _words(10)
    document = _document(source, [("paragraph", 0, len(source))])
    chunk_set = build_chunk_set(_request(document))

    with pytest.raises(ValueError, match="lineage"):
        replace(chunk_set, lineage=LineageBundle())
