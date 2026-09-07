"""Tests for immutable provenance-backed chunk records."""

from collections.abc import Iterable
from dataclasses import replace

import pytest

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    ChunkingOutcome,
    ChunkingPolicy,
    ChunkingRequest,
    ChunkMinimumException,
    ChunkSourceReference,
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    build_chunk_set,
    build_chunking_outcome,
    generate_source_span_id,
)
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.tokenisation import FakeChunkTokeniser
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    InputRole,
    LineageBundle,
    SourceSpanId,
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


def test_build_chunk_set_attaches_exact_source_lineage() -> None:
    source = "alpha beta gamma"
    document = _document(source, [("paragraph", 0, len(source))])

    chunk_set = build_chunk_set(_request(document))

    assert chunk_set.corpus_id.as_str().endswith("203")
    assert chunk_set.document_id == document.document_id
    assert chunk_set.document_version_id == document.document_version_id
    assert chunk_set.parser_version == CURRENT_PARSER_VERSION
    assert chunk_set.chunker_version == CURRENT_CHUNKER_VERSION
    assert len(chunk_set.chunks) == 1
    chunk = chunk_set.chunks[0]
    assert chunk.text == normalise_source_text(source)
    assert chunk.minimum_exception is ChunkMinimumException.SHORT_DOCUMENT
    assert chunk.source_range == (0, len(source))
    assert chunk.source_references == (
        ChunkSourceReference(
            source_span_id=document.blocks[0].source_span.source_span_id,
            document_version_id=document.document_version_id,
            block_ordinal=0,
            block_kind="paragraph",
            source_start=0,
            source_end=len(source),
        ),
    )
    assert chunk.metadata.inputs.source_anchors[0].input_role == InputRole(
        "source-text"
    )
    assert chunk.metadata.inputs.source_anchors[0].input_ordinal == 0
    assert chunk_set.lineage.records == (chunk.metadata,)
    assert len(chunk_set.lineage.edges) == 1
    assert chunk.record_id == chunk.metadata.record_id


def test_repeated_runs_and_corpus_memberships_keep_chunk_identity_stable() -> None:
    source = _words(600)
    document = _document(source, [("paragraph", 0, len(source))])

    first = build_chunk_set(_request(document))
    replay = build_chunk_set(_request(document))
    other_corpus = build_chunk_set(
        _request(document, corpus_value="00000000-0000-4000-8000-000000000204")
    )

    assert first == replay
    assert tuple(chunk.record_id for chunk in first.chunks) == tuple(
        chunk.record_id for chunk in other_corpus.chunks
    )
    assert first.configuration_identity == other_corpus.configuration_identity
    assert first.corpus_id != other_corpus.corpus_id


def test_policy_change_invalidates_configuration_and_chunk_records() -> None:
    source = _words(600)
    document = _document(source, [("paragraph", 0, len(source))])

    baseline = build_chunk_set(_request(document))
    changed = build_chunk_set(_request(document, policy=ChunkingPolicy(min_tokens=128)))

    assert baseline.configuration_identity != changed.configuration_identity
    assert tuple(chunk.record_id for chunk in baseline.chunks) != tuple(
        chunk.record_id for chunk in changed.chunks
    )


def test_source_references_are_clipped_to_each_chunk() -> None:
    first = _words(350)
    second = _words(350)
    source = f"{first}\n{second}"
    first_end = len(first)
    second_start = first_end + 1
    document = _document(
        source,
        (
            ("paragraph", 0, first_end),
            ("paragraph", second_start, len(source)),
        ),
    )

    chunk_set = build_chunk_set(_request(document))

    assert chunk_set.chunks
    for chunk in chunk_set.chunks:
        for reference in chunk.source_references:
            block = document.blocks[reference.block_ordinal]
            assert reference.source_start == max(
                chunk.source_start,
                block.source_span.start_offset,
            )
            assert reference.source_end == min(
                chunk.source_end,
                block.source_span.end_offset,
            )
            assert chunk.source_start <= reference.source_start
            assert reference.source_end <= chunk.source_end


def test_failed_outcome_contains_no_partial_chunks_or_lineage() -> None:
    source = "private secret"
    document = _document(source, ())

    outcome = build_chunking_outcome(_request(document))

    assert isinstance(outcome, ChunkingOutcome)
    assert not outcome.succeeded
    assert outcome.chunk_set is None
    assert outcome.failure is not None
    assert outcome.failure.category.value == "uncovered_content"
    assert "private secret" not in str(outcome)


def test_chunk_set_rejects_missing_lineage_records() -> None:
    source = _words(10)
    document = _document(source, [("paragraph", 0, len(source))])
    chunk_set = build_chunk_set(_request(document))

    with pytest.raises(ValueError, match="lineage"):
        replace(chunk_set, lineage=LineageBundle())


def test_chunk_source_reference_rejects_empty_ranges() -> None:
    document_version_id = DocumentVersionId.generate()
    with pytest.raises(ValueError, match="non-empty"):
        ChunkSourceReference(
            source_span_id=SourceSpanId.generate(),
            document_version_id=document_version_id,
            block_ordinal=0,
            block_kind="paragraph",
            source_start=2,
            source_end=2,
        )
