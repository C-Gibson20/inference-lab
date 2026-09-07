"""Tests for the private deterministic structural-window engine."""

from collections.abc import Iterable

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ChunkingRequest,
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    build_normalisation_boundary_map,
    generate_source_span_id,
)
from inference_lab.ingest._chunk_windows import (
    ChunkWindowError,
    build_chunk_drafts,
)
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.tokenisation import FakeChunkTokeniser
from inference_lab.provenance import CorpusId, DocumentId, DocumentVersionId


def _parsed_document(
    source_text: str,
    spans: Iterable[tuple[str, int, int]],
) -> ParsedDocument:
    document_id = DocumentId.from_str("00000000-0000-4000-8000-000000000101")
    document_version_id = DocumentVersionId.from_str(
        "00000000-0000-4000-8000-000000000102"
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
        paragraphs=tuple(block for block in blocks if block.kind == "paragraph"),
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(normalise_source_text(source_text)),
        token_count=None,
    )


def _request(document: ParsedDocument) -> ChunkingRequest:
    return ChunkingRequest(
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000103"),
        parsed_document=document,
        tokeniser=FakeChunkTokeniser(),
    )


def _words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


def test_long_flat_text_rebalances_a_short_final_tail() -> None:
    source = _words(600)
    drafts = build_chunk_drafts(
        _request(_parsed_document(source, [("other", 0, len(source))]))
    )

    assert len(drafts) == 2
    assert all(256 <= draft.token_count <= 512 for draft in drafts)
    assert drafts[1].overlap_token_count > 0
    assert drafts[1].overlap_token_count <= drafts[1].target_overlap_tokens
    assert all(draft.token_count <= 512 for draft in drafts)
    assert all(draft.minimum_exception is None for draft in drafts)


def test_long_flat_text_uses_exact_standalone_counts_and_progress() -> None:
    source = _words(1200)
    drafts = build_chunk_drafts(
        _request(_parsed_document(source, [("other", 0, len(source))]))
    )

    assert all(0 < draft.token_count <= 512 for draft in drafts)
    assert all(
        current.source_start <= previous.source_end
        and current.source_end > previous.source_end
        for previous, current in zip(drafts, drafts[1:])
    )
    assert all(
        current.overlap_token_count <= current.target_overlap_tokens
        for current in drafts[1:]
    )


def test_nested_structural_overlays_do_not_duplicate_source_text() -> None:
    source = _words(600)
    paragraph_end = source.index("word300")
    spans = (
        ("page", 0, len(source)),
        ("section", 0, len(source)),
        ("paragraph", 0, paragraph_end),
        ("paragraph", paragraph_end + 1, len(source)),
    )
    document = _parsed_document(source, spans)
    drafts = build_chunk_drafts(_request(document))

    assert drafts
    reconstructed = drafts[0].text
    for previous, current in zip(drafts, drafts[1:]):
        overlap = current.overlap_source_end - current.overlap_source_start
        reconstructed += current.text[overlap:]
        assert current.source_start == current.overlap_source_start
        assert current.source_end > previous.source_end
    assert reconstructed == normalise_source_text(source)


def test_fitting_table_is_atomic_and_can_force_a_short_neighbour() -> None:
    before = _words(200)
    table = _words(100)
    after = _words(200)
    source = f"{before}\n{table}\n{after}"
    table_start = len(before) + 1
    table_end = table_start + len(table)
    spans = (
        ("paragraph", 0, len(before)),
        ("table", table_start, table_end),
        ("paragraph", table_end + 1, len(source)),
    )

    drafts = build_chunk_drafts(_request(_parsed_document(source, spans)))

    assert all(not (table_start < draft.source_end < table_end) for draft in drafts)
    assert all(draft.token_count <= 512 for draft in drafts)


def test_oversize_table_is_split_under_the_hard_maximum() -> None:
    source = _words(600)
    drafts = build_chunk_drafts(
        _request(_parsed_document(source, [("table", 0, len(source))]))
    )

    assert len(drafts) == 2
    assert all(draft.token_count <= 512 for draft in drafts)
    assert all(draft.split_reason == "oversize_structure" for draft in drafts)


def test_uncovered_non_whitespace_content_fails_closed() -> None:
    with pytest.raises(ChunkWindowError) as error:
        build_chunk_drafts(_request(_parsed_document("secret", ())))

    assert error.value.category == "uncovered_content"
    assert "secret" not in str(error.value)


def test_empty_and_whitespace_only_documents_emit_no_drafts() -> None:
    empty = _parsed_document("", ())
    whitespace = _parsed_document(" \r\n\t ", (("paragraph", 0, 5),))

    assert build_chunk_drafts(_request(empty)) == ()
    assert build_chunk_drafts(_request(whitespace)) == ()


def test_nfc_source_ranges_remain_exactly_navigable() -> None:
    source = "e\u0301 " * 300
    end = len(source)
    document = _parsed_document(source, [("paragraph", 0, end)])
    drafts = build_chunk_drafts(_request(document))
    boundary_map = build_normalisation_boundary_map(document.text)

    assert drafts
    for draft in drafts:
        assert (
            boundary_map.canonical_text_for_source_range(
                draft.source_start,
                draft.source_end,
            )
            == draft.text
        )
