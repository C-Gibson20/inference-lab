"""Tests for source objects."""

import pytest

from inference_lab.ingest import (
    BoundingBox,
    generate_source_span_id,
    normalise_text,
    SourceSpan,
    StructuralBlock,
    StructuralBlockKind,
    ParsedDocument,
    NormalisedText,
)
from inference_lab.provenance import (
    DocumentVersionId,
    SourceSpanId,
    DocumentId,
)


def _block(
    document_version_id: DocumentVersionId,
    source_text: str,
    *,
    ordinal: int,
    kind: StructuralBlockKind = "paragraph",
    start_offset: int,
    end_offset: int,
    text: str | None = None,
    source_span_id: SourceSpanId | None = None,
) -> StructuralBlock:
    """Build a canonical block, with explicit overrides for negative tests."""
    span = SourceSpan(
        source_span_id=source_span_id
        or generate_source_span_id(
            document_version_id,
            ordinal=ordinal,
            kind=kind,
            start_offset=start_offset,
            end_offset=end_offset,
        ),
        document_version_id=document_version_id,
        start_offset=start_offset,
        end_offset=end_offset,
    )
    block_text = (
        normalise_text(source_text[start_offset:end_offset]) if text is None else text
    )
    return StructuralBlock(kind=kind, text=block_text, source_span=span)


def _parsed_document(
    source_text: str,
    blocks: tuple[StructuralBlock, ...],
    *,
    document_version_id: DocumentVersionId,
    pages: tuple[StructuralBlock, ...] | None = None,
    sections: tuple[StructuralBlock, ...] | None = None,
    paragraphs: tuple[StructuralBlock, ...] | None = None,
) -> ParsedDocument:
    """Build a parsed document for canonical-output validation tests."""
    normalised_text = normalise_text(source_text)
    return ParsedDocument(
        document_id=DocumentId.generate(),
        document_version_id=document_version_id,
        text=NormalisedText(source_text=source_text),
        blocks=blocks,
        pages=pages,
        sections=sections,
        paragraphs=paragraphs,
        language=None,
        page_count=None,
        character_count=len(normalised_text),
        token_count=None,
    )


def test_source_span_uses_half_open_codepoint_offsets() -> None:
    span = SourceSpan(
        source_span_id=SourceSpanId.generate(),
        document_version_id=DocumentVersionId.generate(),
        start_offset=2,
        end_offset=7,
    )

    assert span.start_offset == 2
    assert span.end_offset == 7


def test_source_span_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="source_span_start_must_be_non_negative"):
        SourceSpan(
            source_span_id=SourceSpanId.generate(),
            document_version_id=DocumentVersionId.generate(),
            start_offset=-1,
            end_offset=2,
        )


def test_source_span_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="source_span_end_must_not_precede_start"):
        SourceSpan(
            source_span_id=SourceSpanId.generate(),
            document_version_id=DocumentVersionId.generate(),
            start_offset=3,
            end_offset=2,
        )


def test_bounding_box_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="bounding_box_coordinates_must_be_finite"):
        BoundingBox(left=0.0, top=0.0, right=float("inf"), bottom=1.0)


def test_bounding_box_rejects_non_numeric_coordinates() -> None:
    with pytest.raises(ValueError, match="bounding_box_coordinates_must_be_finite"):
        BoundingBox(
            left="0",  # type: ignore[arg-type]
            top=0.0,
            right=1.0,
            bottom=1.0,
        )


def test_bounding_box_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValueError, match="bounding_box_must_not_be_inverted"):
        BoundingBox(left=2.0, top=0.0, right=1.0, bottom=1.0)


def test_source_span_rejects_invalid_identifier() -> None:
    with pytest.raises(ValueError, match="invalid_source_span_id"):
        SourceSpan(
            source_span_id="not-an-id",  # type: ignore[arg-type]
            document_version_id=DocumentVersionId.generate(),
            start_offset=0,
            end_offset=1,
        )


def test_source_span_rejects_non_integer_offsets() -> None:
    with pytest.raises(ValueError, match="source_span_offsets_must_be_integers"):
        SourceSpan(
            source_span_id=SourceSpanId.generate(),
            document_version_id=DocumentVersionId.generate(),
            start_offset=0.5,  # type: ignore[arg-type]
            end_offset=1,
        )


def test_source_span_rejects_non_integer_page_number() -> None:
    with pytest.raises(ValueError, match="page_number_must_be_integer"):
        SourceSpan(
            source_span_id=SourceSpanId.generate(),
            document_version_id=DocumentVersionId.generate(),
            start_offset=0,
            end_offset=1,
            page_number=1.5,  # type: ignore[arg-type]
        )


def test_structural_block_rejects_invalid_values() -> None:
    span = SourceSpan(
        source_span_id=SourceSpanId.generate(),
        document_version_id=DocumentVersionId.generate(),
        start_offset=0,
        end_offset=1,
    )

    with pytest.raises(ValueError, match="unsupported_structural_block_kind"):
        StructuralBlock(
            kind="unknown",  # type: ignore[arg-type]
            text="A",
            source_span=span,
        )

    with pytest.raises(ValueError, match="structural_block_text_must_be_string"):
        StructuralBlock(
            kind="paragraph",
            text=1,  # type: ignore[arg-type]
            source_span=span,
        )

    with pytest.raises(ValueError, match="invalid_structural_block_source_span"):
        StructuralBlock(
            kind="paragraph",
            text="A",
            source_span="not-a-span",  # type: ignore[arg-type]
        )


def test_parsed_document_rejects_out_of_bounds_source_span() -> None:
    document_version_id = DocumentVersionId.generate()
    span = SourceSpan(
        source_span_id=SourceSpanId.generate(),
        document_version_id=document_version_id,
        start_offset=0,
        end_offset=2,
    )
    block = StructuralBlock(kind="paragraph", text="A", source_span=span)

    with pytest.raises(ValueError, match="source_span_exceeds_source_text"):
        ParsedDocument(
            document_id=DocumentId.generate(),
            document_version_id=document_version_id,
            text=NormalisedText(source_text="A"),
            blocks=(block,),
            pages=None,
            sections=None,
            paragraphs=(block,),
            language=None,
            page_count=None,
            character_count=1,
            token_count=None,
        )


def test_parsed_document_rejects_source_span_version_mismatch() -> None:
    document_version_id = DocumentVersionId.generate()
    span = SourceSpan(
        source_span_id=SourceSpanId.generate(),
        document_version_id=DocumentVersionId.generate(),
        start_offset=0,
        end_offset=1,
    )
    block = StructuralBlock(kind="paragraph", text="A", source_span=span)

    with pytest.raises(ValueError, match="source_span_document_version_mismatch"):
        ParsedDocument(
            document_id=DocumentId.generate(),
            document_version_id=document_version_id,
            text=NormalisedText(source_text="A"),
            blocks=(block,),
            pages=None,
            sections=None,
            paragraphs=(block,),
            language=None,
            page_count=None,
            character_count=1,
            token_count=None,
        )


def test_parsed_document_rejects_invalid_parser_version() -> None:
    with pytest.raises(ValueError, match="invalid_parser_version"):
        ParsedDocument(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            text=NormalisedText(source_text=""),
            blocks=(),
            pages=None,
            sections=None,
            paragraphs=(),
            language=None,
            page_count=None,
            character_count=0,
            token_count=None,
            parser_version="not-a-parser-version",  # type: ignore[arg-type]
        )


def test_parsed_document_rejects_block_text_unrelated_to_source_span() -> None:
    document_version_id = DocumentVersionId.generate()
    block = _block(
        document_version_id,
        "source",
        ordinal=0,
        start_offset=0,
        end_offset=6,
        text="different",
    )

    with pytest.raises(ValueError, match="block_text_source_span_mismatch"):
        _parsed_document(
            "source",
            (block,),
            document_version_id=document_version_id,
            paragraphs=(block,),
        )


def test_parsed_document_rejects_arbitrary_source_span_id() -> None:
    document_version_id = DocumentVersionId.generate()
    block = _block(
        document_version_id,
        "source",
        ordinal=0,
        start_offset=0,
        end_offset=6,
        source_span_id=SourceSpanId.generate(),
    )

    with pytest.raises(ValueError, match="source_span_id_mismatch"):
        _parsed_document(
            "source",
            (block,),
            document_version_id=document_version_id,
            paragraphs=(block,),
        )


def test_parsed_document_rejects_blocks_out_of_source_order() -> None:
    document_version_id = DocumentVersionId.generate()
    blocks = (
        _block(
            document_version_id,
            "A B",
            ordinal=0,
            start_offset=2,
            end_offset=3,
        ),
        _block(
            document_version_id,
            "A B",
            ordinal=1,
            start_offset=0,
            end_offset=1,
        ),
    )

    with pytest.raises(ValueError, match="blocks_must_be_in_source_order"):
        _parsed_document(
            "A B",
            blocks,
            document_version_id=document_version_id,
            paragraphs=blocks,
        )


def test_parsed_document_rejects_typed_collection_drift() -> None:
    document_version_id = DocumentVersionId.generate()
    block = _block(
        document_version_id,
        "source",
        ordinal=0,
        start_offset=0,
        end_offset=6,
    )

    with pytest.raises(ValueError, match="paragraphs_must_match_blocks"):
        _parsed_document(
            "source",
            (block,),
            document_version_id=document_version_id,
            paragraphs=(),
        )


def test_parsed_document_requires_collection_for_typed_blocks() -> None:
    document_version_id = DocumentVersionId.generate()
    block = _block(
        document_version_id,
        "source",
        ordinal=0,
        start_offset=0,
        end_offset=6,
    )

    with pytest.raises(ValueError, match="paragraphs_missing_for_blocks"):
        _parsed_document(
            "source",
            (block,),
            document_version_id=document_version_id,
            paragraphs=None,
        )


def test_parsed_document_requires_immutable_blocks() -> None:
    document_version_id = DocumentVersionId.generate()
    block = _block(
        document_version_id,
        "source",
        ordinal=0,
        start_offset=0,
        end_offset=6,
    )

    with pytest.raises(ValueError, match="blocks_must_be_tuple"):
        _parsed_document(
            "source",
            [block],  # type: ignore[arg-type]
            document_version_id=document_version_id,
            paragraphs=(block,),
        )


def test_parsed_document_requires_immutable_typed_collections() -> None:
    document_version_id = DocumentVersionId.generate()
    block = _block(
        document_version_id,
        "source",
        ordinal=0,
        start_offset=0,
        end_offset=6,
    )

    with pytest.raises(ValueError, match="paragraphs_must_be_tuple"):
        _parsed_document(
            "source",
            (block,),
            document_version_id=document_version_id,
            paragraphs=[block],  # type: ignore[arg-type]
        )
