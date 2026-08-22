"""Tests for parser contract."""

import pytest

from inference_lab.ingest import (
    BoundingBox,
    NormalisedText,
    Parser,
    ParserDecodeError,
    ParserInput,
    ParserInputError,
    normalise_text,
    ParsedDocument,
    prepare_text,
    SourceSpan,
    StructuralBlock,
)
from inference_lab.provenance import DocumentId, DocumentVersionId, SourceSpanId


def test_parser_is_publicly_importable() -> None:
    assert Parser is not None


def test_parser_input_preserves_required_source_metadata() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=b"Hello\n",
        media_type="text/plain",
        encoding="utf-8",
    )

    assert source.content == b"Hello\n"
    assert source.media_type == "text/plain"
    assert source.encoding == "utf-8"


def test_empty_text_content_is_valid() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=b"",
        media_type="text/plain",
        encoding="utf-8",
    )

    assert source.content == b""


def test_non_bytes_content_is_rejected() -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            content="text",  # type: ignore[arg-type]
            media_type="text/plain",
            encoding="utf-8",
        )

    assert error.value.category == "non_bytes_content"


def test_unsupported_media_type_is_rejected() -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            content=b"text",
            media_type="application/pdf",  # type: ignore[arg-type]
            encoding="utf-8",
        )

    assert error.value.category == "unsupported_media_type"


@pytest.mark.parametrize("encoding", ["", "   ", None])
def test_missing_encoding_is_rejected(encoding: object) -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            content=b"text",
            media_type="text/plain",
            encoding=encoding,  # type: ignore[arg-type]
        )

    assert error.value.category == "missing_encoding"


def test_invalid_encoding_is_rejected() -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            content=b"text",
            media_type="text/plain",
            encoding="not-a-real-codec",
        )

    assert error.value.category == "invalid_encoding"


def test_invalid_document_id_is_rejected() -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id="not-an-id",  # type: ignore[arg-type]
            document_version_id=DocumentVersionId.generate(),
            content=b"text",
            media_type="text/plain",
            encoding="utf-8",
        )

    assert error.value.category == "invalid_document_id"


def test_invalid_document_version_id_is_rejected() -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id=DocumentId.generate(),
            document_version_id="not-a-version-id",  # type: ignore[arg-type]
            content=b"text",
            media_type="text/plain",
            encoding="utf-8",
        )

    assert error.value.category == "invalid_document_version_id"


def test_prepare_text_normalizes_newlines_and_unicode() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content="A\r\nB\re\u0301".encode("utf-8"),
        media_type="text/plain",
        encoding="utf-8",
    )

    prepared = prepare_text(source)

    assert prepared.source_text == "A\r\nB\re\u0301"
    assert prepared.text == "A\nB\né"


def test_normalization_preserves_whitespace() -> None:
    assert normalise_text("  A \n\n B\t") == "  A \n\n B\t"


def test_malformed_source_bytes_are_rejected() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=b"\xff",
        media_type="text/plain",
        encoding="utf-8",
    )

    with pytest.raises(ParserDecodeError) as error:
        prepare_text(source)

    assert error.value.category == "decode_failed"
    assert error.value.encoding == "utf-8"


def test_prepare_text_is_deterministic() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=b"A\r\nB",
        media_type="text/plain",
        encoding="utf-8",
    )

    assert prepare_text(source) == prepare_text(source)


def test_prepare_text_preserves_empty_content() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=b"",
        media_type="text/plain",
        encoding="utf-8",
    )

    assert prepare_text(source) == NormalisedText(source_text="", text="")


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


def test_parsed_document_distinguishes_absent_and_empty_structure() -> None:
    parsed = ParsedDocument(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        text=NormalisedText(source_text="", text=""),
        blocks=(),
        pages=None,
        sections=None,
        paragraphs=(),
        language=None,
        page_count=None,
        character_count=0,
        token_count=None,
    )

    assert parsed.pages is None
    assert parsed.sections is None
    assert parsed.paragraphs == ()


def test_parsed_document_rejects_character_count_mismatch() -> None:
    with pytest.raises(ValueError, match="character_count_mismatch"):
        ParsedDocument(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            text=NormalisedText(source_text="A", text="A"),
            blocks=(),
            pages=None,
            sections=None,
            paragraphs=(),
            language=None,
            page_count=None,
            character_count=2,
            token_count=None,
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
            text=NormalisedText(source_text="A", text="A"),
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
            text=NormalisedText(source_text="A", text="A"),
            blocks=(block,),
            pages=None,
            sections=None,
            paragraphs=(block,),
            language=None,
            page_count=None,
            character_count=1,
            token_count=None,
        )
