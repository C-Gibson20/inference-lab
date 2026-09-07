"""Tests for parser contract."""

import traceback

import pytest

from inference_lab.ingest import (
    NormalisedText,
    ParsedDocument,
    Parser,
    ParserDecodeError,
    ParserInput,
    ParserInputError,
    normalise_source_text,
    prepare_text,
)
from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
)


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
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    "encoding",
    ["base64_codec", "hex_codec", "rot_13", "uu_codec", "zlib_codec"],
)
def test_non_text_codec_is_rejected(encoding: str) -> None:
    with pytest.raises(ParserInputError) as error:
        ParserInput(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            content=b"private source",
            media_type="text/plain",
            encoding=encoding,
        )

    assert error.value.category == "invalid_encoding"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


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


def test_prepare_text_normalises_newlines_and_unicode() -> None:
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


def test_normalisation_preserves_whitespace() -> None:
    assert normalise_source_text("  A \n\n B\t") == "  A \n\n B\t"


def test_normalisation_rejects_non_text_input() -> None:
    with pytest.raises(TypeError, match="source_text"):
        normalise_source_text(b"text")  # type: ignore[arg-type]


def test_malformed_source_bytes_are_rejected() -> None:
    source = ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=b"private source \xff",
        media_type="text/plain",
        encoding="utf-8",
    )

    with pytest.raises(ParserDecodeError) as error:
        prepare_text(source)

    assert error.value.category == "decode_failed"
    assert error.value.encoding == "utf-8"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert "private source" not in "".join(traceback.format_exception(error.value))


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

    assert prepare_text(source) == NormalisedText(source_text="")


def test_normalised_text_rejects_non_string_source() -> None:
    with pytest.raises(ValueError, match="source_text_must_be_string"):
        NormalisedText(source_text=b"source")  # type: ignore[arg-type]


def test_normalised_text_derives_canonical_text() -> None:
    text = NormalisedText(source_text="A\r\ne\u0301")

    assert text.source_text == "A\r\ne\u0301"
    assert text.text == "A\né"


def test_parsed_document_distinguishes_absent_and_empty_structure() -> None:
    parsed = ParsedDocument(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        text=NormalisedText(source_text=""),
        blocks=(),
        pages=None,
        sections=None,
        paragraphs=(),
        page_layouts=(),
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
            text=NormalisedText(source_text="A"),
            blocks=(),
            pages=None,
            sections=None,
            paragraphs=(),
            page_layouts=(),
            language=None,
            page_count=None,
            character_count=2,
            token_count=None,
        )
