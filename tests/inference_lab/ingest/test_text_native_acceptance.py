"""Acceptance gate for the deterministic text-native parser baseline."""

import ast
import inspect
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParserInput,
    ParserInputError,
    TextNativeParser,
    serialise_parsed_document,
)
from inference_lab.ingest.versioning import generate_source_span_id

from .fixtures import (  # noqa: TID252
    FIXTURE_DOCUMENT_ID,
    FIXTURE_DOCUMENT_VERSION_ID,
    FIXTURE_ROOT,
    fixture_settings,
    fixture_source,
)


def _pdf_bytes(*pages: str, rotations: tuple[int, ...] = ()) -> bytes:
    """Build a minimal local PDF fixture without adding a PDF-writing dependency."""
    page_count = len(pages)
    page_object_ids = tuple(range(3, 3 + page_count))
    font_object_id = 3 + page_count
    stream_object_ids = tuple(
        range(font_object_id + 1, font_object_id + 1 + page_count)
    )
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{item} 0 R' for item in page_object_ids)}] "
            f"/Count {page_count} >>"
        ).encode("ascii"),
    ]
    for index, (page_object_id, stream_object_id) in enumerate(
        zip(page_object_ids, stream_object_ids, strict=True)
    ):
        rotation = rotations[index] if index < len(rotations) else 0
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_id} 0 R >> >> "
                f"/Contents {stream_object_id} 0 R /Rotate {rotation} >>"
            ).encode("ascii")
        )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    for page in pages:
        escaped = page.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode("ascii")
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        )

    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode("ascii"))
        result.extend(obj)
        result.extend(b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(result)


def _encrypted_pdf_bytes() -> bytes:
    """Return a generated encrypted PDF that must fail without a password."""
    reader = PdfReader(BytesIO(_pdf_bytes("private text")), strict=True)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("fixture-password")
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _blank_pdf_bytes() -> bytes:
    """Return a valid one-page PDF whose page has no content stream."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_text_native_emits_exact_canonical_output_deterministically() -> None:
    source = fixture_source("basic.txt")
    settings = fixture_settings()

    first = TextNativeParser().parse(source, settings=settings)
    second = TextNativeParser().parse(source, settings=settings)

    assert first.status == "succeeded"
    assert first.failures == ()
    assert first.document is not None
    document = first.document
    assert document.document_id == FIXTURE_DOCUMENT_ID
    assert document.document_version_id == FIXTURE_DOCUMENT_VERSION_ID
    assert document.parser_version == CURRENT_PARSER_VERSION
    assert document.text.source_text == "First paragraph.\n\nSecond paragraph.\n"
    assert document.text.text == document.text.source_text
    assert document.blocks == document.paragraphs
    assert document.pages is None
    assert document.sections is None
    assert document.page_layouts == ()
    assert document.language is None
    assert document.page_count is None
    assert document.token_count is None
    assert document.character_count == 36
    assert [block.kind for block in document.blocks] == ["paragraph", "paragraph"]
    assert [block.text for block in document.blocks] == [
        "First paragraph.",
        "Second paragraph.\n",
    ]
    assert [
        (block.source_span.start_offset, block.source_span.end_offset)
        for block in document.blocks
    ] == [(0, 16), (18, 36)]
    assert all(
        block.source_span.document_version_id == source.document_version_id
        for block in document.blocks
    )

    expected = (FIXTURE_ROOT / "golden_basic.json").read_bytes()
    assert serialise_parsed_document(document) == expected
    assert second.document is not None
    assert serialise_parsed_document(second.document) == expected


def test_text_native_normalises_source_and_accepts_empty_documents() -> None:
    normalised = TextNativeParser().parse(
        fixture_source("unicode_newlines.bin"),
        settings=fixture_settings(),
    )
    empty = TextNativeParser().parse(
        fixture_source("empty.txt"),
        settings=fixture_settings(),
    )

    assert normalised.status == "succeeded"
    assert normalised.document is not None
    assert normalised.document.text.source_text == "A\r\nB\re\u0301"
    assert normalised.document.text.text == "A\nB\né"
    assert empty.status == "succeeded"
    assert empty.document is not None
    assert empty.document.blocks == ()
    assert empty.document.paragraphs == ()


def test_text_native_extracts_pdf_pages_and_stable_source_spans() -> None:
    source = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=_pdf_bytes("First PDF page", "Second PDF page", rotations=(0, 90)),
        media_type="application/pdf",
    )

    first = TextNativeParser().parse(source, settings=fixture_settings())
    second = TextNativeParser().parse(source, settings=fixture_settings())

    assert first.status == "succeeded"
    assert first.document is not None
    document = first.document
    assert document.parser_version == CURRENT_PARSER_VERSION
    assert document.page_count == 2
    assert document.pages is not None
    assert len(document.pages) == 2
    assert document.paragraphs is not None
    assert len(document.paragraphs) == 2
    assert document.page_layouts[0].page_width == 612
    assert document.page_layouts[0].page_height == 792
    assert document.page_layouts[1].rotation == 90
    assert [page.source_span.page_number for page in document.pages] == [1, 2]
    assert [block.source_span.page_number for block in document.paragraphs] == [1, 2]
    assert "First PDF page" in document.pages[0].text
    assert "Second PDF page" in document.pages[1].text
    assert document.text.source_text.count("\n\n") >= 1
    for ordinal, block in enumerate(document.blocks):
        assert block.source_span.source_span_id == generate_source_span_id(
            source.document_version_id,
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=ordinal,
            kind=block.kind,
            start_offset=block.source_span.start_offset,
            end_offset=block.source_span.end_offset,
        )
    assert second.document is not None
    assert serialise_parsed_document(document) == serialise_parsed_document(
        second.document
    )


def test_pdf_input_rejects_a_caller_selected_encoding() -> None:
    with pytest.raises(ParserInputError, match="invalid_encoding"):
        ParserInput(
            document_id=FIXTURE_DOCUMENT_ID,
            document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
            content=_pdf_bytes("Text"),
            media_type="application/pdf",
            encoding="utf-8",
        )


def test_text_native_preserves_empty_pdf_pages_without_claiming_ocr() -> None:
    source = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=_pdf_bytes(""),
        media_type="application/pdf",
    )

    outcome = TextNativeParser().parse(source, settings=fixture_settings())

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert outcome.document.page_count == 1
    assert outcome.document.pages is not None
    assert outcome.document.pages[0].text == ""
    assert outcome.document.paragraphs == ()
    assert outcome.document.token_count is None


def test_text_native_accepts_pdf_pages_without_content_streams() -> None:
    source = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=_blank_pdf_bytes(),
        media_type="application/pdf",
    )

    outcome = TextNativeParser().parse(source, settings=fixture_settings())

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert outcome.document.page_count == 1
    assert outcome.document.pages is not None
    assert outcome.document.pages[0].text == ""
    assert outcome.document.paragraphs == ()


def test_text_native_rejects_malformed_pdf_without_exposing_source_bytes() -> None:
    payload_marker = "private-pdf-payload-marker"
    source = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=f"%PDF-1.4 {payload_marker}".encode(),
        media_type="application/pdf",
    )

    with capture_logs() as logs:
        outcome = TextNativeParser().parse(source, settings=fixture_settings())

    assert outcome.status == "failed"
    assert outcome.document is None
    assert [failure.category for failure in outcome.failures] == ["malformed_source"]
    assert payload_marker not in repr(source)
    assert payload_marker not in repr(outcome)
    assert payload_marker not in repr(logs)


def test_text_native_fails_encrypted_and_oversized_pdfs_safely() -> None:
    parser = TextNativeParser()
    encrypted = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=_encrypted_pdf_bytes(),
        media_type="application/pdf",
    )
    oversized = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=_pdf_bytes("Some text"),
        media_type="application/pdf",
    )

    encrypted_outcome = parser.parse(encrypted, settings=fixture_settings())
    oversized_outcome = parser.parse(
        oversized,
        settings=fixture_settings(max_source_bytes=1),
    )

    assert encrypted_outcome.status == "failed"
    assert encrypted_outcome.document is None
    assert [failure.category for failure in encrypted_outcome.failures] == [
        "unsupported_format"
    ]
    assert oversized_outcome.status == "failed"
    assert oversized_outcome.document is None
    assert [failure.category for failure in oversized_outcome.failures] == [
        "source_too_large"
    ]


@pytest.mark.parametrize(
    ("max_source_bytes", "category", "stage"),
    [
        (50 * 1024 * 1024, "decode_failed", "decode"),
        (1, "source_too_large", "input"),
    ],
)
def test_text_native_rejects_malformed_source_without_partial_output(
    max_source_bytes: int,
    category: str,
    stage: str,
) -> None:
    outcome = TextNativeParser().parse(
        fixture_source("malformed_utf8.bin"),
        settings=fixture_settings(max_source_bytes=max_source_bytes),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert [
        (failure.category, failure.severity, failure.stage)
        for failure in outcome.failures
    ] == [(category, "fatal", stage)]


def test_text_native_rejects_invalid_runtime_source() -> None:
    outcome = TextNativeParser().parse(
        "not-parser-input",  # type: ignore[arg-type]
        settings=fixture_settings(),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert [
        (failure.category, failure.severity, failure.stage)
        for failure in outcome.failures
    ] == [("invalid_input", "fatal", "input")]


def test_text_native_failure_output_and_logs_do_not_expose_source_payload() -> None:
    payload_marker = "private-ingest-payload"
    source = ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=f"{payload_marker}-".encode("utf-8") + b"\xff",
        media_type="text/plain",
        encoding="utf-8",
    )

    with capture_logs() as logs:
        outcome = TextNativeParser().parse(source, settings=fixture_settings())

    assert outcome.status == "failed"
    failure_log = next(
        record for record in logs if record["event"] == "document_parse_failed"
    )
    assert failure_log["pipeline_stage"] == "parse"
    assert failure_log["document_id"] == str(source.document_id)
    assert failure_log["error_category"] == "decode_failed"
    assert failure_log["failure_severity"] == "fatal"
    assert payload_marker not in repr(outcome)
    assert payload_marker not in repr(logs)
    assert all("content" not in record and "path" not in record for record in logs)


def test_text_native_parser_boundary_has_no_model_ocr_or_network_imports() -> None:
    source_file = inspect.getsourcefile(TextNativeParser)
    assert source_file is not None

    syntax = ast.parse(Path(source_file).read_text(encoding="utf-8"))
    forbidden_roots = {
        "httpx",
        "ocr",
        "pytesseract",
        "requests",
        "socket",
        "torch",
        "transformers",
    }
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(syntax)
        if isinstance(node, ast.Import) and node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert imported_roots.isdisjoint(forbidden_roots)
