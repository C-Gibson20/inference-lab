"""Acceptance gate for the deterministic text-native parser baseline."""

import ast
import inspect
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParserInput,
    TextNativeParser,
    serialise_parsed_document,
)
from .fixtures import (
    FIXTURE_DOCUMENT_ID,
    FIXTURE_DOCUMENT_VERSION_ID,
    FIXTURE_ROOT,
    fixture_settings,
    fixture_source,
)


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
