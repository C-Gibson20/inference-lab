"""Tests for the deterministic text-native parser."""

import ast
import inspect
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    TextNativeParser,
    serialise_parsed_document,
)
from tests.inference_lab.ingest.fixtures import (
    FIXTURE_DOCUMENT_ID,
    FIXTURE_DOCUMENT_VERSION_ID,
    FIXTURE_ROOT,
    fixture_bytes,
    fixture_settings,
    fixture_source,
)


def test_text_native_parser_emits_ordered_paragraphs() -> None:
    parser = TextNativeParser()

    outcome = parser.parse(
        fixture_source("basic.txt"),
        settings=fixture_settings(),
    )

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == [
        "First paragraph.",
        "Second paragraph.\n",
    ]


def test_single_newline_does_not_split_paragraphs() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("single_paragraph.txt"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    paragraphs = outcome.document.paragraphs
    assert paragraphs is not None
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "First line\nSecond line\n"


def test_whitespace_only_blank_lines_split_paragraphs() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("blank_edges.txt"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == [
        "A",
        "B\n",
    ]


def test_blank_edges_do_not_create_empty_paragraphs() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("edge_blank.txt"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == ["A"]


def test_single_blank_edge_lines_do_not_enter_paragraph_text() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("edge_blank.txt"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == ["A"]


def test_empty_document_is_a_successful_empty_parse() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("empty.txt"),
        settings=fixture_settings(),
    )

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert outcome.document.blocks == ()
    assert outcome.document.paragraphs == ()


def test_paragraph_spans_exclude_blank_line_separator() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("basic.txt"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    spans = [block.source_span for block in outcome.document.blocks]

    assert [(span.start_offset, span.end_offset) for span in spans] == [
        (0, 16),
        (18, 36),
    ]


def test_repeated_parse_has_stable_source_span_ids() -> None:
    source = fixture_source("basic.txt")

    first = TextNativeParser().parse(source, settings=fixture_settings())
    second = TextNativeParser().parse(source, settings=fixture_settings())

    assert first.document is not None
    assert second.document is not None
    assert [block.source_span.source_span_id for block in first.document.blocks] == [
        block.source_span.source_span_id for block in second.document.blocks
    ]


def test_repeated_parse_has_identical_serialized_output() -> None:
    source = fixture_source("basic.txt")
    settings = fixture_settings()

    first = TextNativeParser().parse(source, settings=settings)
    second = TextNativeParser().parse(source, settings=settings)

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert first.document is not None
    assert second.document is not None
    assert serialise_parsed_document(first.document) == serialise_parsed_document(
        second.document
    )


def test_successful_parse_preserves_provenance_on_every_block() -> None:
    source = fixture_source("basic.txt")
    outcome = TextNativeParser().parse(source, settings=fixture_settings())

    assert outcome.document is not None
    assert outcome.document.parser_version.value == (
        "text-native/contract-1.0.0/normalization-1.0.0"
    )
    assert all(
        block.source_span.document_version_id == source.document_version_id
        for block in outcome.document.blocks
    )


def test_text_native_parser_has_no_ocr_model_or_network_imports() -> None:
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


def test_malformed_utf8_fails_without_partial_document() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("malformed_utf8.bin"),
        settings=fixture_settings(),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.failures[0].category == "decode_failed"
    assert outcome.failures[0].severity == "fatal"


def test_invalid_runtime_source_returns_failed_outcome() -> None:
    outcome = TextNativeParser().parse(
        "not-parser-input",  # type: ignore[arg-type]
        settings=fixture_settings(),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.failures[0].category == "invalid_input"


def test_parser_returns_canonical_normalized_text() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("unicode_newlines.bin"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert outcome.document.text.text == "A\nB\né"


def test_source_at_size_limit_is_accepted() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("size_limit_exact.txt"),
        settings=fixture_settings(max_source_bytes=5),
    )

    assert outcome.status == "succeeded"


def test_source_over_size_limit_fails_before_decoding() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("size_limit_over.txt"),
        settings=fixture_settings(max_source_bytes=5),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.failures[0].category == "source_too_large"
    assert outcome.failures[0].severity == "fatal"


def test_size_limit_precedes_decoding() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("malformed_utf8.bin"),
        settings=fixture_settings(max_source_bytes=1),
    )

    assert outcome.failures[0].category == "source_too_large"


def test_parser_emits_privacy_safe_failure_log() -> None:
    source = fixture_source("size_limit_over.txt")
    with capture_logs() as logs:
        outcome = TextNativeParser().parse(
            source,
            settings=fixture_settings(max_source_bytes=5),
        )

    assert outcome.status == "failed"
    failure_log = next(
        record for record in logs if record["event"] == "document_parse_failed"
    )

    assert failure_log["pipeline_stage"] == "parse"
    assert failure_log["document_id"] == str(source.document_id)
    assert failure_log["error_category"] == "source_too_large"
    assert "secret text" not in str(logs)
    assert "content" not in failure_log
    assert "path" not in failure_log


def test_basic_fixture_preserves_order_and_source_spans() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("basic.txt"),
        settings=fixture_settings(),
    )

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert outcome.document.document_id == FIXTURE_DOCUMENT_ID
    assert outcome.document.document_version_id == FIXTURE_DOCUMENT_VERSION_ID

    paragraphs = outcome.document.paragraphs
    assert paragraphs is not None
    assert [paragraph.text for paragraph in paragraphs] == [
        "First paragraph.",
        "Second paragraph.\n",
    ]
    assert [
        (
            paragraph.source_span.start_offset,
            paragraph.source_span.end_offset,
        )
        for paragraph in paragraphs
    ] == [(0, 16), (18, 36)]


def test_unicode_fixture_preserves_source_and_normalizes_output() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("unicode_newlines.bin"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert outcome.document.text.source_text == "A\r\nB\re\u0301"
    assert outcome.document.text.text == "A\nB\né"


def test_empty_fixture_is_a_successful_empty_document() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("empty.txt"),
        settings=fixture_settings(),
    )

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert outcome.document.blocks == ()


def test_malformed_fixture_is_a_fatal_decode_failure() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("malformed_utf8.bin"),
        settings=fixture_settings(),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert [(failure.category, failure.severity) for failure in outcome.failures] == [
        ("decode_failed", "fatal")
    ]


def test_basic_fixture_matches_golden_serialization() -> None:
    outcome = TextNativeParser().parse(
        fixture_source("basic.txt"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None

    actual = serialise_parsed_document(outcome.document)
    expected = (FIXTURE_ROOT / "golden_basic.json").read_bytes()

    assert actual == expected


def test_fixture_loader_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="fixture_name_must_remain_inside"):
        fixture_bytes("../outside.txt")


def test_fixture_loader_rejects_missing_fixture() -> None:
    with pytest.raises(FileNotFoundError, match="fixture_not_found"):
        fixture_bytes("missing.txt")
