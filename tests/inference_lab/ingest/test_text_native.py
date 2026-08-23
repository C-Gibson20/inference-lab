"""Tests for the deterministic text-native parser."""

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    ParserInput,
    TextNativeParser,
)
from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
)
from inference_lab.settings import (
    ModelPaths,
    RuntimeConfig,
    Settings,
    StorageConfig,
    IngestConfig,
)


def make_source(content: bytes) -> ParserInput:
    return ParserInput(
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
        content=content,
        media_type="text/plain",
        encoding="utf-8",
    )


def make_settings(*, max_source_bytes: int = 50 * 1024 * 1024) -> Settings:
    return Settings(
        models=ModelPaths(
            embedder_path=Path("fixture/embedder"),
            reranker_path=Path("fixture/reranker"),
            generator_path=Path("fixture/generator"),
        ),
        runtime=RuntimeConfig(vram_budget_gb=24),
        storage=StorageConfig(
            index_dir=Path("fixture/index"),
            metadata_db_path=Path("fixture/metadata.db"),
        ),
        ingest=IngestConfig(max_source_bytes=max_source_bytes),
    )


def test_text_native_parser_emits_ordered_paragraphs() -> None:
    parser = TextNativeParser()
    source = make_source(b"First paragraph.\n\nSecond paragraph.")

    outcome = parser.parse(source, settings=make_settings())

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_single_newline_does_not_split_paragraphs() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"First line\nSecond line"),
        settings=make_settings(),
    )

    assert outcome.document is not None
    paragraphs = outcome.document.paragraphs
    assert paragraphs is not None
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "First line\nSecond line"


def test_whitespace_only_blank_lines_split_paragraphs() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"A\n \t\nB"),
        settings=make_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == [
        "A",
        "B",
    ]


def test_blank_edges_do_not_create_empty_paragraphs() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"\n\nA\n\n"),
        settings=make_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == ["A"]


@pytest.mark.parametrize("content", [b"\nA\n", b"\r\nA\r\n"])
def test_single_blank_edge_lines_do_not_enter_paragraph_text(
    content: bytes,
) -> None:
    outcome = TextNativeParser().parse(
        make_source(content),
        settings=make_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == ["A"]


def test_empty_document_is_a_successful_empty_parse() -> None:
    outcome = TextNativeParser().parse(
        make_source(b""),
        settings=make_settings(),
    )

    assert outcome.status == "succeeded"
    assert outcome.document is not None
    assert outcome.document.blocks == ()
    assert outcome.document.paragraphs == ()


def test_paragraph_spans_exclude_blank_line_separator() -> None:
    source_text = "A\n\nB"
    outcome = TextNativeParser().parse(
        make_source(source_text.encode("utf-8")),
        settings=make_settings(),
    )

    assert outcome.document is not None
    spans = [block.source_span for block in outcome.document.blocks]

    assert [(span.start_offset, span.end_offset) for span in spans] == [
        (0, 1),
        (3, 4),
    ]


def test_repeated_parse_has_stable_source_span_ids() -> None:
    source = make_source(b"A\n\nB")

    first = TextNativeParser().parse(source, settings=make_settings())
    second = TextNativeParser().parse(source, settings=make_settings())

    assert first.document is not None
    assert second.document is not None
    assert [block.source_span.source_span_id for block in first.document.blocks] == [
        block.source_span.source_span_id for block in second.document.blocks
    ]


def test_malformed_utf8_fails_without_partial_document() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"\xff"),
        settings=make_settings(),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.failures[0].category == "decode_failed"
    assert outcome.failures[0].severity == "fatal"


def test_invalid_runtime_source_returns_failed_outcome() -> None:
    outcome = TextNativeParser().parse(
        "not-parser-input",  # type: ignore[arg-type]
        settings=make_settings(),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.failures[0].category == "invalid_input"


def test_parser_returns_canonical_normalized_text() -> None:
    outcome = TextNativeParser().parse(
        make_source("A\r\nB\re\u0301".encode("utf-8")),
        settings=make_settings(),
    )

    assert outcome.document is not None
    assert outcome.document.text.text == "A\nB\né"


def test_source_at_size_limit_is_accepted() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"A" * 4),
        settings=make_settings(max_source_bytes=4),
    )

    assert outcome.status == "succeeded"


def test_source_over_size_limit_fails_before_decoding() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"A" * 5),
        settings=make_settings(max_source_bytes=4),
    )

    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.failures[0].category == "source_too_large"
    assert outcome.failures[0].severity == "fatal"


def test_size_limit_precedes_decoding() -> None:
    outcome = TextNativeParser().parse(
        make_source(b"\xff\xff"),
        settings=make_settings(max_source_bytes=1),
    )

    assert outcome.failures[0].category == "source_too_large"


def test_parser_emits_privacy_safe_failure_log() -> None:
    source = make_source(b"secret text")

    with capture_logs() as logs:
        outcome = TextNativeParser().parse(
            source,
            settings=make_settings(max_source_bytes=1),
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
