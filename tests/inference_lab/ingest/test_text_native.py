"""Tests for the deterministic text-native parser."""

import pytest

from inference_lab.ingest import (
    ParserInput,
    TextNativeParser,
)
from .fixtures import (
    FIXTURE_DOCUMENT_ID,
    FIXTURE_DOCUMENT_VERSION_ID,
    fixture_bytes,
    fixture_settings,
    fixture_source,
)


def _source(source_text: str) -> ParserInput:
    """Build an inline source with stable fixture provenance."""
    return ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=source_text.encode("utf-8"),
        media_type="text/plain",
        encoding="utf-8",
    )


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


def test_single_leading_blank_line_does_not_create_empty_paragraph() -> None:
    outcome = TextNativeParser().parse(
        _source("\nA\n"),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == ["A\n"]


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_paragraph_boundaries_preserve_content_indentation(
    line_ending: str,
) -> None:
    source_text = f"A{line_ending} \t{line_ending}  B{line_ending}"
    second_start = len(f"A{line_ending} \t{line_ending}")
    outcome = TextNativeParser().parse(
        _source(source_text),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    paragraphs = outcome.document.paragraphs
    assert paragraphs is not None
    assert [paragraph.text for paragraph in paragraphs] == ["A", "  B\n"]
    assert [
        (paragraph.source_span.start_offset, paragraph.source_span.end_offset)
        for paragraph in paragraphs
    ] == [(0, 1), (second_start, len(source_text))]


@pytest.mark.parametrize(
    ("source_text", "expected_text"),
    [
        ("A\n", "A\n"),
        ("A\nB\n", "A\nB\n"),
        ("A\n\n", "A"),
    ],
)
def test_trailing_line_ending_handling_is_content_independent(
    source_text: str,
    expected_text: str,
) -> None:
    outcome = TextNativeParser().parse(
        _source(source_text),
        settings=fixture_settings(),
    )

    assert outcome.document is not None
    assert [block.text for block in outcome.document.paragraphs or ()] == [
        expected_text
    ]


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


def test_fixture_loader_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="fixture_name_must_remain_inside"):
        fixture_bytes("../outside.txt")


def test_fixture_loader_rejects_missing_fixture() -> None:
    with pytest.raises(FileNotFoundError, match="fixture_not_found"):
        fixture_bytes("missing.txt")
