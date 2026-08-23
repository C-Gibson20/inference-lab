"""Test parser outcomes."""

import pytest

from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.errors import ParserFailure
from inference_lab.ingest import (
    generate_source_span_id,
    SourceSpan,
    StructuralBlock,
    NormalisedText,
    ParsedDocument,
)
from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
)


def make_fixture_parsed_document(*, include_paragraph: bool = True) -> ParsedDocument:
    document_version_id = DocumentVersionId.generate()
    span = SourceSpan(
        source_span_id=generate_source_span_id(
            document_version_id,
            ordinal=0,
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        ),
        document_version_id=document_version_id,
        start_offset=0,
        end_offset=1,
    )
    block = StructuralBlock(kind="paragraph", text="A", source_span=span)

    blocks = (block,) if include_paragraph else ()

    return ParsedDocument(
        document_id=DocumentId.generate(),
        document_version_id=document_version_id,
        text=NormalisedText(source_text="A"),
        blocks=blocks,
        pages=None,
        sections=None,
        paragraphs=blocks,
        language=None,
        page_count=None,
        character_count=1,
        token_count=None,
    )


def test_success_requires_document_and_no_failures() -> None:
    outcome = ParseOutcome(
        status="succeeded",
        document=make_fixture_parsed_document(),
        failures=(),
    )

    assert outcome.document is not None


def test_success_without_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="successful_parse_requires_document"):
        ParseOutcome(
            status="succeeded",
            document=None,
            failures=(),
        )


def test_partial_success_requires_recoverable_failure() -> None:
    failure = ParserFailure(
        category="malformed_unit",
        severity="recoverable",
        stage="structure",
        unit_kind="paragraph",
        unit_index=1,
    )

    outcome = ParseOutcome(
        status="completed_with_recoverable_failures",
        document=make_fixture_parsed_document(),
        failures=(failure,),
    )

    assert outcome.document is not None


def test_partial_success_rejects_fatal_failure() -> None:
    failure = ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )

    with pytest.raises(
        ValueError,
        match="partial_parse_must_not_have_fatal_failures",
    ):
        ParseOutcome(
            status="completed_with_recoverable_failures",
            document=make_fixture_parsed_document(),
            failures=(failure,),
        )


def test_failed_outcome_requires_fatal_failure_and_no_document() -> None:
    failure = ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )

    outcome = ParseOutcome(
        status="failed",
        document=None,
        failures=(failure,),
    )

    assert outcome.status == "failed"


def test_failed_outcome_cannot_contain_document() -> None:
    failure = ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )

    with pytest.raises(ValueError, match="failed_parse_must_not_have_document"):
        ParseOutcome(
            status="failed",
            document=make_fixture_parsed_document(),
            failures=(failure,),
        )
