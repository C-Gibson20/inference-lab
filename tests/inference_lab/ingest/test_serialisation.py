"""Tests for document serialisation."""

from inference_lab.ingest import (
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    parsed_document_payload,
    serialise_parsed_document,
)
from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    SourceSpanId,
)


def make_fixture_parsed_document(*, include_paragraph: bool = True) -> ParsedDocument:
    document_version_id = DocumentVersionId.generate()
    span = SourceSpan(
        source_span_id=SourceSpanId.generate(),
        document_version_id=document_version_id,
        start_offset=0,
        end_offset=1,
    )
    block = StructuralBlock(kind="paragraph", text="A", source_span=span)

    blocks = (block,) if include_paragraph else ()

    return ParsedDocument(
        document_id=DocumentId.generate(),
        document_version_id=document_version_id,
        text=NormalisedText(source_text="A", text="A"),
        blocks=blocks,
        pages=None,
        sections=None,
        paragraphs=blocks,
        language=None,
        page_count=None,
        character_count=1,
        token_count=None,
    )


def test_parsed_document_serialization_is_deterministic() -> None:
    document = make_fixture_parsed_document()

    assert serialise_parsed_document(document) == serialise_parsed_document(document)


def test_serialization_preserves_absent_and_empty_structure() -> None:
    payload = parsed_document_payload(
        make_fixture_parsed_document(include_paragraph=False)
    )

    assert payload["pages"] is None
    assert payload["paragraphs"] == []
