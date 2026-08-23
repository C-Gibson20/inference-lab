"""Tests for document serialisation."""

from inference_lab.ingest import (
    parsed_document_payload,
    serialise_parsed_document,
)

from .fixtures import fixture_parsed_document  # noqa: TID252


def test_parsed_document_serialisation_is_deterministic() -> None:
    document = fixture_parsed_document()

    assert serialise_parsed_document(document) == serialise_parsed_document(document)


def test_serialisation_preserves_absent_and_empty_structure() -> None:
    payload = parsed_document_payload(fixture_parsed_document(include_paragraph=False))

    assert payload["pages"] is None
    assert payload["paragraphs"] == []
