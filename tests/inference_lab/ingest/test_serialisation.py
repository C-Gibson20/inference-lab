"""Tests for document serialisation."""

from dataclasses import replace

from inference_lab.ingest import (
    PageLayoutDescriptor,
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
    assert payload["page_layouts"] == []


def test_serialisation_preserves_page_layout_without_coordinate_conversion() -> None:
    layout = PageLayoutDescriptor(
        page_number=1,
        coordinate_unit="image_pixels",
        coordinate_origin="top_left",
        page_width=1200.0,
        page_height=1600.0,
        rotation=270,
    )
    document = replace(
        fixture_parsed_document(),
        page_count=1,
        page_layouts=(layout,),
    )

    payload = parsed_document_payload(document)

    assert payload["page_layouts"] == [
        {
            "page_number": 1,
            "coordinate_unit": "image_pixels",
            "coordinate_origin": "top_left",
            "page_width": 1200.0,
            "page_height": 1600.0,
            "rotation": 270,
        }
    ]
