"""Tests for persisted structure and source-location coordinate contracts."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.storage import (
    BoundingBox,
    CollectionAvailability,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    SourceLocation,
    StructuralBlock,
    StructuralBlockKind,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    OCR_PRODUCER,
    PARSED_RECORD_ID,
    PARSER,
)


def _bounding_box() -> BoundingBox:
    return BoundingBox(left=10.0, top=20.0, right=80.0, bottom=40.0)


def _layout(**changes: object) -> PageLayoutDescriptor:
    values: dict[str, object] = {
        "page_number": 1,
        "coordinate_unit": CoordinateUnit.PDF_POINTS,
        "coordinate_origin": CoordinateOrigin.BOTTOM_LEFT,
        "page_width": 612.0,
        "page_height": 792.0,
        "rotation": PageRotation.DEGREES_90,
        "content_origin": ContentOrigin.NATIVE,
        "producer": PARSER,
    }
    values.update(changes)
    return PageLayoutDescriptor(**values)  # type: ignore[arg-type]


def _location(**changes: object) -> SourceLocation:
    values: dict[str, object] = {
        "source_span_id": FIRST_SPAN_ID,
        "document_version_id": DOCUMENT_VERSION_ID,
        "parsed_record_id": PARSED_RECORD_ID,
        "start_offset": 4,
        "end_offset": 12,
        "content_origin": ContentOrigin.NATIVE,
        "page_number": None,
        "bounding_box": None,
    }
    values.update(changes)
    return SourceLocation(**values)  # type: ignore[arg-type]


def test_coordinate_enums_cover_the_approved_initial_contract() -> None:
    assert {unit.value for unit in CoordinateUnit} == {"pdf_points", "image_pixels"}
    assert {origin.value for origin in CoordinateOrigin} == {
        "top_left",
        "bottom_left",
    }
    assert {rotation.value for rotation in PageRotation} == {0, 90, 180, 270}
    assert {state.value for state in CollectionAvailability} == {
        "available",
        "unavailable",
    }


def test_bounding_box_retains_untransformed_coordinates() -> None:
    box = _bounding_box()

    assert (box.left, box.top, box.right, box.bottom) == (10.0, 20.0, 80.0, 40.0)
    with pytest.raises(FrozenInstanceError):
        box.left = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"right": float("inf")}, ValueError),
        ({"bottom": float("nan")}, ValueError),
        ({"left": True}, TypeError),
        ({"left": 5.0, "right": 4.0}, ValueError),
        ({"top": 5.0, "bottom": 4.0}, ValueError),
    ],
)
def test_bounding_box_rejects_invalid_coordinates(
    changes: dict[str, object], error_type: type[Exception]
) -> None:
    values: dict[str, object] = {
        "left": 0.0,
        "top": 0.0,
        "right": 1.0,
        "bottom": 1.0,
    }
    values.update(changes)

    with pytest.raises(error_type):
        BoundingBox(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("unit", "origin"),
    [
        (CoordinateUnit.PDF_POINTS, CoordinateOrigin.BOTTOM_LEFT),
        (CoordinateUnit.IMAGE_PIXELS, CoordinateOrigin.TOP_LEFT),
    ],
)
def test_page_layout_retains_each_supported_coordinate_space(
    unit: CoordinateUnit,
    origin: CoordinateOrigin,
) -> None:
    layout = _layout(coordinate_unit=unit, coordinate_origin=origin)

    assert layout.coordinate_unit is unit
    assert layout.coordinate_origin is origin
    assert layout.rotation is PageRotation.DEGREES_90


def test_page_layout_can_identify_ocr_produced_layout() -> None:
    layout = _layout(
        content_origin=ContentOrigin.OCR,
        producer=OCR_PRODUCER,
    )

    assert layout.content_origin is ContentOrigin.OCR
    assert layout.producer == OCR_PRODUCER


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"page_number": 0}, ValueError, "page_number"),
        ({"page_number": True}, TypeError, "page_number"),
        ({"page_width": 0.0}, ValueError, "dimensions"),
        ({"page_height": float("inf")}, ValueError, "page_height"),
        ({"coordinate_unit": "points"}, TypeError, "CoordinateUnit"),
        ({"coordinate_origin": "top"}, TypeError, "CoordinateOrigin"),
        ({"rotation": 45}, TypeError, "PageRotation"),
        ({"content_origin": "native"}, TypeError, "ContentOrigin"),
        ({"producer": "parser"}, TypeError, "ProducerIdentity"),
    ],
)
def test_page_layout_rejects_incomplete_or_unsupported_coordinates(
    changes: dict[str, object], error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        _layout(**changes)


def test_source_location_uses_half_open_decoded_text_offsets() -> None:
    location = _location(start_offset=4, end_offset=12)

    assert location.start_offset == 4
    assert location.end_offset == 12


def test_source_location_converts_to_provenance_catalogue_value() -> None:
    location = _location()

    known_span = location.as_known_source_span()

    assert known_span.source_span_id == FIRST_SPAN_ID
    assert known_span.document_version_id == DOCUMENT_VERSION_ID


def test_source_location_retains_page_box_and_content_origin() -> None:
    box = _bounding_box()
    location = _location(
        page_number=1,
        bounding_box=box,
        content_origin=ContentOrigin.OCR,
    )

    assert location.page_number == 1
    assert location.bounding_box is box
    assert location.content_origin is ContentOrigin.OCR


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"source_span_id": "span"}, TypeError, "SourceSpanId"),
        ({"document_version_id": "version"}, TypeError, "DocumentVersionId"),
        ({"parsed_record_id": "record"}, TypeError, "RecordId"),
        ({"start_offset": -1}, ValueError, "start_offset"),
        ({"start_offset": True}, TypeError, "start_offset"),
        ({"start_offset": 5, "end_offset": 4}, ValueError, "precede"),
        ({"page_number": 0}, ValueError, "page_number"),
        ({"bounding_box": _bounding_box()}, ValueError, "page_number"),
        ({"bounding_box": "box"}, TypeError, "BoundingBox"),
        ({"content_origin": "native"}, TypeError, "ContentOrigin"),
    ],
)
def test_source_location_rejects_invalid_identity_or_coordinates(
    changes: dict[str, object], error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        _location(**changes)


def test_structural_block_retains_order_kind_text_and_span() -> None:
    block = StructuralBlock(
        ordinal=0,
        kind=StructuralBlockKind.FIGURE_CAPTION,
        text="Figure 1",
        source_span_id=FIRST_SPAN_ID,
    )

    assert block.ordinal == 0
    assert block.kind is StructuralBlockKind.FIGURE_CAPTION
    assert block.text == "Figure 1"
    assert block.source_span_id == FIRST_SPAN_ID


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"ordinal": -1}, ValueError, "ordinal"),
        ({"ordinal": True}, TypeError, "ordinal"),
        ({"kind": "paragraph"}, TypeError, "StructuralBlockKind"),
        ({"text": None}, TypeError, "text"),
        ({"source_span_id": "span"}, TypeError, "SourceSpanId"),
    ],
)
def test_structural_block_rejects_invalid_values(
    changes: dict[str, object], error_type: type[Exception], match: str
) -> None:
    values: dict[str, object] = {
        "ordinal": 0,
        "kind": StructuralBlockKind.PARAGRAPH,
        "text": "text",
        "source_span_id": FIRST_SPAN_ID,
    }
    values.update(changes)

    with pytest.raises(error_type, match=match):
        StructuralBlock(**values)  # type: ignore[arg-type]
