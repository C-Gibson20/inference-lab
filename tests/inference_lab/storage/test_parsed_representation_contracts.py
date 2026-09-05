"""Tests for the complete immutable parsed-representation contract."""

from dataclasses import FrozenInstanceError, replace

import pytest

from inference_lab.provenance import DocumentVersionId, RecordId
from inference_lab.storage import (
    BoundingBox,
    CollectionAvailability,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    ParsedRepresentation,
    SourceLocation,
    StructuralBlock,
    StructuralBlockKind,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    OCR_PRODUCER,
    PARSED_RECORD_ID,
    PARSER,
    SECOND_SPAN_ID,
    SOURCE_TEXT,
    parsed_representation,
)


def _empty_representation(**changes: object) -> ParsedRepresentation:
    values: dict[str, object] = {
        "record_id": PARSED_RECORD_ID,
        "document_id": DOCUMENT_ID,
        "document_version_id": DOCUMENT_VERSION_ID,
        "source_text": "",
        "text": "",
        "parser": PARSER,
        "normalisation_version": "1.0.0",
        "language": None,
        "page_count": None,
        "character_count": 0,
        "token_count": None,
        "page_collection_availability": CollectionAvailability.UNAVAILABLE,
        "section_collection_availability": CollectionAvailability.UNAVAILABLE,
        "paragraph_collection_availability": CollectionAvailability.UNAVAILABLE,
        "blocks": (),
        "source_locations": (),
        "page_layouts": (),
    }
    values.update(changes)
    return ParsedRepresentation(**values)  # type: ignore[arg-type]


def _layout(**changes: object) -> PageLayoutDescriptor:
    values: dict[str, object] = {
        "page_number": 1,
        "coordinate_unit": CoordinateUnit.PDF_POINTS,
        "coordinate_origin": CoordinateOrigin.BOTTOM_LEFT,
        "page_width": 612.0,
        "page_height": 792.0,
        "rotation": PageRotation.DEGREES_0,
        "content_origin": ContentOrigin.NATIVE,
        "producer": PARSER,
    }
    values.update(changes)
    return PageLayoutDescriptor(**values)  # type: ignore[arg-type]


def _located_representation(
    *,
    location: SourceLocation,
    layout: PageLayoutDescriptor,
    ocr: bool = False,
) -> ParsedRepresentation:
    return ParsedRepresentation(
        record_id=PARSED_RECORD_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        source_text="Page",
        text="Page",
        parser=PARSER,
        normalisation_version="1.0.0",
        language=None,
        page_count=None,
        character_count=4,
        token_count=None,
        page_collection_availability=CollectionAvailability.AVAILABLE,
        section_collection_availability=CollectionAvailability.UNAVAILABLE,
        paragraph_collection_availability=CollectionAvailability.UNAVAILABLE,
        blocks=(
            StructuralBlock(
                ordinal=0,
                kind=StructuralBlockKind.PAGE,
                text="Page",
                source_span_id=FIRST_SPAN_ID,
            ),
        ),
        source_locations=(location,),
        page_layouts=(layout,),
        ocr_producer=OCR_PRODUCER if ocr else None,
    )


def test_representation_retains_decoded_and_canonical_text() -> None:
    representation = parsed_representation()

    assert representation.source_text == SOURCE_TEXT
    assert representation.text == "Café\n\nResults"
    assert representation.source_locations[1].start_offset == 9
    assert representation.blocks[1].text == "Results"
    with pytest.raises(FrozenInstanceError):
        representation.language = "fr"  # type: ignore[misc]


def test_empty_representation_has_no_fabricated_lineage() -> None:
    representation = _empty_representation()

    assert representation.pages is None
    assert representation.sections is None
    assert representation.paragraphs is None
    assert representation.known_source_spans() == ()
    assert not hasattr(representation, "lineage")


def test_typed_collections_are_exact_ordered_projections() -> None:
    representation = parsed_representation()

    assert representation.pages == ()
    assert representation.sections is None
    assert representation.paragraphs == representation.blocks


def test_source_catalogue_converts_directly_to_provenance_values() -> None:
    known_spans = parsed_representation().known_source_spans()

    assert tuple(span.source_span_id for span in known_spans) == (
        FIRST_SPAN_ID,
        SECOND_SPAN_ID,
    )
    assert all(span.document_version_id == DOCUMENT_VERSION_ID for span in known_spans)


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"text": "not canonical"}, ValueError, "canonical form"),
        ({"character_count": 14}, ValueError, "canonical text length"),
        ({"normalisation_version": "1"}, ValueError, "normalisation_version"),
        ({"page_count": -1}, ValueError, "page_count"),
        ({"token_count": True}, TypeError, "token_count"),
        ({"language": "  "}, ValueError, "language"),
        ({"blocks": []}, TypeError, "blocks"),
        ({"record_schema_version": "2.0.0"}, ValueError, "record_schema_version"),
    ],
)
def test_representation_rejects_invalid_scalar_or_collection_fields(
    changes: dict[str, object], error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        replace(parsed_representation(), **changes)  # type: ignore[arg-type]


def test_representation_cannot_supersede_itself() -> None:
    with pytest.raises(ValueError, match="supersede itself"):
        replace(
            parsed_representation(),
            supersedes_record_id=PARSED_RECORD_ID,
        )


def test_blocks_require_contiguous_ordinals_and_source_order() -> None:
    representation = parsed_representation()
    wrong_ordinal = replace(representation.blocks[1], ordinal=2)
    reversed_locations = tuple(reversed(representation.source_locations))

    with pytest.raises(ValueError, match="contiguous"):
        replace(representation, blocks=(representation.blocks[0], wrong_ordinal))
    with pytest.raises(ValueError, match="canonical identifier order"):
        replace(representation, source_locations=reversed_locations)
    with pytest.raises(ValueError, match="source order"):
        replace(
            representation,
            blocks=tuple(
                replace(block, ordinal=ordinal)
                for ordinal, block in enumerate(reversed(representation.blocks))
            ),
        )


def test_blocks_and_locations_must_form_one_exact_catalogue() -> None:
    representation = parsed_representation()

    with pytest.raises(ValueError, match="same spans"):
        replace(representation, blocks=representation.blocks[:1])
    with pytest.raises(ValueError, match="duplicate spans"):
        replace(
            representation,
            source_locations=(
                representation.source_locations[0],
                representation.source_locations[0],
            ),
        )
    with pytest.raises(ValueError, match="canonical source slice"):
        replace(
            representation,
            blocks=(
                replace(representation.blocks[0], text="Cafe"),
                representation.blocks[1],
            ),
        )


def test_locations_must_belong_to_representation_and_decoded_text() -> None:
    representation = parsed_representation()
    other_version = DocumentVersionId.from_str("20000000-0000-4000-8000-000000000002")
    other_record = RecordId.from_str("40000000-0000-4000-8000-000000000002")

    for location, match in (
        (
            replace(
                representation.source_locations[0],
                document_version_id=other_version,
            ),
            "wrong document version",
        ),
        (
            replace(
                representation.source_locations[0],
                parsed_record_id=other_record,
            ),
            "wrong parsed record",
        ),
        (
            replace(
                representation.source_locations[0],
                end_offset=len(SOURCE_TEXT) + 1,
            ),
            "exceeds decoded source text",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            replace(
                representation,
                source_locations=(location, representation.source_locations[1]),
            )


def test_page_location_can_exist_when_total_page_count_is_unknown() -> None:
    box = BoundingBox(left=10.0, top=20.0, right=80.0, bottom=40.0)
    location = SourceLocation(
        source_span_id=FIRST_SPAN_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        parsed_record_id=PARSED_RECORD_ID,
        start_offset=0,
        end_offset=4,
        content_origin=ContentOrigin.NATIVE,
        page_number=1,
        bounding_box=box,
    )

    representation = _located_representation(location=location, layout=_layout())

    assert representation.page_count is None
    assert representation.source_locations[0].bounding_box is box


def test_bounding_box_requires_matching_layout_and_fits_coordinate_space() -> None:
    box = BoundingBox(left=10.0, top=20.0, right=80.0, bottom=40.0)
    location = SourceLocation(
        source_span_id=FIRST_SPAN_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        parsed_record_id=PARSED_RECORD_ID,
        start_offset=0,
        end_offset=4,
        content_origin=ContentOrigin.NATIVE,
        page_number=1,
        bounding_box=box,
    )

    with pytest.raises(ValueError, match="layout descriptor"):
        _empty_representation(
            source_text="Page",
            text="Page",
            character_count=4,
            page_count=1,
            blocks=(
                StructuralBlock(
                    ordinal=0,
                    kind=StructuralBlockKind.PAGE,
                    text="Page",
                    source_span_id=FIRST_SPAN_ID,
                ),
            ),
            source_locations=(location,),
            page_collection_availability=CollectionAvailability.AVAILABLE,
        )
    with pytest.raises(ValueError, match="outside"):
        _located_representation(
            location=replace(
                location,
                bounding_box=BoundingBox(
                    left=10.0,
                    top=20.0,
                    right=700.0,
                    bottom=40.0,
                ),
            ),
            layout=_layout(),
        )


def test_layouts_are_unique_ordered_and_within_known_page_count() -> None:
    representation = _empty_representation(
        page_count=2,
        page_layouts=(_layout(page_number=1), _layout(page_number=2)),
    )

    with pytest.raises(ValueError, match="at most one"):
        replace(representation, page_layouts=(_layout(), _layout()))
    with pytest.raises(ValueError, match="page-number order"):
        replace(
            representation, page_layouts=tuple(reversed(representation.page_layouts))
        )
    with pytest.raises(ValueError, match="outside"):
        replace(representation, page_layouts=(_layout(page_number=3),))


def test_page_blocks_require_page_numbers() -> None:
    representation = parsed_representation()

    with pytest.raises(ValueError, match="page blocks"):
        replace(
            representation,
            blocks=(
                replace(representation.blocks[0], kind=StructuralBlockKind.PAGE),
                representation.blocks[1],
            ),
        )


def test_ocr_locations_and_layouts_require_the_declared_ocr_producer() -> None:
    location = SourceLocation(
        source_span_id=FIRST_SPAN_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        parsed_record_id=PARSED_RECORD_ID,
        start_offset=0,
        end_offset=4,
        content_origin=ContentOrigin.OCR,
        page_number=1,
    )

    with pytest.raises(ValueError, match="OCR source locations"):
        _located_representation(location=location, layout=_layout())
    with pytest.raises(ValueError, match="OCR page layout"):
        _located_representation(
            location=location,
            layout=_layout(content_origin=ContentOrigin.OCR, producer=PARSER),
            ocr=True,
        )
    representation = _located_representation(
        location=location,
        layout=_layout(content_origin=ContentOrigin.OCR, producer=OCR_PRODUCER),
        ocr=True,
    )

    assert representation.ocr_producer == OCR_PRODUCER


def test_unavailable_typed_collection_cannot_contain_matching_blocks() -> None:
    representation = parsed_representation()

    with pytest.raises(ValueError, match="paragraph_collection_availability"):
        replace(
            representation,
            paragraph_collection_availability=CollectionAvailability.UNAVAILABLE,
        )
