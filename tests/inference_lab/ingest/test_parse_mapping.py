"""Tests for pure parser-to-storage record mapping."""

from dataclasses import replace
from typing import get_args

import pytest

import inference_lab.ingest.parse_mapping as parse_mapping
from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    BoundingBox,
    CoordinateOrigin,
    CoordinateUnit,
    MappedParseRecords,
    PageLayoutDescriptor,
    PageRotation,
    ParseOutcome,
    ParsePersistenceRequest,
    ParserFailure,
    ParserFailureCategory,
    ParserFailureSeverity,
    ParserFailureStage,
    ParserFailureUnitKind,
    ParseStatus,
    StructuralBlockKind,
    TextNativeParser,
    map_parse_persistence_request,
)
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    RecordId,
)
from inference_lab.storage import (
    CollectionAvailability,
    ContentOrigin,
    ProducerIdentity,
)
from inference_lab.storage import (
    CoordinateOrigin as StorageCoordinateOrigin,
)
from inference_lab.storage import (
    CoordinateUnit as StorageCoordinateUnit,
)
from inference_lab.storage import (
    PageRotation as StoragePageRotation,
)
from inference_lab.storage import (
    ParseFailureCategory as StorageParseFailureCategory,
)
from inference_lab.storage import (
    ParseFailureSeverity as StorageParseFailureSeverity,
)
from inference_lab.storage import (
    ParseFailureStage as StorageParseFailureStage,
)
from inference_lab.storage import (
    ParseFailureUnitKind as StorageParseFailureUnitKind,
)
from inference_lab.storage import (
    ParseStatus as StorageParseStatus,
)
from inference_lab.storage import (
    StructuralBlockKind as StorageStructuralBlockKind,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252


def _successful_request(
    *,
    source_name: str = "basic.txt",
    parsed_record_id: RecordId | None = None,
    supersedes_record_id: RecordId | None = None,
) -> ParsePersistenceRequest:
    source = fixture_source(source_name)
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    return ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=parsed_record_id or RecordId.generate(),
        supersedes_record_id=supersedes_record_id,
    )


def test_successful_parse_maps_every_representation_and_outcome_field() -> None:
    predecessor = RecordId.generate()
    request = _successful_request(supersedes_record_id=predecessor)

    mapped = map_parse_persistence_request(request)

    representation = mapped.representation
    assert representation is not None
    assert representation.record_id == request.parsed_record_id
    assert representation.document_id == request.parser_input.document_id
    assert (
        representation.document_version_id == request.parser_input.document_version_id
    )
    assert representation.source_text == "First paragraph.\n\nSecond paragraph.\n"
    assert representation.text == representation.source_text
    assert representation.parser.implementation == "text-native"
    assert representation.parser.contract_version == "3.0.0"
    assert representation.parser.artefact_identity is None
    assert representation.parser.artefact_revision is None
    assert representation.normalisation_version == "1.0.0"
    assert representation.language is None
    assert representation.page_count is None
    assert representation.character_count == 36
    assert representation.token_count is None
    assert representation.page_collection_availability is (
        CollectionAvailability.UNAVAILABLE
    )
    assert representation.section_collection_availability is (
        CollectionAvailability.UNAVAILABLE
    )
    assert representation.paragraph_collection_availability is (
        CollectionAvailability.AVAILABLE
    )
    assert representation.ocr_producer is None
    assert representation.supersedes_record_id == predecessor

    assert [block.ordinal for block in representation.blocks] == [0, 1]
    assert [block.kind for block in representation.blocks] == [
        StorageStructuralBlockKind.PARAGRAPH,
        StorageStructuralBlockKind.PARAGRAPH,
    ]
    assert [block.text for block in representation.blocks] == [
        "First paragraph.",
        "Second paragraph.\n",
    ]
    assert [block.source_span_id for block in representation.blocks] != [
        location.source_span_id for location in representation.source_locations
    ]
    assert representation.source_locations == tuple(
        sorted(
            representation.source_locations,
            key=lambda location: location.source_span_id.as_str(),
        )
    )
    assert all(
        location.content_origin is ContentOrigin.NATIVE
        and location.parsed_record_id == representation.record_id
        for location in representation.source_locations
    )
    assert representation.page_layouts == ()

    assert mapped.outcome.parse_id == request.parse_id
    assert mapped.outcome.document_id == representation.document_id
    assert mapped.outcome.document_version_id == representation.document_version_id
    assert mapped.outcome.status is StorageParseStatus.SUCCEEDED
    assert mapped.outcome.parser == representation.parser
    assert mapped.outcome.normalisation_version == "1.0.0"
    assert mapped.outcome.parsed_record_id == representation.record_id
    assert mapped.outcome.failures == ()
    assert mapped.outcome.ocr_producer is None


def test_empty_available_collection_remains_distinct_from_unavailable() -> None:
    mapped = map_parse_persistence_request(_successful_request(source_name="empty.txt"))

    representation = mapped.representation
    assert representation is not None
    assert representation.blocks == ()
    assert representation.pages is None
    assert representation.sections is None
    assert representation.paragraphs == ()


def test_native_page_layout_and_bounding_box_map_without_conversion() -> None:
    request = _successful_request()
    document = request.outcome.document
    assert document is not None
    first_block = document.blocks[0]
    span = replace(
        first_block.source_span,
        page_number=1,
        bounding_box=BoundingBox(
            left=10.5,
            top=20.25,
            right=80.75,
            bottom=90.5,
        ),
    )
    blocks = (replace(first_block, source_span=span), *document.blocks[1:])
    layout = PageLayoutDescriptor(
        page_number=1,
        coordinate_unit="image_pixels",
        coordinate_origin="top_left",
        page_width=1200.0,
        page_height=1600.0,
        rotation=90,
    )
    represented_outcome = replace(
        request.outcome,
        document=replace(
            document,
            blocks=blocks,
            paragraphs=blocks,
            page_layouts=(layout,),
            page_count=1,
        ),
    )
    request = replace(request, outcome=represented_outcome)

    mapped = map_parse_persistence_request(request)

    representation = mapped.representation
    assert representation is not None
    mapped_layout = representation.page_layouts[0]
    assert mapped_layout.page_number == 1
    assert mapped_layout.coordinate_unit is StorageCoordinateUnit.IMAGE_PIXELS
    assert mapped_layout.coordinate_origin is StorageCoordinateOrigin.TOP_LEFT
    assert mapped_layout.page_width == 1200.0
    assert mapped_layout.page_height == 1600.0
    assert mapped_layout.rotation is StoragePageRotation.DEGREES_90
    assert mapped_layout.content_origin is ContentOrigin.NATIVE
    assert mapped_layout.producer == representation.parser

    mapped_location = next(
        location
        for location in representation.source_locations
        if location.source_span_id == span.source_span_id
    )
    assert mapped_location.page_number == 1
    assert mapped_location.bounding_box is not None
    assert (
        mapped_location.bounding_box.left,
        mapped_location.bounding_box.top,
        mapped_location.bounding_box.right,
        mapped_location.bounding_box.bottom,
    ) == (10.5, 20.25, 80.75, 90.5)


@pytest.mark.parametrize(
    "category",
    ["malformed_unit", "unsupported_structure"],
)
def test_recoverable_failures_map_with_omitted_unit_details(
    category: ParserFailureCategory,
) -> None:
    request = _successful_request()
    document = request.outcome.document
    assert document is not None
    failures = (
        ParserFailure(
            category=category,
            severity="recoverable",
            stage="structure",
            unit_kind="table",
            unit_index=3,
        ),
    )
    request = replace(
        request,
        outcome=ParseOutcome(
            status="completed_with_recoverable_failures",
            document=document,
            failures=failures,
        ),
    )

    mapped = map_parse_persistence_request(request)

    assert mapped.representation is not None
    assert mapped.outcome.status is (
        StorageParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES
    )
    assert len(mapped.outcome.failures) == 1
    failure = mapped.outcome.failures[0]
    assert failure.category is StorageParseFailureCategory(category)
    assert failure.severity is StorageParseFailureSeverity.RECOVERABLE
    assert failure.stage is StorageParseFailureStage.STRUCTURE
    assert failure.unit_kind is StorageParseFailureUnitKind.TABLE
    assert failure.unit_index == 3


@pytest.mark.parametrize(
    ("category", "stage"),
    [
        ("invalid_input", "input"),
        ("unsupported_format", "input"),
        ("invalid_encoding", "input"),
        ("source_too_large", "input"),
        ("decode_failed", "decode"),
        ("malformed_source", "structure"),
        ("invalid_output", "output"),
    ],
)
def test_fatal_outcome_maps_without_partial_representation(
    category: ParserFailureCategory,
    stage: ParserFailureStage,
) -> None:
    source = fixture_source("malformed_utf8.bin")
    outcome = ParseOutcome(
        status="failed",
        document=None,
        failures=(ParserFailure(category=category, severity="fatal", stage=stage),),
    )
    request = ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=None,
    )

    mapped = map_parse_persistence_request(request)

    assert mapped.representation is None
    assert mapped.outcome.status is StorageParseStatus.FAILED
    assert mapped.outcome.parsed_record_id is None
    assert mapped.outcome.document_id == source.document_id
    assert mapped.outcome.document_version_id == source.document_version_id
    assert mapped.outcome.parser.implementation == "text-native"
    assert mapped.outcome.parser.contract_version == "3.0.0"
    assert mapped.outcome.normalisation_version == "1.0.0"
    assert mapped.outcome.failures[0].category is StorageParseFailureCategory(category)
    assert mapped.outcome.failures[0].stage is StorageParseFailureStage(stage)
    assert mapped.outcome.failures[0].severity is StorageParseFailureSeverity.FATAL


def test_forward_mapping_rejects_wrong_request_type_without_payload() -> None:
    with pytest.raises(TypeError, match="ParsePersistenceRequest") as error:
        map_parse_persistence_request(object())  # type: ignore[arg-type]

    assert error.value.__cause__ is None


def test_mapped_records_reject_wrong_record_types_without_payload() -> None:
    mapped = map_parse_persistence_request(_successful_request())

    with pytest.raises(TypeError, match="representation") as representation_error:
        MappedParseRecords(
            representation=object(),  # type: ignore[arg-type]
            outcome=mapped.outcome,
        )
    with pytest.raises(TypeError, match="outcome") as outcome_error:
        MappedParseRecords(
            representation=mapped.representation,
            outcome=object(),  # type: ignore[arg-type]
        )

    assert representation_error.value.__cause__ is None
    assert outcome_error.value.__cause__ is None


def test_mapped_records_reject_link_to_missing_representation() -> None:
    mapped = map_parse_persistence_request(_successful_request())

    with pytest.raises(ValueError, match="missing representation") as error:
        MappedParseRecords(representation=None, outcome=mapped.outcome)

    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parsed_record_id", RecordId.generate(), "identities"),
        ("document_id", DocumentId.generate(), "documents"),
        ("document_version_id", DocumentVersionId.generate(), "versions"),
        (
            "parser",
            ProducerIdentity(implementation="other-parser", contract_version="1.0.0"),
            "parsers",
        ),
        ("normalisation_version", "2.0.0", "normalisation"),
    ],
)
def test_mapped_records_reject_inconsistent_linked_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    mapped = map_parse_persistence_request(_successful_request())
    inconsistent_outcome = replace(mapped.outcome, **{field: value})

    with pytest.raises(ValueError, match=message) as error:
        MappedParseRecords(
            representation=mapped.representation,
            outcome=inconsistent_outcome,
        )

    assert error.value.__cause__ is None


def test_forward_mapping_tables_exhaust_both_closed_vocabularies() -> None:
    expected_mappings = (
        (
            parse_mapping._BLOCK_KIND_MAP,
            set(get_args(StructuralBlockKind)),
            set(StorageStructuralBlockKind),
        ),
        (
            parse_mapping._COORDINATE_UNIT_MAP,
            set(get_args(CoordinateUnit)),
            set(StorageCoordinateUnit),
        ),
        (
            parse_mapping._COORDINATE_ORIGIN_MAP,
            set(get_args(CoordinateOrigin)),
            set(StorageCoordinateOrigin),
        ),
        (
            parse_mapping._PAGE_ROTATION_MAP,
            set(get_args(PageRotation)),
            set(StoragePageRotation),
        ),
        (
            parse_mapping._STATUS_MAP,
            set(get_args(ParseStatus)),
            set(StorageParseStatus),
        ),
        (
            parse_mapping._FAILURE_CATEGORY_MAP,
            set(get_args(ParserFailureCategory)),
            set(StorageParseFailureCategory),
        ),
        (
            parse_mapping._FAILURE_SEVERITY_MAP,
            set(get_args(ParserFailureSeverity)),
            set(StorageParseFailureSeverity),
        ),
        (
            parse_mapping._FAILURE_STAGE_MAP,
            set(get_args(ParserFailureStage)),
            set(StorageParseFailureStage),
        ),
        (
            parse_mapping._FAILURE_UNIT_KIND_MAP,
            set(get_args(ParserFailureUnitKind)),
            set(StorageParseFailureUnitKind),
        ),
    )

    for mapping, ingest_values, storage_values in expected_mappings:
        assert set(mapping) == ingest_values
        assert set(mapping.values()) == storage_values
