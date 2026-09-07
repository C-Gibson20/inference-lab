"""Tests for strict storage-to-parser outcome reconstruction."""

from dataclasses import replace
from typing import cast, get_args

import pytest

import inference_lab.ingest.parse_reconstruction as parse_reconstruction
from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    BoundingBox,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    ParseOutcome,
    ParsePersistenceRequest,
    ParseReconstructionError,
    ParserFailure,
    ParserFailureCategory,
    ParserFailureSeverity,
    ParserFailureStage,
    ParserFailureUnitKind,
    ParserVersion,
    ParseStatus,
    ReconstructedParseOutcome,
    StructuralBlockKind,
    TextNativeParser,
    map_parse_persistence_request,
    reconstruct_parse_outcome,
    serialise_parsed_document,
)
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage import ContentOrigin, ParsedRepresentation, ProducerIdentity
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
    ParseOutcome as StorageParseOutcome,
)
from inference_lab.storage import (
    ParseStatus as StorageParseStatus,
)
from inference_lab.storage import (
    StructuralBlockKind as StorageStructuralBlockKind,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252


def _successful_request(source_name: str = "basic.txt") -> ParsePersistenceRequest:
    source = fixture_source(source_name)
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    return ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )


def _request_with_layout() -> ParsePersistenceRequest:
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
    outcome = replace(
        request.outcome,
        document=replace(
            document,
            blocks=blocks,
            paragraphs=blocks,
            page_layouts=(layout,),
            page_count=1,
        ),
    )
    return replace(request, outcome=outcome)


def _recoverable_request() -> ParsePersistenceRequest:
    request = _successful_request()
    document = request.outcome.document
    assert document is not None
    outcome = ParseOutcome(
        status="completed_with_recoverable_failures",
        document=document,
        failures=(
            ParserFailure(
                category="unsupported_structure",
                severity="recoverable",
                stage="structure",
                unit_kind="table",
                unit_index=3,
            ),
        ),
    )
    return replace(request, outcome=outcome)


def _fatal_request() -> ParsePersistenceRequest:
    source = fixture_source("malformed_utf8.bin")
    outcome = ParseOutcome(
        status="failed",
        document=None,
        failures=(
            ParserFailure(
                category="decode_failed",
                severity="fatal",
                stage="decode",
            ),
        ),
    )
    return ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=None,
    )


@pytest.mark.parametrize("source_name", ["basic.txt", "empty.txt"])
def test_represented_outcome_round_trips_exact_parser_contract_bytes(
    source_name: str,
) -> None:
    request = _successful_request(source_name)
    mapped = map_parse_persistence_request(request)

    reconstructed = reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert reconstructed.parser_version == request.parser_version
    assert reconstructed.outcome == request.outcome
    original_document = request.outcome.document
    reconstructed_document = reconstructed.outcome.document
    assert original_document is not None
    assert reconstructed_document is not None
    assert serialise_parsed_document(reconstructed_document) == (
        serialise_parsed_document(original_document)
    )
    assert reconstructed_document.pages is None
    assert reconstructed_document.sections is None
    assert reconstructed_document.paragraphs is not None


def test_page_layout_and_bounding_box_round_trip_without_conversion() -> None:
    request = _request_with_layout()
    mapped = map_parse_persistence_request(request)

    reconstructed = reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert reconstructed.outcome == request.outcome
    document = reconstructed.outcome.document
    assert document is not None
    assert document.page_layouts == request.outcome.document.page_layouts  # type: ignore[union-attr]
    assert document.blocks[0].source_span.bounding_box == BoundingBox(
        left=10.5,
        top=20.25,
        right=80.75,
        bottom=90.5,
    )


def test_recoverable_outcome_round_trips_failure_and_representation() -> None:
    request = _recoverable_request()
    mapped = map_parse_persistence_request(request)

    reconstructed = reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert reconstructed.parser_version == request.parser_version
    assert reconstructed.outcome == request.outcome
    assert reconstructed.outcome.status == "completed_with_recoverable_failures"
    assert reconstructed.outcome.failures[0].unit_kind == "table"
    assert reconstructed.outcome.failures[0].unit_index == 3


def test_fatal_outcome_reconstructs_parser_version_without_representation() -> None:
    request = _fatal_request()
    mapped = map_parse_persistence_request(request)

    reconstructed = reconstruct_parse_outcome(mapped.outcome, None)

    assert reconstructed.parser_version == request.parser_version
    assert reconstructed.outcome == request.outcome
    assert reconstructed.outcome.document is None


@pytest.mark.parametrize(
    ("outcome", "representation", "category"),
    [
        (object(), None, "invalid_outcome"),
        (
            map_parse_persistence_request(_successful_request()).outcome,
            object(),
            "invalid_representation",
        ),
    ],
)
def test_reconstruction_rejects_wrong_record_types(
    outcome: object,
    representation: object | None,
    category: str,
) -> None:
    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(
            cast("StorageParseOutcome", outcome),
            cast("ParsedRepresentation | None", representation),
        )

    assert error.value.category == category
    assert str(error.value) == category
    assert error.value.__cause__ is None


def test_reconstruction_rejects_missing_linked_representation() -> None:
    mapped = map_parse_persistence_request(_successful_request())

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, None)

    assert error.value.category == "missing_representation"


def test_reconstruction_rejects_representation_for_fatal_outcome() -> None:
    successful = map_parse_persistence_request(_successful_request())
    fatal = map_parse_persistence_request(_fatal_request())
    assert successful.representation is not None

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(fatal.outcome, successful.representation)

    assert error.value.category == "unexpected_representation"


@pytest.mark.parametrize(
    ("field", "value", "category"),
    [
        ("parsed_record_id", RecordId.generate(), "identity_mismatch"),
        ("document_id", DocumentId.generate(), "identity_mismatch"),
        ("document_version_id", DocumentVersionId.generate(), "identity_mismatch"),
        (
            "parser",
            ProducerIdentity(implementation="other-parser", contract_version="1.0.0"),
            "producer_mismatch",
        ),
        ("normalisation_version", "2.0.0", "producer_mismatch"),
    ],
)
def test_reconstruction_rejects_disagreeing_linked_records(
    field: str,
    value: object,
    category: str,
) -> None:
    mapped = map_parse_persistence_request(_successful_request())
    inconsistent_outcome = replace(mapped.outcome, **{field: value})

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(inconsistent_outcome, mapped.representation)

    assert error.value.category == category


def test_reconstruction_rejects_ocr_attribution_on_either_record() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    ocr_producer = ProducerIdentity(
        implementation="ocr-engine",
        contract_version="1.0.0",
    )

    with pytest.raises(ParseReconstructionError) as outcome_error:
        reconstruct_parse_outcome(
            replace(mapped.outcome, ocr_producer=ocr_producer),
            mapped.representation,
        )
    with pytest.raises(ParseReconstructionError) as representation_error:
        reconstruct_parse_outcome(
            mapped.outcome,
            replace(mapped.representation, ocr_producer=ocr_producer),
        )

    assert outcome_error.value.category == "unsupported_attribution"
    assert representation_error.value.category == "unsupported_attribution"


def test_reconstruction_rejects_parser_artefact_attribution() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    attributed_parser = ProducerIdentity(
        implementation="text-native",
        contract_version="3.0.0",
        artefact_identity="model",
        artefact_revision="revision",
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(
            replace(mapped.outcome, parser=attributed_parser),
            replace(mapped.representation, parser=attributed_parser),
        )

    assert error.value.category == "unsupported_attribution"


def test_reconstruction_rejects_layout_from_different_producer() -> None:
    mapped = map_parse_persistence_request(_request_with_layout())
    assert mapped.representation is not None
    layout = mapped.representation.page_layouts[0]
    other_producer = ProducerIdentity(
        implementation="layout-engine",
        contract_version="1.0.0",
    )
    representation = replace(
        mapped.representation,
        page_layouts=(replace(layout, producer=other_producer),),
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, representation)

    assert error.value.category == "producer_mismatch"


def test_reconstruction_rejects_noncanonical_location_order() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    representation = mapped.representation
    object.__setattr__(
        representation,
        "source_locations",
        tuple(reversed(representation.source_locations)),
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_noncontiguous_block_ordinals() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    object.__setattr__(mapped.representation.blocks[1], "ordinal", 3)

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_duplicate_source_locations() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    location = mapped.representation.source_locations[0]
    object.__setattr__(mapped.representation, "source_locations", (location, location))

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_extra_source_locations() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    object.__setattr__(
        mapped.representation,
        "blocks",
        mapped.representation.blocks[:1],
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_location_owned_by_another_record() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    object.__setattr__(
        mapped.representation.source_locations[0],
        "parsed_record_id",
        RecordId.generate(),
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "identity_mismatch"


def test_reconstruction_rejects_non_native_source_location() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    object.__setattr__(
        mapped.representation.source_locations[0],
        "content_origin",
        ContentOrigin.OCR,
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "unsupported_attribution"


def test_reconstruction_rejects_non_native_page_layout() -> None:
    mapped = map_parse_persistence_request(_request_with_layout())
    assert mapped.representation is not None
    object.__setattr__(
        mapped.representation.page_layouts[0],
        "content_origin",
        ContentOrigin.OCR,
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "unsupported_attribution"


def test_reconstruction_rejects_out_of_order_page_layouts() -> None:
    mapped = map_parse_persistence_request(_request_with_layout())
    assert mapped.representation is not None
    layout = mapped.representation.page_layouts[0]
    object.__setattr__(
        mapped.representation,
        "page_layouts",
        (replace(layout, page_number=2), layout),
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_invalid_typed_collection_availability() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    object.__setattr__(
        mapped.representation,
        "page_collection_availability",
        "future_availability",
    )

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "unsupported_value"


def test_reconstruction_rejects_invalid_parser_version_components() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    object.__setattr__(mapped.outcome.parser, "contract_version", "invalid")

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_invalid_bounding_box() -> None:
    mapped = map_parse_persistence_request(_request_with_layout())
    assert mapped.representation is not None
    box = next(
        location.bounding_box
        for location in mapped.representation.source_locations
        if location.bounding_box is not None
    )
    object.__setattr__(box, "right", -1.0)

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_invalid_page_layout() -> None:
    mapped = map_parse_persistence_request(_request_with_layout())
    assert mapped.representation is not None
    object.__setattr__(mapped.representation.page_layouts[0], "page_width", -1.0)

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_invalid_storage_failure() -> None:
    mapped = map_parse_persistence_request(_recoverable_request())
    object.__setattr__(mapped.outcome.failures[0], "unit_index", -1)

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_inconsistent_storage_outcome_status() -> None:
    mapped = map_parse_persistence_request(_recoverable_request())
    object.__setattr__(mapped.outcome, "failures", ())

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_unmatched_source_span() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    block = mapped.representation.blocks[0]
    object.__setattr__(block, "source_span_id", SourceSpanId.generate())

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_rejects_unknown_storage_enum_without_coercion() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    block = mapped.representation.blocks[0]
    object.__setattr__(block, "kind", "future_kind")

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "unsupported_value"


def test_reconstruction_does_not_repair_changed_canonical_text() -> None:
    mapped = map_parse_persistence_request(_successful_request())
    assert mapped.representation is not None
    object.__setattr__(mapped.representation, "text", "changed")

    with pytest.raises(ParseReconstructionError) as error:
        reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    assert error.value.category == "invalid_structure"


def test_reconstruction_result_validates_its_public_contract() -> None:
    request = _successful_request()
    mapped = map_parse_persistence_request(request)
    reconstructed = reconstruct_parse_outcome(mapped.outcome, mapped.representation)

    with pytest.raises(TypeError, match="ParserVersion"):
        ReconstructedParseOutcome(
            parser_version=cast("ParserVersion", object()),
            outcome=reconstructed.outcome,
        )
    with pytest.raises(TypeError, match="ParseOutcome"):
        ReconstructedParseOutcome(
            parser_version=reconstructed.parser_version,
            outcome=cast("ParseOutcome", object()),
        )
    with pytest.raises(ValueError, match="must agree"):
        ReconstructedParseOutcome(
            parser_version=ParserVersion(
                implementation="other-parser",
                contract_version="1.0.0",
                normalisation_version="1.0.0",
            ),
            outcome=reconstructed.outcome,
        )


def test_inverse_mapping_tables_exhaust_both_closed_vocabularies() -> None:
    expected_mappings = (
        (
            parse_reconstruction._BLOCK_KIND_MAP,
            set(StorageStructuralBlockKind),
            set(get_args(StructuralBlockKind)),
        ),
        (
            parse_reconstruction._COORDINATE_UNIT_MAP,
            set(StorageCoordinateUnit),
            set(get_args(CoordinateUnit)),
        ),
        (
            parse_reconstruction._COORDINATE_ORIGIN_MAP,
            set(StorageCoordinateOrigin),
            set(get_args(CoordinateOrigin)),
        ),
        (
            parse_reconstruction._PAGE_ROTATION_MAP,
            set(StoragePageRotation),
            set(get_args(PageRotation)),
        ),
        (
            parse_reconstruction._STATUS_MAP,
            set(StorageParseStatus),
            set(get_args(ParseStatus)),
        ),
        (
            parse_reconstruction._FAILURE_CATEGORY_MAP,
            set(StorageParseFailureCategory),
            set(get_args(ParserFailureCategory)),
        ),
        (
            parse_reconstruction._FAILURE_SEVERITY_MAP,
            set(StorageParseFailureSeverity),
            set(get_args(ParserFailureSeverity)),
        ),
        (
            parse_reconstruction._FAILURE_STAGE_MAP,
            set(StorageParseFailureStage),
            set(get_args(ParserFailureStage)),
        ),
        (
            parse_reconstruction._FAILURE_UNIT_KIND_MAP,
            set(StorageParseFailureUnitKind),
            set(get_args(ParserFailureUnitKind)),
        ),
    )

    for mapping, storage_values, ingest_values in expected_mappings:
        assert set(mapping) == storage_values
        assert set(mapping.values()) == ingest_values
