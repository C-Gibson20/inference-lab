"""Strict reconstruction of ingest parser values from storage records."""

from dataclasses import dataclass
from typing import Final, Literal, Mapping, TypeVar

from inference_lab.ingest.errors import (
    ParserFailure,
    ParserFailureCategory,
    ParserFailureSeverity,
    ParserFailureStage,
    ParserFailureUnitKind,
)
from inference_lab.ingest.outcomes import ParseOutcome, ParseStatus
from inference_lab.ingest.parser import NormalisedText, ParsedDocument
from inference_lab.ingest.sources import (
    BoundingBox,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    SourceSpan,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import SourceSpanId

# isort: off
from inference_lab.storage import (
    CollectionAvailability,
    ContentOrigin,
    CoordinateOrigin as StorageCoordinateOrigin,
    CoordinateUnit as StorageCoordinateUnit,
    PageLayoutDescriptor as StoragePageLayoutDescriptor,
    PageRotation as StoragePageRotation,
    ParsedRepresentation,
    ParseFailureCategory as StorageParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity as StorageParseFailureSeverity,
    ParseFailureStage as StorageParseFailureStage,
    ParseFailureUnitKind as StorageParseFailureUnitKind,
    ParseOutcome as StorageParseOutcome,
    ParseStatus as StorageParseStatus,
    ProducerIdentity,
    SourceLocation,
    StructuralBlock as StorageStructuralBlock,
    StructuralBlockKind as StorageStructuralBlockKind,
)
# isort: on

ParseReconstructionErrorCategory = Literal[
    "invalid_outcome",
    "invalid_representation",
    "missing_representation",
    "unexpected_representation",
    "identity_mismatch",
    "producer_mismatch",
    "unsupported_attribution",
    "unsupported_value",
    "invalid_structure",
]


class ParseReconstructionError(ValueError):
    """Reject an incomplete persisted parse aggregate without exposing payloads."""

    def __init__(self, category: ParseReconstructionErrorCategory) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True, slots=True)
class ReconstructedParseOutcome:
    """Pair a reconstructed outcome with its parser version, including failures."""

    parser_version: ParserVersion
    outcome: ParseOutcome

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the representation and outcome are consistent with each other."""
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.outcome, ParseOutcome):
            raise TypeError("outcome must be an ingest ParseOutcome.")
        document = self.outcome.document
        if document is not None and document.parser_version != self.parser_version:
            raise ValueError("outcome and parser version must agree.")


_BLOCK_KIND_MAP: Final[dict[StorageStructuralBlockKind, StructuralBlockKind]] = {
    StorageStructuralBlockKind.PAGE: "page",
    StorageStructuralBlockKind.SECTION: "section",
    StorageStructuralBlockKind.PARAGRAPH: "paragraph",
    StorageStructuralBlockKind.TABLE: "table",
    StorageStructuralBlockKind.FIGURE_CAPTION: "figure_caption",
    StorageStructuralBlockKind.OTHER: "other",
}
_COORDINATE_UNIT_MAP: Final[dict[StorageCoordinateUnit, CoordinateUnit]] = {
    StorageCoordinateUnit.PDF_POINTS: "pdf_points",
    StorageCoordinateUnit.IMAGE_PIXELS: "image_pixels",
}
_COORDINATE_ORIGIN_MAP: Final[dict[StorageCoordinateOrigin, CoordinateOrigin]] = {
    StorageCoordinateOrigin.TOP_LEFT: "top_left",
    StorageCoordinateOrigin.BOTTOM_LEFT: "bottom_left",
}
_PAGE_ROTATION_MAP: Final[dict[StoragePageRotation, PageRotation]] = {
    StoragePageRotation.DEGREES_0: 0,
    StoragePageRotation.DEGREES_90: 90,
    StoragePageRotation.DEGREES_180: 180,
    StoragePageRotation.DEGREES_270: 270,
}
_STATUS_MAP: Final[dict[StorageParseStatus, ParseStatus]] = {
    StorageParseStatus.SUCCEEDED: "succeeded",
    StorageParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES: (
        "completed_with_recoverable_failures"
    ),
    StorageParseStatus.FAILED: "failed",
}
_FAILURE_CATEGORY_MAP: Final[
    dict[StorageParseFailureCategory, ParserFailureCategory]
] = {
    StorageParseFailureCategory.INVALID_INPUT: "invalid_input",
    StorageParseFailureCategory.UNSUPPORTED_FORMAT: "unsupported_format",
    StorageParseFailureCategory.INVALID_ENCODING: "invalid_encoding",
    StorageParseFailureCategory.DECODE_FAILED: "decode_failed",
    StorageParseFailureCategory.MALFORMED_SOURCE: "malformed_source",
    StorageParseFailureCategory.MALFORMED_UNIT: "malformed_unit",
    StorageParseFailureCategory.UNSUPPORTED_STRUCTURE: "unsupported_structure",
    StorageParseFailureCategory.INVALID_OUTPUT: "invalid_output",
    StorageParseFailureCategory.SOURCE_TOO_LARGE: "source_too_large",
}
_FAILURE_SEVERITY_MAP: Final[
    dict[StorageParseFailureSeverity, ParserFailureSeverity]
] = {
    StorageParseFailureSeverity.FATAL: "fatal",
    StorageParseFailureSeverity.RECOVERABLE: "recoverable",
}
_FAILURE_STAGE_MAP: Final[dict[StorageParseFailureStage, ParserFailureStage]] = {
    StorageParseFailureStage.INPUT: "input",
    StorageParseFailureStage.DECODE: "decode",
    StorageParseFailureStage.STRUCTURE: "structure",
    StorageParseFailureStage.OUTPUT: "output",
}
_FAILURE_UNIT_KIND_MAP: Final[
    dict[StorageParseFailureUnitKind, ParserFailureUnitKind]
] = {
    StorageParseFailureUnitKind.PAGE: "page",
    StorageParseFailureUnitKind.SECTION: "section",
    StorageParseFailureUnitKind.PARAGRAPH: "paragraph",
    StorageParseFailureUnitKind.TABLE: "table",
    StorageParseFailureUnitKind.FIGURE_CAPTION: "figure_caption",
    StorageParseFailureUnitKind.OTHER: "other",
}

_MappedKey = TypeVar("_MappedKey")
_MappedValue = TypeVar("_MappedValue")


def _mapped_value(
    mapping: Mapping[_MappedKey, _MappedValue],
    value: object,
    expected_type: type[_MappedKey],
) -> _MappedValue:
    """Return one exact closed-vocabulary mapping without enum coercion."""
    if not isinstance(value, expected_type):
        raise ParseReconstructionError("unsupported_value")
    try:
        return mapping[value]
    except (KeyError, TypeError):
        raise ParseReconstructionError("unsupported_value") from None


def _parser_version(
    producer: ProducerIdentity,
    normalisation_version: str,
) -> ParserVersion:
    if producer.artefact_identity is not None or producer.artefact_revision is not None:
        raise ParseReconstructionError("unsupported_attribution")
    try:
        return ParserVersion(
            implementation=producer.implementation,
            contract_version=producer.contract_version,
            normalisation_version=normalisation_version,
        )
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _bounding_box(location: SourceLocation) -> BoundingBox | None:
    box = location.bounding_box
    if box is None:
        return None
    try:
        return BoundingBox(
            left=box.left,
            top=box.top,
            right=box.right,
            bottom=box.bottom,
        )
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _source_span(location: SourceLocation) -> SourceSpan:
    if location.content_origin is not ContentOrigin.NATIVE:
        raise ParseReconstructionError("unsupported_attribution")
    try:
        return SourceSpan(
            source_span_id=location.source_span_id,
            document_version_id=location.document_version_id,
            start_offset=location.start_offset,
            end_offset=location.end_offset,
            page_number=location.page_number,
            bounding_box=_bounding_box(location),
        )
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _block(
    block: StorageStructuralBlock,
    locations_by_id: Mapping[SourceSpanId, SourceLocation],
) -> StructuralBlock:
    location = locations_by_id.get(block.source_span_id)
    if location is None:
        raise ParseReconstructionError("invalid_structure")
    try:
        return StructuralBlock(
            kind=_mapped_value(
                _BLOCK_KIND_MAP,
                block.kind,
                StorageStructuralBlockKind,
            ),
            text=block.text,
            source_span=_source_span(location),
        )
    except ParseReconstructionError:
        raise
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _page_layout(layout: StoragePageLayoutDescriptor) -> PageLayoutDescriptor:
    if layout.content_origin is not ContentOrigin.NATIVE:
        raise ParseReconstructionError("unsupported_attribution")
    try:
        return PageLayoutDescriptor(
            page_number=layout.page_number,
            coordinate_unit=_mapped_value(
                _COORDINATE_UNIT_MAP,
                layout.coordinate_unit,
                StorageCoordinateUnit,
            ),
            coordinate_origin=_mapped_value(
                _COORDINATE_ORIGIN_MAP,
                layout.coordinate_origin,
                StorageCoordinateOrigin,
            ),
            page_width=layout.page_width,
            page_height=layout.page_height,
            rotation=_mapped_value(
                _PAGE_ROTATION_MAP,
                layout.rotation,
                StoragePageRotation,
            ),
        )
    except ParseReconstructionError:
        raise
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _failure(failure: ParseFailureRecord) -> ParserFailure:
    unit_kind = (
        _mapped_value(
            _FAILURE_UNIT_KIND_MAP,
            failure.unit_kind,
            StorageParseFailureUnitKind,
        )
        if failure.unit_kind is not None
        else None
    )
    try:
        return ParserFailure(
            category=_mapped_value(
                _FAILURE_CATEGORY_MAP,
                failure.category,
                StorageParseFailureCategory,
            ),
            severity=_mapped_value(
                _FAILURE_SEVERITY_MAP,
                failure.severity,
                StorageParseFailureSeverity,
            ),
            stage=_mapped_value(
                _FAILURE_STAGE_MAP,
                failure.stage,
                StorageParseFailureStage,
            ),
            unit_kind=unit_kind,
            unit_index=failure.unit_index,
        )
    except ParseReconstructionError:
        raise
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _typed_collection(
    availability: CollectionAvailability,
    kind: StructuralBlockKind,
    blocks: tuple[StructuralBlock, ...],
) -> tuple[StructuralBlock, ...] | None:
    if availability is CollectionAvailability.UNAVAILABLE:
        return None
    if availability is not CollectionAvailability.AVAILABLE:
        raise ParseReconstructionError("unsupported_value")
    return tuple(block for block in blocks if block.kind == kind)


def _document(
    representation: ParsedRepresentation,
    parser_version: ParserVersion,
) -> ParsedDocument:
    if representation.ocr_producer is not None:
        raise ParseReconstructionError("unsupported_attribution")
    if tuple(block.ordinal for block in representation.blocks) != tuple(
        range(len(representation.blocks))
    ):
        raise ParseReconstructionError("invalid_structure")
    expected_location_order = tuple(
        sorted(
            representation.source_locations,
            key=lambda location: location.source_span_id.as_str(),
        )
    )
    if representation.source_locations != expected_location_order:
        raise ParseReconstructionError("invalid_structure")
    if tuple(layout.page_number for layout in representation.page_layouts) != tuple(
        sorted(layout.page_number for layout in representation.page_layouts)
    ):
        raise ParseReconstructionError("invalid_structure")

    locations_by_id = {
        location.source_span_id: location
        for location in representation.source_locations
    }
    if len(locations_by_id) != len(representation.source_locations):
        raise ParseReconstructionError("invalid_structure")
    if any(
        location.parsed_record_id != representation.record_id
        or location.document_version_id != representation.document_version_id
        for location in representation.source_locations
    ):
        raise ParseReconstructionError("identity_mismatch")
    blocks = tuple(_block(block, locations_by_id) for block in representation.blocks)
    if len(blocks) != len(locations_by_id):
        raise ParseReconstructionError("invalid_structure")
    layouts = tuple(_page_layout(layout) for layout in representation.page_layouts)
    if any(
        layout.producer != representation.parser
        for layout in representation.page_layouts
    ):
        raise ParseReconstructionError("producer_mismatch")

    try:
        text = NormalisedText(source_text=representation.source_text)
        if text.text != representation.text:
            raise ParseReconstructionError("invalid_structure")
        return ParsedDocument(
            document_id=representation.document_id,
            document_version_id=representation.document_version_id,
            text=text,
            blocks=blocks,
            pages=_typed_collection(
                representation.page_collection_availability,
                "page",
                blocks,
            ),
            sections=_typed_collection(
                representation.section_collection_availability,
                "section",
                blocks,
            ),
            paragraphs=_typed_collection(
                representation.paragraph_collection_availability,
                "paragraph",
                blocks,
            ),
            page_layouts=layouts,
            language=representation.language,
            page_count=representation.page_count,
            character_count=representation.character_count,
            token_count=representation.token_count,
            parser_version=parser_version,
        )
    except ParseReconstructionError:
        raise
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None


def _validate_pair(
    outcome: StorageParseOutcome,
    representation: ParsedRepresentation | None,
) -> None:
    represented = outcome.status in (
        StorageParseStatus.SUCCEEDED,
        StorageParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
    )
    if represented and representation is None:
        raise ParseReconstructionError("missing_representation")
    if not represented and representation is not None:
        raise ParseReconstructionError("unexpected_representation")
    if representation is None:
        return
    if (
        outcome.parsed_record_id != representation.record_id
        or outcome.document_id != representation.document_id
        or outcome.document_version_id != representation.document_version_id
    ):
        raise ParseReconstructionError("identity_mismatch")
    if (
        outcome.parser != representation.parser
        or outcome.normalisation_version != representation.normalisation_version
    ):
        raise ParseReconstructionError("producer_mismatch")


def reconstruct_parse_outcome(
    outcome: StorageParseOutcome,
    representation: ParsedRepresentation | None,
) -> ReconstructedParseOutcome:
    """Reconstruct one exact ingest outcome from a complete storage aggregate."""
    if not isinstance(outcome, StorageParseOutcome):
        raise ParseReconstructionError("invalid_outcome")
    if representation is not None and not isinstance(
        representation, ParsedRepresentation
    ):
        raise ParseReconstructionError("invalid_representation")
    if outcome.ocr_producer is not None:
        raise ParseReconstructionError("unsupported_attribution")

    status = _mapped_value(_STATUS_MAP, outcome.status, StorageParseStatus)
    _validate_pair(outcome, representation)
    parser_version = _parser_version(outcome.parser, outcome.normalisation_version)
    document = (
        _document(representation, parser_version)
        if representation is not None
        else None
    )
    failures = tuple(_failure(failure) for failure in outcome.failures)
    try:
        reconstructed_outcome = ParseOutcome(
            status=status,
            document=document,
            failures=failures,
        )
    except (TypeError, ValueError):
        raise ParseReconstructionError("invalid_structure") from None
    return ReconstructedParseOutcome(
        parser_version=parser_version,
        outcome=reconstructed_outcome,
    )
