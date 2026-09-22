"""Pure forward mapping from ingest parser values to storage records."""

from dataclasses import dataclass
from typing import Final

from inference_lab.ingest.errors import (
    ParserFailure,
    ParserFailureCategory,
    ParserFailureSeverity,
    ParserFailureStage,
    ParserFailureUnitKind,
)
from inference_lab.ingest.outcomes import ParseStatus
from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.ingest.sources import (
    BoundingBox,
    CoordinateOrigin,
    CoordinateUnit,
    PageRotation,
    StructuralBlockKind,
)

# isort: off
from inference_lab.storage import (
    BoundingBox as StorageBoundingBox,
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

_BLOCK_KIND_MAP: Final[dict[StructuralBlockKind, StorageStructuralBlockKind]] = {
    "page": StorageStructuralBlockKind.PAGE,
    "section": StorageStructuralBlockKind.SECTION,
    "paragraph": StorageStructuralBlockKind.PARAGRAPH,
    "table": StorageStructuralBlockKind.TABLE,
    "figure_caption": StorageStructuralBlockKind.FIGURE_CAPTION,
    "other": StorageStructuralBlockKind.OTHER,
}
_COORDINATE_UNIT_MAP: Final[dict[CoordinateUnit, StorageCoordinateUnit]] = {
    "pdf_points": StorageCoordinateUnit.PDF_POINTS,
    "image_pixels": StorageCoordinateUnit.IMAGE_PIXELS,
}
_COORDINATE_ORIGIN_MAP: Final[dict[CoordinateOrigin, StorageCoordinateOrigin]] = {
    "top_left": StorageCoordinateOrigin.TOP_LEFT,
    "bottom_left": StorageCoordinateOrigin.BOTTOM_LEFT,
}
_PAGE_ROTATION_MAP: Final[dict[PageRotation, StoragePageRotation]] = {
    0: StoragePageRotation.DEGREES_0,
    90: StoragePageRotation.DEGREES_90,
    180: StoragePageRotation.DEGREES_180,
    270: StoragePageRotation.DEGREES_270,
}
_STATUS_MAP: Final[dict[ParseStatus, StorageParseStatus]] = {
    "succeeded": StorageParseStatus.SUCCEEDED,
    "completed_with_recoverable_failures": (
        StorageParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES
    ),
    "failed": StorageParseStatus.FAILED,
}
_FAILURE_CATEGORY_MAP: Final[
    dict[ParserFailureCategory, StorageParseFailureCategory]
] = {
    "invalid_input": StorageParseFailureCategory.INVALID_INPUT,
    "unsupported_format": StorageParseFailureCategory.UNSUPPORTED_FORMAT,
    "invalid_encoding": StorageParseFailureCategory.INVALID_ENCODING,
    "decode_failed": StorageParseFailureCategory.DECODE_FAILED,
    "malformed_source": StorageParseFailureCategory.MALFORMED_SOURCE,
    "malformed_unit": StorageParseFailureCategory.MALFORMED_UNIT,
    "unsupported_structure": StorageParseFailureCategory.UNSUPPORTED_STRUCTURE,
    "invalid_output": StorageParseFailureCategory.INVALID_OUTPUT,
    "source_too_large": StorageParseFailureCategory.SOURCE_TOO_LARGE,
}
_FAILURE_SEVERITY_MAP: Final[
    dict[ParserFailureSeverity, StorageParseFailureSeverity]
] = {
    "fatal": StorageParseFailureSeverity.FATAL,
    "recoverable": StorageParseFailureSeverity.RECOVERABLE,
}
_FAILURE_STAGE_MAP: Final[dict[ParserFailureStage, StorageParseFailureStage]] = {
    "input": StorageParseFailureStage.INPUT,
    "decode": StorageParseFailureStage.DECODE,
    "structure": StorageParseFailureStage.STRUCTURE,
    "output": StorageParseFailureStage.OUTPUT,
}
_FAILURE_UNIT_KIND_MAP: Final[
    dict[ParserFailureUnitKind, StorageParseFailureUnitKind]
] = {
    "page": StorageParseFailureUnitKind.PAGE,
    "section": StorageParseFailureUnitKind.SECTION,
    "paragraph": StorageParseFailureUnitKind.PARAGRAPH,
    "table": StorageParseFailureUnitKind.TABLE,
    "figure_caption": StorageParseFailureUnitKind.FIGURE_CAPTION,
    "other": StorageParseFailureUnitKind.OTHER,
}


@dataclass(frozen=True, slots=True)
class MappedParseRecords:
    """Pair the optional representation with its immutable parse outcome."""

    representation: ParsedRepresentation | None
    outcome: StorageParseOutcome

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the representation and outcome are consistent with each other."""
        if self.representation is not None and not isinstance(
            self.representation, ParsedRepresentation
        ):
            raise TypeError("representation must be a ParsedRepresentation or None.")
        if not isinstance(self.outcome, StorageParseOutcome):
            raise TypeError("outcome must be a storage ParseOutcome.")
        if self.representation is None:
            if self.outcome.parsed_record_id is not None:
                raise ValueError("an outcome cannot link a missing representation.")
            return
        if self.outcome.parsed_record_id != self.representation.record_id:
            raise ValueError("outcome and representation identities must agree.")
        if self.outcome.document_id != self.representation.document_id:
            raise ValueError("outcome and representation documents must agree.")
        if self.outcome.document_version_id != self.representation.document_version_id:
            raise ValueError("outcome and representation versions must agree.")
        if self.outcome.parser != self.representation.parser:
            raise ValueError("outcome and representation parsers must agree.")
        if (
            self.outcome.normalisation_version
            != self.representation.normalisation_version
        ):
            raise ValueError("outcome and representation normalisation must agree.")


def _producer(request: ParsePersistenceRequest) -> ProducerIdentity:
    return ProducerIdentity(
        implementation=request.parser_version.implementation,
        contract_version=request.parser_version.contract_version,
    )


def _map_bounding_box(box: BoundingBox | None) -> StorageBoundingBox | None:
    if box is None:
        return None
    if not isinstance(box, BoundingBox):
        raise TypeError("box must be an ingest BoundingBox or None.")
    return StorageBoundingBox(
        left=box.left,
        top=box.top,
        right=box.right,
        bottom=box.bottom,
    )


def _map_failure(failure: ParserFailure) -> ParseFailureRecord:
    unit_kind = (
        _FAILURE_UNIT_KIND_MAP[failure.unit_kind]
        if failure.unit_kind is not None
        else None
    )
    return ParseFailureRecord(
        category=_FAILURE_CATEGORY_MAP[failure.category],
        severity=_FAILURE_SEVERITY_MAP[failure.severity],
        stage=_FAILURE_STAGE_MAP[failure.stage],
        unit_kind=unit_kind,
        unit_index=failure.unit_index,
    )


def _map_representation(
    request: ParsePersistenceRequest,
    producer: ProducerIdentity,
) -> ParsedRepresentation | None:
    document = request.outcome.document
    if document is None:
        return None
    assert request.parsed_record_id is not None

    blocks = tuple(
        StorageStructuralBlock(
            ordinal=ordinal,
            kind=_BLOCK_KIND_MAP[block.kind],
            text=block.text,
            source_span_id=block.source_span.source_span_id,
        )
        for ordinal, block in enumerate(document.blocks)
    )
    locations = tuple(
        sorted(
            (
                SourceLocation(
                    source_span_id=block.source_span.source_span_id,
                    document_version_id=block.source_span.document_version_id,
                    parsed_record_id=request.parsed_record_id,
                    start_offset=block.source_span.start_offset,
                    end_offset=block.source_span.end_offset,
                    page_number=block.source_span.page_number,
                    bounding_box=_map_bounding_box(block.source_span.bounding_box),
                    content_origin=ContentOrigin.NATIVE,
                )
                for block in document.blocks
            ),
            key=lambda location: location.source_span_id.as_str(),
        )
    )
    page_layouts = tuple(
        StoragePageLayoutDescriptor(
            page_number=layout.page_number,
            coordinate_unit=_COORDINATE_UNIT_MAP[layout.coordinate_unit],
            coordinate_origin=_COORDINATE_ORIGIN_MAP[layout.coordinate_origin],
            page_width=layout.page_width,
            page_height=layout.page_height,
            rotation=_PAGE_ROTATION_MAP[layout.rotation],
            content_origin=ContentOrigin.NATIVE,
            producer=producer,
        )
        for layout in document.page_layouts
    )
    return ParsedRepresentation(
        record_id=request.parsed_record_id,
        document_id=document.document_id,
        document_version_id=document.document_version_id,
        source_text=document.text.source_text,
        text=document.text.text,
        parser=producer,
        normalisation_version=request.parser_version.normalisation_version,
        language=document.language,
        page_count=document.page_count,
        character_count=document.character_count,
        token_count=document.token_count,
        page_collection_availability=(
            CollectionAvailability.UNAVAILABLE
            if document.pages is None
            else CollectionAvailability.AVAILABLE
        ),
        section_collection_availability=(
            CollectionAvailability.UNAVAILABLE
            if document.sections is None
            else CollectionAvailability.AVAILABLE
        ),
        paragraph_collection_availability=(
            CollectionAvailability.UNAVAILABLE
            if document.paragraphs is None
            else CollectionAvailability.AVAILABLE
        ),
        blocks=blocks,
        source_locations=locations,
        page_layouts=page_layouts,
        ocr_producer=None,
        supersedes_record_id=request.supersedes_record_id,
    )


def map_parse_persistence_request(
    request: ParsePersistenceRequest,
) -> MappedParseRecords:
    """Map one validated ingest request into complete STOR-02 parser records."""
    if not isinstance(request, ParsePersistenceRequest):
        raise TypeError("request must be a ParsePersistenceRequest.")
    producer = _producer(request)
    representation = _map_representation(request, producer)
    outcome = StorageParseOutcome(
        parse_id=request.parse_id,
        document_id=request.parser_input.document_id,
        document_version_id=request.parser_input.document_version_id,
        status=_STATUS_MAP[request.outcome.status],
        parser=producer,
        normalisation_version=request.parser_version.normalisation_version,
        parsed_record_id=request.parsed_record_id,
        failures=tuple(_map_failure(failure) for failure in request.outcome.failures),
        ocr_producer=None,
    )
    return MappedParseRecords(representation=representation, outcome=outcome)
