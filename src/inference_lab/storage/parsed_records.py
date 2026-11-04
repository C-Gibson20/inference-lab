"""Complete immutable parsed-representation storage aggregate."""

import hashlib
from dataclasses import dataclass, field

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab._normalisation import normalise_source_text
from inference_lab._validation import is_canonical_semantic_version
from inference_lab.provenance import (
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    KnownSourceSpan,
    OutputDigest,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage.producers import ProducerIdentity
from inference_lab.storage.source_locations import (
    BoundingBox,
    CollectionAvailability,
    ContentOrigin,
    PageLayoutDescriptor,
    SourceLocation,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.storage.source_versioning import (
    CURRENT_SOURCE_RECORD_SCHEMA_VERSION,
    validate_source_record_schema_version,
)


def _validate_optional_text(value: object, field_name: str) -> None:
    """Validate nullable, non-blank text metadata without echoing its value."""
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None.")
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-blank and contain no NUL.")


def _validate_count(value: object, field_name: str, *, optional: bool) -> None:
    """Validate a strict non-negative count or an optional absent value."""
    if optional and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        suffix = " or None" if optional else ""
        raise TypeError(f"{field_name} must be an integer{suffix}.")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedRepresentation:
    """Own one complete parsed representation and its source-span catalogue."""

    record_id: RecordId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    source_text: str
    text: str
    parser: ProducerIdentity
    normalisation_version: str
    language: str | None
    page_count: int | None
    character_count: int
    token_count: int | None
    page_collection_availability: CollectionAvailability
    section_collection_availability: CollectionAvailability
    paragraph_collection_availability: CollectionAvailability
    blocks: tuple[StructuralBlock, ...]
    source_locations: tuple[SourceLocation, ...]
    page_layouts: tuple[PageLayoutDescriptor, ...]
    ocr_producer: ProducerIdentity | None = None
    supersedes_record_id: RecordId | None = None
    record_schema_version: str = CURRENT_SOURCE_RECORD_SCHEMA_VERSION
    output_digest: OutputDigest = field(init=False)

    def __post_init__(self) -> None:
        self._validate_scalars()
        self._validate_collections()
        object.__setattr__(
            self,
            "output_digest",
            build_parsed_representation_output_digest(self),
        )

    def _validate_scalars(self) -> None:
        """Validate identity, text, version, count, and availability fields."""
        validate_source_record_schema_version(self.record_schema_version)
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.source_text, str):
            raise TypeError("source_text must be a string.")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if self.text != normalise_source_text(self.source_text):
            raise ValueError("text must be the canonical form of source_text.")
        if not isinstance(self.parser, ProducerIdentity):
            raise TypeError("parser must be a ProducerIdentity.")
        if not isinstance(self.normalisation_version, str):
            raise TypeError("normalisation_version must be a string.")
        if not is_canonical_semantic_version(self.normalisation_version):
            raise ValueError(
                "normalisation_version must be a canonical semantic version."
            )
        _validate_optional_text(self.language, "language")
        _validate_count(self.page_count, "page_count", optional=True)
        _validate_count(self.character_count, "character_count", optional=False)
        _validate_count(self.token_count, "token_count", optional=True)
        if self.character_count != len(self.text):
            raise ValueError("character_count must equal canonical text length.")
        for field_name in (
            "page_collection_availability",
            "section_collection_availability",
            "paragraph_collection_availability",
        ):
            if not isinstance(getattr(self, field_name), CollectionAvailability):
                raise TypeError(f"{field_name} must be a CollectionAvailability.")
        if self.ocr_producer is not None and not isinstance(
            self.ocr_producer, ProducerIdentity
        ):
            raise TypeError("ocr_producer must be a ProducerIdentity or None.")
        if self.supersedes_record_id is not None and not isinstance(
            self.supersedes_record_id, RecordId
        ):
            raise TypeError("supersedes_record_id must be a RecordId or None.")
        if self.supersedes_record_id == self.record_id:
            raise ValueError("a parsed representation cannot supersede itself.")

    def _validate_collections(self) -> None:
        """Validate the complete aggregate and its exact source relationships."""
        for field_name, member_type in (
            ("blocks", StructuralBlock),
            ("source_locations", SourceLocation),
            ("page_layouts", PageLayoutDescriptor),
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple.")
            if any(not isinstance(value, member_type) for value in values):
                raise TypeError(f"{field_name} contains an invalid value.")
        if tuple(block.ordinal for block in self.blocks) != tuple(
            range(len(self.blocks))
        ):
            raise ValueError("block ordinals must be contiguous from zero.")
        locations_by_id = self._locations_by_id()
        layouts_by_page = self._layouts_by_page()
        self._validate_locations(layouts_by_page)
        self._validate_blocks(locations_by_id)
        self._validate_availability()

    def _locations_by_id(self) -> dict[SourceSpanId, SourceLocation]:
        """Return unique locations indexed by source-span identity."""
        locations_by_id = {
            location.source_span_id: location for location in self.source_locations
        }
        if len(locations_by_id) != len(self.source_locations):
            raise ValueError("source_locations must not contain duplicate spans.")
        expected_order = tuple(
            sorted(
                self.source_locations,
                key=lambda location: location.source_span_id.as_str(),
            )
        )
        if self.source_locations != expected_order:
            raise ValueError("source_locations must be in canonical identifier order.")
        return locations_by_id

    def _layouts_by_page(self) -> dict[int, PageLayoutDescriptor]:
        """Return unique layouts indexed by page number."""
        layouts_by_page = {layout.page_number: layout for layout in self.page_layouts}
        if len(layouts_by_page) != len(self.page_layouts):
            raise ValueError("page_layouts must contain at most one layout per page.")
        if tuple(layout.page_number for layout in self.page_layouts) != tuple(
            sorted(layouts_by_page)
        ):
            raise ValueError("page_layouts must be in page-number order.")
        return layouts_by_page

    def _validate_blocks(
        self,
        locations_by_id: dict[SourceSpanId, SourceLocation],
    ) -> None:
        """Require one exact location and source slice for every block."""
        block_span_ids = tuple(block.source_span_id for block in self.blocks)
        if len(block_span_ids) != len(set(block_span_ids)):
            raise ValueError("blocks must not reuse a source span.")
        if set(block_span_ids) != set(locations_by_id):
            raise ValueError(
                "blocks and source_locations must reference the same spans."
            )
        previous_start = -1
        for block in self.blocks:
            location = locations_by_id[block.source_span_id]
            if location.start_offset < previous_start:
                raise ValueError("blocks must be in non-decreasing source order.")
            previous_start = location.start_offset
            source_slice = self.source_text[location.start_offset : location.end_offset]
            if block.text != normalise_source_text(source_slice):
                raise ValueError("block text must match its canonical source slice.")
            if block.kind is StructuralBlockKind.PAGE and location.page_number is None:
                raise ValueError("page blocks must identify a page number.")

    def _validate_locations(
        self,
        layouts_by_page: dict[int, PageLayoutDescriptor],
    ) -> None:
        """Validate source ownership, ranges, OCR identity, pages, and boxes."""
        source_length = len(self.source_text)
        for location in self.source_locations:
            if location.document_version_id != self.document_version_id:
                raise ValueError("source location uses the wrong document version.")
            if location.parsed_record_id != self.record_id:
                raise ValueError("source location uses the wrong parsed record.")
            if location.end_offset > source_length:
                raise ValueError("source location exceeds decoded source text.")
            if (
                location.content_origin is ContentOrigin.OCR
                and self.ocr_producer is None
            ):
                raise ValueError("OCR source locations require an OCR producer.")
            if location.page_number is not None:
                self._validate_page_number(location.page_number)
            if location.bounding_box is not None:
                assert location.page_number is not None
                layout = layouts_by_page.get(location.page_number)
                if layout is None:
                    raise ValueError(
                        "a bounding box requires a page layout descriptor."
                    )
                self._validate_box_in_layout(location.bounding_box, layout)
        for layout in self.page_layouts:
            self._validate_page_number(layout.page_number)
            if (
                layout.content_origin is ContentOrigin.OCR
                and self.ocr_producer != layout.producer
            ):
                raise ValueError(
                    "OCR page layout must use the representation OCR producer."
                )

    def _validate_page_number(self, page_number: int) -> None:
        """Require represented pages to fit the declared document page count."""
        if self.page_count is not None and page_number > self.page_count:
            raise ValueError("represented page is outside the declared page count.")

    @staticmethod
    def _validate_box_in_layout(
        bounding_box: BoundingBox,
        layout: PageLayoutDescriptor,
    ) -> None:
        """Require an untransformed box to fit its declared coordinate space."""
        if (
            bounding_box.left < 0
            or bounding_box.top < 0
            or bounding_box.right > layout.page_width
            or bounding_box.bottom > layout.page_height
        ):
            raise ValueError("bounding box lies outside its page coordinate space.")

    def _validate_availability(self) -> None:
        """Require typed block projections to agree with availability state."""
        for availability, kind, field_name in (
            (
                self.page_collection_availability,
                StructuralBlockKind.PAGE,
                "page_collection_availability",
            ),
            (
                self.section_collection_availability,
                StructuralBlockKind.SECTION,
                "section_collection_availability",
            ),
            (
                self.paragraph_collection_availability,
                StructuralBlockKind.PARAGRAPH,
                "paragraph_collection_availability",
            ),
        ):
            has_blocks = any(block.kind is kind for block in self.blocks)
            if availability is CollectionAvailability.UNAVAILABLE and has_blocks:
                raise ValueError(
                    f"{field_name} cannot be unavailable when matching blocks exist."
                )

    def _typed_blocks(
        self,
        availability: CollectionAvailability,
        kind: StructuralBlockKind,
    ) -> tuple[StructuralBlock, ...] | None:
        """Return an exact typed projection while preserving unavailability."""
        if availability is CollectionAvailability.UNAVAILABLE:
            return None
        return tuple(block for block in self.blocks if block.kind is kind)

    @property
    def pages(self) -> tuple[StructuralBlock, ...] | None:
        """Return the exact page-block projection, or unavailable state."""
        return self._typed_blocks(
            self.page_collection_availability,
            StructuralBlockKind.PAGE,
        )

    @property
    def sections(self) -> tuple[StructuralBlock, ...] | None:
        """Return the exact section-block projection, or unavailable state."""
        return self._typed_blocks(
            self.section_collection_availability,
            StructuralBlockKind.SECTION,
        )

    @property
    def paragraphs(self) -> tuple[StructuralBlock, ...] | None:
        """Return the exact paragraph-block projection, or unavailable state."""
        return self._typed_blocks(
            self.paragraph_collection_availability,
            StructuralBlockKind.PARAGRAPH,
        )

    def known_source_spans(self) -> tuple[KnownSourceSpan, ...]:
        """Return the complete lineage-compatible source-span catalogue."""
        return tuple(
            location.as_known_source_span() for location in self.source_locations
        )


def _producer_payload(producer: ProducerIdentity) -> JsonObject:
    """Return canonical data for one producing component identity."""
    return {
        "implementation": producer.implementation,
        "contract_version": producer.contract_version,
        "artefact_identity": producer.artefact_identity,
        "artefact_revision": producer.artefact_revision,
    }


def _bounding_box_payload(bounding_box: BoundingBox | None) -> JsonObject | None:
    """Return canonical data for an optional untransformed bounding box."""
    if bounding_box is None:
        return None
    return {
        "left": bounding_box.left,
        "top": bounding_box.top,
        "right": bounding_box.right,
        "bottom": bounding_box.bottom,
    }


def parsed_representation_output_payload(
    representation: ParsedRepresentation,
) -> JsonObject:
    """Return the canonical immutable output covered by its output digest."""
    if not isinstance(representation, ParsedRepresentation):
        raise TypeError("representation must be a ParsedRepresentation.")
    return {
        "record_schema_version": representation.record_schema_version,
        "document_id": representation.document_id.as_str(),
        "document_version_id": representation.document_version_id.as_str(),
        "source_text": representation.source_text,
        "text": representation.text,
        "parser": _producer_payload(representation.parser),
        "normalisation_version": representation.normalisation_version,
        "ocr_producer": (
            _producer_payload(representation.ocr_producer)
            if representation.ocr_producer is not None
            else None
        ),
        "language": representation.language,
        "page_count": representation.page_count,
        "character_count": representation.character_count,
        "token_count": representation.token_count,
        "collection_availability": {
            "pages": representation.page_collection_availability.value,
            "sections": representation.section_collection_availability.value,
            "paragraphs": representation.paragraph_collection_availability.value,
        },
        "blocks": [
            {
                "ordinal": block.ordinal,
                "kind": block.kind.value,
                "text": block.text,
                "source_span_id": block.source_span_id.as_str(),
            }
            for block in representation.blocks
        ],
        "source_locations": [
            {
                "source_span_id": location.source_span_id.as_str(),
                "document_version_id": location.document_version_id.as_str(),
                "start_offset": location.start_offset,
                "end_offset": location.end_offset,
                "content_origin": location.content_origin.value,
                "page_number": location.page_number,
                "bounding_box": _bounding_box_payload(location.bounding_box),
            }
            for location in representation.source_locations
        ],
        "page_layouts": [
            {
                "page_number": layout.page_number,
                "coordinate_unit": layout.coordinate_unit.value,
                "coordinate_origin": layout.coordinate_origin.value,
                "page_width": layout.page_width,
                "page_height": layout.page_height,
                "rotation": layout.rotation.value,
                "content_origin": layout.content_origin.value,
                "producer": _producer_payload(layout.producer),
            }
            for layout in representation.page_layouts
        ],
    }


def build_parsed_representation_output_digest(
    representation: ParsedRepresentation,
) -> OutputDigest:
    """Build the SHA-256 digest for one canonical parsed output payload."""
    digest = hashlib.sha256(
        canonical_json_bytes(parsed_representation_output_payload(representation))
    ).hexdigest()
    return OutputDigest(DigestAlgorithm.SHA256, digest)
