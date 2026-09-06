"""Immutable parsed structure and source-location coordinate values."""

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    KnownSourceSpan,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage.producers import ProducerIdentity


class StructuralBlockKind(StrEnum):
    """Structural unit kinds retained from parser output."""

    PAGE = "page"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    OTHER = "other"


class CollectionAvailability(StrEnum):
    """Distinguish unavailable structure from an available empty collection."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ContentOrigin(StrEnum):
    """Identify whether persisted text or layout is native or OCR-produced."""

    NATIVE = "native"
    OCR = "ocr"


class CoordinateUnit(StrEnum):
    """Supported page-layout coordinate units."""

    PDF_POINTS = "pdf_points"
    IMAGE_PIXELS = "image_pixels"


class CoordinateOrigin(StrEnum):
    """Supported origins for page-layout coordinate spaces."""

    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"


class PageRotation(IntEnum):
    """Supported clockwise page rotations in quarter turns."""

    DEGREES_0 = 0
    DEGREES_90 = 90
    DEGREES_180 = 180
    DEGREES_270 = 270


def _validate_non_negative_integer(
    value: object,
    field_name: str,
    *,
    optional: bool,
) -> None:
    """Validate a strict non-negative integer or an optional absent value."""
    if optional and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        suffix = " or None" if optional else ""
        raise TypeError(f"{field_name} must be an integer{suffix}.")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative.")


def _validate_finite_number(value: object, field_name: str) -> None:
    """Validate one finite coordinate without coercing its representation."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite number.")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number.")


@dataclass(frozen=True, slots=True, kw_only=True)
class BoundingBox:
    """Retain one untransformed rectangle in a declared page coordinate space."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the bounding box is well-formed."""
        for field_name in ("left", "top", "right", "bottom"):
            _validate_finite_number(getattr(self, field_name), field_name)
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("bounding_box must not be inverted.")


@dataclass(frozen=True, slots=True, kw_only=True)
class PageLayoutDescriptor:
    """Describe the coordinate system for one represented source page."""

    page_number: int
    coordinate_unit: CoordinateUnit
    coordinate_origin: CoordinateOrigin
    page_width: float
    page_height: float
    rotation: PageRotation
    content_origin: ContentOrigin
    producer: ProducerIdentity

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the page layout descriptor is well-formed."""
        _validate_non_negative_integer(self.page_number, "page_number", optional=False)
        if self.page_number < 1:
            raise ValueError("page_number must be positive.")
        if not isinstance(self.coordinate_unit, CoordinateUnit):
            raise TypeError("coordinate_unit must be a CoordinateUnit.")
        if not isinstance(self.coordinate_origin, CoordinateOrigin):
            raise TypeError("coordinate_origin must be a CoordinateOrigin.")
        _validate_finite_number(self.page_width, "page_width")
        _validate_finite_number(self.page_height, "page_height")
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("page dimensions must be positive.")
        if not isinstance(self.rotation, PageRotation):
            raise TypeError("rotation must be a PageRotation.")
        if not isinstance(self.content_origin, ContentOrigin):
            raise TypeError("content_origin must be a ContentOrigin.")
        if not isinstance(self.producer, ProducerIdentity):
            raise TypeError("producer must be a ProducerIdentity.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceLocation:
    """Locate one source span in decoded text and optional page layout."""

    source_span_id: SourceSpanId
    document_version_id: DocumentVersionId
    parsed_record_id: RecordId
    start_offset: int
    end_offset: int
    content_origin: ContentOrigin
    page_number: int | None = None
    bounding_box: BoundingBox | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the source location is self-consistent."""
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.parsed_record_id, RecordId):
            raise TypeError("parsed_record_id must be a RecordId.")
        _validate_non_negative_integer(
            self.start_offset,
            "start_offset",
            optional=False,
        )
        _validate_non_negative_integer(self.end_offset, "end_offset", optional=False)
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not precede start_offset.")
        if not isinstance(self.content_origin, ContentOrigin):
            raise TypeError("content_origin must be a ContentOrigin.")
        _validate_non_negative_integer(self.page_number, "page_number", optional=True)
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive.")
        if self.bounding_box is not None and not isinstance(
            self.bounding_box, BoundingBox
        ):
            raise TypeError("bounding_box must be a BoundingBox or None.")
        if self.bounding_box is not None and self.page_number is None:
            raise ValueError("a bounding_box requires a page_number.")

    def as_known_source_span(self) -> KnownSourceSpan:
        """Convert this location to the PROV-03 source catalogue contract."""
        return KnownSourceSpan(
            document_version_id=self.document_version_id,
            source_span_id=self.source_span_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSourceSpan:
    """Return one source location with its owning document and coordinates."""

    document_id: DocumentId
    location: SourceLocation
    page_layout: PageLayoutDescriptor | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.location, SourceLocation):
            raise TypeError("location must be a SourceLocation.")
        if self.page_layout is not None and not isinstance(
            self.page_layout, PageLayoutDescriptor
        ):
            raise TypeError("page_layout must be a PageLayoutDescriptor or None.")
        if self.location.bounding_box is not None and self.page_layout is None:
            raise ValueError("a resolved bounding box requires its page layout.")
        if self.page_layout is not None and (
            self.location.page_number != self.page_layout.page_number
        ):
            raise ValueError(
                "resolved location and page layout must identify one page."
            )

    def as_known_source_span(self) -> KnownSourceSpan:
        """Return this resolution as a PROV-03 catalogue entry."""
        return self.location.as_known_source_span()


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuralBlock:
    """Retain one ordered parsed block and its source-span reference."""

    ordinal: int
    kind: StructuralBlockKind
    text: str
    source_span_id: SourceSpanId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the structural block is self-consistent."""
        _validate_non_negative_integer(self.ordinal, "ordinal", optional=False)
        if not isinstance(self.kind, StructuralBlockKind):
            raise TypeError("kind must be a StructuralBlockKind.")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")
