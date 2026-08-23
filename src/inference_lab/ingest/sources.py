"""Define the data structures for representing source layout and content."""

import math
from dataclasses import dataclass
from typing import Literal

from inference_lab.provenance import DocumentVersionId, SourceSpanId

StructuralBlockKind = Literal[
    "page",
    "section",
    "paragraph",
    "table",
    "figure_caption",
    "other",
]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Represent an optional source-layout rectangle."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the bounding box coordinates."""

        coordinates = (self.left, self.top, self.right, self.bottom)
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in coordinates
        ):
            raise ValueError("bounding_box_coordinates_must_be_finite")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bounding_box_coordinates_must_be_finite")
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("bounding_box_must_not_be_inverted")


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Locate a text-bearing unit in decoded source text."""

    source_span_id: SourceSpanId
    document_version_id: DocumentVersionId
    start_offset: int
    end_offset: int
    page_number: int | None = None
    bounding_box: BoundingBox | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the source span coordinates."""

        if not isinstance(self.source_span_id, SourceSpanId):
            raise ValueError("invalid_source_span_id")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise ValueError("invalid_document_version_id")
        if not self._is_integer(self.start_offset):
            raise ValueError("source_span_offsets_must_be_integers")
        if not self._is_integer(self.end_offset):
            raise ValueError("source_span_offsets_must_be_integers")
        if self.start_offset < 0:
            raise ValueError("source_span_start_must_be_non_negative")
        if self.end_offset < self.start_offset:
            raise ValueError("source_span_end_must_not_precede_start")
        if self.page_number is not None:
            if not self._is_integer(self.page_number):
                raise ValueError("page_number_must_be_integer")
            if self.page_number < 1:
                raise ValueError("page_number_must_be_positive")
        if self.bounding_box is not None and not isinstance(
            self.bounding_box, BoundingBox
        ):
            raise ValueError("invalid_bounding_box")

    @staticmethod
    def _is_integer(value: object) -> bool:
        """Return whether a runtime value is an integer but not a boolean."""
        return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class StructuralBlock:
    """Represent one ordered text-bearing structural unit."""

    kind: StructuralBlockKind
    text: str
    source_span: SourceSpan

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject invalid structural block values at the contract boundary."""
        if self.kind not in {
            "page",
            "section",
            "paragraph",
            "table",
            "figure_caption",
            "other",
        }:
            raise ValueError("unsupported_structural_block_kind")
        if not isinstance(self.text, str):
            raise ValueError("structural_block_text_must_be_string")
        if not isinstance(self.source_span, SourceSpan):
            raise ValueError("invalid_structural_block_source_span")
