"""Deterministic parsing contracts and text-native document parsing."""

import codecs
from dataclasses import dataclass, field
from typing import Literal, Protocol

import structlog

from inference_lab.ingest.errors import ParserDecodeError
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.sources import (
    BoundingBox,
    PageLayoutDescriptor,
    StructuralBlock,
)
from inference_lab.ingest.versioning import (
    CURRENT_PARSER_VERSION,
    ParserVersion,
    generate_source_span_id,
)
from inference_lab.provenance import DocumentId, DocumentVersionId
from inference_lab.settings import Settings

logger = structlog.get_logger(__name__)

TextNativeFormat = Literal["text/plain"]
ParserInputErrorCategory = Literal[
    "invalid_document_id",
    "invalid_document_version_id",
    "non_bytes_content",
    "unsupported_media_type",
    "missing_encoding",
    "invalid_encoding",
]


class ParserInputError(ValueError):
    """Represent a safe, machine-readable parser-input validation failure."""

    def __init__(self, category: ParserInputErrorCategory) -> None:
        """Create an error without retaining invalid source payloads."""
        super().__init__(category)
        self.category = category


def _is_text_encoding(encoding: str) -> bool:
    """Return whether the named codec decodes bytes into text."""
    try:
        codec = codecs.lookup(encoding)
        decoded, _ = codec.decode(b"", "strict")
    except Exception:
        # Codec implementations are registry extensions. Treat any decoder
        # incompatibility as invalid input rather than leaking it past this boundary.
        return False
    return isinstance(decoded, str)


@dataclass(frozen=True, slots=True)
class NormalisedText:
    """Pair canonical text with the decoded source coordinate space."""

    source_text: str
    text: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_text, str):
            raise ValueError("source_text_must_be_string")
        object.__setattr__(self, "text", normalise_source_text(self.source_text))


@dataclass(frozen=True, slots=True)
class ParserInput:
    """Describe one source document version to be parsed."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    content: bytes
    media_type: TextNativeFormat
    encoding: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject invalid metadata before parsing can begin."""
        if not isinstance(self.document_id, DocumentId):
            raise ParserInputError("invalid_document_id")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise ParserInputError("invalid_document_version_id")
        if not isinstance(self.content, bytes):
            raise ParserInputError("non_bytes_content")
        if self.media_type != "text/plain":
            raise ParserInputError("unsupported_media_type")
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise ParserInputError("missing_encoding")
        if not _is_text_encoding(self.encoding):
            raise ParserInputError("invalid_encoding")


def decode_source(source: ParserInput) -> str:
    """Decode source bytes strictly into Unicode code-point text."""
    try:
        return source.content.decode(source.encoding, errors="strict")
    except UnicodeDecodeError:
        pass
    raise ParserDecodeError(source.encoding)


def prepare_text(source: ParserInput) -> NormalisedText:
    """Decode and normalise one parser input deterministically."""
    source_text = decode_source(source)
    return NormalisedText(source_text=source_text)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Canonical parser output for one document version."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    text: NormalisedText
    blocks: tuple[StructuralBlock, ...]
    pages: tuple[StructuralBlock, ...] | None
    sections: tuple[StructuralBlock, ...] | None
    paragraphs: tuple[StructuralBlock, ...] | None
    page_layouts: tuple[PageLayoutDescriptor, ...]
    language: str | None
    page_count: int | None
    character_count: int
    token_count: int | None
    parser_version: ParserVersion = CURRENT_PARSER_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate deterministic document-level metadata and source ranges."""
        if not isinstance(self.document_id, DocumentId):
            raise ValueError("invalid_document_id")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise ValueError("invalid_document_version_id")
        if not isinstance(self.parser_version, ParserVersion):
            raise ValueError("invalid_parser_version")
        if not isinstance(self.text, NormalisedText):
            raise ValueError("invalid_normalised_text")
        if not isinstance(self.character_count, int) or isinstance(
            self.character_count, bool
        ):
            raise ValueError("character_count_must_be_integer")
        if self.token_count is not None and (
            not isinstance(self.token_count, int) or isinstance(self.token_count, bool)
        ):
            raise ValueError("token_count_must_be_integer")
        if not isinstance(self.blocks, tuple):
            raise ValueError("blocks_must_be_tuple")
        source_length = len(self.text.source_text)

        if self.character_count != len(self.text.text):
            raise ValueError("character_count_mismatch")
        if self.page_count is not None:
            if not isinstance(self.page_count, int) or isinstance(
                self.page_count, bool
            ):
                raise ValueError("page_count_must_be_integer")
            if self.page_count < 0:
                raise ValueError("page_count_must_not_be_negative")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("token_count_must_not_be_negative")

        layouts_by_page = self._validate_page_layouts()

        previous_start_offset = -1
        for ordinal, block in enumerate(self.blocks):
            self._validate_block(
                block,
                ordinal=ordinal,
                source_length=source_length,
                layouts_by_page=layouts_by_page,
            )
            if block.source_span.start_offset < previous_start_offset:
                raise ValueError("blocks_must_be_in_source_order")
            previous_start_offset = block.source_span.start_offset

        self._validate_typed_collection(self.pages, kind="page", name="pages")
        self._validate_typed_collection(
            self.sections,
            kind="section",
            name="sections",
        )
        self._validate_typed_collection(
            self.paragraphs,
            kind="paragraph",
            name="paragraphs",
        )

    def _validate_block(
        self,
        block: StructuralBlock,
        *,
        ordinal: int,
        source_length: int,
        layouts_by_page: dict[int, PageLayoutDescriptor],
    ) -> None:
        """Validate one canonical block and its source provenance."""
        if not isinstance(block, StructuralBlock):
            raise ValueError("invalid_structural_block")
        span = block.source_span
        if span.document_version_id != self.document_version_id:
            raise ValueError("source_span_document_version_mismatch")
        if span.end_offset > source_length:
            raise ValueError("source_span_exceeds_source_text")
        if span.page_number is not None:
            self._validate_page_number(span.page_number)
        if span.bounding_box is not None:
            assert span.page_number is not None
            layout = layouts_by_page.get(span.page_number)
            if layout is None:
                raise ValueError("bounding_box_requires_page_layout")
            self._validate_bounding_box(span.bounding_box, layout)
        source_text = self.text.source_text[span.start_offset : span.end_offset]
        if block.text != normalise_source_text(source_text):
            raise ValueError("block_text_source_span_mismatch")
        expected_span_id = generate_source_span_id(
            self.document_version_id,
            parser_version=self.parser_version,
            ordinal=ordinal,
            kind=block.kind,
            start_offset=span.start_offset,
            end_offset=span.end_offset,
        )
        if span.source_span_id != expected_span_id:
            raise ValueError("source_span_id_mismatch")

    def _validate_typed_collection(
        self,
        collection: tuple[StructuralBlock, ...] | None,
        *,
        kind: Literal["page", "section", "paragraph"],
        name: Literal["pages", "sections", "paragraphs"],
    ) -> None:
        """Require typed collections to be exact immutable block projections."""
        expected = tuple(block for block in self.blocks if block.kind == kind)
        if collection is None:
            if expected:
                raise ValueError(f"{name}_missing_for_blocks")
            return
        if not isinstance(collection, tuple):
            raise ValueError(f"{name}_must_be_tuple")
        if any(block.kind != kind for block in collection):
            raise ValueError(f"{name}_contain_wrong_block_kind")
        if collection != expected:
            raise ValueError(f"{name}_must_match_blocks")

    def _validate_page_layouts(self) -> dict[int, PageLayoutDescriptor]:
        """Require unique page-number-ordered layout descriptors."""
        if not isinstance(self.page_layouts, tuple):
            raise ValueError("page_layouts_must_be_tuple")
        if any(
            not isinstance(layout, PageLayoutDescriptor) for layout in self.page_layouts
        ):
            raise ValueError("invalid_page_layout")
        layouts_by_page = {layout.page_number: layout for layout in self.page_layouts}
        if len(layouts_by_page) != len(self.page_layouts):
            raise ValueError("page_layouts_must_have_unique_page_numbers")
        if tuple(layout.page_number for layout in self.page_layouts) != tuple(
            sorted(layouts_by_page)
        ):
            raise ValueError("page_layouts_must_be_in_page_order")
        for layout in self.page_layouts:
            self._validate_page_number(layout.page_number)
        return layouts_by_page

    def _validate_page_number(self, page_number: int) -> None:
        """Require represented pages to fit the declared page count."""
        if self.page_count is not None and page_number > self.page_count:
            raise ValueError("page_number_exceeds_page_count")

    @staticmethod
    def _validate_bounding_box(
        bounding_box: BoundingBox,
        layout: PageLayoutDescriptor,
    ) -> None:
        """Require a box to fit the declared untransformed coordinate space."""
        if (
            bounding_box.left < 0
            or bounding_box.top < 0
            or bounding_box.right > layout.page_width
            or bounding_box.bottom > layout.page_height
        ):
            raise ValueError("bounding_box_exceeds_page_layout")


class Parser(Protocol):
    """Parse one document version into canonical parser output."""

    def parse(self, source: ParserInput, *, settings: Settings) -> ParseOutcome:
        """Parse a document version into canonical parser output."""
        ...
