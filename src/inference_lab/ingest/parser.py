"""Deterministic parsing contracts and text-native document parsing."""

import codecs
from dataclasses import dataclass
from typing import Literal, Protocol

import structlog

from inference_lab.ingest.sources import StructuralBlock
from inference_lab.ingest.errors import ParserDecodeError
from inference_lab.provenance import DocumentId, DocumentVersionId
from inference_lab.settings import Settings
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.utils import normalise_text
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION, ParserVersion

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
    text: str


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
    return NormalisedText(source_text=source_text, text=normalise_text(source_text))


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
        source_length = len(self.text.source_text)

        if self.character_count != len(self.text.text):
            raise ValueError("character_count_mismatch")
        if self.page_count is not None and self.page_count < 0:
            raise ValueError("page_count_must_not_be_negative")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("token_count_must_not_be_negative")

        for block in self.blocks:
            self._validate_block(block)
            if block.source_span.end_offset > source_length:
                raise ValueError("source_span_exceeds_source_text")

        for collection in (self.pages, self.sections, self.paragraphs):
            if collection is not None:
                for block in collection:
                    self._validate_block(block)
                    if block.source_span.end_offset > source_length:
                        raise ValueError("source_span_exceeds_source_text")

    def _validate_block(self, block: StructuralBlock) -> None:
        """Validate block type and document-version provenance."""
        if not isinstance(block, StructuralBlock):
            raise ValueError("invalid_structural_block")
        if block.source_span.document_version_id != self.document_version_id:
            raise ValueError("source_span_document_version_mismatch")


class Parser(Protocol):
    """Parse one document version into canonical parser output."""

    def parse(self, source: ParserInput, *, settings: Settings) -> ParseOutcome:
        """Parse a document version into canonical parser output."""
        ...
