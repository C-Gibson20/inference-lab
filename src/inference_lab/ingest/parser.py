"""Deterministic parsing contracts and text-native document parsing."""

import codecs
from dataclasses import dataclass
from typing import Literal, Protocol

import structlog
import unicodedata

from inference_lab.ingest.errors import ParserDecodeError
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


def normalise_text(source_text: str) -> str:
    """Apply the deterministic canonical text-normalisation policy."""

    newline_normalised = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", newline_normalised)


class ParserInputError(ValueError):
    """Represent a safe, machine-readable parser-input validation failure."""

    def __init__(self, category: ParserInputErrorCategory) -> None:
        """Create an error without retaining invalid source payloads."""
        super().__init__(category)
        self.category = category


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
        try:
            codecs.lookup(self.encoding)
        except LookupError as error:
            raise ParserInputError("invalid_encoding") from error


class Parser(Protocol):
    """Parse one document version into canonical parser output."""

    def parse(self, source: ParserInput, *, settings: Settings) -> object:
        """Parse a document version into canonical parser output."""
        ...


def decode_source(source: ParserInput) -> str:
    """Decode source bytes strictly into Unicode code-point text."""

    try:
        return source.content.decode(source.encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise ParserDecodeError(source.encoding) from error


def prepare_text(source: ParserInput) -> NormalisedText:
    """Decode and normalise one parser input deterministically."""

    source_text = decode_source(source)
    return NormalisedText(source_text=source_text, text=normalise_text(source_text))
