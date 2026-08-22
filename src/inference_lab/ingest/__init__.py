"""Document ingestion contracts and pipeline stages."""

from inference_lab.ingest.parser import (
    NormalisedText,
    Parser,
    ParsedDocument,
    ParserInput,
    ParserInputError,
    decode_source,
    normalise_text,
    prepare_text,
)
from inference_lab.ingest.sources import (
    BoundingBox,
    SourceSpan,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.ingest.errors import ParserDecodeError

__all__ = [
    "Parser",
    "ParsedDocument",
    "ParserInput",
    "ParserInputError",
    "ParserDecodeError",
    "NormalisedText",
    "decode_source",
    "normalise_text",
    "prepare_text",
    "BoundingBox",
    "SourceSpan",
    "StructuralBlock",
    "StructuralBlockKind",
]
