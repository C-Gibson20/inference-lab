"""Document ingestion contracts and pipeline stages."""

from inference_lab.ingest.parser import (
    NormalisedText,
    Parser,
    ParserInput,
    ParserInputError,
    decode_source,
    normalise_text,
    prepare_text,
)
from inference_lab.ingest.errors import ParserDecodeError

__all__ = [
    "Parser",
    "ParserInput",
    "ParserInputError",
    "ParserDecodeError",
    "NormalisedText",
    "decode_source",
    "normalise_text",
    "prepare_text",
]
