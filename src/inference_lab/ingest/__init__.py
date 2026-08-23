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
from inference_lab.ingest.serialisation import (
    parsed_document_payload,
    serialise_parsed_document,
)
from inference_lab.ingest.versioning import (
    CURRENT_PARSER_VERSION,
    ParserVersion,
    generate_source_span_id,
)
from inference_lab.ingest.errors import (
    ParserDecodeError,
    ParserFailure,
    ParserFailureCategory,
    ParserFailureSeverity,
    ParserFailureStage,
)
from inference_lab.ingest.outcomes import ParseOutcome, ParseStatus
from inference_lab.ingest.text_native import TextNativeParser

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
    "parsed_document_payload",
    "serialise_parsed_document",
    "CURRENT_PARSER_VERSION",
    "ParserVersion",
    "generate_source_span_id",
    "ParserFailure",
    "ParserFailureCategory",
    "ParserFailureSeverity",
    "ParserFailureStage",
    "ParseOutcome",
    "ParseStatus",
    "TextNativeParser",
]
