"""Document ingestion contracts and pipeline stages."""

from inference_lab.ingest.errors import (
    ParserDecodeError,
    ParserFailure,
    ParserFailureCategory,
    ParserFailureSeverity,
    ParserFailureStage,
    ParserFailureUnitKind,
)
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.outcomes import ParseOutcome, ParseStatus
from inference_lab.ingest.parser import (
    NormalisedText,
    ParsedDocument,
    Parser,
    ParserInput,
    ParserInputError,
    decode_source,
    prepare_text,
)
from inference_lab.ingest.serialisation import (
    parsed_document_payload,
    serialise_parsed_document,
)
from inference_lab.ingest.sources import (
    BoundingBox,
    SourceSpan,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.ingest.text_native import TextNativeParser
from inference_lab.ingest.versioning import (
    CURRENT_PARSER_VERSION,
    ParserVersion,
    generate_source_span_id,
)

__all__ = [
    "CURRENT_PARSER_VERSION",
    "BoundingBox",
    "NormalisedText",
    "ParseOutcome",
    "ParseStatus",
    "ParsedDocument",
    "Parser",
    "ParserDecodeError",
    "ParserFailure",
    "ParserFailureCategory",
    "ParserFailureSeverity",
    "ParserFailureStage",
    "ParserFailureUnitKind",
    "ParserInput",
    "ParserInputError",
    "ParserVersion",
    "SourceSpan",
    "StructuralBlock",
    "StructuralBlockKind",
    "TextNativeParser",
    "decode_source",
    "generate_source_span_id",
    "normalise_source_text",
    "parsed_document_payload",
    "prepare_text",
    "serialise_parsed_document",
]
