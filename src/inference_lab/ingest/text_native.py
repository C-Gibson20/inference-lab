"""Deterministic parser for text-native plain-text documents."""

import re

import structlog

from inference_lab.ingest.errors import ParserDecodeError, ParserFailure
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.parser import (
    ParsedDocument,
    ParserInputError,
    ParserInput,
    NormalisedText,
    prepare_text,
)
from inference_lab.ingest.sources import SourceSpan, StructuralBlock
from inference_lab.ingest.utils import normalise_text
from inference_lab.ingest.versioning import (
    CURRENT_PARSER_VERSION,
    generate_source_span_id,
)
from inference_lab.settings import Settings

logger = structlog.get_logger(__name__)

_BLANK_LINE_SEPARATOR = re.compile(r"(?:\r\n|\r|\n)[ \t]*(?:(?:\r\n|\r|\n)[ \t]*)+")
_EDGE_BLANK_LINES = re.compile(r"(?:[ \t]*(?:\r\n|\r|\n))+")


def _paragraph_ranges(source_text: str) -> tuple[tuple[int, int], ...]:
    """Return ordered non-empty paragraph ranges in decoded source text."""

    ranges: list[tuple[int, int]] = []
    cursor = 0
    leading_blank_lines = _EDGE_BLANK_LINES.match(source_text)
    if leading_blank_lines is not None:
        cursor = leading_blank_lines.end()

    trailing_blank_lines = _EDGE_BLANK_LINES.search(source_text, cursor)
    content_end = (
        trailing_blank_lines.start()
        if trailing_blank_lines is not None
        and trailing_blank_lines.end() == len(source_text)
        else len(source_text)
    )

    for separator in _BLANK_LINE_SEPARATOR.finditer(source_text, cursor, content_end):
        if separator.start() > cursor:
            ranges.append((cursor, separator.start()))
        cursor = separator.end()

    if cursor < content_end:
        ranges.append((cursor, content_end))

    return tuple(ranges)


def _paragraph_blocks(
    source: ParserInput,
    source_text: str,
) -> tuple[StructuralBlock, ...]:
    """Create ordered paragraph blocks with deterministic source spans."""

    blocks: list[StructuralBlock] = []

    for ordinal, (start_offset, end_offset) in enumerate(
        _paragraph_ranges(source_text)
    ):
        source_span_id = generate_source_span_id(
            source.document_version_id,
            ordinal=ordinal,
            kind="paragraph",
            start_offset=start_offset,
            end_offset=end_offset,
        )
        span = SourceSpan(
            source_span_id=source_span_id,
            document_version_id=source.document_version_id,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        block = StructuralBlock(
            kind="paragraph",
            text=normalise_text(source_text[start_offset:end_offset]),
            source_span=span,
        )
        blocks.append(block)

    return tuple(blocks)


def _build_document(
    source: ParserInput,
    prepared_text: NormalisedText,
) -> ParsedDocument:
    """Build canonical parsed output for one text-native document."""
    blocks = _paragraph_blocks(source, prepared_text.source_text)

    return ParsedDocument(
        document_id=source.document_id,
        document_version_id=source.document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        text=prepared_text,
        blocks=blocks,
        pages=None,
        sections=None,
        paragraphs=blocks,
        language=None,
        page_count=None,
        character_count=len(prepared_text.text),
        token_count=None,
    )


def _decode_failure() -> ParserFailure:
    """Create the safe failure record for malformed source bytes."""
    return ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )


def _failure_outcome(failure: ParserFailure) -> ParseOutcome:
    """Return a failed outcome containing one safe fatal failure."""
    return ParseOutcome(status="failed", document=None, failures=(failure,))


class TextNativeParser:
    """Parse deterministic text/plain document inputs."""

    def parse(
        self,
        source: ParserInput,
        *,
        settings: Settings,
    ) -> ParseOutcome:
        """Parse one text-native document into a validated outcome."""
        del settings

        document_id = (
            str(source.document_id) if isinstance(source, ParserInput) else None
        )
        log = logger.bind(
            pipeline_stage="parse",
            document_id=document_id,
        )

        if not isinstance(source, ParserInput):
            failure = ParserFailure(
                category="invalid_input",
                severity="fatal",
                stage="input",
            )
            log.error(
                "document_parse_failed",
                error_category=failure.category,
                failure_severity=failure.severity,
            )
            return _failure_outcome(failure)

        log.info("document_started")

        try:
            prepared_text = prepare_text(source)
        except ParserDecodeError:
            failure = _decode_failure()
            log.error(
                "document_parse_failed",
                error_category=failure.category,
                failure_severity=failure.severity,
            )
            return _failure_outcome(failure)

        try:
            document = _build_document(source, prepared_text)
        except (ParserInputError, ValueError):
            failure = ParserFailure(
                category="invalid_output",
                severity="fatal",
                stage="output",
            )
            log.error(
                "document_parse_failed",
                error_category=failure.category,
                failure_severity=failure.severity,
            )
            return _failure_outcome(failure)

        log.info(
            "document_completed",
            parser_version=document.parser_version.value,
            character_count=document.character_count,
            records_emitted=len(document.blocks),
        )

        return ParseOutcome(
            status="succeeded",
            document=document,
            failures=(),
        )
