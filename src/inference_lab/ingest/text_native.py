"""Deterministic parser for plain text and text-layer PDF documents."""

import re

import structlog

from inference_lab.ingest._pdf_text import (
    PdfExtractionUnsupportedError,
    PdfPageText,
    extract_pdf_pages,
)
from inference_lab.ingest.errors import ParserDecodeError, ParserFailure
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.parser import (
    NormalisedText,
    ParsedDocument,
    ParserInput,
    ParserInputError,
    prepare_text,
)
from inference_lab.ingest.sources import (
    SourceSpan,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.ingest.versioning import (
    CURRENT_PARSER_VERSION,
    generate_source_span_id,
)
from inference_lab.settings import Settings

logger = structlog.get_logger(__name__)

_LINE_ENDING = r"(?:\r\n|\r(?!\n)|\n)"
_BLANK_LINE_SEPARATOR = re.compile(rf"{_LINE_ENDING}(?:[ \t]*{_LINE_ENDING})+")
_LEADING_BLANK_LINES = re.compile(rf"(?:[ \t]*{_LINE_ENDING})+")


def _paragraph_ranges(source_text: str) -> tuple[tuple[int, int], ...]:
    """Return ordered non-empty paragraph ranges in decoded source text."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    leading_blank_lines = _LEADING_BLANK_LINES.match(source_text)
    if leading_blank_lines is not None:
        cursor = leading_blank_lines.end()

    for separator in _BLANK_LINE_SEPARATOR.finditer(source_text, cursor):
        if separator.start() > cursor:
            ranges.append((cursor, separator.start()))
        cursor = separator.end()

    if cursor < len(source_text):
        ranges.append((cursor, len(source_text)))

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
            parser_version=CURRENT_PARSER_VERSION,
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
            text=normalise_source_text(source_text[start_offset:end_offset]),
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
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(prepared_text.text),
        token_count=None,
    )


def _build_pdf_document(
    source: ParserInput,
    extracted_pages: tuple[PdfPageText, ...],
) -> ParsedDocument:
    """Build page-aware canonical output from extracted PDF text."""
    page_texts = tuple(page.text for page in extracted_pages)
    source_text = "\n\n".join(page_texts)
    prepared_text = NormalisedText(source_text=source_text)
    layouts = tuple(page.layout for page in extracted_pages)

    blocks: list[StructuralBlock] = []
    page_blocks: list[StructuralBlock] = []
    paragraph_blocks: list[StructuralBlock] = []
    page_ranges: list[tuple[int, int]] = []
    cursor = 0
    for index, page_text in enumerate(page_texts):
        start_offset = cursor
        end_offset = start_offset + len(page_text)
        page_ranges.append((start_offset, end_offset))
        cursor = end_offset + (2 if index < len(page_texts) - 1 else 0)

    for page_number, (page, page_range) in enumerate(
        zip(extracted_pages, page_ranges, strict=True),
        start=1,
    ):
        page_start, page_end = page_range
        page_block = _pdf_block(
            source,
            prepared_text.source_text,
            ordinal=len(blocks),
            kind="page",
            start_offset=page_start,
            end_offset=page_end,
            page_number=page_number,
        )
        blocks.append(page_block)
        page_blocks.append(page_block)

        for start, end in _paragraph_ranges(page.text):
            paragraph_start = page_start + start
            paragraph_end = page_start + end
            paragraph_block = _pdf_block(
                source,
                prepared_text.source_text,
                ordinal=len(blocks),
                kind="paragraph",
                start_offset=paragraph_start,
                end_offset=paragraph_end,
                page_number=page_number,
            )
            blocks.append(paragraph_block)
            paragraph_blocks.append(paragraph_block)

    return ParsedDocument(
        document_id=source.document_id,
        document_version_id=source.document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        text=prepared_text,
        blocks=tuple(blocks),
        pages=tuple(page_blocks),
        sections=None,
        paragraphs=tuple(paragraph_blocks),
        page_layouts=layouts,
        language=None,
        page_count=len(extracted_pages),
        character_count=len(prepared_text.text),
        token_count=None,
    )


def _pdf_block(
    source: ParserInput,
    source_text: str,
    *,
    ordinal: int,
    kind: StructuralBlockKind,
    start_offset: int,
    end_offset: int,
    page_number: int,
) -> StructuralBlock:
    """Create one ordered page or paragraph block with stable provenance."""
    source_span_id = generate_source_span_id(
        source.document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=ordinal,
        kind=kind,
        start_offset=start_offset,
        end_offset=end_offset,
    )
    span = SourceSpan(
        source_span_id=source_span_id,
        document_version_id=source.document_version_id,
        start_offset=start_offset,
        end_offset=end_offset,
        page_number=page_number,
    )
    return StructuralBlock(
        kind=kind,
        text=normalise_source_text(source_text[start_offset:end_offset]),
        source_span=span,
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


def _source_size_failure() -> ParserFailure:
    """Create the safe failure record for an oversized source."""
    return ParserFailure(
        category="source_too_large",
        severity="fatal",
        stage="input",
    )


class TextNativeParser:
    """Parse plain text and extractable PDF text without OCR."""

    def parse(
        self,
        source: ParserInput,
        *,
        settings: Settings,
    ) -> ParseOutcome:
        """Parse one text-native document into a validated outcome."""
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

        source_size_bytes = len(source.content)
        max_source_bytes = settings.ingest.max_source_bytes

        if source_size_bytes > max_source_bytes:
            failure = _source_size_failure()
            log.error(
                "document_parse_failed",
                error_category=failure.category,
                failure_severity=failure.severity,
                source_size_bytes=source_size_bytes,
                max_source_bytes=max_source_bytes,
            )
            return _failure_outcome(failure)

        log.info("document_started")

        if source.media_type == "application/pdf":
            try:
                extracted_pages = extract_pdf_pages(source.content)
            except PdfExtractionUnsupportedError:
                failure = ParserFailure(
                    category="unsupported_format",
                    severity="fatal",
                    stage="input",
                )
                log.error(
                    "document_parse_failed",
                    error_category=failure.category,
                    failure_severity=failure.severity,
                )
                return _failure_outcome(failure)
            except Exception:
                failure = ParserFailure(
                    category="malformed_source",
                    severity="fatal",
                    stage="structure",
                )
                log.error(
                    "document_parse_failed",
                    error_category=failure.category,
                    failure_severity=failure.severity,
                )
                return _failure_outcome(failure)

            try:
                document = _build_pdf_document(source, extracted_pages)
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
        else:
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
            source_size_bytes=source_size_bytes,
        )

        return ParseOutcome(
            status="succeeded",
            document=document,
            failures=(),
        )
