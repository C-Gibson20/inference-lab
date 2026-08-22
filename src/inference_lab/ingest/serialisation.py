"""Canonical serialisation for parser output."""

import json
from typing import Any

from inference_lab.ingest.parser import ParsedDocument
from inference_lab.ingest.sources import SourceSpan, StructuralBlock


def _source_span_payload(span: SourceSpan) -> dict[str, Any]:
    """Convert a SourceSpan to a canonical JSON-compatible mapping."""

    bounding_box = span.bounding_box
    return {
        "source_span_id": str(span.source_span_id),
        "document_version_id": str(span.document_version_id),
        "start_offset": span.start_offset,
        "end_offset": span.end_offset,
        "page_number": span.page_number,
        "bounding_box": (
            {
                "left": bounding_box.left,
                "top": bounding_box.top,
                "right": bounding_box.right,
                "bottom": bounding_box.bottom,
            }
            if bounding_box is not None
            else None
        ),
    }


def _block_payload(block: StructuralBlock) -> dict[str, Any]:
    """Convert one structural block to a canonical mapping."""
    return {
        "kind": block.kind,
        "text": block.text,
        "source_span": _source_span_payload(block.source_span),
    }


def parsed_document_payload(document: ParsedDocument) -> dict[str, Any]:
    """Return the canonical semantic payload for parser output."""
    return {
        "document_id": str(document.document_id),
        "document_version_id": str(document.document_version_id),
        "parser_version": {
            "implementation": document.parser_version.implementation,
            "contract_version": document.parser_version.contract_version,
            "normalization_version": document.parser_version.normalisation_version,
        },
        "text": {
            "text": document.text.text,
            "source_text": document.text.source_text,
        },
        "blocks": [_block_payload(block) for block in document.blocks],
        "pages": (
            [_block_payload(block) for block in document.pages]
            if document.pages is not None
            else None
        ),
        "sections": (
            [_block_payload(block) for block in document.sections]
            if document.sections is not None
            else None
        ),
        "paragraphs": (
            [_block_payload(block) for block in document.paragraphs]
            if document.paragraphs is not None
            else None
        ),
        "language": document.language,
        "page_count": document.page_count,
        "character_count": document.character_count,
        "token_count": document.token_count,
    }


def serialise_parsed_document(document: ParsedDocument) -> bytes:
    """Serialize parser output as canonical UTF-8 JSON with a trailing newline."""

    payload = parsed_document_payload(document)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{text}\n".encode("utf-8")
