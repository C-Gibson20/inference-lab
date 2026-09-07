"""Deterministic values shared by source-artefact contract tests."""

import hashlib

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage import (
    CollectionAvailability,
    ContentOrigin,
    ParsedRepresentation,
    ProducerIdentity,
    SourceLocation,
    StructuralBlock,
    StructuralBlockKind,
)

DOCUMENT_ID = DocumentId.from_str("10000000-0000-4000-8000-000000000001")
DOCUMENT_VERSION_ID = DocumentVersionId.from_str("20000000-0000-4000-8000-000000000001")
PARSE_ID = RecordId.from_str("30000000-0000-4000-8000-000000000001")
PARSED_RECORD_ID = RecordId.from_str("40000000-0000-4000-8000-000000000001")
FIRST_SPAN_ID = SourceSpanId.from_str("50000000-0000-4000-8000-000000000001")
SECOND_SPAN_ID = SourceSpanId.from_str("50000000-0000-4000-8000-000000000002")
SUPERSEDED_RECORD_ID = RecordId.from_str("60000000-0000-4000-8000-000000000001")
PARSER = ProducerIdentity(implementation="text-native", contract_version="2.0.0")
OCR_PRODUCER = ProducerIdentity(
    implementation="local-ocr",
    contract_version="1.0.0",
    artefact_identity="ocr-model",
    artefact_revision="revision-1",
)
SOURCE_BYTES = b"source bytes"
SOURCE_TEXT = "Cafe\u0301\r\n\r\nResults"
SOURCE_HASH = ContentHash(
    HashAlgorithm.SHA256,
    hashlib.sha256(SOURCE_BYTES).hexdigest(),
)


def parsed_representation(
    *,
    record_id: RecordId = PARSED_RECORD_ID,
    supersedes_record_id: RecordId | None = None,
) -> ParsedRepresentation:
    """Build a deterministic representation with decoded-text offsets."""
    return ParsedRepresentation(
        record_id=record_id,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        source_text=SOURCE_TEXT,
        text="Café\n\nResults",
        parser=PARSER,
        normalisation_version="1.0.0",
        language="en",
        page_count=None,
        character_count=13,
        token_count=2,
        page_collection_availability=CollectionAvailability.AVAILABLE,
        section_collection_availability=CollectionAvailability.UNAVAILABLE,
        paragraph_collection_availability=CollectionAvailability.AVAILABLE,
        blocks=(
            StructuralBlock(
                ordinal=0,
                kind=StructuralBlockKind.PARAGRAPH,
                text="Café",
                source_span_id=FIRST_SPAN_ID,
            ),
            StructuralBlock(
                ordinal=1,
                kind=StructuralBlockKind.PARAGRAPH,
                text="Results",
                source_span_id=SECOND_SPAN_ID,
            ),
        ),
        source_locations=(
            SourceLocation(
                source_span_id=FIRST_SPAN_ID,
                document_version_id=DOCUMENT_VERSION_ID,
                parsed_record_id=record_id,
                start_offset=0,
                end_offset=5,
                content_origin=ContentOrigin.NATIVE,
            ),
            SourceLocation(
                source_span_id=SECOND_SPAN_ID,
                document_version_id=DOCUMENT_VERSION_ID,
                parsed_record_id=record_id,
                start_offset=9,
                end_offset=len(SOURCE_TEXT),
                content_origin=ContentOrigin.NATIVE,
            ),
        ),
        page_layouts=(),
        supersedes_record_id=supersedes_record_id,
    )
