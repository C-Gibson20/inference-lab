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
from inference_lab.storage import ProducerIdentity

DOCUMENT_ID = DocumentId.from_str("10000000-0000-4000-8000-000000000001")
DOCUMENT_VERSION_ID = DocumentVersionId.from_str("20000000-0000-4000-8000-000000000001")
PARSE_ID = RecordId.from_str("30000000-0000-4000-8000-000000000001")
PARSED_RECORD_ID = RecordId.from_str("40000000-0000-4000-8000-000000000001")
FIRST_SPAN_ID = SourceSpanId.from_str("50000000-0000-4000-8000-000000000001")
PARSER = ProducerIdentity(implementation="text-native", contract_version="2.0.0")
OCR_PRODUCER = ProducerIdentity(
    implementation="local-ocr",
    contract_version="1.0.0",
    artefact_identity="ocr-model",
    artefact_revision="revision-1",
)
SOURCE_BYTES = b"source bytes"
SOURCE_HASH = ContentHash(
    HashAlgorithm.SHA256,
    hashlib.sha256(SOURCE_BYTES).hexdigest(),
)
