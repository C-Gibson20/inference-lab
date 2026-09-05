"""Deterministic values shared by source-artefact contract tests."""

import hashlib

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentVersionId,
    HashAlgorithm,
)

DOCUMENT_ID = DocumentId.from_str("10000000-0000-4000-8000-000000000001")
DOCUMENT_VERSION_ID = DocumentVersionId.from_str("20000000-0000-4000-8000-000000000001")
SOURCE_BYTES = b"source bytes"
SOURCE_HASH = ContentHash(
    HashAlgorithm.SHA256,
    hashlib.sha256(SOURCE_BYTES).hexdigest(),
)
