"""Immutable storage contracts for managed original source artefacts."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from inference_lab.provenance import (
    ContentHash,
    DocumentId,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage.source_versioning import (
    CURRENT_SOURCE_RECORD_SCHEMA_VERSION,
    validate_source_record_schema_version,
)

_MEDIA_TYPE: Final = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceArtefact:
    """Describe one immutable original held by managed source storage."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    content_hash: ContentHash
    byte_count: int
    media_type: str
    storage_key: str
    record_schema_version: str = CURRENT_SOURCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the source artefact's attributes."""
        validate_source_record_schema_version(self.record_schema_version)
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")
        if self.content_hash.algorithm is not HashAlgorithm.SHA256:
            raise ValueError("source artefacts require a SHA-256 content hash.")
        if not isinstance(self.byte_count, int) or isinstance(self.byte_count, bool):
            raise TypeError("byte_count must be an integer.")
        if self.byte_count < 0:
            raise ValueError("byte_count must not be negative.")
        if not isinstance(self.media_type, str):
            raise TypeError("media_type must be a string.")
        if _MEDIA_TYPE.fullmatch(self.media_type) is None:
            raise ValueError("media_type must be a canonical type and subtype.")
        self._validate_storage_key()

    def _validate_storage_key(self) -> None:
        """Require an opaque canonical relative POSIX storage key."""
        if not isinstance(self.storage_key, str):
            raise TypeError("storage_key must be a string.")
        if (
            not self.storage_key
            or "\x00" in self.storage_key
            or "\\" in self.storage_key
        ):
            raise ValueError("storage_key must be a canonical relative storage key.")
        key = PurePosixPath(self.storage_key)
        if (
            key.is_absolute()
            or key.as_posix() != self.storage_key
            or any(part in {"", ".", ".."} for part in key.parts)
        ):
            raise ValueError("storage_key must be a canonical relative storage key.")
