"""Immutable document identity and source-version records."""

from dataclasses import dataclass

from inference_lab.provenance.identifiers import (
    DocumentId,
    DocumentVersionId,
)
from inference_lab.provenance.values import (
    ContentHash,
    ExternalIdentifier,
    validate_external_identifiers,
)


@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    """Identify one logical document across immutable source versions."""

    document_id: DocumentId

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """Identify one immutable source version of a logical document."""

    document_version_id: DocumentVersionId
    document_id: DocumentId
    content_hash: ContentHash
    external_identifiers: tuple[ExternalIdentifier, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")

        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")

        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")

        validate_external_identifiers(self.external_identifiers)
