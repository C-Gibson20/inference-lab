"""Immutable duplicate and document-family relationship records."""

from dataclasses import dataclass

from inference_lab.provenance.identifiers import (
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.provenance.values import DuplicateClass


class InvalidRelationshipError(ValueError):
    """Raised when a relationship violates provenance invariants."""


def _ordered_versions(
    first: DocumentVersionId,
    second: DocumentVersionId,
) -> tuple[DocumentVersionId, DocumentVersionId]:
    """Return duplicate endpoints in canonical order."""
    if first == second:
        raise InvalidRelationshipError(
            "a duplicate relationship cannot link a version to itself."
        )

    if first.as_str() < second.as_str():
        return first, second

    return second, first


@dataclass(frozen=True, slots=True)
class DuplicateRelationship:
    """Relate two independently identifiable source versions as duplicates."""

    first_version_id: DocumentVersionId
    second_version_id: DocumentVersionId
    duplicate_class: DuplicateClass

    def __post_init__(self) -> None:
        if not isinstance(self.first_version_id, DocumentVersionId):
            raise TypeError("first_version_id must be a DocumentVersionId.")

        if not isinstance(self.second_version_id, DocumentVersionId):
            raise TypeError("second_version_id must be a DocumentVersionId.")

        if not isinstance(self.duplicate_class, DuplicateClass):
            raise TypeError("duplicate_class must be a DuplicateClass.")

        first, second = _ordered_versions(
            self.first_version_id,
            self.second_version_id,
        )

        if self.first_version_id != first or self.second_version_id != second:
            raise InvalidRelationshipError(
                "duplicate relationship endpoints must be in canonical order."
            )

    def key(self) -> tuple[str, str, str]:
        """Return the deterministic identity key for this relationship."""
        return (
            self.duplicate_class.value,
            self.first_version_id.as_str(),
            self.second_version_id.as_str(),
        )


@dataclass(frozen=True, slots=True)
class DocumentFamilyMembership:
    """Associate a document, optionally a source version, with a family."""

    family_id: DocumentFamilyId
    document_id: DocumentId
    document_version_id: DocumentVersionId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, DocumentFamilyId):
            raise TypeError("family_id must be a DocumentFamilyId.")

        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")

        if self.document_version_id is not None and not isinstance(
            self.document_version_id,
            DocumentVersionId,
        ):
            raise TypeError("document_version_id must be a DocumentVersionId or None.")

    def key(self) -> tuple[str, str, str | None]:
        """Return the deterministic membership key."""
        return (
            self.family_id.as_str(),
            self.document_id.as_str(),
            (
                self.document_version_id.as_str()
                if self.document_version_id is not None
                else None
            ),
        )
