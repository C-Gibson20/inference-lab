"""Immutable document identity and source-version records."""

import re
from dataclasses import dataclass
from typing import Final

from inference_lab.provenance.identifiers import (
    DocumentId,
    DocumentVersionId,
)
from inference_lab.provenance.values import ContentHash

_IDENTIFIER_NAMESPACE: Final = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class ExternalIdentifier:
    """A stable identifier supplied by a publication or patent source."""

    namespace: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str):
            raise TypeError("namespace must be a string.")

        if not isinstance(self.value, str):
            raise TypeError("value must be a string.")

        if _IDENTIFIER_NAMESPACE.fullmatch(self.namespace) is None:
            raise ValueError("namespace must be a lowercase identifier namespace.")

        if not self.value.strip():
            raise ValueError("value must not be empty or whitespace.")

    def as_tuple(self) -> tuple[str, str]:
        """Return the identifier as a namespace/value tuple."""
        return self.namespace, self.value


def _validate_external_identifiers(
    identifiers: tuple[ExternalIdentifier, ...],
) -> None:
    """Validate an immutable, canonically ordered identifier collection."""
    if not isinstance(identifiers, tuple):
        raise TypeError("external_identifiers must be a tuple.")

    if any(
        not isinstance(identifier, ExternalIdentifier) for identifier in identifiers
    ):
        raise TypeError(
            "external_identifiers must contain only ExternalIdentifier values."
        )

    keys = tuple(identifier.as_tuple() for identifier in identifiers)

    if len(keys) != len(set(keys)):
        raise ValueError("external_identifiers must not contain duplicates.")

    if keys != tuple(sorted(keys)):
        raise ValueError("external_identifiers must be in canonical order.")


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

        _validate_external_identifiers(self.external_identifiers)
