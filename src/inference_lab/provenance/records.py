"""Immutable document identity and source-version records."""

from dataclasses import dataclass

from inference_lab.provenance.identifiers import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    SourceObservationId,
)
from inference_lab.provenance.values import (
    ContentHash,
    ExternalIdentifier,
    validate_external_identifiers,
)


@dataclass(frozen=True, slots=True)
class CorpusMetadata:
    """Describe one named corpus without using its name as identity."""

    corpus_id: CorpusId
    display_name: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the fields of the CorpusMetadata."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.display_name, str):
            raise TypeError("display_name must be a string.")
        if not self.display_name.strip():
            raise ValueError("display_name must not be blank.")
        if "\x00" in self.display_name:
            raise ValueError("display_name must not contain NUL characters.")


@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    """Identify one logical document across immutable source versions."""

    document_id: DocumentId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the fields of the DocumentIdentity."""
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
        self._validate()

    def _validate(self) -> None:
        """Validate the fields of the DocumentVersion."""
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")

        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")

        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")

        validate_external_identifiers(self.external_identifiers)


@dataclass(frozen=True, slots=True)
class SourceObservationState:
    """Record one append-only revision of a source-to-version association."""

    source_observation_id: SourceObservationId
    observation_revision: int
    document_id: DocumentId
    document_version_id: DocumentVersionId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the fields of the SourceObservationState."""
        if not isinstance(self.source_observation_id, SourceObservationId):
            raise TypeError("source_observation_id must be a SourceObservationId.")
        if not isinstance(self.observation_revision, int) or isinstance(
            self.observation_revision, bool
        ):
            raise TypeError("observation_revision must be an integer.")
        if self.observation_revision < 1:
            raise ValueError("observation_revision must be at least one.")
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
