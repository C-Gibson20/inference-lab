"""Read-only validation for parser-output persistence."""

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
)
from inference_lab.storage import ParsedRepresentation, SourceArtefact


class ParsePreflightErrorCategory(StrEnum):
    """Bounded reasons why a parse request is unsafe to publish."""

    SOURCE_NOT_AVAILABLE = "source_not_available"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    SOURCE_TOO_LARGE = "source_too_large"
    SOURCE_CONTENT_MISMATCH = "source_content_mismatch"
    SOURCE_MEDIA_TYPE_MISMATCH = "source_media_type_mismatch"
    INVALID_PREDECESSOR = "invalid_predecessor"


class ParsePreflightError(ValueError):
    """Reject unsafe parser persistence without retaining source payloads."""

    def __init__(self, category: ParsePreflightErrorCategory) -> None:
        super().__init__("Parse persistence preflight failed.")
        self.category = category


class ParsePreflightReader(Protocol):
    """Read the corpus-scoped records required before parse publication."""

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        """Return exact catalogue membership without probing corpus existence."""
        ...

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        """Return the immutable source version after membership is established."""
        ...

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        """Return any corpus-visible source metadata for consistency checks."""
        ...

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        """Return one corpus-visible predecessor, or ``None``."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsePersistencePreflight:
    """Retain validated metadata needed to construct a source parse bundle."""

    membership: CorpusDocumentVersionMembership = field(repr=False)
    document_version: DocumentVersion = field(repr=False)
    source_byte_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the preflight is internally consistent and well-typed."""
        if not isinstance(self.membership, CorpusDocumentVersionMembership):
            raise TypeError("membership must be a CorpusDocumentVersionMembership.")
        if not isinstance(self.document_version, DocumentVersion):
            raise TypeError("document_version must be a DocumentVersion.")
        if not isinstance(self.source_byte_count, int) or isinstance(
            self.source_byte_count,
            bool,
        ):
            raise TypeError("source_byte_count must be an integer.")
        if self.source_byte_count < 0:
            raise ValueError("source_byte_count must not be negative.")
        if (
            self.membership.document_id != self.document_version.document_id
            or self.membership.document_version_id
            != self.document_version.document_version_id
        ):
            raise ValueError("membership and document version must agree.")


def _validate_reader(reader: object) -> None:
    for method_name in (
        "get_corpus_document_version_membership",
        "get_document_version",
        "get_source_artefact",
        "get_parsed_representation",
    ):
        if not callable(getattr(reader, method_name, None)):
            raise TypeError("reader must provide parse preflight reads.")


def _source_not_available() -> ParsePreflightError:
    return ParsePreflightError(ParsePreflightErrorCategory.SOURCE_NOT_AVAILABLE)


def _load_membership(
    reader: ParsePreflightReader,
    request: ParsePersistenceRequest,
) -> CorpusDocumentVersionMembership:
    membership = reader.get_corpus_document_version_membership(
        request.corpus_id,
        request.parser_input.document_version_id,
    )
    if membership is None:
        raise _source_not_available()
    if (
        membership.corpus_id != request.corpus_id
        or membership.document_version_id != request.parser_input.document_version_id
    ):
        raise _source_not_available()
    if membership.document_id != request.parser_input.document_id:
        raise ParsePreflightError(ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH)
    return membership


def _load_document_version(
    reader: ParsePreflightReader,
    membership: CorpusDocumentVersionMembership,
) -> DocumentVersion:
    document_version = reader.get_document_version(membership.document_version_id)
    if document_version is None:
        raise _source_not_available()
    if (
        document_version.document_version_id != membership.document_version_id
        or document_version.document_id != membership.document_id
    ):
        raise ParsePreflightError(ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH)
    return document_version


def _validate_source_bytes(
    request: ParsePersistenceRequest,
    document_version: DocumentVersion,
    *,
    max_source_bytes: int,
) -> int:
    source = request.parser_input.content
    source_byte_count = len(source)
    if source_byte_count > max_source_bytes:
        raise ParsePreflightError(ParsePreflightErrorCategory.SOURCE_TOO_LARGE)
    digest = hashlib.sha256(source).hexdigest()
    if (
        document_version.content_hash.algorithm is not HashAlgorithm.SHA256
        or document_version.content_hash.digest != digest
    ):
        raise ParsePreflightError(ParsePreflightErrorCategory.SOURCE_CONTENT_MISMATCH)
    return source_byte_count


def _validate_existing_source(
    reader: ParsePreflightReader,
    request: ParsePersistenceRequest,
    document_version: DocumentVersion,
    *,
    source_byte_count: int,
) -> None:
    artefact = reader.get_source_artefact(
        request.corpus_id,
        document_version.document_version_id,
    )
    if artefact is None:
        return
    if (
        artefact.document_id != document_version.document_id
        or artefact.document_version_id != document_version.document_version_id
        or artefact.content_hash != document_version.content_hash
        or artefact.byte_count != source_byte_count
    ):
        raise ParsePreflightError(ParsePreflightErrorCategory.SOURCE_IDENTITY_MISMATCH)
    if artefact.media_type != request.parser_input.media_type:
        raise ParsePreflightError(
            ParsePreflightErrorCategory.SOURCE_MEDIA_TYPE_MISMATCH
        )


def _validate_predecessor(
    reader: ParsePreflightReader,
    request: ParsePersistenceRequest,
) -> None:
    predecessor_id = request.supersedes_record_id
    if predecessor_id is None:
        return
    predecessor = reader.get_parsed_representation(
        request.corpus_id,
        predecessor_id,
    )
    if predecessor is None:
        raise ParsePreflightError(ParsePreflightErrorCategory.INVALID_PREDECESSOR)
    if (
        predecessor.record_id != predecessor_id
        or predecessor.document_id != request.parser_input.document_id
        or predecessor.document_version_id != request.parser_input.document_version_id
        or predecessor.ocr_producer is not None
        or predecessor.parser.artefact_identity is not None
        or predecessor.parser.artefact_revision is not None
    ):
        raise ParsePreflightError(ParsePreflightErrorCategory.INVALID_PREDECESSOR)
    if (
        predecessor.parser.implementation == request.parser_version.implementation
        and predecessor.parser.contract_version
        == request.parser_version.contract_version
        and predecessor.normalisation_version
        == request.parser_version.normalisation_version
    ):
        raise ParsePreflightError(ParsePreflightErrorCategory.INVALID_PREDECESSOR)


def preflight_parse_persistence(
    reader: ParsePreflightReader,
    request: ParsePersistenceRequest,
    *,
    max_source_bytes: int,
) -> ParsePersistencePreflight:
    """Validate one parse request completely before publication can begin."""
    _validate_reader(reader)
    if not isinstance(request, ParsePersistenceRequest):
        raise TypeError("request must be a ParsePersistenceRequest.")
    if not isinstance(max_source_bytes, int) or isinstance(max_source_bytes, bool):
        raise TypeError("max_source_bytes must be an integer.")
    if max_source_bytes <= 0:
        raise ValueError("max_source_bytes must be positive.")

    membership = _load_membership(reader, request)
    document_version = _load_document_version(reader, membership)
    source_byte_count = _validate_source_bytes(
        request,
        document_version,
        max_source_bytes=max_source_bytes,
    )
    _validate_existing_source(
        reader,
        request,
        document_version,
        source_byte_count=source_byte_count,
    )
    _validate_predecessor(reader, request)
    return ParsePersistencePreflight(
        membership=membership,
        document_version=document_version,
        source_byte_count=source_byte_count,
    )
