"""Coordinate publication of one validated source parse bundle."""

from dataclasses import dataclass
from typing import Iterable, Protocol

from inference_lab.ingest.outcomes import ParseStatus
from inference_lab.ingest.parse_mapping import map_parse_persistence_request
from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.ingest.parse_preflight import (
    ParsePreflightReader,
    preflight_parse_persistence,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentVersionId,
    RecordId,
)
from inference_lab.settings import Settings
from inference_lab.storage import (
    ParsedRepresentation,
    SourceArtefact,
    build_source_blob_storage_key,
)
from inference_lab.storage import (
    ParseOutcome as StorageParseOutcome,
)


class ParsePublicationRepository(ParsePreflightReader, Protocol):
    """Provide the reads and single bundled write used by parse publication."""

    def persist_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
        parse_outcome: StorageParseOutcome | None = None,
        parsed_representation: ParsedRepresentation | None = None,
    ) -> None:
        """Publish source bytes, metadata, and parse records atomically."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsePersistenceResult:
    """Report stable public identities after a completed publication."""

    parse_id: RecordId
    parsed_record_id: RecordId | None
    status: ParseStatus

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Keep result identities consistent with the parse status."""
        if not isinstance(self.parse_id, RecordId):
            raise TypeError("parse_id must be a RecordId.")
        if self.parsed_record_id is not None and not isinstance(
            self.parsed_record_id,
            RecordId,
        ):
            raise TypeError("parsed_record_id must be a RecordId or None.")
        if self.status not in (
            "succeeded",
            "completed_with_recoverable_failures",
            "failed",
        ):
            raise ValueError("invalid parse persistence status.")
        if self.status == "failed" and self.parsed_record_id is not None:
            raise ValueError("a failed parse must not report a parsed record.")
        if self.status != "failed" and self.parsed_record_id is None:
            raise ValueError("a represented parse must report a parsed record.")


def _validate_repository(repository: object) -> None:
    if not callable(getattr(repository, "persist_source_artefact", None)):
        raise TypeError("repository must provide bundled parse publication.")


def _build_source_artefact(
    request: ParsePersistenceRequest,
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    content_hash: ContentHash,
    source_byte_count: int,
) -> SourceArtefact:
    return SourceArtefact(
        document_id=document_id,
        document_version_id=document_version_id,
        content_hash=content_hash,
        byte_count=source_byte_count,
        media_type=request.parser_input.media_type,
        storage_key=build_source_blob_storage_key(content_hash),
    )


def persist_parse_request(
    repository: ParsePublicationRepository,
    request: ParsePersistenceRequest,
    *,
    settings: Settings,
) -> ParsePersistenceResult:
    """Validate, map, and publish one immutable source parse bundle."""
    _validate_repository(repository)
    if not isinstance(request, ParsePersistenceRequest):
        raise TypeError("request must be a ParsePersistenceRequest.")
    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings.")

    max_source_bytes = settings.ingest.max_source_bytes
    preflight = preflight_parse_persistence(
        repository,
        request,
        max_source_bytes=max_source_bytes,
    )
    mapped = map_parse_persistence_request(request)
    document_version = preflight.document_version
    artefact = _build_source_artefact(
        request,
        document_id=document_version.document_id,
        document_version_id=document_version.document_version_id,
        content_hash=document_version.content_hash,
        source_byte_count=preflight.source_byte_count,
    )

    repository.persist_source_artefact(
        request.corpus_id,
        artefact,
        (request.parser_input.content,),
        max_source_bytes=max_source_bytes,
        parse_outcome=mapped.outcome,
        parsed_representation=mapped.representation,
    )
    return ParsePersistenceResult(
        parse_id=request.parse_id,
        parsed_record_id=request.parsed_record_id,
        status=request.outcome.status,
    )
