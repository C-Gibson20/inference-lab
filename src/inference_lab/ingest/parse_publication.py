"""Coordinate publication of one validated source parse bundle."""

from dataclasses import dataclass
from typing import Iterable, Protocol

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab.ingest.outcomes import ParseStatus
from inference_lab.ingest.parse_mapping import map_parse_persistence_request
from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.ingest.parse_preflight import (
    ParsePreflightError,
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
    StorageError,
    build_source_blob_storage_key,
)
from inference_lab.storage import (
    ParseOutcome as StorageParseOutcome,
)

_LOGGER = structlog.get_logger(__name__)


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


def _emit_event(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded parse metadata without changing publication outcomes."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="ingest", **fields)
    except Exception:
        pass


def _request_log_fields(request: ParsePersistenceRequest) -> dict[str, object]:
    parser_version = request.parser_version
    parsed_record_id = request.parsed_record_id
    return {
        "parser_implementation": parser_version.implementation,
        "parser_contract_version": parser_version.contract_version,
        "normalisation_version": parser_version.normalisation_version,
        "parse_status": request.outcome.status,
        "parse_id": request.parse_id.as_str(),
        "parsed_record_id": (
            parsed_record_id.as_str() if parsed_record_id is not None else None
        ),
    }


def _failure_log_fields(error: Exception) -> dict[str, object]:
    if isinstance(error, ParsePreflightError):
        return {
            "error_category": error.category.value,
            "retryable": False,
        }
    if isinstance(error, StorageError):
        return {
            "error_category": error.category.value,
            "retryable": error.retryable,
        }
    return {"error_category": "unexpected_error", "retryable": False}


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


def _persist_parse_request(
    repository: ParsePublicationRepository,
    request: ParsePersistenceRequest,
    *,
    max_source_bytes: int,
) -> tuple[
    ParsePersistenceResult, int, StorageParseOutcome, ParsedRepresentation | None
]:
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
    result = ParsePersistenceResult(
        parse_id=request.parse_id,
        parsed_record_id=request.parsed_record_id,
        status=request.outcome.status,
    )
    return (
        result,
        preflight.source_byte_count,
        mapped.outcome,
        mapped.representation,
    )


def persist_parse_request(
    repository: ParsePublicationRepository,
    request: ParsePersistenceRequest,
    *,
    settings: Settings,
) -> ParsePersistenceResult:
    """Validate, map, and publish one observable immutable parse bundle."""
    _validate_repository(repository)
    if not isinstance(request, ParsePersistenceRequest):
        raise TypeError("request must be a ParsePersistenceRequest.")
    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings.")

    log_fields = _request_log_fields(request)
    with bound_contextvars(
        corpus_id=request.corpus_id.as_str(),
        document_id=request.parser_input.document_id.as_str(),
        corpus_version=None,
    ):
        _emit_event("info", "parse_persistence_started", **log_fields)
        try:
            result, source_byte_count, outcome, representation = _persist_parse_request(
                repository,
                request,
                max_source_bytes=settings.ingest.max_source_bytes,
            )
        except Exception as error:
            _emit_event(
                "error",
                "parse_persistence_failed",
                **log_fields,
                **_failure_log_fields(error),
            )
            raise

        _emit_event(
            "info",
            "parse_persistence_completed",
            **log_fields,
            source_byte_count=source_byte_count,
            record_count=2 + int(representation is not None),
            block_count=len(representation.blocks) if representation is not None else 0,
            span_count=(
                len(representation.source_locations)
                if representation is not None
                else 0
            ),
            failure_count=len(outcome.failures),
        )
        return result
