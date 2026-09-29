"""Execute one resumable, corpus-scoped baseline parse stage."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab.ingest.baseline_checkpoints import BaselineWorkKey
from inference_lab.ingest.baseline_contracts import (
    BaselineStage,
    BaselineStageDisposition,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.ingest.discovery import DiscoveredSource
from inference_lab.ingest.discovery_failures import DiscoveryFailure
from inference_lab.ingest.hashing import StableReadFileSystem, read_verified_source
from inference_lab.ingest.parse_persistence import ParsePersistenceRequest
from inference_lab.ingest.parse_publication import (
    ParsePublicationRepository,
    persist_parse_request,
)
from inference_lab.ingest.parse_reconstruction import (
    ParseReconstructionError,
    ReconstructedParseOutcome,
    reconstruct_parse_outcome,
)
from inference_lab.ingest.parser import (
    ParseOutcome,
    Parser,
    ParserInput,
    TextNativeFormat,
)
from inference_lab.ingest.parser_selection import (
    CURRENT_PARSER_REGISTRY,
    ParserRegistry,
    ParserSelectionError,
    select_parser,
)
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    RecordId,
)
from inference_lab.settings import Settings
from inference_lab.storage import ParsedRepresentation
from inference_lab.storage import ParseOutcome as StorageParseOutcome

_LOGGER = structlog.get_logger(__name__)


class BaselineParseErrorCategory(StrEnum):
    """Bounded reasons why a baseline parse stage cannot complete."""

    SOURCE_ACQUISITION_FAILED = "source_acquisition_failed"
    PARSER_UNAVAILABLE = "parser_unavailable"
    PARSE_FAILED = "parse_failed"
    PARSE_PERSISTENCE_FAILED = "parse_persistence_failed"
    CORRUPT_STATE = "corrupt_state"


class BaselineParseExecutionError(RuntimeError):
    """Represent a payload-free, document-scoped parse-stage failure."""

    def __init__(self, category: BaselineParseErrorCategory) -> None:
        """Create an error containing only its bounded category."""
        if not isinstance(category, BaselineParseErrorCategory):
            raise TypeError("category must be a BaselineParseErrorCategory.")
        super().__init__(category.value)
        self.category = category


class BaselineParseRepository(ParsePublicationRepository, Protocol):
    """Provide parse publication plus exact corpus-scoped outcome reads."""

    def get_parse_outcome(
        self,
        corpus_id: CorpusId,
        parse_id: RecordId,
    ) -> StorageParseOutcome | None:
        """Return one corpus-visible persisted parser outcome, if present."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineParseRequest:
    """Carry one discovered source through the resumable parse stage."""

    work_key: BaselineWorkKey
    source: DiscoveredSource
    parser_version: ParserVersion
    parse_id: RecordId
    parsed_record_id: RecordId | None
    supersedes_record_id: RecordId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate stage identity, source identity, and stable record IDs."""
        if not isinstance(self.work_key, BaselineWorkKey):
            raise TypeError("work_key must be a BaselineWorkKey.")
        if self.work_key.stage is not BaselineStage.PARSE:
            raise ValueError("work_key must identify the parse stage.")
        if not isinstance(self.source, DiscoveredSource):
            raise TypeError("source must be a DiscoveredSource.")
        version = self.source.identity_outcome.version
        if not isinstance(version, DocumentVersion):
            raise ValueError("source identity outcome must assign a document version.")
        document = self.source.identity_outcome.document
        if not isinstance(document, DocumentIdentity):
            raise ValueError("source identity outcome must assign a document.")
        if document.document_id != version.document_id:
            raise ValueError("source identity document and version must agree.")
        if version.document_version_id != self.work_key.document_version_id:
            raise ValueError("work key and source must identify the same version.")
        if version.content_hash != self.source.snapshot.content_hash:
            raise ValueError("source snapshot and identity version must agree.")
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.parse_id, RecordId):
            raise TypeError("parse_id must be a RecordId.")
        if self.parsed_record_id is not None and not isinstance(
            self.parsed_record_id,
            RecordId,
        ):
            raise TypeError("parsed_record_id must be a RecordId or None.")
        if self.supersedes_record_id is not None and not isinstance(
            self.supersedes_record_id,
            RecordId,
        ):
            raise TypeError("supersedes_record_id must be a RecordId or None.")
        ids = tuple(
            value
            for value in (
                self.parse_id,
                self.parsed_record_id,
                self.supersedes_record_id,
            )
            if value is not None
        )
        if len(ids) != len(set(ids)):
            raise ValueError("parse record identities must be distinct.")

    @property
    def document_version(self) -> DocumentVersion:
        """Return the identity-assigned document version after validation."""
        version = self.source.identity_outcome.version
        assert version is not None
        return version


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineParseResult:
    """Report a reconstructed parse outcome and whether it was reused."""

    corpus_id: CorpusId
    work_key: BaselineWorkKey
    parser_version: ParserVersion
    parse_id: RecordId
    parsed_record_id: RecordId | None
    disposition: BaselineStageDisposition
    reconstructed: ReconstructedParseOutcome

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate replay identity and represented-outcome invariants."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.work_key, BaselineWorkKey):
            raise TypeError("work_key must be a BaselineWorkKey.")
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.parse_id, RecordId):
            raise TypeError("parse_id must be a RecordId.")
        if self.parsed_record_id is not None and not isinstance(
            self.parsed_record_id,
            RecordId,
        ):
            raise TypeError("parsed_record_id must be a RecordId or None.")
        if self.disposition not in (
            BaselineStageDisposition.REUSED,
            BaselineStageDisposition.COMPLETED,
        ):
            raise ValueError("parse result must be reused or completed.")
        if not isinstance(self.reconstructed, ReconstructedParseOutcome):
            raise TypeError("reconstructed must be a ReconstructedParseOutcome.")
        if self.reconstructed.parser_version != self.parser_version:
            raise ValueError("reconstructed parser version must match the request.")
        document = self.reconstructed.outcome.document
        if document is None:
            if self.parsed_record_id is not None:
                raise ValueError("failed outcomes must not name a parsed record.")
        elif self.parsed_record_id is None:
            raise ValueError("represented outcomes require a parsed record.")


def _validate_repository(repository: object) -> None:
    """Validate the narrow repository boundary before any source work."""
    required_methods = (
        "get_parse_outcome",
        "get_corpus_document_version_membership",
        "get_document_version",
        "get_source_artefact",
        "get_parsed_representation",
        "persist_source_artefact",
    )
    if any(not callable(getattr(repository, name, None)) for name in required_methods):
        raise TypeError(
            "repository must provide baseline parse persistence operations."
        )


def _assigned_document_identity(
    request: BaselineParseRequest,
) -> tuple[DocumentId, DocumentVersionId]:
    """Return the assigned document and version without exposing source paths."""
    document = request.source.identity_outcome.document
    assert isinstance(document, DocumentIdentity)
    version = request.document_version
    return document.document_id, version.document_version_id


def _result_from_reconstructed(
    request: BaselineParseRequest,
    reconstructed: ReconstructedParseOutcome,
    *,
    disposition: BaselineStageDisposition,
) -> BaselineParseResult:
    """Build a public result after the replay boundary has validated storage."""
    document = reconstructed.outcome.document
    if document is not None:
        if document.document_id != request.document_version.document_id:
            raise BaselineParseExecutionError(BaselineParseErrorCategory.CORRUPT_STATE)
        if document.document_version_id != request.document_version.document_version_id:
            raise BaselineParseExecutionError(BaselineParseErrorCategory.CORRUPT_STATE)
        parsed_record_id = request.parsed_record_id
    else:
        parsed_record_id = None
    return BaselineParseResult(
        corpus_id=request.source.corpus_id,
        work_key=request.work_key,
        parser_version=request.parser_version,
        parse_id=request.parse_id,
        parsed_record_id=parsed_record_id,
        disposition=disposition,
        reconstructed=reconstructed,
    )


def _load_reconstructed(
    repository: BaselineParseRepository,
    request: BaselineParseRequest,
    *,
    require_existing: bool,
) -> ReconstructedParseOutcome | None:
    """Read and reconstruct the complete persisted parser aggregate."""
    try:
        storage_outcome = repository.get_parse_outcome(
            request.source.corpus_id,
            request.parse_id,
        )
    except Exception as error:
        raise BaselineParseExecutionError(
            BaselineParseErrorCategory.PARSE_PERSISTENCE_FAILED
        ) from error
    if storage_outcome is None:
        if require_existing:
            raise BaselineParseExecutionError(BaselineParseErrorCategory.CORRUPT_STATE)
        return None
    if not isinstance(storage_outcome, StorageParseOutcome):
        raise BaselineParseExecutionError(BaselineParseErrorCategory.CORRUPT_STATE)
    if (
        storage_outcome.parse_id != request.parse_id
        or storage_outcome.document_id != request.document_version.document_id
        or storage_outcome.document_version_id
        != request.document_version.document_version_id
        or (
            request.parsed_record_id is None
            and storage_outcome.parsed_record_id is not None
        )
        or (
            request.parsed_record_id is not None
            and storage_outcome.parsed_record_id != request.parsed_record_id
        )
    ):
        raise BaselineParseExecutionError(BaselineParseErrorCategory.CORRUPT_STATE)

    representation: ParsedRepresentation | None = None
    if storage_outcome.parsed_record_id is not None:
        try:
            representation = repository.get_parsed_representation(
                request.source.corpus_id,
                storage_outcome.parsed_record_id,
            )
        except Exception as error:
            raise BaselineParseExecutionError(
                BaselineParseErrorCategory.PARSE_PERSISTENCE_FAILED
            ) from error
    try:
        reconstructed = reconstruct_parse_outcome(storage_outcome, representation)
    except ParseReconstructionError as error:
        raise BaselineParseExecutionError(
            BaselineParseErrorCategory.CORRUPT_STATE
        ) from error
    if reconstructed.parser_version != request.parser_version:
        raise BaselineParseExecutionError(BaselineParseErrorCategory.CORRUPT_STATE)
    return reconstructed


def _build_parser_input(
    request: BaselineParseRequest,
    content: bytes,
) -> ParserInput:
    """Build parser input only after source bytes pass stable verification."""
    encoding = request.source.snapshot.encoding
    media_type = request.source.snapshot.media_type
    if media_type == "text/plain" and encoding is None:
        raise BaselineParseExecutionError(BaselineParseErrorCategory.PARSE_FAILED)
    try:
        return ParserInput(
            document_id=request.document_version.document_id,
            document_version_id=request.document_version.document_version_id,
            content=content,
            media_type=cast(TextNativeFormat, media_type),
            encoding=encoding,
        )
    except (TypeError, ValueError) as error:
        raise BaselineParseExecutionError(
            BaselineParseErrorCategory.PARSE_FAILED
        ) from error


def _run_parser(
    parser: Parser,
    parser_input: ParserInput,
    *,
    settings: Settings,
) -> ParseOutcome:
    """Run one parser and convert malformed adapter behaviour to a safe failure."""
    try:
        outcome = parser.parse(parser_input, settings=settings)
    except Exception as error:
        raise BaselineParseExecutionError(
            BaselineParseErrorCategory.PARSE_FAILED
        ) from error
    if not isinstance(outcome, ParseOutcome):
        raise BaselineParseExecutionError(BaselineParseErrorCategory.PARSE_FAILED)
    return outcome


def _publish_parse(
    repository: BaselineParseRepository,
    request: BaselineParseRequest,
    *,
    settings: Settings,
    parser_registry: ParserRegistry,
    filesystem: StableReadFileSystem | None,
    profiling: BaselineProfilingContext,
    document_id: DocumentId,
    source_stage_id: str,
    parse_stage_id: str,
    parse_persistence_stage_id: str,
    parent_stage_id: str | None,
) -> BaselineParseResult:
    """Acquire, parse, publish, and replay one new parse attempt."""
    content: bytes | None = None
    with profiling.observed_stage(
        stage="source_acquisition",
        stage_id=source_stage_id,
        document_id=document_id.as_str(),
        corpus_id=request.source.corpus_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
    ) as source_stage:
        source_stage.set_counts(items_processed=1, records_emitted=0)
        content_result = read_verified_source(
            request.source.snapshot,
            ordinal=request.source.ordinal,
            max_source_bytes=settings.ingest.max_source_bytes,
            filesystem=filesystem,
        )
        if isinstance(content_result, DiscoveryFailure):
            raise BaselineParseExecutionError(
                BaselineParseErrorCategory.SOURCE_ACQUISITION_FAILED
            )
        content = content_result
        source_stage.set_counts(items_processed=1, records_emitted=1)

    assert content is not None
    parser_input: ParserInput | None = None
    outcome: ParseOutcome | None = None
    with profiling.observed_stage(
        stage="parse",
        stage_id=parse_stage_id,
        document_id=document_id.as_str(),
        corpus_id=request.source.corpus_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
    ) as parse_stage:
        parse_stage.set_counts(items_processed=1, records_emitted=0)
        try:
            parser = select_parser(
                request.source.classification,
                parser_version=request.parser_version,
                registry=parser_registry,
            )
        except ParserSelectionError as error:
            raise BaselineParseExecutionError(
                BaselineParseErrorCategory.PARSER_UNAVAILABLE
            ) from error

        parser_input = _build_parser_input(request, content)
        outcome = _run_parser(parser, parser_input, settings=settings)
        parse_stage.set_counts(items_processed=1, records_emitted=1)

    assert parser_input is not None
    assert outcome is not None
    reconstructed: ReconstructedParseOutcome | None = None
    with profiling.observed_stage(
        stage="parse_persistence",
        stage_id=parse_persistence_stage_id,
        document_id=document_id.as_str(),
        corpus_id=request.source.corpus_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        parent_stage_id=parse_stage_id,
    ) as persistence_stage:
        persistence_stage.set_counts(items_processed=1, records_emitted=0)
        if outcome.document is not None and request.parsed_record_id is None:
            raise BaselineParseExecutionError(
                BaselineParseErrorCategory.PARSE_PERSISTENCE_FAILED
            )
        try:
            persistence_request = ParsePersistenceRequest(
                corpus_id=request.source.corpus_id,
                parser_input=parser_input,
                parser_version=request.parser_version,
                outcome=outcome,
                parse_id=request.parse_id,
                parsed_record_id=request.parsed_record_id,
                supersedes_record_id=request.supersedes_record_id,
            )
            persist_parse_request(
                repository,
                persistence_request,
                settings=settings,
                emit_lifecycle_events=False,
            )
        except Exception as error:
            raise BaselineParseExecutionError(
                BaselineParseErrorCategory.PARSE_PERSISTENCE_FAILED
            ) from error

        reconstructed = _load_reconstructed(
            repository,
            request,
            require_existing=True,
        )
        assert reconstructed is not None
        persistence_stage.set_counts(items_processed=1, records_emitted=1)
    assert reconstructed is not None
    return _result_from_reconstructed(
        request,
        reconstructed,
        disposition=BaselineStageDisposition.COMPLETED,
    )


def _log_fields(request: BaselineParseRequest) -> dict[str, object]:
    """Return bounded identifiers and parser metadata for stage events."""
    return {
        "parser_implementation": request.parser_version.implementation,
        "parser_contract_version": request.parser_version.contract_version,
        "normalisation_version": request.parser_version.normalisation_version,
        "parse_id": request.parse_id.as_str(),
    }


def _emit_event(method: str, event: str, **fields: object) -> None:
    """Keep observability best-effort without replacing stage outcomes."""
    try:
        getattr(_LOGGER, method)(event, pipeline_stage="parse", **fields)
    except Exception:
        pass


def execute_baseline_parse(
    repository: BaselineParseRepository,
    request: BaselineParseRequest,
    *,
    settings: Settings,
    parser_registry: ParserRegistry = CURRENT_PARSER_REGISTRY,
    filesystem: StableReadFileSystem | None = None,
    profiling: BaselineProfilingContext | None = None,
    parent_stage_id: str | None = None,
) -> BaselineParseResult:
    """Reuse or complete one stable, persist-and-reconstruct parse attempt."""
    _validate_repository(repository)
    if not isinstance(request, BaselineParseRequest):
        raise TypeError("request must be a BaselineParseRequest.")
    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings.")
    if not isinstance(parser_registry, ParserRegistry):
        raise TypeError("parser_registry must be a ParserRegistry.")
    if profiling is not None and not isinstance(profiling, BaselineProfilingContext):
        raise TypeError("profiling must be a BaselineProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    selected_profiling = (
        profiling if profiling is not None else BaselineProfilingContext.create()
    )
    document_id, _ = _assigned_document_identity(request)
    ordinal = str(request.source.ordinal)
    parse_stage_id = selected_profiling.stage_id("parse", ordinal)
    source_stage_id = selected_profiling.stage_id("source_acquisition", ordinal)
    parse_persistence_preflight_stage_id = selected_profiling.stage_id(
        "parse_persistence", f"{ordinal}:preflight"
    )
    parse_persistence_publish_stage_id = selected_profiling.stage_id(
        "parse_persistence", f"{ordinal}:publish"
    )
    context = bound_contextvars(
        corpus_id=request.source.corpus_id.as_str(),
        document_id=document_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        pipeline_stage="parse",
        run_id=selected_profiling.run_id,
        request_id=selected_profiling.request_id,
    )
    with context:
        fields = _log_fields(request)
        with selected_profiling.observed_stage(
            stage="parse_persistence",
            stage_id=parse_persistence_preflight_stage_id,
            document_id=document_id.as_str(),
            corpus_id=request.source.corpus_id.as_str(),
            corpus_version=request.work_key.proposed_corpus_version.as_str(),
            parent_stage_id=parent_stage_id,
        ) as persistence_stage:
            persistence_stage.set_counts(items_processed=0, records_emitted=0)
            reconstructed = _load_reconstructed(
                repository,
                request,
                require_existing=False,
            )
            if reconstructed is not None:
                persistence_stage.set_counts(items_processed=1, records_emitted=1)
                result = _result_from_reconstructed(
                    request,
                    reconstructed,
                    disposition=BaselineStageDisposition.REUSED,
                )
                _emit_event(
                    "info",
                    "baseline_parse_reused",
                    **fields,
                    run_id=selected_profiling.run_id,
                    stage_id=parse_persistence_preflight_stage_id,
                    parse_status=result.reconstructed.outcome.status,
                )
                return result

        return _publish_parse(
            repository,
            request,
            settings=settings,
            parser_registry=parser_registry,
            filesystem=filesystem,
            profiling=selected_profiling,
            document_id=document_id,
            source_stage_id=source_stage_id,
            parse_stage_id=parse_stage_id,
            parse_persistence_stage_id=parse_persistence_publish_stage_id,
            parent_stage_id=parent_stage_id,
        )
