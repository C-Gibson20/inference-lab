"""Execute one resumable, corpus-scoped baseline chunk stage."""

from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Protocol, Self

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab.ingest._chunk_builder import build_chunking_outcome
from inference_lab.ingest._chunk_records import ChunkingFailureCategory, ChunkSet
from inference_lab.ingest.baseline_checkpoints import BaselineWorkKey
from inference_lab.ingest.baseline_contracts import (
    BaselineStage,
    BaselineStageDisposition,
)
from inference_lab.ingest.baseline_parse import BaselineParseResult
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.ingest.chunk_mapping import ChunkMappingError, map_chunk_set
from inference_lab.ingest.chunk_serialisation import (
    InvalidChunkSerialisationError,
    deserialise_chunk_set_selection,
)
from inference_lab.ingest.chunking import (
    CURRENT_CHUNKER_VERSION,
    ChunkerVersion,
    ChunkingPolicy,
    ChunkingRequest,
)
from inference_lab.ingest.tokenisation import (
    ChunkTokeniser,
    validate_chunk_tokeniser,
)
from inference_lab.provenance import CorpusId
from inference_lab.storage import (
    ChunkSetReader,
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    WriteChunkSetOutcome,
)

_LOGGER = structlog.get_logger(__name__)


class BaselineChunkErrorCategory(StrEnum):
    """Bounded reasons why a baseline chunk stage cannot complete."""

    CHUNK_FAILED = "chunk_failed"
    CHUNK_PERSISTENCE_FAILED = "chunk_persistence_failed"
    CORRUPT_STATE = "corrupt_state"


class BaselineChunkExecutionError(RuntimeError):
    """Represent a payload-free, document-scoped chunk-stage failure."""

    def __init__(
        self,
        category: BaselineChunkErrorCategory,
        *,
        chunking_category: ChunkingFailureCategory | None = None,
    ) -> None:
        """Create an error with bounded stage and optional engine categories."""
        if not isinstance(category, BaselineChunkErrorCategory):
            raise TypeError("category must be a BaselineChunkErrorCategory.")
        if chunking_category is not None and not isinstance(
            chunking_category,
            ChunkingFailureCategory,
        ):
            raise TypeError(
                "chunking_category must be a ChunkingFailureCategory or None."
            )
        super().__init__(category.value)
        self.category = category
        self.chunking_category = chunking_category


class BaselineChunkTransaction(Protocol):
    """Minimal atomic transaction required by the chunk-stage executor."""

    def __enter__(self) -> Self:
        """Start the transaction and return its write surface."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Commit on success or roll back after an exception."""
        ...

    def write_chunk_set(
        self,
        request: ChunkSetWriteRequest,
    ) -> WriteChunkSetOutcome:
        """Write one immutable set inside this transaction."""
        ...


class BaselineChunkRepository(ChunkSetReader, Protocol):
    """Provide reads plus an atomic transaction for immutable chunk writes."""

    def transaction(self) -> BaselineChunkTransaction:
        """Return a transaction whose successful exit publishes the write."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineChunkRequest:
    """Carry one represented parser result through the chunk stage."""

    corpus_id: CorpusId
    work_key: BaselineWorkKey
    parse_result: BaselineParseResult
    tokeniser: ChunkTokeniser
    policy: ChunkingPolicy
    chunker_version: ChunkerVersion = CURRENT_CHUNKER_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate source identity and prevent chunking failed parses."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.work_key, BaselineWorkKey):
            raise TypeError("work_key must be a BaselineWorkKey.")
        if self.work_key.stage is not BaselineStage.CHUNK:
            raise ValueError("work_key must identify the chunk stage.")
        if not isinstance(self.parse_result, BaselineParseResult):
            raise TypeError("parse_result must be a BaselineParseResult.")
        if self.parse_result.corpus_id != self.corpus_id:
            raise ValueError("corpus and parse result must identify the same corpus.")
        document = self.parse_result.reconstructed.outcome.document
        if document is None:
            raise ValueError("chunking requires a represented parse outcome.")
        if document.document_version_id != self.work_key.document_version_id:
            raise ValueError(
                "work key and parse result must identify the same version."
            )
        if not isinstance(self.policy, ChunkingPolicy):
            raise TypeError("policy must be a ChunkingPolicy.")
        if not isinstance(self.chunker_version, ChunkerVersion):
            raise TypeError("chunker_version must be a ChunkerVersion.")
        if self.chunker_version != CURRENT_CHUNKER_VERSION:
            raise ValueError("unsupported chunker version.")


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineChunkResult:
    """Report a complete, reconstructed chunk set and its disposition."""

    corpus_id: CorpusId
    work_key: BaselineWorkKey
    chunk_set: ChunkSet
    disposition: BaselineStageDisposition
    storage_replayed: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate exact chunk identity and stage disposition."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.work_key, BaselineWorkKey):
            raise TypeError("work_key must be a BaselineWorkKey.")
        if self.work_key.stage is not BaselineStage.CHUNK:
            raise ValueError("work key must identify the chunk stage.")
        if not isinstance(self.chunk_set, ChunkSet):
            raise TypeError("chunk_set must be a ChunkSet.")
        if self.chunk_set.corpus_id != self.corpus_id:
            raise ValueError("chunk set and result corpus must agree.")
        if self.chunk_set.document_version_id != self.work_key.document_version_id:
            raise ValueError("chunk set and work key must identify the same version.")
        if (
            self.chunk_set.configuration_identity
            != self.work_key.configuration_identity
        ):
            raise ValueError("chunk set and work key configuration must agree.")
        if self.disposition not in (
            BaselineStageDisposition.REUSED,
            BaselineStageDisposition.COMPLETED,
        ):
            raise ValueError("chunk result must be reused or completed.")
        if type(self.storage_replayed) is not bool:
            raise TypeError("storage_replayed must be a boolean.")


def _validate_repository(repository: object) -> None:
    """Validate the narrow chunk repository boundary before chunking."""
    if any(
        not callable(getattr(repository, name, None))
        for name in ("get_chunk_set_selection", "transaction")
    ):
        raise TypeError("repository must provide chunk-set reads and transactions.")


def _validated_tokeniser(request: BaselineChunkRequest) -> ChunkTokeniser:
    """Validate the tokeniser identity before reuse or construction."""
    try:
        return validate_chunk_tokeniser(request.tokeniser)
    except Exception as error:
        raise BaselineChunkExecutionError(
            BaselineChunkErrorCategory.CHUNK_FAILED,
        ) from error


def _validate_selection_metadata(
    selection: ChunkSetSelectionRecord,
    request: BaselineChunkRequest,
    tokeniser: ChunkTokeniser,
) -> ChunkSet:
    """Reconstruct and validate every immutable input that affects chunks."""
    if not isinstance(selection, ChunkSetSelectionRecord):
        raise BaselineChunkExecutionError(BaselineChunkErrorCategory.CORRUPT_STATE)
    if (
        selection.document_version_id != request.work_key.document_version_id
        or selection.configuration_identity != request.work_key.configuration_identity
    ):
        raise BaselineChunkExecutionError(BaselineChunkErrorCategory.CORRUPT_STATE)
    try:
        chunk_set = deserialise_chunk_set_selection(
            selection.canonical_payload,
            corpus_id=request.corpus_id,
        )
    except (InvalidChunkSerialisationError, TypeError, ValueError) as error:
        raise BaselineChunkExecutionError(
            BaselineChunkErrorCategory.CORRUPT_STATE
        ) from error
    document = request.parse_result.reconstructed.outcome.document
    assert document is not None
    if (
        chunk_set.document_id != document.document_id
        or chunk_set.document_version_id != document.document_version_id
        or chunk_set.parser_version != request.parse_result.parser_version
        or chunk_set.chunker_version != request.chunker_version
        or chunk_set.tokeniser_identity != tokeniser.identity
        or chunk_set.policy != request.policy
        or chunk_set.configuration_identity != request.work_key.configuration_identity
    ):
        raise BaselineChunkExecutionError(BaselineChunkErrorCategory.CORRUPT_STATE)
    return chunk_set


def _read_selection(
    repository: BaselineChunkRepository,
    request: BaselineChunkRequest,
    tokeniser: ChunkTokeniser,
) -> ChunkSet | None:
    """Read and reconstruct an exact visible selection, if available."""
    try:
        selection = repository.get_chunk_set_selection(
            request.corpus_id,
            request.work_key.document_version_id,
            request.work_key.configuration_identity,
        )
    except Exception as error:
        raise BaselineChunkExecutionError(
            BaselineChunkErrorCategory.CHUNK_PERSISTENCE_FAILED
        ) from error
    if selection is None:
        return None
    return _validate_selection_metadata(selection, request, tokeniser)


def _build_and_publish(
    repository: BaselineChunkRepository,
    request: BaselineChunkRequest,
    tokeniser: ChunkTokeniser,
    *,
    profiling: BaselineProfilingContext,
    chunk_stage_id: str,
    chunk_persistence_stage_id: str,
    parent_stage_id: str | None,
) -> BaselineChunkResult:
    """Build an all-or-nothing set, write it, and replay its public payload."""
    document = request.parse_result.reconstructed.outcome.document
    assert document is not None
    chunking_request = ChunkingRequest(
        corpus_id=request.corpus_id,
        parsed_document=document,
        tokeniser=tokeniser,
        policy=request.policy,
    )
    chunk_set: ChunkSet | None = None
    with profiling.observed_stage(
        stage="chunk",
        stage_id=chunk_stage_id,
        document_id=document.document_id.as_str(),
        corpus_id=request.corpus_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        parent_stage_id=parent_stage_id,
    ) as chunk_stage:
        chunk_stage.set_counts(items_processed=1, records_emitted=0)
        try:
            outcome = build_chunking_outcome(chunking_request)
        except Exception as error:
            raise BaselineChunkExecutionError(
                BaselineChunkErrorCategory.CHUNK_FAILED
            ) from error
        if not outcome.succeeded:
            assert outcome.failure is not None
            raise BaselineChunkExecutionError(
                BaselineChunkErrorCategory.CHUNK_FAILED,
                chunking_category=outcome.failure.category,
            )
        chunk_set = outcome.chunk_set
        assert chunk_set is not None
        if (
            chunk_set.chunker_version != request.chunker_version
            or chunk_set.tokeniser_identity != tokeniser.identity
            or chunk_set.policy != request.policy
            or chunk_set.configuration_identity
            != request.work_key.configuration_identity
        ):
            raise BaselineChunkExecutionError(BaselineChunkErrorCategory.CORRUPT_STATE)
        chunk_stage.set_counts(
            items_processed=1,
            records_emitted=len(chunk_set.chunks),
        )

    assert chunk_set is not None
    write_outcome: WriteChunkSetOutcome | None = None
    replayed: ChunkSet | None = None
    with profiling.observed_stage(
        stage="chunk_persistence",
        stage_id=chunk_persistence_stage_id,
        document_id=document.document_id.as_str(),
        corpus_id=request.corpus_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        parent_stage_id=chunk_stage_id,
    ) as persistence_stage:
        persistence_stage.set_counts(
            items_processed=len(chunk_set.chunks),
            records_emitted=0,
        )
        try:
            selection = map_chunk_set(chunk_set)
            with repository.transaction() as transaction:
                write_outcome = transaction.write_chunk_set(
                    ChunkSetWriteRequest(
                        corpus_id=request.corpus_id,
                        selection=selection,
                    )
                )
                if not isinstance(write_outcome, WriteChunkSetOutcome):
                    raise BaselineChunkExecutionError(
                        BaselineChunkErrorCategory.CORRUPT_STATE
                    )
                if write_outcome.selection != selection:
                    raise BaselineChunkExecutionError(
                        BaselineChunkErrorCategory.CORRUPT_STATE
                    )
        except BaselineChunkExecutionError:
            raise
        except (ChunkMappingError, TypeError, ValueError) as error:
            raise BaselineChunkExecutionError(
                BaselineChunkErrorCategory.CHUNK_PERSISTENCE_FAILED
            ) from error
        except Exception as error:
            raise BaselineChunkExecutionError(
                BaselineChunkErrorCategory.CHUNK_PERSISTENCE_FAILED
            ) from error
        if write_outcome is None:
            raise BaselineChunkExecutionError(BaselineChunkErrorCategory.CORRUPT_STATE)
        replayed = _read_selection(repository, request, tokeniser)
        if replayed is None:
            raise BaselineChunkExecutionError(BaselineChunkErrorCategory.CORRUPT_STATE)
        persistence_stage.set_counts(
            items_processed=len(chunk_set.chunks),
            records_emitted=len(replayed.chunks),
        )
    assert write_outcome is not None
    assert replayed is not None
    return BaselineChunkResult(
        corpus_id=request.corpus_id,
        work_key=request.work_key,
        chunk_set=replayed,
        disposition=BaselineStageDisposition.COMPLETED,
        storage_replayed=write_outcome.replayed,
    )


def _log_fields(
    request: BaselineChunkRequest, tokeniser: ChunkTokeniser
) -> dict[str, object]:
    """Return bounded chunk identities for structured lifecycle events."""
    document = request.parse_result.reconstructed.outcome.document
    assert document is not None
    return {
        "document_version_id": document.document_version_id.as_str(),
        "chunker_implementation": request.chunker_version.implementation,
        "chunker_contract_version": request.chunker_version.contract_version,
        "tokeniser_implementation": tokeniser.identity.implementation,
        "tokeniser_contract_version": tokeniser.identity.contract_version,
        "configuration_identity": request.work_key.configuration_identity.as_str(),
    }


def _emit_event(method: str, event: str, **fields: object) -> None:
    """Keep observability best-effort without changing stage outcomes."""
    try:
        getattr(_LOGGER, method)(event, pipeline_stage="chunk", **fields)
    except Exception:
        pass


def execute_baseline_chunk(
    repository: BaselineChunkRepository,
    request: BaselineChunkRequest,
    *,
    profiling: BaselineProfilingContext | None = None,
    parent_stage_id: str | None = None,
    profiling_identity: str | None = None,
) -> BaselineChunkResult:
    """Reuse or complete one deterministic, atomically persisted chunk set."""
    _validate_repository(repository)
    if not isinstance(request, BaselineChunkRequest):
        raise TypeError("request must be a BaselineChunkRequest.")
    if profiling is not None and not isinstance(profiling, BaselineProfilingContext):
        raise TypeError("profiling must be a BaselineProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")
    if profiling_identity is not None and not isinstance(profiling_identity, str):
        raise TypeError("profiling_identity must be a string or None.")
    tokeniser = _validated_tokeniser(request)
    document = request.parse_result.reconstructed.outcome.document
    assert document is not None
    selected_profiling = (
        profiling if profiling is not None else BaselineProfilingContext.create()
    )
    identity = (
        profiling_identity
        if profiling_identity is not None
        else document.document_version_id.as_str()
    )
    chunk_stage_id = selected_profiling.stage_id("chunk", identity)
    chunk_persistence_preflight_stage_id = selected_profiling.stage_id(
        "chunk_persistence", f"{identity}:preflight"
    )
    chunk_persistence_publish_stage_id = selected_profiling.stage_id(
        "chunk_persistence", f"{identity}:publish"
    )
    context = bound_contextvars(
        corpus_id=request.corpus_id.as_str(),
        document_id=document.document_id.as_str(),
        corpus_version=request.work_key.proposed_corpus_version.as_str(),
        pipeline_stage="chunk",
        run_id=selected_profiling.run_id,
        request_id=selected_profiling.request_id,
    )
    with context:
        fields = _log_fields(request, tokeniser)
        with selected_profiling.observed_stage(
            stage="chunk_persistence",
            stage_id=chunk_persistence_preflight_stage_id,
            document_id=document.document_id.as_str(),
            corpus_id=request.corpus_id.as_str(),
            corpus_version=request.work_key.proposed_corpus_version.as_str(),
            parent_stage_id=parent_stage_id,
        ) as persistence_stage:
            persistence_stage.set_counts(items_processed=0, records_emitted=0)
            existing = _read_selection(repository, request, tokeniser)
            if existing is not None:
                persistence_stage.set_counts(
                    items_processed=len(existing.chunks),
                    records_emitted=len(existing.chunks),
                )
                result = BaselineChunkResult(
                    corpus_id=request.corpus_id,
                    work_key=request.work_key,
                    chunk_set=existing,
                    disposition=BaselineStageDisposition.REUSED,
                )
                _emit_event(
                    "info",
                    "baseline_chunk_reused",
                    **fields,
                    run_id=selected_profiling.run_id,
                    stage_id=chunk_persistence_preflight_stage_id,
                    chunk_count=len(existing.chunks),
                )
                return result
        return _build_and_publish(
            repository,
            request,
            tokeniser,
            profiling=selected_profiling,
            chunk_stage_id=chunk_stage_id,
            chunk_persistence_stage_id=chunk_persistence_publish_stage_id,
            parent_stage_id=parent_stage_id,
        )
