"""Document execution and checkpoint orchestration for serving ingest."""

from __future__ import annotations

from collections.abc import Callable
from time import sleep
from typing import TypeVar

from inference_lab.ingest import (
    BaselineChunkRequest,
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineDocumentTask,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineParseRequest,
    BaselineRequestExecutionError,
    BaselineStage,
    BaselineStageDisposition,
    BaselineWorkKey,
    DiscoveredSource,
    DiscoveryFailure,
    execute_baseline_chunk,
    execute_baseline_documents,
    execute_baseline_parse,
)
from inference_lab.ingest.baseline_checkpoint_mapping import (
    baseline_checkpoint_from_record,
    baseline_checkpoint_to_record,
)
from inference_lab.ingest.baseline_checkpoints import (
    BaselineCheckpoint,
    BaselineCheckpointStatus,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
)
from inference_lab.serving.ingest_contracts import (
    ServingIngestError,
    ServingIngestFailureCategory,
    ServingIngestRequest,
    _ServingRepository,
)
from inference_lab.serving.ingest_helpers import (
    discovery_failure_category,
    failure,
    parse_failure_category,
    record_id,
    resolution_kind,
)
from inference_lab.settings import Settings
from inference_lab.storage import (
    BaselineCheckpointRecordStage,
    BaselineCheckpointRecordStatus,
    StorageConcurrencyError,
)

_Result = TypeVar("_Result")


def _is_storage_contention(error: BaseException) -> bool:
    """Recognise typed SQLite lock expiry through bounded stage wrappers."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, StorageConcurrencyError):
            return True
        current = current.__cause__
    return False


def _retry_storage_contention(operation: Callable[[], _Result]) -> _Result:
    """Retry complete idempotent stage attempts after transient lock expiry."""
    retries = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if not _is_storage_contention(error) or retries >= 3:
                raise
            sleep(0.01 * (2**retries))
            retries += 1


def _request_contention_failure(
    stage: BaselineStage,
    category: BaselineFailureCategory,
) -> BaselineRequestExecutionError:
    """Keep exhausted contention retryable without sealing the work key."""
    return BaselineRequestExecutionError(
        BaselineFailure(
            stage=stage,
            category=category,
            scope=BaselineFailureScope.REQUEST,
            severity=BaselineFailureSeverity.FATAL,
            retryable=True,
        )
    )


def _retry_repository_operation(
    operation: Callable[[], _Result],
    *,
    stage: BaselineStage,
) -> _Result:
    """Keep exhausted repository contention a retryable request failure."""
    try:
        return _retry_storage_contention(operation)
    except Exception as error:
        if _is_storage_contention(error):
            raise _request_contention_failure(
                stage,
                BaselineFailureCategory.STORAGE_UNAVAILABLE,
            ) from None
        raise


def _checkpoint_persistence_stage(stage: BaselineStage) -> BaselineStage:
    """Map the two durable document work keys to their storage stage."""
    if stage is BaselineStage.PARSE:
        return BaselineStage.PARSE_PERSISTENCE
    if stage is BaselineStage.CHUNK:
        return BaselineStage.CHUNK_PERSISTENCE
    raise ValueError("checkpoint stage must be parse or chunk.")


class ServingDocumentRunner:
    """Run document-level baseline work against one serving repository."""

    def __init__(self, repository: _ServingRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def execute(
        self,
        request: ServingIngestRequest,
        discovery: DiscoveryResult,
        profiling: BaselineProfilingContext,
        parent_memberships: tuple[CorpusDocumentVersionMembership, ...],
        *,
        parent_stage_id: str | None = None,
    ) -> tuple[BaselineDocumentOutcome, ...]:
        """Create and execute bounded document tasks for one discovery result."""
        tasks = self.tasks(
            request,
            discovery,
            profiling,
            parent_memberships,
            parent_stage_id=parent_stage_id,
        )
        return execute_baseline_documents(
            tasks,
            settings=self._settings,
            corpus_id=request.corpus_id,
            proposed_corpus_version=request.proposed_corpus_version,
            profiling=profiling,
            parent_stage_id=parent_stage_id,
        )

    def tasks(
        self,
        request: ServingIngestRequest,
        discovery: DiscoveryResult,
        profiling: BaselineProfilingContext,
        parent_memberships: tuple[CorpusDocumentVersionMembership, ...],
        *,
        parent_stage_id: str | None = None,
    ) -> tuple[BaselineDocumentTask, ...]:
        """Build stable document tasks for sources and discovery failures."""
        by_ordinal = {source.ordinal: source for source in discovery.sources}
        failures_by_ordinal = {
            failure.ordinal: failure for failure in discovery.failures
        }
        ordinals = tuple(sorted((*by_ordinal, *failures_by_ordinal)))
        tasks: list[BaselineDocumentTask] = []
        for ordinal in ordinals:
            source = by_ordinal.get(ordinal)
            failure_value = failures_by_ordinal.get(ordinal)
            document_id = (
                source.identity_outcome.document.document_id
                if source is not None and source.identity_outcome.document is not None
                else None
            )

            def worker(
                cancellation: object,
                *,
                ordinal: int = ordinal,
                source: DiscoveredSource | None = source,
                failure_value: DiscoveryFailure | None = failure_value,
            ) -> BaselineDocumentOutcome:
                del cancellation
                if isinstance(failure_value, DiscoveryFailure):
                    return BaselineDocumentOutcome(
                        ordinal=ordinal,
                        kind=BaselineDocumentOutcomeKind.FAILED,
                        failures=(
                            failure(
                                stage=BaselineStage.DISCOVERY,
                                category=discovery_failure_category(failure_value),
                                scope=BaselineFailureScope.DOCUMENT,
                                severity=BaselineFailureSeverity.RECOVERABLE,
                            ),
                        ),
                    )
                assert source is not None
                if self._settings.ingest.max_concurrent_documents > 1:
                    return self.process_source_on_worker(
                        request,
                        source,
                        profiling,
                        parent_memberships,
                        parent_stage_id=parent_stage_id,
                    )
                return self.process_source(
                    request,
                    source,
                    profiling,
                    parent_memberships,
                    parent_stage_id=parent_stage_id,
                )

            tasks.append(
                BaselineDocumentTask(
                    ordinal=ordinal,
                    worker=worker,
                    document_id=document_id,
                )
            )
        return tuple(tasks)

    def process_source_on_worker(
        self,
        request: ServingIngestRequest,
        source: DiscoveredSource,
        profiling: BaselineProfilingContext,
        parent_memberships: tuple[CorpusDocumentVersionMembership, ...],
        *,
        parent_stage_id: str | None = None,
    ) -> BaselineDocumentOutcome:
        """Own a validated repository only for this document worker's lifetime."""
        repository = _retry_repository_operation(
            self._repository.open_worker_repository,
            stage=BaselineStage.PARSE_PERSISTENCE,
        )
        runner = ServingDocumentRunner(repository, self._settings)
        try:
            outcome = runner.process_source(
                request,
                source,
                profiling,
                parent_memberships,
                parent_stage_id=parent_stage_id,
            )
        except BaseException:
            try:
                repository.close()
            except Exception:
                pass
            raise
        repository.close()
        return outcome

    def process_source(
        self,
        request: ServingIngestRequest,
        source: DiscoveredSource,
        profiling: BaselineProfilingContext,
        parent_memberships: tuple[CorpusDocumentVersionMembership, ...],
        *,
        parent_stage_id: str | None = None,
    ) -> BaselineDocumentOutcome:
        """Parse, persist, chunk, and checkpoint one discovered source."""
        resolved = source.identity_outcome
        kind = resolution_kind(resolved.kind)
        if kind in {
            BaselineDocumentOutcomeKind.AMBIGUOUS,
            BaselineDocumentOutcomeKind.CONFLICTING,
        }:
            return BaselineDocumentOutcome(ordinal=source.ordinal, kind=kind)
        document = resolved.document
        version = resolved.version
        assert isinstance(document, DocumentIdentity)
        assert isinstance(version, DocumentVersion)
        if kind is BaselineDocumentOutcomeKind.UNCHANGED_REUSED:
            parent = next(
                (
                    membership
                    for membership in parent_memberships
                    if membership.document_id == document.document_id
                ),
                None,
            )
            if parent is None:
                kind = BaselineDocumentOutcomeKind.NEW_DOCUMENT
            elif parent.document_version_id != version.document_version_id:
                kind = BaselineDocumentOutcomeKind.CHANGED_VERSION
        parse_id = record_id(
            "parse",
            version.document_version_id.as_str(),
            request.parser_version.value,
        )
        parsed_record_id = record_id(
            "parsed-representation",
            version.document_version_id.as_str(),
            request.parser_version.value,
        )
        parse_key = BaselineWorkKey(
            proposed_corpus_version=request.proposed_corpus_version,
            document_version_id=version.document_version_id,
            stage=BaselineStage.PARSE,
            configuration_identity=request.configuration_identity,
        )
        parse_checkpoint = self.start_checkpoint(
            request,
            BaselineCheckpoint(
                work_key=parse_key,
                document_id=document.document_id,
                input_identities=(version.document_version_id,),
                output_identities=(),
                status=BaselineCheckpointStatus.STARTED,
            ),
        )
        if parse_checkpoint.status is BaselineCheckpointStatus.FAILED:
            assert parse_checkpoint.failure_category is not None
            return self.failed_outcome(
                source,
                document.document_id,
                version.document_version_id,
                stage=BaselineStage.PARSE,
                category=parse_checkpoint.failure_category,
            )
        if (
            parse_checkpoint.status is BaselineCheckpointStatus.COMPLETED
            and _retry_repository_operation(
                lambda: self._repository.get_parse_outcome(request.corpus_id, parse_id),
                stage=BaselineStage.PARSE_PERSISTENCE,
            )
            is None
        ):
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)
        try:
            parsed = _retry_storage_contention(
                lambda: execute_baseline_parse(
                    self._repository,
                    BaselineParseRequest(
                        work_key=parse_key,
                        source=source,
                        parser_version=request.parser_version,
                        parse_id=parse_id,
                        parsed_record_id=parsed_record_id,
                    ),
                    settings=self._settings,
                    parser_registry=request.parser_registry,
                    filesystem=request.source_filesystem,
                    profiling=profiling,
                    parent_stage_id=parent_stage_id,
                )
            )
        except Exception as error:
            if _is_storage_contention(error):
                raise _request_contention_failure(
                    BaselineStage.PARSE_PERSISTENCE,
                    BaselineFailureCategory.PARSE_PERSISTENCE_FAILED,
                ) from None
            if parse_checkpoint.status is BaselineCheckpointStatus.COMPLETED:
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                ) from None
            category = parse_failure_category(getattr(error, "category", None))
            self.write_checkpoint(
                request,
                BaselineCheckpoint(
                    work_key=parse_key,
                    document_id=document.document_id,
                    input_identities=(version.document_version_id,),
                    output_identities=(),
                    status=BaselineCheckpointStatus.FAILED,
                    failure_category=category,
                ),
                expected_status=BaselineCheckpointStatus.STARTED,
            )
            return self.failed_outcome(
                source,
                document.document_id,
                version.document_version_id,
                stage=BaselineStage.PARSE,
                category=category,
            )
        parse_outputs = (parse_id,)
        if parsed.parsed_record_id is not None:
            parse_outputs = (parse_id, parsed.parsed_record_id)
        completed_parse = BaselineCheckpoint(
            work_key=parse_key,
            document_id=document.document_id,
            input_identities=(version.document_version_id,),
            output_identities=parse_outputs,
            status=BaselineCheckpointStatus.COMPLETED,
        )
        if parse_checkpoint.status is BaselineCheckpointStatus.COMPLETED:
            if (
                parsed.disposition is not BaselineStageDisposition.REUSED
                or parse_checkpoint != completed_parse
            ):
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )
        else:
            self.write_checkpoint(
                request,
                completed_parse,
                expected_status=BaselineCheckpointStatus.STARTED,
            )
        if parsed.parsed_record_id is None:
            return self.failed_outcome(
                source,
                document.document_id,
                version.document_version_id,
                stage=BaselineStage.PARSE,
                category=BaselineFailureCategory.PARSE_FAILED,
            )

        chunk_key = BaselineWorkKey(
            proposed_corpus_version=request.proposed_corpus_version,
            document_version_id=version.document_version_id,
            stage=BaselineStage.CHUNK,
            configuration_identity=request.configuration_identity,
        )
        chunk_checkpoint = self.start_checkpoint(
            request,
            BaselineCheckpoint(
                work_key=chunk_key,
                document_id=document.document_id,
                input_identities=(parsed_record_id,),
                output_identities=(),
                status=BaselineCheckpointStatus.STARTED,
            ),
        )
        if chunk_checkpoint.status is BaselineCheckpointStatus.FAILED:
            assert chunk_checkpoint.failure_category is not None
            return self.failed_outcome(
                source,
                document.document_id,
                version.document_version_id,
                stage=BaselineStage.CHUNK,
                category=chunk_checkpoint.failure_category,
                parse_disposition=parsed.disposition,
            )
        if (
            chunk_checkpoint.status is BaselineCheckpointStatus.COMPLETED
            and _retry_repository_operation(
                lambda: self._repository.get_chunk_set_selection(
                    request.corpus_id,
                    version.document_version_id,
                    request.configuration_identity,
                ),
                stage=BaselineStage.CHUNK_PERSISTENCE,
            )
            is None
        ):
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)
        try:
            chunked = _retry_storage_contention(
                lambda: execute_baseline_chunk(
                    self._repository,
                    BaselineChunkRequest(
                        corpus_id=request.corpus_id,
                        work_key=chunk_key,
                        parse_result=parsed,
                        tokeniser=request.tokeniser,
                        policy=request.chunking_policy,
                        chunker_version=request.chunker_version,
                    ),
                    profiling=profiling,
                    parent_stage_id=parent_stage_id,
                )
            )
        except Exception as error:
            if _is_storage_contention(error):
                raise _request_contention_failure(
                    BaselineStage.CHUNK_PERSISTENCE,
                    BaselineFailureCategory.CHUNK_PERSISTENCE_FAILED,
                ) from None
            if chunk_checkpoint.status is BaselineCheckpointStatus.COMPLETED:
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                ) from None
            category = parse_failure_category(getattr(error, "category", None))
            self.write_checkpoint(
                request,
                BaselineCheckpoint(
                    work_key=chunk_key,
                    document_id=document.document_id,
                    input_identities=(parsed_record_id,),
                    output_identities=(),
                    status=BaselineCheckpointStatus.FAILED,
                    failure_category=category,
                ),
                expected_status=BaselineCheckpointStatus.STARTED,
            )
            return self.failed_outcome(
                source,
                document.document_id,
                version.document_version_id,
                stage=BaselineStage.CHUNK,
                category=category,
                parse_disposition=parsed.disposition,
            )
        chunk_outputs = tuple(chunk.record_id for chunk in chunked.chunk_set.chunks)
        completed_chunk = BaselineCheckpoint(
            work_key=chunk_key,
            document_id=document.document_id,
            input_identities=(parsed_record_id,),
            output_identities=chunk_outputs,
            status=BaselineCheckpointStatus.COMPLETED,
        )
        if chunk_checkpoint.status is BaselineCheckpointStatus.COMPLETED:
            if (
                chunked.disposition is not BaselineStageDisposition.REUSED
                or chunk_checkpoint != completed_chunk
            ):
                raise ServingIngestError(
                    ServingIngestFailureCategory.INCOMPATIBLE_RESUME
                )
        else:
            self.write_checkpoint(
                request,
                completed_chunk,
                expected_status=BaselineCheckpointStatus.STARTED,
            )
        return BaselineDocumentOutcome(
            ordinal=source.ordinal,
            kind=kind,
            document_id=document.document_id,
            document_version_id=version.document_version_id,
            parse_disposition=parsed.disposition,
            chunk_disposition=chunked.disposition,
        )

    def start_checkpoint(
        self,
        request: ServingIngestRequest,
        started: BaselineCheckpoint,
    ) -> BaselineCheckpoint:
        """Recover an exact stage attempt or create its first durable marker."""
        record = _retry_repository_operation(
            lambda: self._repository.get_baseline_checkpoint(
                request.corpus_id,
                request.proposed_corpus_version,
                started.work_key.document_version_id,
                BaselineCheckpointRecordStage(started.work_key.stage.value),
                request.configuration_identity,
            ),
            stage=_checkpoint_persistence_stage(started.work_key.stage),
        )
        if record is None:
            self.write_checkpoint(request, started)
            return started
        checkpoint = baseline_checkpoint_from_record(
            record,
            expected_corpus_id=request.corpus_id,
        )
        if (
            checkpoint.work_key != started.work_key
            or checkpoint.document_id != started.document_id
            or checkpoint.input_identities != started.input_identities
        ):
            raise ServingIngestError(ServingIngestFailureCategory.INCOMPATIBLE_RESUME)
        return checkpoint

    @staticmethod
    def failed_outcome(
        source: DiscoveredSource,
        document_id: DocumentId,
        document_version_id: DocumentVersionId,
        *,
        stage: BaselineStage,
        category: BaselineFailureCategory,
        parse_disposition: BaselineStageDisposition = BaselineStageDisposition.FAILED,
    ) -> BaselineDocumentOutcome:
        """Keep a failed checkpoint terminal within its proposed version."""
        return BaselineDocumentOutcome(
            ordinal=source.ordinal,
            kind=BaselineDocumentOutcomeKind.FAILED,
            document_id=document_id,
            document_version_id=document_version_id,
            parse_disposition=parse_disposition,
            chunk_disposition=(
                BaselineStageDisposition.FAILED
                if stage is BaselineStage.CHUNK
                else BaselineStageDisposition.NOT_ATTEMPTED
            ),
            failures=(
                failure(
                    stage=stage,
                    category=category,
                    scope=BaselineFailureScope.DOCUMENT,
                    severity=BaselineFailureSeverity.RECOVERABLE,
                ),
            ),
        )

    def write_checkpoint(
        self,
        request: ServingIngestRequest,
        checkpoint: BaselineCheckpoint,
        *,
        expected_status: BaselineCheckpointStatus | None = None,
    ) -> None:
        """Persist one exact checkpoint transition."""
        record = baseline_checkpoint_to_record(checkpoint, corpus_id=request.corpus_id)

        def write() -> None:
            with self._repository.transaction() as transaction:
                transaction.write_baseline_checkpoint(
                    record,
                    expected_status=(
                        None
                        if expected_status is None
                        else BaselineCheckpointRecordStatus(expected_status.value)
                    ),
                )

        _retry_repository_operation(
            write,
            stage=_checkpoint_persistence_stage(checkpoint.work_key.stage),
        )
