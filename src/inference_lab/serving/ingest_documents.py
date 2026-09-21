"""Document execution and checkpoint orchestration for serving ingest."""

from __future__ import annotations

from inference_lab.ingest import (
    BaselineChunkRequest,
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineDocumentTask,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineParseRequest,
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
    baseline_checkpoint_to_record,
)
from inference_lab.ingest.baseline_checkpoints import (
    BaselineCheckpoint,
    BaselineCheckpointStatus,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.ingest.discovery_results import DiscoveryResult
from inference_lab.provenance import DocumentIdentity, DocumentVersion
from inference_lab.serving.ingest_contracts import (
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
from inference_lab.storage import BaselineCheckpointRecordStatus


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
    ) -> tuple[BaselineDocumentOutcome, ...]:
        """Create and execute bounded document tasks for one discovery result."""
        tasks = self.tasks(request, discovery, profiling)
        return execute_baseline_documents(
            tasks,
            settings=self._settings,
            corpus_id=request.corpus_id,
            proposed_corpus_version=request.proposed_corpus_version,
            profiling=profiling,
            parent_stage_id=None,
        )

    def tasks(
        self,
        request: ServingIngestRequest,
        discovery: DiscoveryResult,
        profiling: BaselineProfilingContext,
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
                return self.process_source(request, source, profiling)

            tasks.append(
                BaselineDocumentTask(
                    ordinal=ordinal,
                    worker=worker,
                    document_id=document_id,
                )
            )
        return tuple(tasks)

    def process_source(
        self,
        request: ServingIngestRequest,
        source: DiscoveredSource,
        profiling: BaselineProfilingContext,
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
        self.write_checkpoint(
            request,
            BaselineCheckpoint(
                work_key=parse_key,
                document_id=document.document_id,
                input_identities=(version.document_version_id,),
                output_identities=(),
                status=BaselineCheckpointStatus.STARTED,
            ),
        )
        try:
            parsed = execute_baseline_parse(
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
            )
            self.write_checkpoint(
                request,
                BaselineCheckpoint(
                    work_key=parse_key,
                    document_id=document.document_id,
                    input_identities=(version.document_version_id,),
                    output_identities=(parse_id, parsed_record_id),
                    status=BaselineCheckpointStatus.COMPLETED,
                ),
                expected_status=BaselineCheckpointStatus.STARTED,
            )
        except Exception as error:
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
            return BaselineDocumentOutcome(
                ordinal=source.ordinal,
                kind=BaselineDocumentOutcomeKind.FAILED,
                document_id=document.document_id,
                document_version_id=version.document_version_id,
                parse_disposition=BaselineStageDisposition.FAILED,
                failures=(
                    failure(
                        stage=BaselineStage.PARSE,
                        category=category,
                        scope=BaselineFailureScope.DOCUMENT,
                        severity=BaselineFailureSeverity.RECOVERABLE,
                    ),
                ),
            )

        chunk_key = BaselineWorkKey(
            proposed_corpus_version=request.proposed_corpus_version,
            document_version_id=version.document_version_id,
            stage=BaselineStage.CHUNK,
            configuration_identity=request.configuration_identity,
        )
        self.write_checkpoint(
            request,
            BaselineCheckpoint(
                work_key=chunk_key,
                document_id=document.document_id,
                input_identities=(parsed_record_id,),
                output_identities=(),
                status=BaselineCheckpointStatus.STARTED,
            ),
        )
        try:
            chunked = execute_baseline_chunk(
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
            )
            self.write_checkpoint(
                request,
                BaselineCheckpoint(
                    work_key=chunk_key,
                    document_id=document.document_id,
                    input_identities=(parsed_record_id,),
                    output_identities=tuple(
                        chunk.record_id for chunk in chunked.chunk_set.chunks
                    ),
                    status=BaselineCheckpointStatus.COMPLETED,
                ),
                expected_status=BaselineCheckpointStatus.STARTED,
            )
        except Exception as error:
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
            return BaselineDocumentOutcome(
                ordinal=source.ordinal,
                kind=BaselineDocumentOutcomeKind.FAILED,
                document_id=document.document_id,
                document_version_id=version.document_version_id,
                parse_disposition=parsed.disposition,
                chunk_disposition=BaselineStageDisposition.FAILED,
                failures=(
                    failure(
                        stage=BaselineStage.CHUNK,
                        category=category,
                        scope=BaselineFailureScope.DOCUMENT,
                        severity=BaselineFailureSeverity.RECOVERABLE,
                    ),
                ),
            )
        return BaselineDocumentOutcome(
            ordinal=source.ordinal,
            kind=kind,
            document_id=document.document_id,
            document_version_id=version.document_version_id,
            parse_disposition=parsed.disposition,
            chunk_disposition=chunked.disposition,
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
        with self._repository.transaction() as transaction:
            transaction.write_baseline_checkpoint(
                record,
                expected_status=(
                    None
                    if expected_status is None
                    else BaselineCheckpointRecordStatus(expected_status.value)
                ),
            )
