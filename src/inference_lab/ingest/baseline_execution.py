"""Bounded, deterministic execution for independent baseline documents."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event
from typing import Protocol

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab.ingest.baseline_contracts import (
    BaselineDocumentOutcome,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineStage,
)
from inference_lab.ingest.baseline_profiling import BaselineProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId, DocumentId
from inference_lab.settings import Settings

_LOGGER = structlog.get_logger(__name__)


class BaselineDocumentWorker(Protocol):
    """Run one document while observing the shared cancellation signal."""

    def __call__(self, cancellation: Event) -> BaselineDocumentOutcome:
        """Return one bounded document outcome or raise a request failure."""
        ...


class BaselineRequestExecutionError(RuntimeError):
    """Represent a request-wide failure without exposing worker payloads."""

    def __init__(self, failure: BaselineFailure) -> None:
        """Create a request failure with no raw exception text."""
        if not isinstance(failure, BaselineFailure):
            raise TypeError("failure must be a BaselineFailure.")
        if failure.scope is not BaselineFailureScope.REQUEST:
            raise ValueError("request execution errors require request scope.")
        if failure.severity is not BaselineFailureSeverity.FATAL:
            raise ValueError("request execution errors require fatal severity.")
        super().__init__(failure.category.value)
        self.failure = failure


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineDocumentTask:
    """Describe one independently executable document ordinal."""

    ordinal: int
    worker: BaselineDocumentWorker
    document_id: DocumentId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the bounded task identity and callable boundary."""
        if type(self.ordinal) is not int:
            raise TypeError("ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")
        if not callable(self.worker):
            raise TypeError("worker must be callable.")
        if self.document_id is not None and not isinstance(
            self.document_id,
            DocumentId,
        ):
            raise TypeError("document_id must be a DocumentId or None.")


def _failure(
    category: BaselineFailureCategory,
) -> BaselineRequestExecutionError:
    """Create a non-sensitive fatal request failure."""
    return BaselineRequestExecutionError(
        BaselineFailure(
            stage=BaselineStage.PARSE,
            category=category,
            scope=BaselineFailureScope.REQUEST,
            severity=BaselineFailureSeverity.FATAL,
            retryable=False,
        )
    )


def _validate_tasks(tasks: tuple[BaselineDocumentTask, ...]) -> None:
    """Require unique tasks in stable discovery order."""
    if not isinstance(tasks, tuple):
        raise TypeError("tasks must be a tuple.")
    if any(not isinstance(task, BaselineDocumentTask) for task in tasks):
        raise TypeError("tasks must contain BaselineDocumentTask values.")
    ordinals = tuple(task.ordinal for task in tasks)
    if ordinals != tuple(sorted(ordinals)):
        raise ValueError("tasks must be in discovery order.")
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("tasks must contain unique ordinals.")


def _validate_outcome(
    task: BaselineDocumentTask,
    outcome: object,
) -> BaselineDocumentOutcome:
    """Validate that a worker returned its own bounded result."""
    if not isinstance(outcome, BaselineDocumentOutcome):
        raise _failure(BaselineFailureCategory.CORRUPT_STATE)
    if outcome.ordinal != task.ordinal:
        raise _failure(BaselineFailureCategory.CORRUPT_STATE)
    if (
        task.document_id is not None
        and outcome.document_id is not None
        and outcome.document_id != task.document_id
    ):
        raise _failure(BaselineFailureCategory.CORRUPT_STATE)
    return outcome


def _emit_document_event(
    event: str,
    *,
    task: BaselineDocumentTask,
    outcome: BaselineDocumentOutcome,
) -> None:
    """Emit one bounded document event after canonical result ordering."""
    fields: dict[str, object] = {
        "document_ordinal": task.ordinal,
        "outcome_kind": outcome.kind.value,
    }
    if outcome.document_id is not None:
        fields["document_id"] = outcome.document_id.as_str()
    try:
        _LOGGER.info(event, pipeline_stage="document", **fields)
    except Exception:
        pass


def execute_baseline_documents(
    tasks: tuple[BaselineDocumentTask, ...],
    *,
    settings: Settings,
    corpus_id: CorpusId,
    proposed_corpus_version: CorpusVersionId,
    cancellation: Event | None = None,
    profiling: BaselineProfilingContext | None = None,
    parent_stage_id: str | None = None,
) -> tuple[BaselineDocumentOutcome, ...]:
    """Run independent document work with bounded admission and stable output."""
    _validate_tasks(tasks)
    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
    if cancellation is not None and not isinstance(cancellation, Event):
        raise TypeError("cancellation must be an Event or None.")
    if profiling is not None and not isinstance(profiling, BaselineProfilingContext):
        raise TypeError("profiling must be a BaselineProfilingContext or None.")
    if parent_stage_id is not None and not isinstance(parent_stage_id, str):
        raise TypeError("parent_stage_id must be a string or None.")

    cancel_event = cancellation if cancellation is not None else Event()
    selected_profiling = (
        profiling if profiling is not None else BaselineProfilingContext.create()
    )
    maximum_workers = settings.ingest.max_concurrent_documents
    context = bound_contextvars(
        corpus_id=corpus_id.as_str(),
        corpus_version=proposed_corpus_version.as_str(),
        pipeline_stage="document",
        run_id=selected_profiling.run_id,
        request_id=selected_profiling.request_id,
        parent_stage_id=parent_stage_id,
    )
    with context:
        if cancel_event.is_set():
            raise _failure(BaselineFailureCategory.CANCELLED)

        executor = ThreadPoolExecutor(max_workers=maximum_workers)
        pending: dict[Future[BaselineDocumentOutcome], BaselineDocumentTask] = {}
        results: dict[int, BaselineDocumentOutcome] = {}
        next_index = 0
        request_failure: BaselineRequestExecutionError | None = None

        def run_task(task: BaselineDocumentTask) -> BaselineDocumentOutcome:
            """Run one worker with explicit correlation context in its thread."""
            with bound_contextvars(
                corpus_id=corpus_id.as_str(),
                corpus_version=proposed_corpus_version.as_str(),
                pipeline_stage="document",
                run_id=selected_profiling.run_id,
                request_id=selected_profiling.request_id,
                parent_stage_id=parent_stage_id,
                document_ordinal=task.ordinal,
                document_id=(
                    task.document_id.as_str() if task.document_id is not None else None
                ),
            ):
                return task.worker(cancel_event)

        def admit() -> None:
            """Admit only the bounded number of not-yet-started tasks."""
            nonlocal next_index
            while (
                request_failure is None
                and not cancel_event.is_set()
                and len(pending) < maximum_workers
                and next_index < len(tasks)
            ):
                task = tasks[next_index]
                next_index += 1
                pending[executor.submit(run_task, task)] = task

        try:
            admit()
            while pending:
                completed, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    task = pending.pop(future)
                    try:
                        results[task.ordinal] = _validate_outcome(
                            task,
                            future.result(),
                        )
                    except BaselineRequestExecutionError as error:
                        request_failure = error
                        cancel_event.set()
                    except Exception as error:
                        request_failure = _failure(
                            BaselineFailureCategory.CORRUPT_STATE
                        )
                        cancel_event.set()
                        del error
                if request_failure is None and not cancel_event.is_set():
                    admit()

            if request_failure is not None:
                try:
                    _LOGGER.error(
                        "baseline_documents_failed",
                        error_category=request_failure.failure.category.value,
                        retryable=request_failure.failure.retryable,
                    )
                except Exception:
                    pass
                raise request_failure

            if cancel_event.is_set():
                raise _failure(BaselineFailureCategory.CANCELLED)

            ordered = tuple(results[task.ordinal] for task in tasks)
            for task in tasks:
                _emit_document_event(
                    "baseline_document_completed",
                    task=task,
                    outcome=results[task.ordinal],
                )
            return ordered
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
