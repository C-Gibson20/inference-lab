"""Tests for bounded, cancellation-aware baseline document execution."""

from threading import Event, Lock
from time import sleep

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

from inference_lab.ingest import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineDocumentTask,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineProfilingContext,
    BaselineRequestExecutionError,
    BaselineStage,
    execute_baseline_documents,
)
from inference_lab.provenance import CorpusId, CorpusVersionId, DocumentId

from .fixtures import fixture_settings  # noqa: TID252


def _outcome(ordinal: int) -> BaselineDocumentOutcome:
    return BaselineDocumentOutcome(
        ordinal=ordinal,
        kind=BaselineDocumentOutcomeKind.UNSUPPORTED,
    )


def _request_failure(
    category: BaselineFailureCategory,
) -> BaselineRequestExecutionError:
    return BaselineRequestExecutionError(
        BaselineFailure(
            stage=BaselineStage.DISCOVERY,
            category=category,
            scope=BaselineFailureScope.REQUEST,
            severity=BaselineFailureSeverity.FATAL,
            retryable=False,
        )
    )


def test_document_execution_is_bounded_and_returns_discovery_order() -> None:
    active = 0
    maximum_active = 0
    lock = Lock()

    def worker(ordinal: int):
        def run(cancellation: Event) -> BaselineDocumentOutcome:
            nonlocal active, maximum_active
            assert not cancellation.is_set()
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            sleep(0.01 * (3 - ordinal))
            with lock:
                active -= 1
            return _outcome(ordinal)

        return run

    tasks = tuple(
        BaselineDocumentTask(ordinal=ordinal, worker=worker(ordinal))
        for ordinal in range(4)
    )
    settings = fixture_settings(max_concurrent_documents=2)

    with capture_logs() as logs:
        outcomes = execute_baseline_documents(
            tasks,
            settings=settings,
            corpus_id=CorpusId.generate(),
            proposed_corpus_version=CorpusVersionId.generate(),
        )

    assert tuple(outcome.ordinal for outcome in outcomes) == (0, 1, 2, 3)
    assert maximum_active == 2
    assert [record["event"] for record in logs] == [
        "baseline_document_completed",
        "baseline_document_completed",
        "baseline_document_completed",
        "baseline_document_completed",
    ]
    assert [record["document_ordinal"] for record in logs] == [0, 1, 2, 3]


def test_document_workers_receive_explicit_run_and_parent_context() -> None:
    seen: list[tuple[int, dict[str, object]]] = []
    lock = Lock()
    document_ids = (DocumentId.generate(), DocumentId.generate())

    def worker(ordinal: int):
        def run(cancellation: Event) -> BaselineDocumentOutcome:
            del cancellation
            with lock:
                seen.append((ordinal, get_contextvars()))
            return _outcome(ordinal)

        return run

    profiling = BaselineProfilingContext.create(run_id="run-documents")
    execute_baseline_documents(
        tuple(
            BaselineDocumentTask(
                ordinal=ordinal,
                worker=worker(ordinal),
                document_id=document_ids[ordinal],
            )
            for ordinal in range(2)
        ),
        settings=fixture_settings(max_concurrent_documents=2),
        corpus_id=CorpusId.generate(),
        proposed_corpus_version=CorpusVersionId.generate(),
        profiling=profiling,
        parent_stage_id="run-documents:discovery:request",
    )

    assert {ordinal for ordinal, _ in seen} == {0, 1}
    for ordinal, contextvars in seen:
        assert contextvars["run_id"] == "run-documents"
        assert contextvars["parent_stage_id"] == "run-documents:discovery:request"
        assert contextvars["document_ordinal"] == ordinal
        assert contextvars["document_id"] == document_ids[ordinal].as_str()


def test_request_failure_stops_new_admission_and_is_payload_free() -> None:
    admitted: list[int] = []

    def failing_worker(cancellation: Event) -> BaselineDocumentOutcome:
        del cancellation
        admitted.append(0)
        raise _request_failure(BaselineFailureCategory.INVALID_REQUEST)

    def later_worker(ordinal: int):
        def run(cancellation: Event) -> BaselineDocumentOutcome:
            del cancellation
            admitted.append(ordinal)
            return _outcome(ordinal)

        return run

    tasks = (
        BaselineDocumentTask(ordinal=0, worker=failing_worker),
        BaselineDocumentTask(ordinal=1, worker=later_worker(1)),
        BaselineDocumentTask(ordinal=2, worker=later_worker(2)),
    )

    with pytest.raises(BaselineRequestExecutionError) as error:
        execute_baseline_documents(
            tasks,
            settings=fixture_settings(max_concurrent_documents=1),
            corpus_id=CorpusId.generate(),
            proposed_corpus_version=CorpusVersionId.generate(),
        )

    assert error.value.failure.category is BaselineFailureCategory.INVALID_REQUEST
    assert admitted == [0]
    assert "private" not in str(error.value)


def test_pre_cancelled_execution_does_not_admit_work() -> None:
    cancellation = Event()
    cancellation.set()
    called = False

    def worker(cancellation: Event) -> BaselineDocumentOutcome:
        nonlocal called
        del cancellation
        called = True
        return _outcome(0)

    with pytest.raises(BaselineRequestExecutionError) as error:
        execute_baseline_documents(
            (BaselineDocumentTask(ordinal=0, worker=worker),),
            settings=fixture_settings(),
            corpus_id=CorpusId.generate(),
            proposed_corpus_version=CorpusVersionId.generate(),
            cancellation=cancellation,
        )

    assert error.value.failure.category is BaselineFailureCategory.CANCELLED
    assert called is False


def test_worker_returning_wrong_ordinal_fails_closed() -> None:
    with pytest.raises(BaselineRequestExecutionError) as error:
        execute_baseline_documents(
            (
                BaselineDocumentTask(
                    ordinal=4,
                    worker=lambda cancellation: _outcome(5),
                ),
            ),
            settings=fixture_settings(),
            corpus_id=CorpusId.generate(),
            proposed_corpus_version=CorpusVersionId.generate(),
        )

    assert error.value.failure.category is BaselineFailureCategory.CORRUPT_STATE
