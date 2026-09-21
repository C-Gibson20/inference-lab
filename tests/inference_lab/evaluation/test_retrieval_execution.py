"""Tests for deterministic retrieval-evaluation execution."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.evaluation import (
    DuplicateRelationshipSnapshot,
    RetrievalAdapterResponse,
    RetrievalComponentIdentity,
    RetrievalEvaluationContractError,
    RetrievalEvaluationExecutionError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRequest,
    RetrievalEvaluationRunResult,
    RetrievalSystemIdentity,
    run_retrieval_evaluation,
)
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.loader import load_dataset
from inference_lab.profiling import ProfilingContext, ProfilingSink
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    FailureCategory,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalFailure,
    RetrievalLimit,
)

from .fixtures import (  # noqa: TID252
    CORPUS_ID,
    CORPUS_VERSION,
    case_payload,
    write_dataset_fixture,
)


def checksum(seed: str) -> str:
    """Return a deterministic test checksum."""
    return f"sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


def system_identity(seed: str = "stable") -> RetrievalSystemIdentity:
    """Build one path-free system identity for a fake adapter."""
    return RetrievalSystemIdentity(
        adapter_implementation="fake-adapter",
        adapter_contract_version="1.0.0",
        query_construction_algorithm="verbatim",
        query_construction_version="1.0.0",
        retrieval_algorithm="fake-ranked-results",
        retrieval_algorithm_version="1.0.0",
        configuration_digest=checksum(f"adapter-{seed}"),
        components=(
            RetrievalComponentIdentity(
                component="fake-index",
                implementation="fixture-index",
                format_version="1.0.0",
                contract_version="1.0.0",
                configuration_digest=checksum(f"index-{seed}"),
                generation_id=f"generation-{seed}",
                source_manifest_digest=checksum(f"manifest-{seed}"),
            ),
        ),
    )


def snapshot() -> DuplicateRelationshipSnapshot:
    """Build the corpus-scoped relationship snapshot for the dataset fixture."""
    corpus_id = CorpusId.from_str(CORPUS_ID)
    corpus_version = CorpusVersionId.from_str(CORPUS_VERSION)
    return DuplicateRelationshipSnapshot(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        memberships=(
            CorpusDocumentVersionMembership(
                corpus_id,
                DocumentId.from_str("00000000-0000-4000-8000-000000000003"),
                DocumentVersionId.from_str("00000000-0000-4000-8000-000000000004"),
            ),
        ),
    )


def result(
    *,
    status: ResultStatus = ResultStatus.COMPLETED,
    requested_limit: int = 10,
    corpus_version: str = CORPUS_VERSION,
    with_hit: bool = False,
    failure: FailureCategory | None = None,
) -> EvidenceResult:
    """Build one bounded RET-01 result for fake adapter behavior."""
    failures = ()
    if failure is not None:
        failures = (RetrievalFailure(failure, retryable=False),)
    hits = ()
    if with_hit:
        hits = (
            EvidenceHit(
                candidate=RetrievalCandidate(
                    record_id=RecordId.from_str("00000000-0000-4000-8000-000000000006"),
                    kind=CandidateKind.ORIGINAL_PASSAGE,
                    content="candidate content sentinel",
                    sources=(
                        EvidenceSource(
                            DocumentId.from_str("00000000-0000-4000-8000-000000000003"),
                            DocumentVersionId.from_str(
                                "00000000-0000-4000-8000-000000000004"
                            ),
                            SourceSpanId.from_str(
                                "00000000-0000-4000-8000-000000000005"
                            ),
                        ),
                    ),
                ),
                contributions=(ChannelScore(RetrievalChannel.LEXICAL, 1),),
                rank=1,
            ),
        )
    return EvidenceResult(
        corpus_version=CorpusVersionId.from_str(corpus_version),
        requested_limit=RetrievalLimit(requested_limit),
        hits=hits,
        status=status,
        failures=failures,
    )


class NullProfilingSink:
    """Discard profiling records while proving one context was propagated."""

    def emit(self, record: object) -> None:
        """Accept one profiling record without inspecting its payload."""


@dataclass
class FakeAdapter:
    """Deterministic adapter fixture covering successful and failed outcomes."""

    identity: RetrievalSystemIdentity
    behavior: dict[str, str]
    changed_identity: RetrievalSystemIdentity | None = None

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, ProfilingContext, str, RetrievalLimit]] = []
        self._current_identity = self.identity

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the currently declared identity."""
        return self._current_identity

    def execute(
        self,
        case: EvaluationCase,
        *,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        candidate_depth: RetrievalLimit,
        profiling: ProfilingContext,
        profiling_identity: str,
    ) -> RetrievalAdapterResponse:
        """Return the configured payload-free fixture outcome."""
        del corpus_id, corpus_version
        case_id = case.case_id
        self.calls.append((case_id, profiling, profiling_identity, candidate_depth))
        behavior = self.behavior[case_id]
        if behavior == "raised":
            raise RuntimeError("question and candidate content sentinel")
        if behavior == "invalid":
            return cast(RetrievalAdapterResponse, object())
        if behavior == "foreign":
            return RetrievalAdapterResponse(
                result(
                    corpus_version="00000000-0000-4000-8000-000000000099",
                ),
                self._current_identity,
            )
        if behavior == "wrong_depth":
            return RetrievalAdapterResponse(
                result(requested_limit=5),
                self._current_identity,
            )
        if behavior == "empty":
            response = result()
        elif behavior == "partial":
            response = result(
                status=ResultStatus.PARTIAL,
                with_hit=True,
                failure=FailureCategory.CHANNEL_EXECUTION_FAILURE,
            )
        elif behavior == "failed":
            response = result(
                status=ResultStatus.FAILED,
                failure=FailureCategory.CHANNEL_UNAVAILABLE,
            )
        else:
            response = result(with_hit=True)
        response_identity = self._current_identity
        if self.changed_identity is not None and len(self.calls) == 1:
            self._current_identity = self.changed_identity
        return RetrievalAdapterResponse(response, response_identity)


def request_for(
    tmp_path: Path,
    adapter: FakeAdapter,
    *,
    case_ids: tuple[str, ...],
    dataset_case_ids: tuple[str, ...] | None = None,
    profiling: ProfilingContext | None = None,
) -> RetrievalEvaluationRequest:
    """Build one complete runner request from a loaded temporary dataset."""
    manifest_path = write_dataset_fixture(
        tmp_path,
        cases=[case_payload(case_id) for case_id in (dataset_case_ids or case_ids)],
    )
    dataset = load_dataset(manifest_path)
    return RetrievalEvaluationRequest(
        dataset=dataset,
        selected_case_ids=case_ids,
        selected_split="development",
        corpus_id=CorpusId.from_str(CORPUS_ID),
        corpus_version=CorpusVersionId.from_str(CORPUS_VERSION),
        policy=RetrievalEvaluationPolicy(
            passage_cutoffs=(1, 5, 10), candidate_depth=10
        ),
        relationship_snapshot=snapshot(),
        adapter=adapter,
        profiling=profiling,
    )


def test_runner_canonicalises_order_and_propagates_one_profiling_context(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(
        system_identity(),
        {
            "case-001": "completed",
            "case-002": "empty",
            "case-003": "partial",
        },
    )
    profiling = ProfilingContext.create(
        run_id="profiling-run",
        sink=cast(ProfilingSink, NullProfilingSink()),
    )
    request = request_for(
        tmp_path,
        adapter,
        case_ids=("case-003", "case-001", "case-002"),
        dataset_case_ids=("case-001", "case-002", "case-003"),
        profiling=profiling,
    )

    with capture_logs() as logs:
        run = run_retrieval_evaluation(request)

    assert isinstance(run, RetrievalEvaluationRunResult)
    assert tuple(item.case_id for item in run.case_results) == (
        "case-001",
        "case-002",
        "case-003",
    )
    assert [call[0] for call in adapter.calls] == [
        "case-001",
        "case-002",
        "case-003",
    ]
    assert [call[1] for call in adapter.calls] == [profiling, profiling, profiling]
    assert [call[2] for call in adapter.calls] == [
        "case-00000001",
        "case-00000002",
        "case-00000003",
    ]
    assert all(call[3] == RetrievalLimit(10) for call in adapter.calls)
    assert [record["event"] for record in logs] == [
        "evaluation_run_started",
        "evaluation_case_completed",
        "evaluation_case_completed",
        "evaluation_case_completed",
        "evaluation_run_completed",
    ]
    assert "question and candidate content sentinel" not in repr(logs)
    assert run.aggregate.case_count == 3


@pytest.mark.parametrize(
    ("behavior", "expected_category"),
    [
        ("raised", FailureCategory.CHANNEL_EXECUTION_FAILURE),
        ("invalid", FailureCategory.INVALID_CHANNEL_RESPONSE),
        ("foreign", FailureCategory.CORPUS_VERSION_MISMATCH),
        ("wrong_depth", FailureCategory.INVALID_CHANNEL_RESPONSE),
    ],
)
def test_runner_converts_case_failures_and_continues(
    tmp_path: Path,
    behavior: str,
    expected_category: FailureCategory,
) -> None:
    adapter = FakeAdapter(
        system_identity(), {"case-001": behavior, "case-002": "empty"}
    )
    run = run_retrieval_evaluation(
        request_for(tmp_path, adapter, case_ids=("case-001", "case-002"))
    )

    assert [item.status for item in run.case_results] == [
        ResultStatus.FAILED,
        ResultStatus.COMPLETED,
    ]
    assert run.case_results[0].failures[0].category is expected_category
    assert run.case_results[1].status is ResultStatus.COMPLETED
    assert run.aggregate.document.failed_case_count == 1
    assert run.aggregate.document.applicable_case_count == 2


def test_runner_aborts_on_identity_change_before_scoring_next_case(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(
        system_identity("stable"),
        {"case-001": "completed", "case-002": "completed"},
        changed_identity=system_identity("changed"),
    )

    with pytest.raises(RetrievalEvaluationExecutionError) as error:
        run_retrieval_evaluation(
            request_for(tmp_path, adapter, case_ids=("case-001", "case-002"))
        )

    assert error.value.category == "system_identity_changed"
    assert [call[0] for call in adapter.calls] == ["case-001"]


def test_runner_rejects_invalid_run_boundary_before_adapter_calls(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(system_identity(), {"case-001": "completed"})
    request = request_for(tmp_path, adapter, case_ids=("case-001",))
    invalid_request = RetrievalEvaluationRequest(
        dataset=request.dataset,
        selected_case_ids=request.selected_case_ids,
        selected_split=request.selected_split,
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000099"),
        corpus_version=request.corpus_version,
        policy=request.policy,
        relationship_snapshot=request.relationship_snapshot,
        adapter=request.adapter,
        profiling=request.profiling,
    )

    with pytest.raises(RetrievalEvaluationContractError):
        run_retrieval_evaluation(invalid_request)

    assert adapter.calls == []
