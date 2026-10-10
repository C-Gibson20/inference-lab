"""Deterministic retrieval-evaluation execution and profiling propagation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import structlog
from structlog.contextvars import bound_contextvars

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.contracts import SplitName
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import LoadedDataset
from inference_lab.evaluation.retrieval_contracts import (
    DuplicateRelationshipSnapshot,
    RetrievalCaseResult,
    RetrievalEvaluationAggregate,
    RetrievalEvaluationContractError,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
)
from inference_lab.evaluation.retrieval_scoring import (
    aggregate_retrieval_case_results,
    build_relevance_targets,
    score_retrieval_case,
)
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
    RetrievalLimit,
)

_LOGGER = structlog.get_logger(__name__)
_EXPECTED_FAILURE_CATEGORIES: Final[frozenset[FailureCategory]] = frozenset(
    FailureCategory
)


class RetrievalEvaluationExecutionError(RuntimeError):
    """Report a run-level execution failure without retaining private payloads."""

    def __init__(self, category: str = "execution_failure") -> None:
        """Create a fixed-message run-level error with a bounded category."""
        if not isinstance(category, str) or not category:
            raise TypeError("category must be a non-empty string.")
        super().__init__("retrieval evaluation execution failed")
        self.category = category


class _CaseResponseError(ValueError):
    """Classify one invalid adapter response without retaining its details."""

    def __init__(self, category: FailureCategory) -> None:
        super().__init__(category.value)
        self.category = category


@dataclass(frozen=True, slots=True)
class RetrievalAdapterResponse:
    """One typed adapter response containing result and system identity."""

    result: EvidenceResult
    system_identity: RetrievalSystemIdentity

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the adapter response boundary."""
        if not isinstance(self.result, EvidenceResult):
            raise TypeError("result must be an EvidenceResult.")
        if not isinstance(self.system_identity, RetrievalSystemIdentity):
            raise TypeError("system_identity must be a RetrievalSystemIdentity.")


@runtime_checkable
class RetrievalEvaluationAdapter(Protocol):
    """Local retrieval seam owned by the caller's channel implementation."""

    @property
    def system_identity(self) -> RetrievalSystemIdentity:
        """Return the immutable path-free identity declared for this adapter."""
        ...

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
        """Execute one private query and return its validated result and identity."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRequest:
    """Complete validated input for one deterministic evaluation run."""

    dataset: LoadedDataset
    selected_case_ids: tuple[str, ...]
    selected_split: SplitName
    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    policy: RetrievalEvaluationPolicy
    relationship_snapshot: DuplicateRelationshipSnapshot
    adapter: RetrievalEvaluationAdapter
    profiling: ProfilingContext | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate request fields before any adapter call is possible."""
        if not isinstance(self.dataset, LoadedDataset):
            raise TypeError("dataset must be a LoadedDataset.")
        if not isinstance(self.selected_case_ids, tuple):
            raise TypeError("selected_case_ids must be a tuple.")
        if not self.selected_case_ids:
            raise RetrievalEvaluationContractError(
                "selected_case_ids must not be empty."
            )
        if any(
            not isinstance(case_id, str) or not case_id
            for case_id in self.selected_case_ids
        ):
            raise TypeError("selected_case_ids must contain non-empty strings.")
        if len(self.selected_case_ids) != len(set(self.selected_case_ids)):
            raise RetrievalEvaluationContractError("selected_case_ids must be unique.")
        if self.selected_split not in {"train", "development", "held_out"}:
            raise RetrievalEvaluationContractError("selected_split is not supported.")
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.policy, RetrievalEvaluationPolicy):
            raise TypeError("policy must be a RetrievalEvaluationPolicy.")
        if not isinstance(
            self.relationship_snapshot,
            DuplicateRelationshipSnapshot,
        ):
            raise TypeError(
                "relationship_snapshot must be a DuplicateRelationshipSnapshot."
            )
        if not callable(getattr(self.adapter, "execute", None)):
            raise TypeError("adapter must provide execute.")
        if self.profiling is not None and not isinstance(
            self.profiling,
            ProfilingContext,
        ):
            raise TypeError("profiling must be a ProfilingContext or None.")


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRunResult:
    """Terminal identity, ordered case results, and aggregate for one run."""

    identity: RetrievalEvaluationRunIdentity
    case_results: tuple[RetrievalCaseResult, ...]
    aggregate: RetrievalEvaluationAggregate

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate result order and aggregate reconciliation."""
        if not isinstance(self.identity, RetrievalEvaluationRunIdentity):
            raise TypeError("identity must be a RetrievalEvaluationRunIdentity.")
        if not isinstance(self.case_results, tuple):
            raise TypeError("case_results must be a tuple.")
        if any(
            not isinstance(case_result, RetrievalCaseResult)
            for case_result in self.case_results
        ):
            raise TypeError(
                "case_results must contain only RetrievalCaseResult values."
            )
        if not isinstance(self.aggregate, RetrievalEvaluationAggregate):
            raise TypeError("aggregate must be a RetrievalEvaluationAggregate.")
        case_ids = tuple(case_result.case_id for case_result in self.case_results)
        if case_ids != self.identity.selected_case_ids:
            raise RetrievalEvaluationContractError(
                "case results must use the canonical selected-case order."
            )
        if self.aggregate.case_count != len(self.case_results):
            raise RetrievalEvaluationContractError(
                "aggregate case count must match the case results."
            )


def _emit_event(method_name: str, event: str, **fields: object) -> None:
    """Emit bounded lifecycle metadata without changing the run outcome."""
    try:
        getattr(_LOGGER, method_name)(event, pipeline_stage="evaluation", **fields)
    except Exception:
        pass


def _manifest_case_ids(
    manifest: DatasetManifest, selected_split: SplitName
) -> tuple[str, ...]:
    """Return the manifest's canonical IDs for one selected split."""
    try:
        values = manifest.splits[selected_split]
    except KeyError:
        raise RetrievalEvaluationContractError(
            "selected_split is not present in the dataset manifest."
        ) from None
    return tuple(values)


def _canonical_cases(request: RetrievalEvaluationRequest) -> tuple[EvaluationCase, ...]:
    """Validate selection membership and return cases in persisted dataset order."""
    manifest = request.dataset.manifest
    if len(request.dataset.cases) != manifest.case_count:
        raise RetrievalEvaluationContractError(
            "loaded case count does not match the dataset manifest."
        )
    if len({case.case_id for case in request.dataset.cases}) != len(
        request.dataset.cases
    ):
        raise RetrievalEvaluationContractError("loaded case IDs must be unique.")
    split_ids = set(_manifest_case_ids(manifest, request.selected_split))
    requested_ids = set(request.selected_case_ids)
    if not requested_ids <= split_ids:
        raise RetrievalEvaluationContractError(
            "selected cases must belong to the selected dataset split."
        )

    cases_by_id: Mapping[str, EvaluationCase] = {
        case.case_id: case for case in request.dataset.cases
    }
    if set(cases_by_id) != {
        case_id
        for split_case_ids in manifest.splits.values()
        for case_id in split_case_ids
    }:
        raise RetrievalEvaluationContractError(
            "loaded cases do not match the dataset manifest."
        )
    expected_split_by_case = {
        case_id: split
        for split, split_case_ids in manifest.splits.items()
        for case_id in split_case_ids
    }
    for case in request.dataset.cases:
        if expected_split_by_case[case.case_id] != case.split:
            raise RetrievalEvaluationContractError(
                "loaded case split does not match the dataset manifest."
            )
        if any(
            evidence.corpus_version != manifest.corpus_version
            for evidence in case.evidence
        ):
            raise RetrievalEvaluationContractError(
                "loaded evidence corpus version does not match the dataset manifest."
            )

    selected_cases = tuple(
        case
        for case in request.dataset.cases
        if case.case_id in requested_ids and case.split == request.selected_split
    )
    if len(selected_cases) != len(request.selected_case_ids):
        raise RetrievalEvaluationContractError(
            "selected cases are missing from the loaded dataset."
        )
    return selected_cases


def _validate_request_boundary(
    request: RetrievalEvaluationRequest,
    cases: tuple[EvaluationCase, ...],
) -> None:
    """Validate one-corpus ownership and every selected gold-evidence boundary."""
    manifest = request.dataset.manifest
    manifest_corpus_id = CorpusId.from_str(manifest.corpus_id)
    manifest_corpus_version = CorpusVersionId.from_str(manifest.corpus_version)
    if request.corpus_id != manifest_corpus_id:
        raise RetrievalEvaluationContractError(
            "request corpus does not match the dataset manifest."
        )
    if request.corpus_version != manifest_corpus_version:
        raise RetrievalEvaluationContractError(
            "request corpus version does not match the dataset manifest."
        )
    snapshot = request.relationship_snapshot
    if snapshot.corpus_id != request.corpus_id:
        raise RetrievalEvaluationContractError(
            "relationship snapshot corpus does not match the request."
        )
    if snapshot.corpus_version != request.corpus_version:
        raise RetrievalEvaluationContractError(
            "relationship snapshot version does not match the request."
        )
    if snapshot.content_digest is None:
        raise RetrievalEvaluationContractError(
            "relationship snapshot must have a content digest."
        )
    if not manifest.checksums.get(manifest.case_file):
        raise RetrievalEvaluationContractError(
            "dataset manifest must declare the case-file checksum."
        )

    for case in cases:
        # Target construction validates corpus ownership and visible provenance
        # before the first private query can reach the adapter.
        build_relevance_targets(case, snapshot)


def _build_identity(
    request: RetrievalEvaluationRequest,
    cases: tuple[EvaluationCase, ...],
    system_identity: RetrievalSystemIdentity,
) -> RetrievalEvaluationRunIdentity:
    """Build the complete semantic identity after request preflight."""
    manifest = request.dataset.manifest
    checksum = manifest.checksums.get(manifest.case_file)
    assert checksum is not None
    snapshot_digest = request.relationship_snapshot.content_digest
    assert snapshot_digest is not None
    return RetrievalEvaluationRunIdentity(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_schema_version=manifest.schema_version,
        case_file_checksum=checksum,
        corpus_id=request.corpus_id,
        corpus_version=request.corpus_version,
        selected_split=request.selected_split,
        selected_case_ids=tuple(case.case_id for case in cases),
        policy=request.policy,
        relationship_snapshot_digest=snapshot_digest,
        system_identity=system_identity,
    )


def _declared_identity(
    adapter: RetrievalEvaluationAdapter,
) -> RetrievalSystemIdentity:
    """Read and validate the adapter identity without exposing adapter errors."""
    try:
        identity = adapter.system_identity
    except Exception:
        raise RetrievalEvaluationExecutionError("invalid_adapter_identity") from None
    if not isinstance(identity, RetrievalSystemIdentity):
        raise RetrievalEvaluationExecutionError("invalid_adapter_identity")
    return identity


def _failed_result(
    request: RetrievalEvaluationRequest,
    category: FailureCategory,
) -> EvidenceResult:
    """Create a payload-free failed retrieval result pinned to the run corpus."""
    if category not in _EXPECTED_FAILURE_CATEGORIES:
        raise ValueError("unsupported failure category")
    return EvidenceResult(
        corpus_version=request.corpus_version,
        requested_limit=RetrievalLimit(request.policy.candidate_depth),
        hits=(),
        status=ResultStatus.FAILED,
        failures=(RetrievalFailure(category=category, retryable=False),),
    )


def _score_failure(
    request: RetrievalEvaluationRequest,
    case: EvaluationCase,
    ordinal: int,
    category: FailureCategory,
) -> RetrievalCaseResult:
    """Score one synthetic failed result without retaining exception details."""
    return score_retrieval_case(
        case=case,
        result=_failed_result(request, category),
        corpus_id=request.corpus_id,
        snapshot=request.relationship_snapshot,
        policy=request.policy,
        case_ordinal=ordinal,
    )


def _validate_adapter_response(
    response: object,
    *,
    request: RetrievalEvaluationRequest,
    declared_identity: RetrievalSystemIdentity,
) -> EvidenceResult:
    """Validate one typed adapter response before handing it to the scorer."""
    if not isinstance(response, RetrievalAdapterResponse):
        raise _CaseResponseError(FailureCategory.INVALID_CHANNEL_RESPONSE)
    if response.system_identity != declared_identity:
        raise RetrievalEvaluationExecutionError("system_identity_changed")
    result = response.result
    expected_limit = RetrievalLimit(request.policy.candidate_depth)
    if result.requested_limit != expected_limit:
        raise _CaseResponseError(FailureCategory.INVALID_CHANNEL_RESPONSE)
    if result.corpus_version != request.corpus_version:
        raise _CaseResponseError(FailureCategory.CORPUS_VERSION_MISMATCH)
    return result


def _log_case_result(
    result: RetrievalCaseResult,
    *,
    ordinal: int,
    policy: RetrievalEvaluationPolicy,
) -> None:
    """Emit the bounded lifecycle event for one terminal case result."""
    if result.status is ResultStatus.PARTIAL:
        method = "warning"
        event = "evaluation_case_completed"
    elif result.status is ResultStatus.FAILED:
        method = "error"
        event = "evaluation_case_failed"
    else:
        method = "info"
        event = "evaluation_case_completed"
    _emit_event(
        method,
        event,
        case_ordinal=ordinal,
        status=result.status.value,
        hit_count=len(result.hits),
        failure_categories=tuple(failure.category.value for failure in result.failures),
        passage_cutoffs=policy.passage_cutoffs,
        document_depth_exhausted=result.document_depth_exhausted,
    )


def run_retrieval_evaluation(
    request: RetrievalEvaluationRequest,
) -> RetrievalEvaluationRunResult:
    """Run selected cases serially through one stable local adapter."""
    if not isinstance(request, RetrievalEvaluationRequest):
        raise TypeError("request must be a RetrievalEvaluationRequest.")

    cases = _canonical_cases(request)
    _validate_request_boundary(request, cases)
    declared_identity = _declared_identity(request.adapter)
    identity = _build_identity(request, cases, declared_identity)
    profiling = (
        request.profiling
        if request.profiling is not None
        else ProfilingContext.create()
    )
    expected_limit = RetrievalLimit(request.policy.candidate_depth)
    base_log_fields = {
        "run_fingerprint": identity.run_fingerprint,
        "selected_split": request.selected_split,
        "selected_case_count": len(cases),
        "candidate_depth": expected_limit.value,
        "passage_cutoffs": request.policy.passage_cutoffs,
    }

    case_results: list[RetrievalCaseResult] = []
    with bound_contextvars(
        run_id=profiling.run_id,
        corpus_id=request.corpus_id.as_str(),
        corpus_version=request.corpus_version.as_str(),
        document_id=None,
    ):
        _emit_event("info", "evaluation_run_started", **base_log_fields)
        try:
            for ordinal, case in enumerate(cases):
                current_identity = _declared_identity(request.adapter)
                if current_identity != declared_identity:
                    raise RetrievalEvaluationExecutionError("system_identity_changed")
                profiling_identity = f"case-{ordinal + 1:08d}"
                try:
                    response = request.adapter.execute(
                        case,
                        corpus_id=request.corpus_id,
                        corpus_version=request.corpus_version,
                        candidate_depth=expected_limit,
                        profiling=profiling,
                        profiling_identity=profiling_identity,
                    )
                    result = _validate_adapter_response(
                        response,
                        request=request,
                        declared_identity=declared_identity,
                    )
                    if _declared_identity(request.adapter) != declared_identity:
                        raise RetrievalEvaluationExecutionError(
                            "system_identity_changed"
                        )
                    case_result = score_retrieval_case(
                        case=case,
                        result=result,
                        corpus_id=request.corpus_id,
                        snapshot=request.relationship_snapshot,
                        policy=request.policy,
                        case_ordinal=ordinal,
                    )
                except RetrievalEvaluationExecutionError:
                    raise
                except _CaseResponseError as error:
                    case_result = _score_failure(
                        request,
                        case,
                        ordinal,
                        error.category,
                    )
                except Exception:
                    case_result = _score_failure(
                        request,
                        case,
                        ordinal,
                        FailureCategory.CHANNEL_EXECUTION_FAILURE,
                    )
                case_results.append(case_result)
                _log_case_result(
                    case_result,
                    ordinal=ordinal + 1,
                    policy=request.policy,
                )

            result_tuple = tuple(case_results)
            aggregate = aggregate_retrieval_case_results(result_tuple, request.policy)
            run_result = RetrievalEvaluationRunResult(
                identity=identity,
                case_results=result_tuple,
                aggregate=aggregate,
            )
        except RetrievalEvaluationExecutionError as error:
            _emit_event(
                "error",
                "evaluation_run_failed",
                **base_log_fields,
                failure_category=error.category,
            )
            raise
        except Exception:
            _emit_event(
                "error",
                "evaluation_run_failed",
                **base_log_fields,
                failure_category="execution_failure",
            )
            raise RetrievalEvaluationExecutionError() from None

        _emit_event(
            "info",
            "evaluation_run_completed",
            **base_log_fields,
            completed_case_count=sum(
                result.status is ResultStatus.COMPLETED for result in case_results
            ),
            partial_case_count=sum(
                result.status is ResultStatus.PARTIAL for result in case_results
            ),
            failed_case_count=sum(
                result.status is ResultStatus.FAILED for result in case_results
            ),
        )
        return run_result
