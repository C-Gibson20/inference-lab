"""Run frozen EVAL-03 channel adapters and assemble an EVAL-04 report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.heldout_gate import (
    HeldOutChannel,
    HeldOutChannelIdentity,
    HeldOutRunContract,
)
from inference_lab.evaluation.heldout_publication import (
    HeldOutReportManifest,
    publish_heldout_baseline_report,
)
from inference_lab.evaluation.heldout_reporting import (
    HeldOutBaselineReport,
    HeldOutCaseObservation,
    HeldOutChannelCaseResult,
    build_held_out_baseline_report,
)
from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.retrieval_contracts import (
    RetrievalEvaluationContractError,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
)
from inference_lab.evaluation.retrieval_execution import (
    RetrievalEvaluationExecutionError,
    RetrievalEvaluationRunResult,
)
from inference_lab.evaluation.splitting import build_leakage_groups


@dataclass(frozen=True, slots=True)
class HeldOutChannelRun:
    """One channel's EVAL-03 output plus RET-06 degradation case IDs."""

    channel: HeldOutChannel
    result: RetrievalEvaluationRunResult
    degraded_case_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.channel, HeldOutChannel):
            raise TypeError("channel must be a HeldOutChannel.")
        if not isinstance(self.result, RetrievalEvaluationRunResult):
            raise TypeError("result must be a RetrievalEvaluationRunResult.")
        if not isinstance(self.degraded_case_ids, tuple) or any(
            not isinstance(case_id, str) or not case_id
            for case_id in self.degraded_case_ids
        ):
            raise TypeError("degraded_case_ids must be a tuple of non-empty IDs.")
        if len(self.degraded_case_ids) != len(set(self.degraded_case_ids)):
            raise RetrievalEvaluationContractError("degraded_case_ids must be unique.")


class HeldOutChannelRunner(Protocol):
    """Deployment-owned execution of one pinned channel through EVAL-03."""

    def system_identity(self, channel: HeldOutChannel) -> RetrievalSystemIdentity:
        """Declare a channel's identity without executing a retrieval query."""
        ...

    def __call__(self, channel: HeldOutChannel) -> HeldOutChannelRun:
        """Run one channel using the frozen dataset and corpus snapshot."""
        ...


@dataclass(frozen=True, slots=True)
class HeldOutBaselineRunResult:
    """Aggregate report and manifest returned by the operator workflow."""

    report: HeldOutBaselineReport
    manifest: HeldOutReportManifest

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.report, HeldOutBaselineReport):
            raise TypeError("report must be a HeldOutBaselineReport.")
        if not isinstance(self.manifest, HeldOutReportManifest):
            raise TypeError("manifest must be a HeldOutReportManifest.")
        if (
            self.manifest.run_fingerprint != self.report.contract.fingerprint
            or self.manifest.report_digest != self.report.digest
        ):
            raise RetrievalEvaluationContractError(
                "publication manifest must identify the returned baseline report."
            )


def _channel_identities(
    contract: HeldOutRunContract,
) -> dict[HeldOutChannel, HeldOutChannelIdentity]:
    return {identity.channel: identity for identity in contract.channel_identities}


def _validate_cases(
    contract: HeldOutRunContract,
    cases: tuple[EvaluationCase, ...],
) -> dict[str, str]:
    """Validate held-out order and calculate opaque leakage-group identities."""
    if not isinstance(cases, tuple) or any(
        not isinstance(case, EvaluationCase) for case in cases
    ):
        raise TypeError("cases must be a tuple of EvaluationCase values.")
    case_ids = tuple(case.case_id for case in cases)
    if case_ids != contract.evaluation_identity.selected_case_ids:
        raise RetrievalEvaluationContractError(
            "cases must match the frozen held-out membership and order."
        )
    if any(case.split != "held_out" for case in cases):
        raise RetrievalEvaluationContractError(
            "the EVAL-04 runner accepts held_out cases only."
        )
    groups = build_leakage_groups(cases)
    group_by_case = {
        case.case_id: f"sha256:{group.group_key}"
        for group in groups
        for case in group.cases
    }
    if set(group_by_case) != set(case_ids):
        raise RetrievalEvaluationContractError(
            "leakage groups must cover every frozen held-out case."
        )
    return group_by_case


def _same_run_scope(
    expected: RetrievalEvaluationRunIdentity,
    observed: RetrievalEvaluationRunIdentity,
) -> bool:
    """Compare channel runs while allowing their system identities to differ."""
    return (
        expected.dataset_id == observed.dataset_id
        and expected.dataset_version == observed.dataset_version
        and expected.dataset_schema_version == observed.dataset_schema_version
        and expected.case_file_checksum == observed.case_file_checksum
        and expected.corpus_id == observed.corpus_id
        and expected.corpus_version == observed.corpus_version
        and expected.selected_split == observed.selected_split
        and expected.selected_case_ids == observed.selected_case_ids
        and expected.policy == observed.policy
        and expected.relationship_snapshot_digest
        == observed.relationship_snapshot_digest
        and expected.result_schema_version == observed.result_schema_version
        and expected.implementation == observed.implementation
        and expected.implementation_version == observed.implementation_version
    )


def run_held_out_baseline(
    contract: HeldOutRunContract,
    cases: tuple[EvaluationCase, ...],
    run_channel: HeldOutChannelRunner,
    *,
    limitations: tuple[str, ...] = (),
) -> HeldOutBaselineReport:
    """Run all frozen channels and build their deterministic aggregate report.

    The deployment supplies ``run_channel`` to adapt each Phase 6 path to the
    EVAL-03 runner. This boundary keeps corpus access, model construction, and
    RET-06 backend lifecycle deployment-local while the orchestration enforces
    one dataset, one corpus version, complete channel coverage, and fixed order.
    """
    if not isinstance(contract, HeldOutRunContract):
        raise TypeError("contract must be a HeldOutRunContract.")
    if not isinstance(run_channel, Callable):
        raise TypeError("run_channel must be callable.")
    group_by_case = _validate_cases(contract, cases)
    expected_identity = contract.evaluation_identity
    identities = _channel_identities(contract)
    identity_provider = getattr(run_channel, "system_identity", None)
    if not callable(identity_provider):
        raise TypeError("run_channel must declare system_identity for every channel.")
    for channel in sorted(HeldOutChannel, key=lambda item: item.value):
        try:
            declared_identity = identity_provider(channel)
        except Exception:
            raise RetrievalEvaluationExecutionError(
                "channel_identity_failure"
            ) from None
        if declared_identity != identities[channel].system_identity:
            raise RetrievalEvaluationContractError(
                "declared channel identity differs from the frozen run contract."
            )
    case_results_by_channel: dict[
        HeldOutChannel, tuple[HeldOutChannelCaseResult, ...]
    ] = {}

    for channel in sorted(HeldOutChannel, key=lambda item: item.value):
        try:
            channel_run = run_channel(channel)
        except RetrievalEvaluationExecutionError:
            raise
        except Exception:
            raise RetrievalEvaluationExecutionError("channel_runner_failure") from None
        if not isinstance(channel_run, HeldOutChannelRun):
            raise TypeError("run_channel must return a HeldOutChannelRun.")
        if channel_run.channel is not channel:
            raise RetrievalEvaluationContractError(
                "channel runner returned results for a different channel."
            )
        result = channel_run.result
        if not _same_run_scope(expected_identity, result.identity):
            raise RetrievalEvaluationContractError(
                "channel runs must use the frozen dataset, policy, and corpus version."
            )
        if result.identity.system_identity != identities[channel].system_identity:
            raise RetrievalEvaluationContractError(
                "channel run identity differs from the frozen channel identity."
            )
        case_ids = tuple(item.case_id for item in result.case_results)
        if case_ids != expected_identity.selected_case_ids:
            raise RetrievalEvaluationContractError(
                "channel results must retain frozen case order and membership."
            )
        unknown_degraded_ids = set(channel_run.degraded_case_ids) - set(case_ids)
        if unknown_degraded_ids:
            raise RetrievalEvaluationContractError(
                "degraded case IDs must belong to the frozen held-out set."
            )
        degraded_case_ids = set(channel_run.degraded_case_ids)
        case_results_by_channel[channel] = tuple(
            HeldOutChannelCaseResult(
                channel=channel,
                result=case_result,
                degraded=case_result.case_id in degraded_case_ids,
            )
            for case_result in result.case_results
        )

    observations = tuple(
        HeldOutCaseObservation(
            case_id=case.case_id,
            question_type=case.question_type,
            expected_abstention=case.abstention.expected,
            leakage_group_id=group_by_case[case.case_id],
            channel_results=tuple(
                case_results_by_channel[channel][ordinal] for channel in HeldOutChannel
            ),
        )
        for ordinal, case in enumerate(cases)
    )
    return build_held_out_baseline_report(
        contract,
        observations,
        limitations=limitations,
    )


def run_and_publish_held_out_baseline(
    contract: HeldOutRunContract,
    cases: tuple[EvaluationCase, ...],
    run_channel: HeldOutChannelRunner,
    destination: str | Path,
    *,
    limitations: tuple[str, ...] = (),
) -> HeldOutBaselineRunResult:
    """Run the frozen channels and atomically publish their aggregate report.

    Validate the destination before executing any held-out query. The caller
    owns the deployment-specific adapters and pinned corpus lifecycle.
    """
    if not isinstance(destination, (str, Path)):
        raise TypeError("destination must be a string or Path.")
    raw_destination = Path(destination).expanduser()
    local_destination = validate_local_path(destination, field_name="destination")
    if (
        raw_destination.is_symlink()
        or local_destination.exists()
        or local_destination.is_symlink()
    ):
        raise FileExistsError("Held-out report destination already exists")
    report = run_held_out_baseline(
        contract,
        cases,
        run_channel,
        limitations=limitations,
    )
    manifest = publish_heldout_baseline_report(local_destination, report)
    return HeldOutBaselineRunResult(report=report, manifest=manifest)
