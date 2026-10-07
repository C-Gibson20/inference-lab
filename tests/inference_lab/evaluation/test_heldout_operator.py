"""Tests for the local held-out baseline command."""

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from runpy import run_path
from types import FunctionType, SimpleNamespace
from typing import cast

from inference_lab.evaluation import (
    EvaluationCase,
    HeldOutChannel,
    LocalHeldOutChannelRunner,
    RetrievalAdapterResponse,
    RetrievalEvaluationRequest,
    UnavailableMetadataEvaluationAdapter,
    load_dataset,
)
from inference_lab.evaluation.heldout_reporting import HeldOutBaselineReport
from inference_lab.profiling import ProfilingContext
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import RetrievalLimit
from tests.inference_lab.evaluation.fixtures import case_payload, write_dataset_fixture
from tests.inference_lab.evaluation.test_heldout_reporting import _contract, _report
from tests.inference_lab.evaluation.test_retrieval_execution import (
    FakeAdapter,
    snapshot,
)

_RUNNER_SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/inference_lab/evaluation/run-held-out-baseline.py"
)


def _script_namespace() -> dict[str, object]:
    """Load the command into a namespace whose integration seams can be tested."""
    return run_path(str(_RUNNER_SCRIPT), run_name="heldout_runner_script")


def _request(report: HeldOutBaselineReport) -> SimpleNamespace:
    """Build a deployment request shell for command orchestration tests."""
    return SimpleNamespace(
        contract=report.contract,
        cases=(),
        run_channel=lambda channel: None,
        limitations=(),
    )


def _install_report_runner(
    namespace: dict[str, object],
    report: HeldOutBaselineReport,
) -> None:
    runner = cast(FunctionType, namespace["_run"])
    runner_globals = cast(dict[str, object], runner.__globals__)
    runner_globals["_load_request"] = lambda _: _request(report)
    runner_globals["run_held_out_baseline"] = lambda *args, **kwargs: report


def test_command_publishes_aggregate_report_and_repeats_identically(
    tmp_path: Path,
    capsys,
) -> None:
    """The command publishes a report and compares a repeat before publication."""
    report = _report()
    namespace = _script_namespace()
    _install_report_runner(namespace, report)
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])
    first_destination = tmp_path / "baseline-001"

    assert (
        main(
            [
                "--request-factory",
                "private_request:build_request",
                "--output",
                str(first_destination),
            ]
        )
        == 0
    )
    first_output = capsys.readouterr()
    result = json.loads(first_output.out)
    assert first_output.err == ""
    assert result["comparison"] == {"status": "not_requested"}
    assert result["outcome"] == report.outcome.value
    assert tuple(sorted(path.name for path in first_destination.iterdir())) == (
        "manifest.json",
        "report.json",
        "report.md",
    )

    second_destination = tmp_path / "baseline-002"
    assert (
        main(
            [
                "--request-factory",
                "private_request:build_request",
                "--compare-to",
                str(first_destination),
                "--output",
                str(second_destination),
            ]
        )
        == 0
    )
    second_output = capsys.readouterr()
    second_result = json.loads(second_output.out)
    assert second_output.err == ""
    assert second_result["comparison"] == {"equivalent": True}
    assert {path.name: path.read_bytes() for path in first_destination.iterdir()} == {
        path.name: path.read_bytes() for path in second_destination.iterdir()
    }
    assert "question" not in second_output.out
    assert "candidate" not in second_output.out


def test_command_leaves_output_absent_when_repeat_does_not_match(
    tmp_path: Path,
    capsys,
) -> None:
    """A changed report returns a bounded mismatch without publishing it."""
    first_report = _report()
    changed_report = _report(fail_metadata_case="case-001")
    namespace = _script_namespace()
    _install_report_runner(namespace, first_report)
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])
    first_destination = tmp_path / "baseline-001"
    assert (
        main(
            [
                "--request-factory",
                "private_request:build_request",
                "--output",
                str(first_destination),
            ]
        )
        == 0
    )
    capsys.readouterr()

    _install_report_runner(namespace, changed_report)
    mismatch_destination = tmp_path / "baseline-mismatch"
    assert (
        main(
            [
                "--request-factory",
                "private_request:build_request",
                "--compare-to",
                str(first_destination),
                "--output",
                str(mismatch_destination),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    assert result["comparison"] == {"equivalent": False}
    assert not mismatch_destination.exists()
    assert "question" not in output.out
    assert "candidate" not in output.out


def test_command_rejects_invalid_factory_result_with_fixed_error(
    tmp_path: Path,
    capsys,
) -> None:
    """Factory and request errors cross the command boundary safely."""
    namespace = _script_namespace()
    runner = cast(FunctionType, namespace["_run"])
    runner_globals = cast(dict[str, object], runner.__globals__)
    runner_globals["_load_request"] = lambda _: object()
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])

    assert (
        main(
            [
                "--request-factory",
                "private_request:build_request",
                "--output",
                str(tmp_path / "baseline"),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip() == "held-out baseline runner failed"


def test_local_channel_runner_preserves_unavailable_metadata(
    tmp_path: Path,
) -> None:
    """Metadata-only absence remains an explicit failed channel result."""
    manifest_path = write_dataset_fixture(
        tmp_path,
        cases=[case_payload("case-001", split="held_out")],
        manifest_overrides={"splits": {"development": [], "held_out": ["case-001"]}},
    )
    dataset = load_dataset(manifest_path)
    relationship_snapshot = snapshot()
    baseline_contract = _contract(("case-001",))
    unavailable_metadata = UnavailableMetadataEvaluationAdapter()
    channel_identities = tuple(
        replace(identity, system_identity=unavailable_metadata.system_identity)
        if identity.channel is HeldOutChannel.METADATA
        else identity
        for identity in baseline_contract.channel_identities
    )
    initial_identity = baseline_contract.evaluation_identity
    manifest = dataset.manifest
    contract = replace(
        baseline_contract,
        evaluation_identity=replace(
            initial_identity,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.dataset_version,
            dataset_schema_version=manifest.schema_version,
            case_file_checksum=manifest.checksums[manifest.case_file],
            relationship_snapshot_digest=cast(
                str, relationship_snapshot.content_digest
            ),
            run_fingerprint=None,
        ),
        channel_identities=channel_identities,
        fingerprint=None,
    )
    identities = {item.channel: item for item in contract.channel_identities}

    class AdapterWithPolicy:
        def __init__(self, channel: HeldOutChannel) -> None:
            self.system_identity = identities[channel].system_identity
            self.fusion_policy = identities[channel].fusion_policy
            self.reranking_identity = identities[channel].reranking_identity
            self.delegate = FakeAdapter(
                self.system_identity,
                {"case-001": "completed"},
            )

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
            return self.delegate.execute(
                case,
                corpus_id=corpus_id,
                corpus_version=corpus_version,
                candidate_depth=candidate_depth,
                profiling=profiling,
                profiling_identity=profiling_identity,
            )

    adapters = {
        channel: (
            unavailable_metadata
            if channel is HeldOutChannel.METADATA
            else AdapterWithPolicy(channel)
        )
        for channel in HeldOutChannel
    }
    request = RetrievalEvaluationRequest(
        dataset=dataset,
        selected_case_ids=("case-001",),
        selected_split="held_out",
        corpus_id=CorpusId.from_str(manifest.corpus_id),
        corpus_version=CorpusVersionId.from_str(manifest.corpus_version),
        policy=initial_identity.policy,
        relationship_snapshot=relationship_snapshot,
        adapter=adapters[HeldOutChannel.FUSED_PRE_RERANK],
    )
    channel_runner = LocalHeldOutChannelRunner(contract, request, adapters)

    metadata_run = channel_runner(HeldOutChannel.METADATA)

    assert metadata_run.channel is HeldOutChannel.METADATA
    assert metadata_run.result.case_results[0].status.value == "failed"
    assert tuple(
        failure.category.value
        for failure in metadata_run.result.case_results[0].failures
    ) == ("channel_unavailable",)
