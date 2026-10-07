"""Tests for immutable aggregate-only EVAL-04 report publication."""

from pathlib import Path
from typing import cast

import pytest

from inference_lab.evaluation import heldout_runner
from inference_lab.evaluation.heldout_publication import (
    HeldOutPublicationMismatchError,
    compare_heldout_baseline_publication,
    deserialise_heldout_report_manifest,
    publish_heldout_baseline_report,
    serialise_heldout_report_manifest,
    verify_heldout_baseline_publication,
)
from inference_lab.evaluation.heldout_reporting import build_held_out_baseline_report
from inference_lab.evaluation.heldout_runner import HeldOutChannelRunner

from .test_heldout_reporting import (  # noqa: TID252
    _contract,
    _observation_fixture,
)


def _report():
    observations = _observation_fixture()
    contract = _contract(tuple(item.case_id for item in observations))
    return build_held_out_baseline_report(contract, observations)


def test_publish_verifies_canonical_bundle_and_exact_rerun(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "run"

    manifest = publish_heldout_baseline_report(destination, report)

    assert manifest.report_digest == report.digest
    assert verify_heldout_baseline_publication(destination) == manifest
    assert compare_heldout_baseline_publication(destination, report)
    assert (
        deserialise_heldout_report_manifest(serialise_heldout_report_manifest(manifest))
        == manifest
    )
    assert {path.name for path in destination.iterdir()} == {
        "report.json",
        "report.md",
        "manifest.json",
    }


def test_publication_is_immutable_and_mismatch_is_rejected(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "run"
    publish_heldout_baseline_report(destination, report)

    with pytest.raises(FileExistsError):
        publish_heldout_baseline_report(destination, report)

    observations = _observation_fixture(
        degrade_dense_case=_observation_fixture()[0].case_id
    )
    altered = build_held_out_baseline_report(report.contract, observations)
    with pytest.raises(HeldOutPublicationMismatchError):
        compare_heldout_baseline_publication(destination, altered)


def test_verification_rejects_tampered_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    publish_heldout_baseline_report(destination, _report())
    (destination / "report.md").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        verify_heldout_baseline_publication(destination)


def test_operator_workflow_runs_then_publishes(tmp_path: Path, monkeypatch) -> None:
    report = _report()
    calls: list[str] = []
    runner = object()

    def fake_run(*args, **kwargs):
        calls.append("run")
        return report

    monkeypatch.setattr(heldout_runner, "run_held_out_baseline", fake_run)
    destination = tmp_path / "workflow"
    result = heldout_runner.run_and_publish_held_out_baseline(
        report.contract,
        (),
        cast(HeldOutChannelRunner, runner),
        destination,
        limitations=("sample-only",),
    )

    assert calls == ["run"]
    assert result.report == report
    assert result.manifest.report_digest == report.digest
    assert verify_heldout_baseline_publication(destination) == result.manifest


def test_operator_workflow_rejects_destination_before_running(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()

    def unexpected_run(*args, **kwargs):
        pytest.fail("held-out retrieval must not start for an occupied destination")

    monkeypatch.setattr(heldout_runner, "run_held_out_baseline", unexpected_run)
    with pytest.raises(FileExistsError):
        heldout_runner.run_and_publish_held_out_baseline(
            _report().contract,
            (),
            cast(HeldOutChannelRunner, object()),
            destination,
        )
