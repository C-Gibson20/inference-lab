"""Tests for the storage-neutral corpus-audit operator runner."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from runpy import run_path
from typing import cast

import pytest

from inference_lab.evaluation import (
    AuditRunResult,
    run_and_publish_audit,
    serialise_audit_report,
)

from .test_audit_golden_matrix import _golden_request  # noqa: TID252

_RUNNER_SCRIPT = (
    Path(__file__).parents[3] / "scripts/inference_lab/evaluation/run-corpus-audit.py"
)


def _script_main() -> Callable[[Sequence[str] | None], int]:
    namespace = run_path(str(_RUNNER_SCRIPT), run_name="audit_runner_script")
    return cast(Callable[[Sequence[str] | None], int], namespace["main"])


def test_run_and_publish_audit_returns_verified_result(tmp_path: Path) -> None:
    destination = tmp_path / "audit"

    result = run_and_publish_audit(_golden_request(), destination)

    assert isinstance(result, AuditRunResult)
    assert result.manifest.report_id == result.report.report_id
    assert result.manifest.input_digest == result.report.input_digest
    assert (destination / "report.json").read_bytes() == serialise_audit_report(
        result.report
    )


def test_run_and_publish_audit_does_not_replace_an_existing_report(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "audit"
    first = run_and_publish_audit(_golden_request(), destination)
    report_bytes = (destination / "report.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        run_and_publish_audit(_golden_request(), destination)

    assert (destination / "report.json").read_bytes() == report_bytes
    assert first.report.report_id.startswith("sha256:")


def test_run_and_publish_audit_rejects_wrong_request_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="CorpusAuditRequest"):
        run_and_publish_audit(object(), tmp_path / "audit")  # type: ignore[arg-type]


def test_operator_script_publishes_and_compares_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = tmp_path / "request_factory.py"
    factory.write_text(
        "from tests.inference_lab.evaluation.test_audit_golden_matrix import "
        "_golden_request\n\n"
        "def build_request():\n"
        "    return _golden_request()\n",
        encoding="utf-8",
    )
    first_destination = tmp_path / "run-001"
    second_destination = tmp_path / "run-002"
    main = _script_main()

    arguments = [
        "--request-factory",
        f"{factory}:build_request",
        "--output",
        str(first_destination),
    ]
    assert main(arguments) == 0
    first_output = capsys.readouterr()
    assert first_output.err == ""
    assert "report_id" in first_output.out
    first_result = json.loads(first_output.out.splitlines()[-1])

    assert (
        main(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--compare-to",
                str(first_destination),
                "--output",
                str(second_destination),
            ]
        )
        == 0
    )
    second_output = capsys.readouterr()
    assert second_output.err == ""
    second_result = json.loads(second_output.out.splitlines()[-1])
    assert second_result == first_result
    assert (second_destination / "report.json").read_bytes() == (
        first_destination / "report.json"
    ).read_bytes()


def test_operator_script_suppresses_request_factory_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = tmp_path / "request_factory.py"
    factory.write_text(
        "def build_request():\n    raise RuntimeError('private-source-canary')\n",
        encoding="utf-8",
    )

    assert (
        _script_main()(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "input_digest" not in output.out
    assert output.err.strip() == "corpus audit runner failed"


def test_operator_script_does_not_publish_a_mismatched_rerun(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_factory = tmp_path / "first_factory.py"
    first_factory.write_text(
        "from tests.inference_lab.evaluation.test_audit_golden_matrix import "
        "_golden_request\n\n"
        "def build_request():\n"
        "    return _golden_request()\n",
        encoding="utf-8",
    )
    changed_factory = tmp_path / "changed_factory.py"
    changed_factory.write_text(
        "from tests.inference_lab.evaluation.test_audit_golden_matrix import "
        "_empty_request\n\n"
        "def build_request():\n"
        "    return _empty_request()\n",
        encoding="utf-8",
    )
    first_destination = tmp_path / "run-001"
    changed_destination = tmp_path / "run-changed"
    main = _script_main()

    assert (
        main(
            [
                "--request-factory",
                f"{first_factory}:build_request",
                "--output",
                str(first_destination),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--request-factory",
                f"{changed_factory}:build_request",
                "--compare-to",
                str(first_destination),
                "--output",
                str(changed_destination),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "input_digest" not in output.out
    assert output.err.strip() == "corpus audit runner failed"
    assert not changed_destination.exists()
