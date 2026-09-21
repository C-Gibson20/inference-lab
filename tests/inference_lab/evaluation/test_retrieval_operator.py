"""Tests for the local retrieval-evaluation request-factory command."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from runpy import run_path
from typing import cast

_RUNNER_SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/inference_lab/evaluation/run-retrieval-evaluation.py"
)


def _script_main() -> Callable[[Sequence[str] | None], int]:
    """Load the operator script's testable main function."""
    namespace = run_path(str(_RUNNER_SCRIPT), run_name="retrieval_runner_script")
    return cast(Callable[[Sequence[str] | None], int], namespace["main"])


def _write_factory(
    path: Path,
    *,
    behavior: str = "completed",
    identity_seed: str = "stable",
) -> None:
    """Write a deployment-local factory with private setup kept off the CLI."""
    dataset_root = path.parent / f"dataset-{behavior}-{identity_seed}"
    path.write_text(
        "from pathlib import Path\n"
        "from tests.inference_lab.evaluation.test_retrieval_execution import "
        "FakeAdapter, request_for, system_identity\n\n"
        "def build_request():\n"
        f"    root = Path({str(dataset_root)!r})\n"
        "    root.mkdir(parents=True, exist_ok=True)\n"
        f"    adapter = FakeAdapter(system_identity({identity_seed!r}), "
        f"{{'case-001': {behavior!r}, 'case-002': 'empty'}})\n"
        "    return request_for(root, adapter, case_ids=('case-001', 'case-002'))\n",
        encoding="utf-8",
    )


def test_operator_publishes_safe_identity_output(
    tmp_path: Path,
    capsys,
) -> None:
    """A successful command publishes five artefacts and only safe output."""
    factory = tmp_path / "request_factory.py"
    _write_factory(factory)
    destination = tmp_path / "run-001"

    assert (
        _script_main()(
            [
                "--request-factory",
                f"{factory}:build_request",
                "--output",
                str(destination),
            ]
        )
        == 0
    )

    output = capsys.readouterr()
    result = json.loads(output.out.splitlines()[-1])
    assert output.err == ""
    assert set(result) == {"comparison", "result_digest", "run_fingerprint"}
    assert result["comparison"] == {"status": "not_requested"}
    assert tuple(sorted(path.name for path in destination.iterdir())) == (
        "cases.jsonl",
        "manifest.json",
        "report.json",
        "report.md",
        "run.json",
    )
    assert "question" not in output.out
    assert "candidate" not in output.out
    assert "private" not in output.out


def test_operator_compares_before_publishing_and_reports_bounded_mismatch(
    tmp_path: Path,
    capsys,
) -> None:
    """A mismatch returns safe categories and leaves the new destination absent."""
    first_factory = tmp_path / "first_factory.py"
    changed_factory = tmp_path / "changed_factory.py"
    _write_factory(first_factory)
    _write_factory(changed_factory, behavior="empty")
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
    result = json.loads(output.out.splitlines()[-1])
    assert output.err == ""
    assert result["comparison"]["equivalent"] is False
    assert result["comparison"]["mismatch_categories"] == [
        "case_results",
        "aggregate",
        "result_digest",
    ]
    assert not changed_destination.exists()
    assert "question" not in output.out
    assert "candidate" not in output.out


def test_operator_suppresses_factory_and_corruption_errors(
    tmp_path: Path,
    capsys,
) -> None:
    """Factory failures and corrupt targets cross the CLI as fixed messages."""
    factory = tmp_path / "bad_factory.py"
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
    assert output.out == ""
    assert output.err.strip() == "retrieval evaluation runner failed"
    assert "private-source-canary" not in output.err
