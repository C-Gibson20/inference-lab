import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from inference_lab.tooling import graph as graph_module


def test_require_graphify_fails_when_executable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_module.shutil, "which", lambda _: None)
    log = Mock()

    assert graph_module.require_graphify(log) is False
    log.error.assert_called_once_with(
        "graphify_not_found",
        hint="uv tool install graphifyy",
    )


def test_require_scan_scope_requires_src_and_tests(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    graph_module.GRAPHIFY_ROOT = tmp_path
    log = Mock()

    assert graph_module.require_scan_scope(log) is False
    log.error.assert_called_once_with(
        "scan_scope_missing",
        missing_roots=["tests"],
    )


def test_run_graphify_command_sets_working_directory_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["graphify", "--version"],
        returncode=0,
        stdout="graphify 0.9.47",
        stderr="",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return completed

    monkeypatch.setattr(graph_module.subprocess, "run", fake_run)
    graph_module.REPO_ROOT = tmp_path

    result = graph_module.run_graphify_command(["--version"])

    assert result is completed
    assert calls == [
        (
            ["graphify", "--version"],
            {
                "capture_output": True,
                "text": True,
                "cwd": tmp_path,
                "timeout": graph_module.GRAPHIFY_TIMEOUT_SECONDS,
            },
        )
    ]


def test_run_graphify_command_converts_timeout_to_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="graphify", timeout=300)

    monkeypatch.setattr(graph_module.subprocess, "run", fake_run)

    with pytest.raises(graph_module.GraphifyFailure) as exc_info:
        graph_module.run_graphify_command(["extract", "."])

    assert exc_info.value.category == "timeout"
    assert exc_info.value.stage == "command"


def test_failed_graphify_stage_does_not_log_stderr_or_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".dev"
    output_root.mkdir()
    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_root / "graphify-out"
    graph_module.GRAPHIFY_ROOT = tmp_path
    log = Mock()
    secret_path = str(tmp_path / "private-source.py")

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            arguments,
            17,
            "",
            f"failed while reading {secret_path}",
        )

    monkeypatch.setattr(graph_module.subprocess, "run", fake_run)

    assert graph_module.refresh_graph(log) is False

    log.error.assert_called_once_with(
        "graph_refresh_failed",
        failure_category="subprocess_failed",
        stage="extract",
        returncode=17,
    )
    assert secret_path not in str(log.mock_calls)
    assert str(tmp_path) not in str(log.mock_calls)


def test_graph_success_logs_repository_relative_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".dev"
    output_root.mkdir()
    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_root / "graphify-out"
    graph_module.GRAPHIFY_ROOT = tmp_path
    log = Mock()

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "extract" in arguments:
            stage_root = Path(arguments[arguments.index("--out") + 1])
            staged_dir = stage_root / "graphify-out"
            staged_dir.mkdir()
            (staged_dir / "graph.json").write_text('{"nodes": [], "edges": []}')
        elif "cluster-only" in arguments:
            stage_root = Path(arguments[arguments.index("cluster-only") + 1])
            (stage_root / "graphify-out" / "graph.html").write_text("<html></html>")
        else:
            tree_path = Path(arguments[arguments.index("--output") + 1])
            tree_path.write_text("<html></html>")

        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(graph_module.subprocess, "run", fake_run)

    assert graph_module.refresh_graph(log) is True, log.mock_calls

    assert log.info.call_args_list[0].kwargs == {
        "graphify_root": ".",
        "included_scope": ["src", "tests"],
        "output_dir": ".dev/graphify-out",
    }
    assert log.info.call_args_list[1].kwargs == {
        "output_dir": ".dev/graphify-out",
    }
    assert str(tmp_path) not in str(log.mock_calls)


def test_build_staged_graph_validates_graph_and_tree_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".dev"
    output_root.mkdir()
    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_root / "graphify-out"
    graph_module.GRAPHIFY_ROOT = tmp_path
    log = Mock()

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "extract":
            stage_root = Path(arguments[arguments.index("--out") + 1])
            staged_dir = stage_root / "graphify-out"
            staged_dir.mkdir()
            (staged_dir / "graph.json").write_text('{"nodes": [], "edges": []}')
        elif arguments[0] == "cluster-only":
            stage_root = Path(arguments[1])
            (stage_root / "graphify-out" / "graph.html").write_text("<html></html>")
        else:
            tree_path = Path(arguments[arguments.index("--output") + 1])
            tree_path.write_text("<html></html>")

        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(graph_module, "run_graphify_command", fake_run)

    stage_root, staged_dir = graph_module.build_staged_graph(log)

    assert (staged_dir / "graph.json").is_file()
    assert (staged_dir / "GRAPH_TREE.html").is_file()

    graph_module.shutil.rmtree(stage_root)


@pytest.mark.parametrize(
    ("graph_contents", "report_contents", "tree_contents", "stage", "category"),
    [
        ("not json", "<html></html>", "<html></html>", "extract", "invalid_output"),
        (
            '{"nodes": {}}',
            "<html></html>",
            "<html></html>",
            "extract",
            "invalid_output",
        ),
        (
            '{"nodes": [], "edges": []}',
            "",
            "<html></html>",
            "cluster",
            "invalid_output",
        ),
        ('{"nodes": [], "edges": []}', "<html></html>", "  ", "tree", "invalid_output"),
    ],
)
def test_build_staged_graph_rejects_invalid_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    graph_contents: str,
    report_contents: str,
    tree_contents: str,
    stage: str,
    category: str,
) -> None:
    output_root = tmp_path / ".dev"
    output_root.mkdir()
    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_root / "graphify-out"
    graph_module.GRAPHIFY_ROOT = tmp_path
    log = Mock()

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "extract" in arguments:
            stage_root = Path(arguments[arguments.index("--out") + 1])
            staged_dir = stage_root / "graphify-out"
            staged_dir.mkdir()
            (staged_dir / "graph.json").write_text(graph_contents)
        elif "cluster-only" in arguments:
            stage_root = Path(arguments[arguments.index("cluster-only") + 1])
            (stage_root / "graphify-out" / "graph.html").write_text(report_contents)
        else:
            tree_path = Path(arguments[arguments.index("--output") + 1])
            tree_path.write_text(tree_contents)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(graph_module.subprocess, "run", fake_run)

    with pytest.raises(graph_module.GraphifyFailure) as exc_info:
        graph_module.build_staged_graph(log)

    assert exc_info.value.category == category
    assert exc_info.value.stage == stage


def test_publish_graph_keeps_backup_when_publish_and_rollback_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".dev"
    output_dir = output_root / "graphify-out"
    stage_root = output_root / ".graphify-refresh-test"
    staged_dir = stage_root / "graphify-out"
    output_dir.mkdir(parents=True)
    stage_root.mkdir()
    staged_dir.mkdir()
    (output_dir / "graph.json").write_text("known-good")
    (staged_dir / "graph.json").write_text("fresh")

    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_dir
    real_replace = graph_module.os.replace
    calls = 0

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        if calls == 3:
            raise OSError("rollback failed")
        real_replace(source, destination)

    monkeypatch.setattr(graph_module.os, "replace", failing_replace)

    with pytest.raises(graph_module.GraphifyFailure) as exc_info:
        graph_module.publish_graph(stage_root, staged_dir)

    assert exc_info.value.category == "publish_failed"
    assert not output_dir.exists()
    assert (
        output_root / ".graphify-previous" / "graph.json"
    ).read_text() == "known-good"


def test_publish_graph_replaces_stale_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".dev"
    output_dir = output_root / "graphify-out"
    stage_root = output_root / ".graphify-refresh-test"
    staged_dir = stage_root / "graphify-out"
    output_dir.mkdir(parents=True)
    stage_root.mkdir()
    staged_dir.mkdir()
    (output_dir / "stale.json").write_text("stale")
    (staged_dir / "graph.json").write_text("fresh")

    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_dir

    graph_module.publish_graph(stage_root, staged_dir)

    assert (output_dir / "graph.json").read_text() == "fresh"
    assert not (output_dir / "stale.json").exists()
    assert not (output_root / ".graphify-previous").exists()
    assert not stage_root.exists()
