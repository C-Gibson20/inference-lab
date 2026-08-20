import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "graph" / "generate_graph.py"


@pytest.fixture
def graph_module() -> ModuleType:
    """Load the standalone graph generator without executing its CLI."""
    spec = importlib.util.spec_from_file_location("generate_graph", SCRIPT)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_require_graphify_fails_when_executable_is_missing(
    graph_module: ModuleType,
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
    graph_module: ModuleType,
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
    graph_module: ModuleType,
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
    graph_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="graphify", timeout=300)

    monkeypatch.setattr(graph_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Graphify timed out"):
        graph_module.run_graphify_command(["extract", "."])


def test_build_staged_graph_validates_graph_and_tree_outputs(
    graph_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".dev"
    output_root.mkdir()
    graph_module.OUTPUT_ROOT = output_root
    graph_module.OUTPUT_DIR = output_root / "graphify-out"
    graph_module.GRAPHIFY_ROOT = tmp_path
    log = Mock()

    def fake_run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "extract":
            stage_root = Path(arguments[arguments.index("--out") + 1])
            staged_dir = stage_root / "graphify-out"
            staged_dir.mkdir()
            (staged_dir / "graph.json").write_text('{"nodes": [], "edges": []}')
        else:
            tree_path = Path(arguments[arguments.index("--output") + 1])
            tree_path.write_text("<html></html>")

        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(graph_module, "run_graphify_command", fake_run)

    stage_root, staged_dir = graph_module.build_staged_graph(log)

    assert (staged_dir / "graph.json").is_file()
    assert (staged_dir / "GRAPH_TREE.html").is_file()

    graph_module.shutil.rmtree(stage_root)


def test_publish_graph_replaces_stale_output(
    graph_module: ModuleType,
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
