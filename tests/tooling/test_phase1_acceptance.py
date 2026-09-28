"""TOOL-01–05 and DOCS-01 phase journeys for tooling and operator setup."""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
import structlog
from pydantic import ValidationError

import inference_lab.tooling.graph as graph_module
from inference_lab import main
from inference_lab.logging.logging_config import configure_logging
from inference_lab.settings import LoggingConfig, Settings, get_settings

REPO_ROOT = Path(__file__).parents[2]
VALIDATOR = REPO_ROOT / "scripts" / "validate-commit-msg.sh"
PRE_COMMIT = REPO_ROOT / ".githooks" / "pre-commit"
COMMIT_MSG = REPO_ROOT / ".githooks" / "commit-msg"
SETUP_HOOKS = REPO_ROOT / "scripts" / "setup-hooks.sh"


def _config_text(root: Path) -> str:
    return f"""
        [models]
        embedder_path = "{root / "models/embedder"}"
        reranker_path = "{root / "models/reranker"}"
        generator_path = "{root / "models/generator"}"

        [runtime]
        vram_budget_gb = 24
        max_context_tokens = 8192

        [storage]
        index_dir = "{root / "state/indexes"}"
        metadata_db_path = "{root / "state/metadata.sqlite3"}"
        source_dir = "{root / "state/source-data"}"

        [logging]
        level = "DEBUG"
        format = "json"

        [ingest]
        max_source_bytes = 1048576
        max_concurrent_documents = 3
        """


@pytest.fixture(autouse=True)
def isolated_phase1_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    (tmp_path / "config.toml").write_text(_config_text(tmp_path), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in (
        "MODELS__GENERATOR_PATH",
        "RUNTIME__VRAM_BUDGET_GB",
        "STORAGE__SOURCE_DIR",
        "LOGGING__LEVEL",
        "LOGGING__FORMAT",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_application_handlers() -> Iterator[None]:
    root_logger = logging.getLogger()
    existing_handlers = set(root_logger.handlers)
    yield
    for handler in root_logger.handlers[:]:
        if handler not in existing_handlers:
            root_logger.removeHandler(handler)
            handler.close()


def test_configuration_merges_toml_environment_and_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOOL-02: precedence, nested overrides, defaults, and cache lifecycle."""
    configured = get_settings()
    assert configured.runtime.vram_budget_gb == 24
    assert configured.storage.metadata_db_path.name == "metadata.sqlite3"
    assert configured.ingest.max_concurrent_documents == 3

    monkeypatch.setenv("RUNTIME__VRAM_BUDGET_GB", "12")
    monkeypatch.setenv("STORAGE__SOURCE_DIR", "/environment/source-data")
    get_settings.cache_clear()
    environment = get_settings()
    assert environment.runtime.vram_budget_gb == 12
    assert environment.storage.source_dir == Path("/environment/source-data")

    explicit = Settings(  # pyright: ignore[reportCallIssue]
        runtime={"vram_budget_gb": 8},
        models={"generator_path": "/explicit/generator"},
    )
    assert explicit.runtime.vram_budget_gb == 8
    assert explicit.runtime.max_context_tokens == 8192
    assert explicit.models.generator_path == Path("/explicit/generator")

    monkeypatch.setenv("RUNTIME__VRAM_BUDGET_GB", "16")
    assert get_settings().runtime.vram_budget_gb == 12
    get_settings.cache_clear()
    assert get_settings().runtime.vram_budget_gb == 16

    config_file = Path.cwd() / "config.toml"
    config_file.write_text(_config_text(Path.cwd()).split("[logging]")[0])
    get_settings.cache_clear()
    defaults = get_settings()
    assert defaults.logging.level == "INFO"
    assert defaults.logging.format == "auto"
    assert defaults.ingest.max_source_bytes == 50 * 1024 * 1024
    assert defaults.ingest.max_concurrent_documents == 1


def test_configuration_rejects_missing_invalid_unknown_and_overlapping_values(
    tmp_path: Path,
) -> None:
    """TOOL-02: invalid values, local paths, unknown fields, and missing paths fail."""
    configured = get_settings()
    valid = configured.model_dump()

    cases: list[tuple[str, dict[str, object]]] = []
    invalid_runtime = copy.deepcopy(valid)
    invalid_runtime["runtime"]["vram_budget_gb"] = 0  # type: ignore[index]
    cases.append(("non-positive runtime limit", invalid_runtime))

    non_local_model = copy.deepcopy(valid)
    non_local_model["models"]["generator_path"] = "https://example.test/model"  # type: ignore[index]
    cases.append(("non-local model path", non_local_model))

    overlapping_paths = copy.deepcopy(valid)
    overlapping_paths["storage"]["source_dir"] = (  # type: ignore[index]
        overlapping_paths["storage"]["index_dir"]  # type: ignore[index]
    )
    cases.append(("overlapping storage paths", overlapping_paths))

    unknown_setting = copy.deepcopy(valid)
    unknown_setting["storage"]["backup_dir"] = (  # type: ignore[index]
        tmp_path / "backup"
    )
    cases.append(("unknown storage setting", unknown_setting))

    for label, values in cases:
        try:
            Settings(**values)  # type: ignore[arg-type]
        except ValidationError:
            continue
        pytest.fail(f"settings accepted {label}")

    incomplete_toml = tmp_path / "incomplete.toml"
    incomplete_toml.write_text(
        """
        [models]
        embedder_path = "/models/embedder"
        reranker_path = "/models/reranker"

        [runtime]
        vram_budget_gb = 24

        [storage]
        index_dir = "/data/indexes"
        metadata_db_path = "/data/metadata.sqlite3"
        source_dir = "/data/sources"
        """,
        encoding="utf-8",
    )

    class IncompleteSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["toml_file"] = incomplete_toml

    with pytest.raises(ValidationError):
        IncompleteSettings()  # pyright: ignore[reportCallIssue]


def test_application_startup_loads_settings_and_emits_configured_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TOOL-02/03: startup uses settings before emitting a complete event."""
    main()

    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    startup = next(
        record for record in records if record["event"] == "startup_completed"
    )
    assert startup["level"] == "info"
    assert startup["module"] == "inference_lab"
    assert startup["document_id"] is None
    assert startup["corpus_id"] is None
    assert startup["corpus_version"] is None
    assert startup["pipeline_stage"] is None


def test_logging_policy_covers_renderers_context_privacy_and_reconfiguration() -> None:
    """TOOL-03: renderers, shared levels, context, privacy, and reconfiguration."""

    class Stream(StringIO):
        def __init__(self, *, is_tty: bool) -> None:
            super().__init__()
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    json_stream = StringIO()
    config = LoggingConfig(level="INFO", format="json")
    configure_logging(config, stream=json_stream)
    log = structlog.get_logger("phase1")
    log.debug("suppressed_event")
    log.info("phase_completed", corpus_id="corpus-a", corpus_version="version-1")
    logging.getLogger("dependency").warning("dependency_degraded")
    records = [json.loads(line) for line in json_stream.getvalue().splitlines()]
    assert [record["event"] for record in records] == [
        "phase_completed",
        "dependency_degraded",
    ]
    assert records[0]["corpus_id"] == "corpus-a"
    assert records[0]["corpus_version"] == "version-1"
    assert records[0]["document_id"] is None
    assert records[1]["module"] == "dependency"
    assert records[1]["level"] == "warning"

    with pytest.raises(ValueError, match="sensitive content fields"):
        log.info("evaluation_completed", model_output="private answer")
    with pytest.raises(ValueError, match="sensitive content fields"):
        log.info("model_output")

    console_stream = StringIO()
    configure_logging(
        LoggingConfig(level="INFO", format="console"),
        stream=console_stream,
    )
    structlog.get_logger("phase1").warning("dependency_degraded")
    assert "dependency_degraded" in console_stream.getvalue()

    tty_stream = Stream(is_tty=True)
    configure_logging(
        LoggingConfig(level="INFO", format="auto"),
        stream=tty_stream,
    )
    structlog.get_logger("phase1").info("automatic_console_record")
    assert "automatic_console_record" in tty_stream.getvalue()
    assert not tty_stream.getvalue().lstrip().startswith("{")

    automatic_stream = Stream(is_tty=False)
    configure_logging(
        LoggingConfig(level="INFO", format="auto"),
        stream=automatic_stream,
    )
    structlog.get_logger("phase1").info("automatic_json_record")
    assert json.loads(automatic_stream.getvalue())["event"] == "automatic_json_record"

    repeated_stream = StringIO()
    configure_logging(config, stream=repeated_stream)
    configure_logging(config, stream=repeated_stream)
    structlog.get_logger("phase1").info("one_record")
    records = [json.loads(line) for line in repeated_stream.getvalue().splitlines()]
    assert [item["event"] for item in records] == ["one_record"]


def test_clean_checkout_ci_hooks_and_commit_validation_work_together(
    tmp_path: Path,
) -> None:
    """TOOL-01: CI events, validation, commit hooks, and clean setup agree."""
    workflow = (REPO_ROOT / ".github/workflows/ci.yaml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    for event in ("opened", "synchronize", "reopened", "ready_for_review"):
        assert event in workflow
    assert "branches: [main]" in workflow
    assert "contents: read" in workflow
    assert "github.ref" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "uv sync --dev --locked" in workflow
    assert "uv run pytest --cov" in workflow

    valid_and_merge = (
        "[tooling] test: validate the repository workflow\n"
        "Merge pull request #43 from example/feature\n"
    )
    accepted = subprocess.run(
        [str(VALIDATOR), "--stdin"],
        input=valid_and_merge,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr

    rejected = subprocess.run(
        [str(VALIDATOR), "--stdin"],
        input="not a valid commit subject\n",
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1

    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(
        "[tooling] test: validate hook\n\nCommit body.\n", encoding="utf-8"
    )
    hook = subprocess.run(
        [str(COMMIT_MSG), str(message_file)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert hook.returncode == 0, hook.stderr

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "uv-calls"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n")
    fake_uv.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    pre_commit = subprocess.run(
        [str(PRE_COMMIT)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
    )
    assert pre_commit.returncode == 0, pre_commit.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "run ruff check .",
        "run ruff format --check .",
        "run pyright",
    ]

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    nested = repository / "nested"
    nested.mkdir()
    setup = subprocess.run(
        [str(SETUP_HOOKS)], cwd=nested, capture_output=True, text=True
    )
    assert setup.returncode == 0, setup.stderr
    hooks_path = subprocess.run(
        ["git", "config", "--local", "core.hooksPath"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert hooks_path.stdout.strip() == ".githooks"


def _configure_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    output_root = tmp_path / ".dev"
    output_dir = output_root / "graphify-out"
    graph_root = tmp_path / "checkout"
    (graph_root / "src").mkdir(parents=True)
    (graph_root / "tests").mkdir()
    monkeypatch.setattr(graph_module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(graph_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(graph_module, "GRAPHIFY_ROOT", graph_root)
    monkeypatch.setattr(graph_module, "REPO_ROOT", graph_root)
    monkeypatch.setattr(graph_module.shutil, "which", lambda _: "/fake/graphify")
    return output_root, output_dir


def _fake_graphify_run(
    arguments: list[str],
    **_: object,
) -> subprocess.CompletedProcess[str]:
    command = arguments[1]
    if command == "extract":
        stage_root = Path(arguments[arguments.index("--out") + 1])
        staged_dir = stage_root / "graphify-out"
        staged_dir.mkdir()
        (staged_dir / "graph.json").write_text(
            '{"nodes": [], "edges": []}', encoding="utf-8"
        )
    elif command == "cluster-only":
        stage_root = Path(arguments[2])
        (stage_root / "graphify-out" / "graph.html").write_text(
            "<html>graph</html>", encoding="utf-8"
        )
    elif command == "tree":
        tree_path = Path(arguments[arguments.index("--output") + 1])
        tree_path.write_text("<html>tree</html>", encoding="utf-8")
    else:
        raise AssertionError(f"unexpected graph command: {command}")
    return subprocess.CompletedProcess(arguments, 0, "", "")


def test_local_graph_refresh_builds_validated_outputs_before_publishing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, output_dir = _configure_graph(monkeypatch, tmp_path)
    monkeypatch.setattr(graph_module.subprocess, "run", _fake_graphify_run)
    log = Mock()

    assert graph_module.refresh_graph(log)
    assert (output_dir / "graph.json").is_file()
    assert (output_dir / "graph.html").read_text(encoding="utf-8") == (
        "<html>graph</html>"
    )
    assert (output_dir / "GRAPH_TREE.html").is_file()
    assert not (output_dir.parent / ".graphify-previous").exists()
    assert str(tmp_path) not in str(log.mock_calls)


def test_graph_refresh_failure_keeps_previous_output_and_hides_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, output_dir = _configure_graph(monkeypatch, tmp_path)
    output_dir.mkdir(parents=True)
    marker = output_dir / "previous.txt"
    marker.write_text("previous graph", encoding="utf-8")
    private_path = str(tmp_path / "private-source.py")
    log = Mock()

    def fail_cluster(
        arguments: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        if arguments[1] == "extract":
            stage_root = Path(arguments[arguments.index("--out") + 1])
            staged_dir = stage_root / "graphify-out"
            staged_dir.mkdir()
            (staged_dir / "graph.json").write_text(
                '{"nodes": [], "edges": []}', encoding="utf-8"
            )
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(
            arguments, 17, "", f"failed while reading {private_path}"
        )

    monkeypatch.setattr(graph_module.subprocess, "run", fail_cluster)
    assert not graph_module.refresh_graph(log)
    assert marker.read_text(encoding="utf-8") == "previous graph"
    assert private_path not in str(log.mock_calls)

    monkeypatch.setattr(graph_module.subprocess, "run", _fake_graphify_run)
    real_replace = os.replace
    failed_publish = False

    def fail_first_publish(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        nonlocal failed_publish
        if Path(source).name == "graphify-out" and Path(destination) == output_dir:
            failed_publish = True
            raise OSError("private publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(graph_module.os, "replace", fail_first_publish)
    assert not graph_module.refresh_graph(log)
    assert failed_publish
    assert marker.read_text(encoding="utf-8") == "previous graph"
    assert not (output_dir.parent / ".graphify-previous").exists()
