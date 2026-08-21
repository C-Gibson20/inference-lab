import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
PRE_COMMIT = REPO_ROOT / ".githooks" / "pre-commit"
COMMIT_MSG = REPO_ROOT / ".githooks" / "commit-msg"
SETUP_HOOKS = REPO_ROOT / "scripts" / "setup-hooks.sh"


def run_script(
    script: Path,
    *args: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_pre_commit_runs_lint_and_format_checks(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "uv-calls"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n")
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = run_script(PRE_COMMIT, env=env)

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        "run ruff check .",
        "run ruff format --check .",
    ]


def test_commit_msg_hook_validates_message_file(tmp_path: Path) -> None:
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text("[tooling] test: exercise commit hook\n")

    result = run_script(COMMIT_MSG, str(message_file))

    assert result.returncode == 0, result.stderr


def test_commit_msg_hook_rejects_invalid_message_file(tmp_path: Path) -> None:
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text("not a valid commit message\n")

    result = run_script(COMMIT_MSG, str(message_file))

    assert result.returncode == 1
    assert "Invalid commit message" in result.stderr


def test_setup_hooks_configures_versioned_hooks_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    nested_directory = repository / "nested"
    nested_directory.mkdir()

    result = run_script(SETUP_HOOKS, cwd=nested_directory)

    assert result.returncode == 0, result.stderr
    configured_path = subprocess.run(
        ["git", "config", "--local", "core.hooksPath"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert configured_path.stdout.strip() == ".githooks"
