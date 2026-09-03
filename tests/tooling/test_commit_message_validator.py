"""Tests for commit-message validation."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "validate-commit-msg.sh"

VALID_MESSAGES = [
    "[retrieval] fix: correct off-by-one in chunk overlap",
    "[ingest] refactor: simplify document loader interface",
    "[storage] test: add fixture for atomic version advancement",
    "[runtime] perf: reduce KV-cache allocation overhead",
    "[serving] feat!: change citation response schema",
    "[provenance] feat: preserve source spans",
    "[evaluation] test: add retrieval recall case",
    "[profiling] feat: record stage timing",
    "[docs] docs: update README with setup instructions",
    "[tooling] chore: bump ruff version",
    "[tooling] build: add uv lockfile check",
    "[tooling] ci: add coverage reporting step",
]

MERGE_MESSAGES = [
    "Merge pull request #43 from C-Gibson20/feature/stor-01-05-acceptance",
    "Merge pull request #42 from C-Gibson20/feature/stor-01-04-relationships",
    "Merge pull request #41 from C-Gibson20/feature/stor-01-03-identity",
    "Merge pull request #40 from C-Gibson20/feature/stor-01-02-schema",
    "Merge pull request #39 from C-Gibson20/feature/stor-01-01-contracts",
    "Merge branch 'feature/stor-01' into feature/storage",
    "Merge remote-tracking branch 'origin/main' into feature/storage",
]

INVALID_MESSAGES = [
    "add raw document loader",
    "[ingest] add raw document loader",
    "ingest: add raw document loader",
    "[ingest] feature: add raw document loader",
    "[nonexistent] feat: add something",
    "[ingest] feat:",
    "[ingest]feat: no space after bracket",
    "[Ingest] feat: wrong case module",
    "[ingest] FEAT: uppercase type",
    "Merge",
    "merge pull request #43 from example/branch",
]


def run_validator(message: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), "--stdin"],
        input=f"{message}\n",
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("message", VALID_MESSAGES)
def test_valid_commit_messages_pass(message):
    result = run_validator(message)

    assert result.returncode == 0, (
        f"expected valid message to pass: {message!r}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize("message", MERGE_MESSAGES)
def test_generated_merge_commit_messages_pass(message):
    result = run_validator(message)

    assert result.returncode == 0, (
        f"expected merge message to pass: {message!r}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize("message", INVALID_MESSAGES)
def test_invalid_commit_messages_fail(message):
    result = run_validator(message)

    assert result.returncode == 1, f"expected invalid message to fail: {message!r}"


def test_file_mode_validates_only_the_subject(tmp_path):
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(
        "[docs] docs: update workflow notes\n\nThis is the commit body.\n"
    )

    result = subprocess.run(
        [str(SCRIPT), "--file", str(message_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_file_mode_rejects_invalid_subject(tmp_path):
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text("not a valid commit subject\n")

    result = subprocess.run(
        [str(SCRIPT), "--file", str(message_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Invalid commit message" in result.stderr


def test_file_mode_rejects_empty_subject(tmp_path):
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text("")

    result = subprocess.run(
        [str(SCRIPT), "--file", str(message_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Invalid commit message" in result.stderr


def test_stdin_mode_accepts_multiple_valid_messages():
    messages = "\n".join([*VALID_MESSAGES, *MERGE_MESSAGES]) + "\n"
    result = subprocess.run(
        [str(SCRIPT), "--stdin"],
        input=messages,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_stdin_mode_accepts_unterminated_final_line():
    result = subprocess.run(
        [str(SCRIPT), "--stdin"],
        input=VALID_MESSAGES[0],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_stdin_mode_rejects_if_any_message_invalid():
    messages = "\n".join(VALID_MESSAGES + ["not a valid message"]) + "\n"
    result = subprocess.run(
        [str(SCRIPT), "--stdin"],
        input=messages,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1


def test_stdin_mode_ignores_blank_lines():
    messages = "\n".join([VALID_MESSAGES[0], "", VALID_MESSAGES[1], ""]) + "\n"
    result = subprocess.run(
        [str(SCRIPT), "--stdin"],
        input=messages,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("args", "expected_returncode"),
    [
        (("--help",), 0),
        (("--stdin", "unexpected"), 2),
        (("--file",), 2),
        (("--file", "/does/not/exist"), 2),
        ((), 2),
    ],
)
def test_cli_errors_and_help(args, expected_returncode):
    result = subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_returncode


def test_script_is_executable():
    assert SCRIPT.exists(), f"script not found at {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "script is not executable (chmod +x)"
