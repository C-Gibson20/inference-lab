"""Fresh-process fixtures for documented local serving workflows."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUERYING_RUNBOOK = PROJECT_ROOT / "docs" / "operator" / "querying.md"
LOCAL_TOKENISER_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "backends" / "runtime" / "local_mechanics"
)


def example(name: str) -> str:
    """Extract the exact executable Python block from the operator runbook."""
    runbook = QUERYING_RUNBOOK.read_text(encoding="utf-8")
    pattern = (
        rf"<!-- operator-{name}-example:start -->\n"
        r"```python\n(.*?)\n```\n"
        rf"<!-- operator-{name}-example:end -->"
    )
    match = re.search(pattern, runbook, flags=re.DOTALL)
    assert match is not None
    return match.group(1) + "\n"


def write_config(tmp_path: Path) -> None:
    """Configure an isolated deployment with operational logs enabled."""
    values = (
        "[models]",
        f"embedder_path = {json.dumps(str(tmp_path / 'models' / 'embedder'))}",
        f"reranker_path = {json.dumps(str(tmp_path / 'models' / 'reranker'))}",
        f"generator_path = {json.dumps(str(tmp_path / 'models' / 'generator'))}",
        "[runtime]",
        "vram_budget_gb = 24",
        "[storage]",
        f"index_dir = {json.dumps(str(tmp_path / 'index'))}",
        f"metadata_db_path = {json.dumps(str(tmp_path / 'metadata.sqlite3'))}",
        f"source_dir = {json.dumps(str(tmp_path / 'source-data'))}",
        "[logging]",
        'level = "INFO"',
        'format = "json"',
    )
    (tmp_path / "config.toml").write_text("\n".join(values) + "\n", encoding="utf-8")


def run_example(
    tmp_path: Path,
    script: Path,
    variables: dict[str, str],
) -> tuple[list[dict[str, Any]], str]:
    """Run one documented program with a fresh interpreter and deployment."""
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment(variables),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert lines
    return [json.loads(line) for line in lines], completed.stderr


def environment(variables: dict[str, str]) -> dict[str, str]:
    """Build the same importable fresh-process environment for examples."""
    result = os.environ.copy()
    result.update(variables)
    source_root = str(PROJECT_ROOT / "src")
    result["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, result.get("PYTHONPATH", "")) if part
    )
    return result
