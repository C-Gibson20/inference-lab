"""Execute the documented serving API workflow across fresh processes."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import open_metadata_repository

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUERYING_RUNBOOK = PROJECT_ROOT / "docs" / "operator" / "querying.md"
LOCAL_TOKENISER_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "backends" / "runtime" / "local_mechanics"
)


def _example(name: str) -> str:
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


def _write_config(tmp_path: Path) -> None:
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


def _run_example(
    tmp_path: Path,
    script: Path,
    variables: dict[str, str],
) -> tuple[list[dict[str, Any]], str]:
    """Run one documented program with a fresh interpreter and deployment."""
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_environment(variables),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert lines
    return [json.loads(line) for line in lines], completed.stderr


def _environment(variables: dict[str, str]) -> dict[str, str]:
    """Build the same importable fresh-process environment for examples."""
    environment = os.environ.copy()
    environment.update(variables)
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH", "")) if part
    )
    return environment


def test_documented_api_create_ingest_replay_update_and_filter(
    tmp_path: Path,
) -> None:
    """Prove every published operator step works after a process restart."""
    _write_config(tmp_path)
    create_script = tmp_path / "create-corpus.py"
    ingest_script = tmp_path / "ingest-and-search.py"
    create_script.write_text(_example("create"), encoding="utf-8")
    ingest_script.write_text(_example("ingest"), encoding="utf-8")
    source = tmp_path / "paper.txt"
    source.write_text("alpha method " * 180, encoding="utf-8")

    created, create_logs = _run_example(
        tmp_path, create_script, {"CORPUS_NAME": "First corpus"}
    )
    assert len(created) == 1
    corpus_id = str(created[0]["corpus_id"])
    assert "First corpus" not in create_logs
    first_version = CorpusVersionId.generate().as_str()
    variables = {
        "CORPUS_ID": corpus_id,
        "SOURCE_FILE": str(source),
        "TOKENISER_DIR": str(LOCAL_TOKENISER_FIXTURE),
        "PROPOSED_VERSION": first_version,
        "QUERY": "alpha method",
    }

    first, logs = _run_example(tmp_path, ingest_script, variables)
    assert len(first) == 2
    assert first[0]["execution_status"] == "completed"
    assert first[0]["publication_status"] == "committed"
    assert first[0]["document_outcomes"] == 1
    assert first[1]["resolved_version"] == first_version
    assert first[1]["hits"]
    locations = first[1]["hits"][0]["source_locations"]
    assert locations[0]["source_span_id"]
    assert locations[0]["parsed_record_id"]
    assert locations[0]["end_offset"] > locations[0]["start_offset"]
    assert str(source) not in logs
    assert "alpha method" not in logs

    replayed, _ = _run_example(tmp_path, ingest_script, variables)
    assert replayed[0]["publication_status"] == "committed"
    assert replayed[0]["document_outcomes"] == 0
    assert replayed[1] == first[1]

    empty, _ = _run_example(
        tmp_path,
        ingest_script,
        {**variables, "DOCUMENT_TYPE": "application/pdf"},
    )
    assert empty[1]["hits"] == []

    source.write_text("beta method " * 180, encoding="utf-8")
    second_version = CorpusVersionId.generate().as_str()
    updated, _ = _run_example(
        tmp_path,
        ingest_script,
        {
            **variables,
            "PROPOSED_VERSION": second_version,
            "EXPECTED_VERSION": first_version,
            "QUERY": "beta method",
        },
    )
    assert updated[0]["publication_status"] == "committed"
    assert updated[1]["resolved_version"] == second_version
    assert updated[1]["hits"]

    search_script = PROJECT_ROOT / "scripts" / "inference_lab" / "serving-search.py"
    searched = subprocess.run(
        [
            sys.executable,
            str(search_script),
            "--corpus-id",
            corpus_id,
            "--query",
            "beta method",
        ],
        cwd=tmp_path,
        env=_environment({}),
        capture_output=True,
        check=False,
        text=True,
    )
    assert searched.returncode == 0, searched.stderr
    assert len(searched.stdout.strip().splitlines()) == 1
    assert json.loads(searched.stdout)["hits"]
    assert '"event":' in searched.stderr
    assert "beta method" not in searched.stderr
    assert str(source) not in searched.stderr

    invalid = subprocess.run(
        [
            sys.executable,
            str(search_script),
            "--corpus-id",
            "invalid-id",
            "--query",
            "private query",
        ],
        cwd=tmp_path,
        env=_environment({}),
        capture_output=True,
        check=False,
        text=True,
    )
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert json.loads(invalid.stderr.strip()) == {"error": "invalid_request"}
    assert "private query" not in invalid.stderr

    repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3", source_dir=tmp_path / "source-data"
    )
    try:
        assert len(repository.list_corpus_versions(CorpusId.from_str(corpus_id))) == 2
    finally:
        repository.close()
