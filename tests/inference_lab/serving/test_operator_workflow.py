"""SERV-01/ING-05/RET-02/RET-03 documented API workflow across fresh processes."""

import json
import subprocess
import sys
from pathlib import Path

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import open_metadata_repository
from tests.inference_lab.serving.operator_fixtures import (  # noqa: TID252
    LOCAL_TOKENISER_FIXTURE,
    PROJECT_ROOT,
    environment,
    example,
    run_example,
    write_config,
)


def test_documented_api_create_ingest_replay_update_and_filter(
    tmp_path: Path,
) -> None:
    """Prove every published operator step works after a process restart."""
    write_config(tmp_path)
    create_script = tmp_path / "create-corpus.py"
    ingest_script = tmp_path / "ingest-and-search.py"
    create_script.write_text(example("create"), encoding="utf-8")
    ingest_script.write_text(example("ingest"), encoding="utf-8")
    source = tmp_path / "paper.txt"
    source.write_text("alpha method " * 180, encoding="utf-8")

    created, create_logs = run_example(
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

    first, logs = run_example(tmp_path, ingest_script, variables)
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

    replayed, _ = run_example(tmp_path, ingest_script, variables)
    assert replayed[0]["publication_status"] == "committed"
    assert replayed[0]["document_outcomes"] == 0
    assert replayed[1] == first[1]

    empty, _ = run_example(
        tmp_path,
        ingest_script,
        {**variables, "DOCUMENT_TYPE": "application/pdf"},
    )
    assert empty[1]["hits"] == []

    source.write_text("beta method " * 180, encoding="utf-8")
    second_version = CorpusVersionId.generate().as_str()
    updated, _ = run_example(
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
        env=environment({}),
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
        env=environment({}),
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
