"""Acceptance for private, version-pinned corpus preparation."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from runpy import run_path
from typing import cast

from inference_lab.provenance import CorpusId, CorpusVersionId
from tests.inference_lab.serving.operator_fixtures import (  # noqa: TID252
    LOCAL_TOKENISER_FIXTURE,
    run_example,
    write_config,
)

_SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/inference_lab/evaluation/prepare_local_evaluation_corpus.py"
)


def test_preparation_ingests_an_explicit_directory_without_leaking_payloads(
    tmp_path: Path,
) -> None:
    """One private directory becomes one corpus version with bounded output."""
    write_config(tmp_path)
    source = tmp_path / "approved-public-corpus"
    source.mkdir()
    document_text = "SARM1 literature passage " * 180
    (source / "paper.txt").write_text(document_text, encoding="utf-8")
    corpus_id = CorpusId.generate().as_str()
    version = CorpusVersionId.generate().as_str()

    output, stderr = run_example(
        tmp_path,
        _SCRIPT,
        {
            "EVALUATION_CORPUS_SOURCE_DIRECTORY": str(source),
            "EVALUATION_CORPUS_TOKENISER_DIRECTORY": str(LOCAL_TOKENISER_FIXTURE),
            "EVALUATION_CORPUS_ID": corpus_id,
            "EVALUATION_CORPUS_PROPOSED_VERSION": version,
        },
    )

    assert output == [
        {
            "corpus_id": corpus_id,
            "document_outcome_count": 1,
            "execution_status": "completed",
            "failure_categories": [],
            "proposed_version": version,
            "publication_status": "committed",
        }
    ]
    safe_output = json.dumps(output) + stderr
    assert str(source) not in safe_output
    assert document_text not in safe_output
    assert "paper.txt" not in safe_output


def test_preparation_fails_closed_when_config_is_absent(
    capsys,
) -> None:
    """Missing deployment configuration produces one fixed safe message."""
    namespace = run_path(str(_SCRIPT), run_name="prepare_local_corpus_script")
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])
    assert main(None) == 1
    assert capsys.readouterr().err.strip() == (
        "local evaluation corpus preparation failed"
    )
