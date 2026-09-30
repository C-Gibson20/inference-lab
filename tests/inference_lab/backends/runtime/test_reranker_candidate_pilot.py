"""Validate the local reranker candidate-pilot fixture and metrics."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[4]
    / "scripts"
    / "inference_lab"
    / "backends"
    / "benchmark_local_reranker_candidates.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "reranker_candidate_pilot", _SCRIPT
)
if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:
    raise RuntimeError("candidate-pilot benchmark script is unavailable")
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)


def test_candidate_fixture_is_complete_and_bounded() -> None:
    """The fixed fixture has unique cases and a positive per query."""
    cases = _MODULE.load_fixture(_MODULE._FIXTURE)

    assert len(cases) == 8
    assert sum(len(case.candidates) for case in cases) == 40
    assert all(
        any(candidate.relevant for candidate in case.candidates) for case in cases
    )


def test_ranking_metrics_use_stable_candidate_id_ties() -> None:
    """Equal scores use ascending candidate identity for metric ranks."""
    case = _MODULE.Case(
        "tie",
        "query",
        (
            _MODULE.Candidate("a-negative", "negative", False),
            _MODULE.Candidate("z-positive", "positive", True),
        ),
    )
    scores = {"a-negative": 0.0, "z-positive": 0.0}

    metrics = _MODULE.ranking_metrics((case,), scores)

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_5"] == pytest.approx(1 / math.log2(3))


def test_fixture_rejects_query_without_relevant_candidate(tmp_path: Path) -> None:
    """Invalid relevance labels fail before model loading."""
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        '{"case_id":"case","query":"query","candidates":'
        '[{"candidate_id":"one","text":"passage","relevant":false},'
        '{"candidate_id":"two","text":"other","relevant":false}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no relevant candidate"):
        _MODULE.load_fixture(fixture)
