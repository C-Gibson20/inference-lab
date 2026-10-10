"""Decoding tests for the approved held-out retrieval gate artefact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from inference_lab._json import JsonFieldsError, JsonStructureError
from inference_lab.evaluation import (
    HELD_OUT_UNCERTAINTY_METHOD,
    MetricFraction,
    deserialise_approved_held_out_gate,
)
from inference_lab.evaluation.retrieval_contracts import (
    RetrievalEvaluationContractError,
)

_ARTEFACT = (
    Path(__file__).resolve().parents[3] / "docs/operator/retrieval-baseline-gate.json"
)


def _artefact_payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_shipped_operator_artefact_decodes_to_the_approved_gate() -> None:
    approved = deserialise_approved_held_out_gate(_ARTEFACT.read_bytes())

    assert approved.binding.dataset_id == "sarm1-retrieval-question-set"
    assert approved.binding.case_file_checksum.startswith("sha256:")
    assert approved.policy.minimum_document_recall_at_10 == MetricFraction(9, 10)
    assert approved.policy.maximum_failure_rate == MetricFraction(1, 20)
    assert approved.policy.maximum_degraded_rate == MetricFraction(1, 20)
    assert approved.policy.minimum_category_cases == 3
    assert approved.policy.minimum_category_leakage_groups == 3
    assert approved.policy.confidence_level_percent == 95
    assert approved.policy.bootstrap_replicates == 10000
    assert approved.policy.uncertainty_method == HELD_OUT_UNCERTAINTY_METHOD
    assert approved.policy.rationale.startswith("The product acceptance floor")


def _without_rationale(payload: dict[str, Any]) -> None:
    del payload["rationale"]


def _with_unknown_field(payload: dict[str, Any]) -> None:
    payload["minimum_fused_recall"] = {"numerator": 1, "denominator": 2}


def _with_unknown_coverage_field(payload: dict[str, Any]) -> None:
    payload["category_coverage"]["minimum_families"] = 2


def _with_future_schema_version(payload: dict[str, Any]) -> None:
    payload["schema_version"] = "2.0.0"


def _with_development_split(payload: dict[str, Any]) -> None:
    payload["split"] = "development"


def _with_post_rerank_channel(payload: dict[str, Any]) -> None:
    payload["primary_gate"]["channel"] = "reranked"


def _with_relaxed_cutoff(payload: dict[str, Any]) -> None:
    payload["primary_gate"]["cutoff"] = 100


def _with_silent_denominator_change(payload: dict[str, Any]) -> None:
    payload["primary_gate"]["denominator"] = "relevant targets in successful cases"


def _with_unapproved_uncertainty_method(payload: dict[str, Any]) -> None:
    payload["uncertainty"]["method"] = "normal-approximation"


def _with_string_threshold(payload: dict[str, Any]) -> None:
    payload["primary_gate"]["minimum"]["numerator"] = "9"


def _with_unprefixed_checksum(payload: dict[str, Any]) -> None:
    payload["dataset"]["case_file_checksum"] = "e949" * 16


def _with_impossible_threshold(payload: dict[str, Any]) -> None:
    payload["primary_gate"]["minimum"] = {"numerator": 11, "denominator": 10}


@pytest.mark.parametrize(
    ("corrupt", "expected_error"),
    [
        (_without_rationale, JsonFieldsError),
        (_with_unknown_field, JsonFieldsError),
        (_with_unknown_coverage_field, JsonFieldsError),
        (_with_future_schema_version, RetrievalEvaluationContractError),
        (_with_development_split, RetrievalEvaluationContractError),
        (_with_post_rerank_channel, RetrievalEvaluationContractError),
        (_with_relaxed_cutoff, RetrievalEvaluationContractError),
        (_with_silent_denominator_change, RetrievalEvaluationContractError),
        (_with_unapproved_uncertainty_method, RetrievalEvaluationContractError),
        (_with_string_threshold, JsonStructureError),
        (_with_unprefixed_checksum, RetrievalEvaluationContractError),
        (_with_impossible_threshold, RetrievalEvaluationContractError),
    ],
    ids=lambda value: value.__name__.lstrip("_") if callable(value) else value.__name__,
)
def test_gate_artefact_rejects_unapproved_or_malformed_policies(
    corrupt: Callable[[dict[str, Any]], None],
    expected_error: type[Exception],
) -> None:
    payload = _artefact_payload()
    corrupt(payload)

    with pytest.raises(expected_error):
        deserialise_approved_held_out_gate(json.dumps(payload).encode("utf-8"))
