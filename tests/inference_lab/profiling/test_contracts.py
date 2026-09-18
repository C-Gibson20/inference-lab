"""Tests for the closed baseline stage and condition contracts."""

import pytest

from inference_lab.profiling import (
    BASELINE_STAGES,
    STAGE_CONDITIONS,
    validate_baseline_stage,
    validate_stage_condition,
)


def test_baseline_stage_vocabulary_is_closed() -> None:
    assert BASELINE_STAGES == {
        "discovery",
        "source_acquisition",
        "parse",
        "parse_persistence",
        "chunk",
        "chunk_persistence",
        "membership",
        "publication",
        "lexical_index",
        "lexical_retrieval",
        "metadata_filter",
        "serving",
    }


@pytest.mark.parametrize("condition", ["cold", "warm", "uncontrolled"])
def test_stage_condition_accepts_only_declared_values(condition: str) -> None:
    assert validate_stage_condition(condition) == condition


@pytest.mark.parametrize("value", ["hot", "", None, 1])
def test_stage_condition_rejects_undeclared_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="condition"):
        validate_stage_condition(value)


def test_stage_validator_rejects_query_payloads_and_unknown_stages() -> None:
    with pytest.raises(ValueError, match="supported baseline stage"):
        validate_baseline_stage("parse?query=secret")

    with pytest.raises(ValueError, match="supported baseline stage"):
        validate_baseline_stage("unknown_stage")


def test_condition_vocabulary_is_exported_and_closed() -> None:
    assert STAGE_CONDITIONS == {"cold", "warm", "uncontrolled"}
