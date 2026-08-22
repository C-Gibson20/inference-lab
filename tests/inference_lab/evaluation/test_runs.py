"""Tests for reproducible evaluation-run identities and governance."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.loader import load_dataset
from inference_lab.evaluation.runs import (
    EvaluationRunRequest,
    configuration_digest,
    create_evaluation_run,
)


FIXTURE_MANIFEST = (
    Path(__file__).parents[3] / "benchmarks" / "evaluation" / "manifest.json"
)
CREATED_AT = datetime(2026, 8, 22, tzinfo=timezone.utc)


def test_configuration_digest_is_order_independent_and_safe() -> None:
    assert configuration_digest({"temperature": 0, "top_k": 5}) == configuration_digest(
        {"top_k": 5, "temperature": 0}
    )

    with pytest.raises(ValueError, match="finite JSON"):
        configuration_digest({"temperature": float("nan")})


@pytest.mark.parametrize(
    "payload",
    [
        {"split": "held_out", "purpose": "development", "tuning": True},
        {
            "split": "held_out",
            "purpose": "development",
            "model_selection": True,
        },
        {"split": "development", "purpose": "final"},
    ],
)
def test_run_request_rejects_invalid_governance(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EvaluationRunRequest.model_validate(payload)


def test_create_run_derives_digests_and_excludes_evaluation_content() -> None:
    dataset = load_dataset(FIXTURE_MANIFEST)
    request = EvaluationRunRequest(split="held_out", purpose="final")
    digest = configuration_digest({"model": "example", "temperature": 0})

    first = create_evaluation_run(
        dataset,
        request,
        system_version="system-1",
        configuration_digest=digest,
        algorithm_versions={"matching": "1.0.0"},
        created_at=CREATED_AT,
    )
    second = create_evaluation_run(
        dataset,
        request,
        system_version="system-1",
        configuration_digest=digest,
        algorithm_versions={"matching": "1.0.0"},
        created_at=CREATED_AT.replace(hour=1),
    )

    assert first.run_id == second.run_id
    assert first.case_file_digest == dataset.manifest.checksums["cases.jsonl"]
    assert "question" not in first.model_dump()
    assert "expected_answer" not in first.model_dump()


def test_create_run_rejects_asserted_digest_that_does_not_match_dataset() -> None:
    dataset = load_dataset(FIXTURE_MANIFEST)
    request = EvaluationRunRequest(split="development", purpose="development")

    with pytest.raises(ValueError, match="case_file_digest"):
        create_evaluation_run(
            dataset,
            request,
            system_version="system-1",
            configuration_digest="sha256:" + "0" * 64,
            algorithm_versions={"matching": "1.0.0"},
            case_file_digest="sha256:" + "0" * 64,
            created_at=CREATED_AT,
        )
