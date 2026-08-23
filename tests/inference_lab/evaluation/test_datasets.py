"""Tests for evaluation dataset manifests."""

import pytest
from pydantic import ValidationError

from inference_lab.evaluation.datasets import DatasetManifest


BASE = {
    "dataset_id": "baseline-v1",
    "dataset_version": "1",
    "description": "Evaluation fixture",
    "corpus_id": "00000000-0000-4000-8000-000000000001",
    "corpus_version": "00000000-0000-4000-8000-000000000002",
    "case_file": "cases.jsonl",
    "case_count": 2,
    "splits": {
        "development": ("case-001",),
        "held_out": ("case-002",),
    },
    "created_at": "2026-08-22T00:00:00Z",
    "updated_at": "2026-08-22T00:00:00Z",
}


def test_manifest_accepts_explicit_disjoint_splits() -> None:
    manifest = DatasetManifest(**BASE)

    assert manifest.case_count == 2


def test_manifest_rejects_duplicate_split_membership() -> None:
    invalid = dict(BASE)
    invalid["splits"] = {
        "development": ("case-001",),
        "held_out": ("case-001",),
    }

    with pytest.raises(ValidationError, match="exactly one split"):
        DatasetManifest(**invalid)


def test_manifest_rejects_unsupported_schema_version() -> None:
    invalid = dict(BASE, schema_version="9.0.0")

    with pytest.raises(ValidationError, match="Unsupported evaluation schema"):
        DatasetManifest(**invalid)


def test_manifest_rejects_coerced_counts_and_reversed_timestamps() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest(**dict(BASE, case_count=True))

    invalid = dict(BASE, updated_at="2026-08-21T23:59:59Z")
    with pytest.raises(ValidationError, match="earlier than created_at"):
        DatasetManifest(**invalid)


def test_manifest_rejects_blank_fields_and_incomplete_split_inventory() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        DatasetManifest(**dict(BASE, description=" "))

    missing_held_out = dict(BASE, splits={"development": ("case-001", "case-002")})
    with pytest.raises(ValidationError, match="include 'held_out'"):
        DatasetManifest(**missing_held_out)

    wrong_count = dict(BASE, case_count=3)
    with pytest.raises(ValidationError, match="case_count"):
        DatasetManifest(**wrong_count)
