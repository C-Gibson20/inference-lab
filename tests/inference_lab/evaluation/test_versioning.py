"""Tests for evaluation schema-version handling."""

import pytest

from inference_lab.evaluation.versioning import (
    CURRENT_SCHEMA_VERSION,
    migrate_case_payload,
    migrate_manifest_payload,
)


def test_current_case_payload_is_accepted_unchanged() -> None:
    payload = {"schema_version": CURRENT_SCHEMA_VERSION, "case_id": "case-001"}

    assert migrate_case_payload(payload) is payload


def test_current_manifest_payload_is_accepted_unchanged() -> None:
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "dataset_id": "baseline-v1",
    }

    assert migrate_manifest_payload(payload) is payload


@pytest.mark.parametrize("migrate", [migrate_case_payload, migrate_manifest_payload])
def test_missing_or_unsupported_schema_versions_are_rejected(migrate) -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation schema version"):
        migrate({})

    with pytest.raises(ValueError, match="Unsupported evaluation schema version"):
        migrate({"schema_version": "9.0.0"})


@pytest.mark.parametrize("migrate", [migrate_case_payload, migrate_manifest_payload])
def test_non_object_payloads_are_rejected(migrate) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        migrate([])  # type: ignore[arg-type]
