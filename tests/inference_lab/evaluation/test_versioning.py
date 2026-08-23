"""Tests for evaluation schema-version handling."""

import pytest

from inference_lab.evaluation.versioning import (
    CURRENT_SCHEMA_VERSION,
    CURRENT_SPLIT_ALGORITHM_VERSION,
    validate_case_payload_version,
    validate_manifest_payload_version,
    validate_schema_version,
    validate_split_algorithm_version,
)


def test_current_payload_versions_are_accepted_unchanged() -> None:
    case = {"schema_version": CURRENT_SCHEMA_VERSION, "case_id": "case-001"}
    manifest = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "dataset_id": "baseline-v1",
    }

    assert validate_case_payload_version(case) is case
    assert validate_manifest_payload_version(manifest) is manifest
    assert validate_schema_version(CURRENT_SCHEMA_VERSION) == CURRENT_SCHEMA_VERSION


@pytest.mark.parametrize(
    "validate",
    [validate_case_payload_version, validate_manifest_payload_version],
)
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": "9.0.0"},
        {"schema_version": "1.0.0"},
        {"schema_version": [CURRENT_SCHEMA_VERSION]},
    ],
)
def test_missing_malformed_or_unsupported_schema_versions_are_rejected(
    validate,
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation schema version"):
        validate(payload)


@pytest.mark.parametrize(
    "validate",
    [validate_case_payload_version, validate_manifest_payload_version],
)
def test_non_object_payloads_are_rejected(validate) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate([])


def test_split_algorithm_versions_are_explicit() -> None:
    assert (
        validate_split_algorithm_version(CURRENT_SPLIT_ALGORITHM_VERSION)
        == CURRENT_SPLIT_ALGORITHM_VERSION
    )

    with pytest.raises(ValueError, match="Unsupported split algorithm version"):
        validate_split_algorithm_version("9.0.0")

    with pytest.raises(ValueError, match="Unsupported split algorithm version"):
        validate_split_algorithm_version([CURRENT_SPLIT_ALGORITHM_VERSION])
