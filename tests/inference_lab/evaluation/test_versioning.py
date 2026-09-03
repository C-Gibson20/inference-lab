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
        {"schema_version": "9.0.0"},
        {"schema_version": "1.0.0"},
        {"schema_version": "2.0.0"},
    ],
)
def test_unsupported_schema_versions_are_rejected(
    validate,
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation schema version"):
        validate(payload)


@pytest.mark.parametrize(
    "validate",
    [validate_case_payload_version, validate_manifest_payload_version],
)
@pytest.mark.parametrize(
    "payload",
    [{}, {"schema_version": [CURRENT_SCHEMA_VERSION]}],
)
def test_missing_or_wrong_type_schema_versions_raise_type_error(
    validate,
    payload: object,
) -> None:
    with pytest.raises(TypeError, match="evaluation schema version must be a string"):
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

    with pytest.raises(TypeError, match="split algorithm version must be a string"):
        validate_split_algorithm_version([CURRENT_SPLIT_ALGORITHM_VERSION])


def test_evaluation_schema_version_requires_a_string() -> None:
    with pytest.raises(TypeError, match="evaluation schema version must be a string"):
        validate_schema_version(3)
