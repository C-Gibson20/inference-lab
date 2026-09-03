"""Tests for profiling record schema-version policy."""

import pytest

from inference_lab.profiling import (
    CURRENT_RECORD_SCHEMA_VERSION,
    SUPPORTED_RECORD_SCHEMA_VERSIONS,
)
from inference_lab.profiling.versioning import validate_record_schema_version


def test_current_schema_version_is_supported() -> None:
    validate_record_schema_version(CURRENT_RECORD_SCHEMA_VERSION)

    assert SUPPORTED_RECORD_SCHEMA_VERSIONS == frozenset(
        {CURRENT_RECORD_SCHEMA_VERSION}
    )


@pytest.mark.parametrize(
    "version",
    ["1", "1.0", "01.0.0", "3.0.0-01", "9.0.0"],
)
def test_unsupported_schema_version_is_rejected(version: object) -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        validate_record_schema_version(version)


@pytest.mark.parametrize("version", [0, True, 1.5, None])
def test_schema_version_requires_a_string(version: object) -> None:
    with pytest.raises(TypeError, match="record_schema_version must be a string"):
        validate_record_schema_version(version)
