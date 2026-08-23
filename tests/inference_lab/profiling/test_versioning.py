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


@pytest.mark.parametrize("version", [0, -1, 1, 999, True, 1.5, "1"])
def test_unsupported_schema_version_is_rejected(version: object) -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        validate_record_schema_version(version)
