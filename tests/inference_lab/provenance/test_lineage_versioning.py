"""Tests for derived-record lineage schema-version policy."""

import pytest

from inference_lab.provenance.errors import (
    InvalidLineageError,
    UnsupportedLineageSchemaVersionError,
)
from inference_lab.provenance.versioning import (
    CURRENT_LINEAGE_SCHEMA_VERSION,
    SUPPORTED_LINEAGE_SCHEMA_VERSIONS,
    validate_lineage_schema_version,
)


def test_current_lineage_schema_version_is_supported() -> None:
    assert (
        validate_lineage_schema_version(CURRENT_LINEAGE_SCHEMA_VERSION)
        == CURRENT_LINEAGE_SCHEMA_VERSION
    )
    assert SUPPORTED_LINEAGE_SCHEMA_VERSIONS == frozenset(
        {CURRENT_LINEAGE_SCHEMA_VERSION}
    )


@pytest.mark.parametrize("version", [None, True, 1, 1.0, ["1.0.0"]])
def test_lineage_schema_version_rejects_wrong_runtime_types(version: object) -> None:
    with pytest.raises(TypeError, match="lineage_schema_version must be a string"):
        validate_lineage_schema_version(version)


@pytest.mark.parametrize("version", ["", "1", "1.0", "1.0.0+local", "2.0.0"])
def test_lineage_schema_version_rejects_unsupported_values(version: str) -> None:
    with pytest.raises(
        UnsupportedLineageSchemaVersionError,
        match="unsupported lineage_schema_version",
    ) as captured:
        validate_lineage_schema_version(version)

    assert isinstance(captured.value, InvalidLineageError)


def test_unsupported_lineage_schema_error_does_not_echo_rejected_value() -> None:
    rejected_value = "private-version-marker"

    with pytest.raises(UnsupportedLineageSchemaVersionError) as captured:
        validate_lineage_schema_version(rejected_value)

    assert rejected_value not in str(captured.value)
