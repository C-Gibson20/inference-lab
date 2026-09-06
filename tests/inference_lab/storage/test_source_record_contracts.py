"""Tests for versioned managed source-artefact contracts."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.storage import (
    CURRENT_SOURCE_RECORD_SCHEMA_VERSION,
    SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS,
    SourceArtefact,
    validate_source_record_schema_version,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
)


def _source_artefact(**changes: object) -> SourceArtefact:
    values: dict[str, object] = {
        "document_id": DOCUMENT_ID,
        "document_version_id": DOCUMENT_VERSION_ID,
        "content_hash": SOURCE_HASH,
        "byte_count": len(SOURCE_BYTES),
        "media_type": "text/plain",
        "storage_key": f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
    }
    values.update(changes)
    return SourceArtefact(**values)  # type: ignore[arg-type]


def test_source_record_schema_version_is_explicit_and_supported() -> None:
    assert CURRENT_SOURCE_RECORD_SCHEMA_VERSION == "1.0.0"
    assert SUPPORTED_SOURCE_RECORD_SCHEMA_VERSIONS == frozenset({"1.0.0"})
    assert validate_source_record_schema_version("1.0.0") == "1.0.0"


@pytest.mark.parametrize("value", [None, 1, True])
def test_source_record_schema_version_rejects_wrong_types(value: object) -> None:
    with pytest.raises(TypeError, match="record_schema_version"):
        validate_source_record_schema_version(value)


def test_source_record_schema_version_rejects_unsupported_versions() -> None:
    with pytest.raises(ValueError, match="not supported"):
        validate_source_record_schema_version("2.0.0")


def test_source_artefact_retains_managed_metadata_without_source_path() -> None:
    artefact = _source_artefact()

    assert artefact.content_hash == SOURCE_HASH
    assert artefact.byte_count == len(SOURCE_BYTES)
    assert artefact.record_schema_version == CURRENT_SOURCE_RECORD_SCHEMA_VERSION
    assert not hasattr(artefact, "source_path")

    with pytest.raises(FrozenInstanceError):
        artefact.byte_count = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "storage_key",
    ["", "/absolute/blob", "../blob", "sha256/../blob", "sha256\\blob", "./blob"],
)
def test_source_artefact_rejects_non_canonical_storage_keys(
    storage_key: str,
) -> None:
    with pytest.raises(ValueError, match="storage_key"):
        _source_artefact(storage_key=storage_key)


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"byte_count": -1}, ValueError, "byte_count"),
        ({"byte_count": True}, TypeError, "byte_count"),
        ({"byte_count": 1.5}, TypeError, "byte_count"),
        ({"media_type": "Text/Plain"}, ValueError, "media_type"),
        ({"document_id": "document"}, TypeError, "document_id"),
        ({"record_schema_version": "2.0.0"}, ValueError, "record_schema_version"),
    ],
)
def test_source_artefact_rejects_invalid_fields(
    changes: dict[str, object], error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        _source_artefact(**changes)
