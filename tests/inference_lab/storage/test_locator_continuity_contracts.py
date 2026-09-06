"""Tests for private source-locator continuity value contracts."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    HashAlgorithm,
    SourceObservationId,
)
from inference_lab.storage import (
    SourceFileIdentityHint,
    SourceFingerprint,
    SourceLocatorRevision,
)

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_SOURCE_ID = SourceObservationId("22222222-2222-4222-8222-222222222222")
_CONTENT_HASH = ContentHash(HashAlgorithm.SHA256, "a" * 64)


def _fingerprint() -> SourceFingerprint:
    return SourceFingerprint(
        content_hash=_CONTENT_HASH,
        byte_count=42,
        modified_time_ns=123_456_789,
    )


def _revision() -> SourceLocatorRevision:
    return SourceLocatorRevision(
        corpus_id=_CORPUS_ID,
        source_observation_id=_SOURCE_ID,
        locator_revision=1,
        path=Path("/private/corpus/source.txt"),
        identity_hint=SourceFileIdentityHint(device_id=7, file_id=11),
        fingerprint=_fingerprint(),
    )


def test_locator_revision_retains_typed_continuity_evidence() -> None:
    revision = _revision()

    assert revision.corpus_id == _CORPUS_ID
    assert revision.source_observation_id == _SOURCE_ID
    assert revision.locator_revision == 1
    assert revision.path == Path("/private/corpus/source.txt")
    assert revision.identity_hint == SourceFileIdentityHint(7, 11)
    assert revision.fingerprint == _fingerprint()


def test_filesystem_identity_hint_is_optional_and_never_defines_identity() -> None:
    revision = replace(_revision(), identity_hint=None)

    assert revision.identity_hint is None
    assert revision.source_observation_id == _SOURCE_ID


def test_locator_contracts_are_immutable_and_hide_sensitive_evidence() -> None:
    private_path = "/private/confidential-client/source.txt"
    private_digest = "a" * 64
    revision = replace(_revision(), path=Path(private_path))

    with pytest.raises(FrozenInstanceError):
        revision.locator_revision = 2  # type: ignore[misc]

    rendered = repr((revision, revision.fingerprint))
    assert private_path not in rendered
    assert private_digest not in rendered
    assert "path=<private>" in rendered
    assert "content_hash=<private>" in rendered


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("device_id", True, TypeError),
        ("file_id", -1, ValueError),
        ("device_id", 2**63, ValueError),
    ],
)
def test_filesystem_identity_hint_rejects_unpersistable_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {"device_id": 1, "file_id": 2}
    values[field_name] = value

    with pytest.raises(error_type, match=field_name):
        SourceFileIdentityHint(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("content_hash", object(), TypeError),
        ("byte_count", True, TypeError),
        ("byte_count", -1, ValueError),
        ("modified_time_ns", 2**63, ValueError),
    ],
)
def test_fingerprint_rejects_invalid_or_unpersistable_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "content_hash": _CONTENT_HASH,
        "byte_count": 1,
        "modified_time_ns": 2,
    }
    values[field_name] = value

    with pytest.raises(error_type, match=field_name):
        SourceFingerprint(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("corpus_id", object(), TypeError),
        ("source_observation_id", object(), TypeError),
        ("locator_revision", True, TypeError),
        ("locator_revision", 0, ValueError),
        ("path", "/private/source.txt", TypeError),
        ("path", Path("relative/source.txt"), ValueError),
        ("path", Path("/private/../source.txt"), ValueError),
        ("identity_hint", object(), TypeError),
        ("fingerprint", object(), TypeError),
    ],
)
def test_locator_revision_rejects_invalid_boundary_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "corpus_id": _CORPUS_ID,
        "source_observation_id": _SOURCE_ID,
        "locator_revision": 1,
        "path": Path("/private/source.txt"),
        "identity_hint": None,
        "fingerprint": _fingerprint(),
    }
    values[field_name] = value

    with pytest.raises(error_type, match=field_name):
        SourceLocatorRevision(**values)  # type: ignore[arg-type]
