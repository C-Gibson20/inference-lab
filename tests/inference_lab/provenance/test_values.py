"""Tests for document identity-resolution value types."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance import (
    ContentHash,
    DuplicateClass,
    HashAlgorithm,
    ResolutionKind,
)

VALID_DIGEST = "a" * 64


def test_content_hash_validates_and_serialises() -> None:
    content_hash = ContentHash(HashAlgorithm.SHA256, VALID_DIGEST)

    assert content_hash.as_str() == f"sha256:{VALID_DIGEST}"
    assert ContentHash.from_str(content_hash.as_str()) == content_hash


@pytest.mark.parametrize("algorithm", ["sha256", None, 1])
def test_content_hash_rejects_invalid_algorithm_type(algorithm: object) -> None:
    with pytest.raises(TypeError, match="algorithm"):
        ContentHash(algorithm, VALID_DIGEST)  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", [None, 1, b"a" * 64])
def test_content_hash_rejects_invalid_digest_type(digest: object) -> None:
    with pytest.raises(TypeError, match="digest"):
        ContentHash(HashAlgorithm.SHA256, digest)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "digest",
    ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 63 + "G"],
)
def test_content_hash_rejects_invalid_digest_value(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ContentHash(HashAlgorithm.SHA256, digest)


@pytest.mark.parametrize(
    "value",
    ["sha256", "sha256:", "SHA256:" + VALID_DIGEST, "md5:" + VALID_DIGEST],
)
def test_content_hash_from_str_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        ContentHash.from_str(value)


def test_content_hash_from_str_rejects_non_string_value() -> None:
    with pytest.raises(TypeError, match="value"):
        ContentHash.from_str(None)  # type: ignore[arg-type]


def test_content_hash_is_immutable() -> None:
    content_hash = ContentHash(HashAlgorithm.SHA256, VALID_DIGEST)

    with pytest.raises(FrozenInstanceError):
        content_hash.digest = "b" * 64  # type: ignore[misc]


def test_duplicate_and_resolution_values_are_stable() -> None:
    assert DuplicateClass.BYTE_IDENTICAL.value == "byte_identical"
    assert {kind.value for kind in ResolutionKind} == {
        "unchanged",
        "new_version",
        "new_document",
        "duplicate",
        "ambiguous",
        "conflict",
    }
