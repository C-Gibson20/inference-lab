"""Focused persisted content-hash value boundary."""

import pytest

from inference_lab.provenance import ContentHash, HashAlgorithm


def test_content_hash_round_trips_canonically_and_rejects_invalid_values() -> None:
    digest = "a" * 64
    value = ContentHash(HashAlgorithm.SHA256, digest)

    assert value.as_str() == f"sha256:{digest}"
    assert ContentHash.from_str(value.as_str()) == value

    with pytest.raises(ValueError):
        ContentHash(HashAlgorithm.SHA256, "g" * 64)
    with pytest.raises(ValueError):
        ContentHash.from_str(f"md5:{digest}")
