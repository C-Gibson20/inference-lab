"""Tests for identity-resolution storage failures."""

from pathlib import Path

import pytest

from inference_lab.storage import (
    CorruptStorageError,
)
from tests.inference_lab.ingest.identity_resolution_fixtures import (
    _Allocator,
    _locator,
    _observation,
    _request,
    _resolve,
    _source_case,
    _Transaction,
    _version,
)


def test_missing_continuity_version_is_reported_as_corrupt_storage(
    tmp_path: Path,
) -> None:
    source, snapshot, inventory = _source_case(tmp_path)
    version = _version()
    transaction = _Transaction(
        observations=(_observation(version=version),),
        locators=(_locator(snapshot.locator.path),),
    )

    with pytest.raises(CorruptStorageError):
        _resolve(
            source,
            snapshot,
            inventory,
            transaction,
            _Allocator(),
            _request(tmp_path),
        )
