"""Tests for filesystem continuity during identity resolution."""

from pathlib import Path

from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
)
from inference_lab.provenance import (
    ResolutionKind,
)
from tests.inference_lab.ingest.identity_resolution_fixtures import (
    _ALLOCATED_SOURCE_ID,
    _CORPUS_ID,
    _FIRST_VERSION_ID,
    _HASH_B,
    _SOURCE_ID,
    _Allocator,
    _locator,
    _observation,
    _request,
    _resolve,
    _source_case,
    _Transaction,
    _version,
)


def test_missing_old_locator_resolves_existing_observation(tmp_path: Path) -> None:
    source, snapshot, inventory = _source_case(tmp_path)
    version = _version()
    old_locator = _locator(snapshot.locator.path.with_name("old.txt"))
    transaction = _Transaction(
        versions=(version,),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
        observations=(_observation(version=version),),
        locators=(old_locator,),
    )

    result = _resolve(
        source,
        snapshot,
        inventory,
        transaction,
        _Allocator(),
        _request(tmp_path),
    )

    assert not isinstance(result, DiscoveryFailure)
    assert result.source_observation_id == _SOURCE_ID
    assert result.identity_outcome.kind is ResolutionKind.UNCHANGED


def test_live_hard_link_resolves_as_independent_duplicate(tmp_path: Path) -> None:
    root = (tmp_path / "sources").resolve()
    old_locator = _locator(root / "still-live.txt")
    source, snapshot, inventory = _source_case(
        tmp_path,
        live_locators=(old_locator,),
    )
    version = _version()
    transaction = _Transaction(
        versions=(version,),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
        observations=(_observation(version=version),),
        locators=(old_locator,),
    )

    result = _resolve(
        source,
        snapshot,
        inventory,
        transaction,
        _Allocator(),
        _request(tmp_path),
    )

    assert not isinstance(result, DiscoveryFailure)
    assert result.source_observation_id == _ALLOCATED_SOURCE_ID
    assert result.identity_outcome.kind is ResolutionKind.DUPLICATE


def test_traversal_gap_stops_before_provenance_resolution(tmp_path: Path) -> None:
    source, snapshot, inventory = _source_case(tmp_path, complete=False)
    version = _version(content_hash=_HASH_B)
    old_locator = _locator(
        snapshot.locator.path.with_name("possibly-missing.txt"),
        content_hash=_HASH_B,
    )
    transaction = _Transaction(
        versions=(version,),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
        observations=(_observation(version=version),),
        locators=(old_locator,),
    )

    with capture_logs() as logs:
        result = _resolve(
            source,
            snapshot,
            inventory,
            transaction,
            _Allocator(),
            _request(tmp_path),
        )

    assert isinstance(result, DiscoveryFailure)
    assert result.category is DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("source_failed", "error")
    ]
    assert logs[0]["error_category"] == "observation_ambiguous"
    assert logs[0]["failure_stage"] == "continuity"
    assert snapshot.locator.relative_path.name not in repr(logs)
