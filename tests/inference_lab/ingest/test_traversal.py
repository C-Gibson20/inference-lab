"""Tests for deterministic, content-free local source traversal."""

import os
from pathlib import Path

import pytest

from inference_lab.ingest import (
    DiscoveryFailureCategory,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
)
from inference_lab.ingest.traversal import (
    TraversalCandidate,
    TraversalInventory,
    canonical_path_key,
    inventory_local_sources,
)
from inference_lab.provenance import CorpusId, ExternalIdentifier


def _request(*inputs: DiscoveryInput) -> DiscoveryRequest:
    return DiscoveryRequest(corpus_id=CorpusId.generate(), inputs=inputs)


def _file_input(
    path: Path,
    *,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> DiscoveryInput:
    return DiscoveryInput(
        root=path,
        kind=DiscoveryInputKind.FILE,
        external_identifiers=external_identifiers,
    )


def _directory_input(path: Path) -> DiscoveryInput:
    return DiscoveryInput(root=path, kind=DiscoveryInputKind.DIRECTORY)


def _relative_locations(inventory: TraversalInventory) -> tuple[str, ...]:
    return tuple(
        candidate.locator.relative_path.as_posix() for candidate in inventory.candidates
    )


def test_explicit_regular_file_becomes_content_free_candidate(tmp_path: Path) -> None:
    source = tmp_path / "private-name.txt"
    source.write_bytes(b"private payload")
    identifier = ExternalIdentifier("doi", "10.1000/private")

    inventory = inventory_local_sources(
        _request(_file_input(source, external_identifiers=(identifier,))),
        excluded_paths=(),
    )

    assert len(inventory.candidates) == 1
    candidate = inventory.candidates[0]
    assert candidate.ordinal == 0
    assert candidate.external_identifiers == (identifier,)
    assert candidate.stat_evidence.byte_count == len(b"private payload")
    assert candidate.stat_evidence.identity_hint is not None
    assert inventory.failures == ()
    assert inventory.complete_root_ordinals == (0,)
    assert set(TraversalCandidate.__dataclass_fields__) == {
        "ordinal",
        "locator",
        "stat_evidence",
        "external_identifiers",
        "family_candidates",
    }


def test_nested_directory_is_sorted_and_includes_hidden_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    (root / "nested").mkdir(parents=True)
    (root / ".hidden").mkdir()
    for relative in ("z.txt", "a.txt", "nested/b.txt", ".hidden/c.txt"):
        (root / relative).write_text(relative, encoding="utf-8")

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(),
    )

    expected = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.txt"), key=canonical_path_key)
    )
    assert _relative_locations(inventory) == expected
    assert inventory.complete_root_ordinals == (0,)


def test_empty_directory_is_a_complete_zero_candidate_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(),
    )

    assert inventory.candidates == ()
    assert inventory.failures == ()
    assert inventory.complete_root_ordinals == (0,)


def test_overlapping_roots_deduplicate_resolved_entry(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "paper.txt"
    source.write_text("payload", encoding="utf-8")
    linked = nested / "linked.txt"
    try:
        linked.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")
    identifier = ExternalIdentifier("doi", "10.1000/explicit")

    inventory = inventory_local_sources(
        _request(
            _directory_input(root),
            _directory_input(nested),
            _file_input(source, external_identifiers=(identifier,)),
        ),
        excluded_paths=(),
    )

    assert len(inventory.candidates) == 1
    assert inventory.candidates[0].external_identifiers == (identifier,)
    assert inventory.candidates[0].locator.request_root_ordinal == 2
    assert len(inventory.failures) == 1
    assert inventory.failures[0].category is DiscoveryFailureCategory.SOURCE_SYMLINK
    assert inventory.complete_root_ordinals == (0, 1, 2)


def test_duplicate_identical_roots_are_scanned_once(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("payload", encoding="utf-8")
    discovery_input = _file_input(source)

    inventory = inventory_local_sources(
        _request(discovery_input, discovery_input),
        excluded_paths=(),
    )

    assert inventory.root_count == 1
    assert len(inventory.candidates) == 1


def test_duplicate_root_with_conflicting_metadata_fails_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("payload", encoding="utf-8")

    inventory = inventory_local_sources(
        _request(
            _file_input(
                source,
                external_identifiers=(ExternalIdentifier("doi", "10.1000/one"),),
            ),
            _file_input(
                source,
                external_identifiers=(ExternalIdentifier("doi", "10.1000/two"),),
            ),
        ),
        excluded_paths=(),
    )

    assert inventory.candidates == ()
    assert len(inventory.failures) == 1
    assert inventory.failures[0].category is DiscoveryFailureCategory.INVALID_REQUEST
    assert inventory.complete_root_ordinals == ()


def test_hard_links_remain_independently_observed_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same bytes", encoding="utf-8")
    try:
        os.link(first, second)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {type(error).__name__}")

    inventory = inventory_local_sources(
        _request(_directory_input(tmp_path)),
        excluded_paths=(),
    )

    assert _relative_locations(inventory) == ("first.txt", "second.txt")
    first_hint = inventory.candidates[0].stat_evidence.identity_hint
    second_hint = inventory.candidates[1].stat_evidence.identity_hint
    assert first_hint == second_hint


def test_nested_symlink_is_reported_without_following_target(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(),
    )

    assert inventory.candidates == ()
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.SOURCE_SYMLINK,
    )
    assert inventory.complete_root_ordinals == (0,)


def test_explicit_symlink_root_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("content", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")

    inventory = inventory_local_sources(
        _request(_file_input(link)),
        excluded_paths=(),
    )

    assert inventory.candidates == ()
    assert inventory.failures[0].category is DiscoveryFailureCategory.INVALID_ROOT
    assert inventory.complete_root_ordinals == ()


def test_special_file_is_reported_without_becoming_candidate(tmp_path: Path) -> None:
    fifo = tmp_path / "source.pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError) as error:
        pytest.skip(f"FIFOs unavailable: {type(error).__name__}")

    inventory = inventory_local_sources(
        _request(_directory_input(tmp_path)),
        excluded_paths=(),
    )

    assert inventory.candidates == ()
    assert inventory.failures[0].category is (
        DiscoveryFailureCategory.SOURCE_SPECIAL_FILE
    )


@pytest.mark.parametrize("excluded_relative", [".", "active/indexes"])
def test_storage_overlap_fails_root_before_traversal(
    tmp_path: Path,
    excluded_relative: str,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    excluded = root if excluded_relative == "." else root / excluded_relative

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(excluded,),
    )

    assert inventory.candidates == ()
    assert inventory.failures[0].category is (
        DiscoveryFailureCategory.STORAGE_ROOT_OVERLAP
    )


def test_missing_and_kind_mismatched_roots_fail_request(tmp_path: Path) -> None:
    missing = inventory_local_sources(
        _request(_file_input(tmp_path / "missing.txt")),
        excluded_paths=(),
    )
    assert missing.failures[0].category is DiscoveryFailureCategory.ROOT_NOT_FOUND

    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    mismatched = inventory_local_sources(
        _request(_directory_input(source)),
        excluded_paths=(),
    )
    assert mismatched.failures[0].category is DiscoveryFailureCategory.INVALID_ROOT


def test_root_input_order_does_not_change_candidate_order(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    forward = inventory_local_sources(
        _request(_file_input(first), _file_input(second)),
        excluded_paths=(),
    )
    reverse = inventory_local_sources(
        _request(_file_input(second), _file_input(first)),
        excluded_paths=(),
    )

    assert _relative_locations(forward) == _relative_locations(reverse)
    assert tuple(item.ordinal for item in forward.candidates) == (0, 1)
    assert tuple(item.ordinal for item in reverse.candidates) == (0, 1)


def test_deep_directory_traversal_is_iterative(tmp_path: Path) -> None:
    root = tmp_path / "deep"
    root.mkdir()
    directory = root
    for _ in range(80):
        directory /= "d"
        directory.mkdir()
    (directory / "leaf.txt").write_text("leaf", encoding="utf-8")

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(),
    )

    assert len(inventory.candidates) == 1
    assert inventory.failures == ()


def test_traversal_api_requires_typed_request_and_exclusions(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="request must be a DiscoveryRequest"):
        inventory_local_sources(
            "not-a-request",  # pyright: ignore[reportArgumentType]
            excluded_paths=(),
        )
    with pytest.raises(TypeError, match="excluded_paths must be a tuple"):
        inventory_local_sources(
            _request(_directory_input(tmp_path)),
            excluded_paths=[],  # pyright: ignore[reportArgumentType]
        )


def test_canonical_path_key_rejects_untyped_or_remote_paths() -> None:
    with pytest.raises(TypeError, match="path must be a Path"):
        canonical_path_key("source.txt")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="local filesystem path"):
        canonical_path_key(Path("https:/example.test/source"))


def test_failure_and_candidate_representations_hide_source_paths(
    tmp_path: Path,
) -> None:
    marker = "confidential-source-7182.txt"
    source = tmp_path / marker
    source.write_text("payload", encoding="utf-8")
    inventory = inventory_local_sources(
        _request(_file_input(source)),
        excluded_paths=(),
    )

    assert marker not in repr(inventory.candidates[0])
