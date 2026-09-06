"""Tests for deterministic, content-free local source traversal."""

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryFailureCategory,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    LocalTraversalFileSystem,
    TraversalCandidate,
    TraversalFileSystem,
    TraversalInventory,
    TraversalPathMetadata,
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


class _FaultInjectingFileSystem(LocalTraversalFileSystem):
    """Expose deterministic path races and host-specific traversal failures."""

    def __init__(self) -> None:
        self.inaccessible_metadata: set[Path] = set()
        self.missing_metadata: set[Path] = set()
        self.inaccessible_directories: set[Path] = set()
        self.junctions: set[Path] = set()
        self.device_overrides: dict[Path, int] = {}
        self.after_snapshot: dict[Path, Callable[[], None]] = {}
        self.reverse_snapshots: set[Path] = set()

    def lstat(self, path: Path) -> TraversalPathMetadata:
        if path in self.missing_metadata:
            raise FileNotFoundError("injected missing path")
        if path in self.inaccessible_metadata:
            raise PermissionError("injected inaccessible path")
        metadata = super().lstat(path)
        device_id = self.device_overrides.get(path)
        return metadata if device_id is None else replace(metadata, device_id=device_id)

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        if path in self.inaccessible_directories:
            raise PermissionError("injected inaccessible directory")
        snapshot = super().list_directory(path)
        action = self.after_snapshot.pop(path, None)
        if action is not None:
            action()
        return tuple(reversed(snapshot)) if path in self.reverse_snapshots else snapshot

    def is_junction(self, path: Path) -> bool:
        return path in self.junctions or super().is_junction(path)


def test_explicit_regular_file_becomes_content_free_candidate(tmp_path: Path) -> None:
    source = tmp_path / "private-name.txt"
    source.write_bytes(b"private payload")
    identifier = ExternalIdentifier("doi", "10.1000/private")

    with capture_logs() as logs:
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
    assert len(inventory.root_scopes) == 1
    assert inventory.root_scopes[0].path == source.resolve()
    assert inventory.root_scopes[0].kind is DiscoveryInputKind.FILE
    assert str(source.resolve()) not in repr(inventory.root_scopes[0])
    assert set(TraversalCandidate.__dataclass_fields__) == {
        "ordinal",
        "locator",
        "stat_evidence",
        "external_identifiers",
        "family_candidates",
    }
    assert logs == [
        {
            "pipeline_stage": "ingest",
            "source_ordinal": 0,
            "file_kind": "regular_file",
            "byte_count": len(b"private payload"),
            "event": "source_discovered",
            "log_level": "debug",
        }
    ]
    log_payload = repr(logs)
    assert source.name not in log_payload
    assert identifier.value not in log_payload
    assert "private payload" not in log_payload


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
    assert inventory.root_scopes[0].path == root.resolve()
    assert inventory.root_scopes[0].kind is DiscoveryInputKind.DIRECTORY


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


def test_nested_symlink_loop_is_reported_without_traversal(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(root, target_is_directory=True)
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


def test_inaccessible_root_fails_the_request_without_leaking_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-root"
    root.mkdir()
    filesystem = _FaultInjectingFileSystem()
    filesystem.inaccessible_metadata.add(root)

    with capture_logs() as logs:
        inventory = inventory_local_sources(
            _request(_directory_input(root)),
            excluded_paths=(),
            filesystem=filesystem,
        )

    assert inventory.candidates == ()
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.ROOT_INACCESSIBLE,
    )
    assert inventory.complete_root_ordinals == ()
    assert "private-root" not in repr(inventory.failures[0])
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("discovery_failed", "error")
    ]
    assert logs[0]["error_category"] == "root_inaccessible"
    assert root.name not in repr(logs)


def test_inaccessible_subtree_and_file_do_not_block_safe_sibling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    blocked = root / "a-blocked"
    blocked.mkdir(parents=True)
    inaccessible_file = root / "b-inaccessible.txt"
    inaccessible_file.write_text("private", encoding="utf-8")
    safe_file = root / "c-safe.txt"
    safe_file.write_text("safe", encoding="utf-8")
    filesystem = _FaultInjectingFileSystem()
    filesystem.inaccessible_directories.add(blocked)
    filesystem.inaccessible_metadata.add(inaccessible_file)

    with capture_logs() as logs:
        inventory = inventory_local_sources(
            _request(_directory_input(root)),
            excluded_paths=(),
            filesystem=filesystem,
        )

    assert _relative_locations(inventory) == ("c-safe.txt",)
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.TRAVERSAL_FAILED,
        DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
    )
    assert tuple(failure.ordinal for failure in inventory.failures) == (0, 1)
    assert inventory.candidates[0].ordinal == 2
    assert inventory.complete_root_ordinals == ()
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("source_skipped", "warning"),
        ("source_skipped", "warning"),
        ("source_discovered", "debug"),
    ]
    assert [record["source_ordinal"] for record in logs] == [0, 1, 2]
    assert logs[0]["error_category"] == "traversal_failed"
    assert logs[1]["error_category"] == "source_inaccessible"
    log_payload = repr(logs)
    assert blocked.name not in log_payload
    assert inaccessible_file.name not in log_payload
    assert safe_file.name not in log_payload


def test_request_fatal_traversal_logs_bounded_failure_without_private_path(
    tmp_path: Path,
) -> None:
    marker = "confidential-missing-root-9813"

    with capture_logs() as logs:
        inventory = inventory_local_sources(
            _request(_directory_input(tmp_path / marker)),
            excluded_paths=(),
        )

    assert inventory.failures[0].category is DiscoveryFailureCategory.ROOT_NOT_FOUND
    assert logs == [
        {
            "pipeline_stage": "ingest",
            "source_ordinal": 0,
            "error_category": "root_not_found",
            "failure_scope": "request",
            "failure_stage": "input",
            "retryable": True,
            "event": "discovery_failed",
            "log_level": "error",
        }
    ]
    assert marker not in repr(logs)


def test_entry_removed_after_snapshot_is_a_bounded_source_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    vanished = root / "a-vanished.txt"
    vanished.write_text("temporary", encoding="utf-8")
    (root / "b-safe.txt").write_text("safe", encoding="utf-8")
    filesystem = _FaultInjectingFileSystem()
    filesystem.after_snapshot[root] = vanished.unlink

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(),
        filesystem=filesystem,
    )

    assert _relative_locations(inventory) == ("b-safe.txt",)
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.SOURCE_NOT_FOUND,
    )
    assert inventory.complete_root_ordinals == ()


def test_entry_added_after_snapshot_is_deferred_until_the_next_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "existing.txt").write_text("existing", encoding="utf-8")
    added = root / "added.txt"
    filesystem = _FaultInjectingFileSystem()

    def add_entry() -> None:
        added.write_text("added", encoding="utf-8")

    filesystem.after_snapshot[root] = add_entry
    request = _request(_directory_input(root))

    first = inventory_local_sources(
        request,
        excluded_paths=(),
        filesystem=filesystem,
    )
    second = inventory_local_sources(request, excluded_paths=())

    assert _relative_locations(first) == ("existing.txt",)
    assert first.failures == ()
    assert first.complete_root_ordinals == (0,)
    assert _relative_locations(second) == ("added.txt", "existing.txt")


def test_nested_mount_is_reported_without_crossing_device_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    nested_mount = root / "a-mounted"
    nested_mount.mkdir(parents=True)
    (nested_mount / "excluded.txt").write_text("excluded", encoding="utf-8")
    (root / "b-local.txt").write_text("local", encoding="utf-8")
    filesystem = _FaultInjectingFileSystem()
    root_device = filesystem.lstat(root).device_id
    filesystem.device_overrides[nested_mount] = root_device + 1

    with capture_logs() as logs:
        inventory = inventory_local_sources(
            _request(_directory_input(root)),
            excluded_paths=(),
            filesystem=filesystem,
        )

    assert _relative_locations(inventory) == ("b-local.txt",)
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.NESTED_MOUNT,
    )
    assert inventory.complete_root_ordinals == (0,)
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("source_skipped", "debug"),
        ("source_discovered", "debug"),
    ]
    assert logs[0]["error_category"] == "nested_mount"
    assert nested_mount.name not in repr(logs)


def test_junction_is_reported_without_traversing_its_target(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    junction = root / "junction"
    junction.mkdir(parents=True)
    (junction / "excluded.txt").write_text("excluded", encoding="utf-8")
    filesystem = _FaultInjectingFileSystem()
    filesystem.junctions.add(junction)

    inventory = inventory_local_sources(
        _request(_directory_input(root)),
        excluded_paths=(),
        filesystem=filesystem,
    )

    assert inventory.candidates == ()
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.SOURCE_SYMLINK,
    )
    assert inventory.complete_root_ordinals == (0,)


def test_unicode_case_order_is_stable_across_adapter_and_repeated_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    names = ("a.txt", "A.txt", "é.txt", "e\N{COMBINING ACUTE ACCENT}.txt", "ß.txt")
    for name in names:
        (root / name).write_text(name, encoding="utf-8")
    filesystem = _FaultInjectingFileSystem()
    filesystem.reverse_snapshots.add(root)
    request = _request(_directory_input(root))

    first = inventory_local_sources(
        request,
        excluded_paths=(),
        filesystem=filesystem,
    )
    second = inventory_local_sources(request, excluded_paths=())
    expected = tuple(
        path.name for path in sorted(root.iterdir(), key=canonical_path_key)
    )

    assert _relative_locations(first) == expected
    assert first == second


def test_traversal_api_requires_typed_request_and_exclusions(tmp_path: Path) -> None:
    assert isinstance(LocalTraversalFileSystem(), TraversalFileSystem)
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
    with pytest.raises(TypeError, match="filesystem must implement"):
        inventory_local_sources(
            _request(_directory_input(tmp_path)),
            excluded_paths=(),
            filesystem=object(),  # pyright: ignore[reportArgumentType]
        )


def test_canonical_path_key_rejects_untyped_or_remote_paths() -> None:
    with pytest.raises(TypeError, match="path must be a Path"):
        canonical_path_key("source.txt")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="local filesystem path"):
        canonical_path_key(Path("https:/example.test/source"))


def test_traversal_path_metadata_rejects_untyped_or_negative_values() -> None:
    valid = {
        "mode": 1,
        "byte_count": 2,
        "modified_time_ns": 3,
        "device_id": 4,
        "file_id": 5,
    }
    with pytest.raises(TypeError, match="mode must be an integer"):
        TraversalPathMetadata(**(valid | {"mode": True}))
    with pytest.raises(ValueError, match="byte_count must be non-negative"):
        TraversalPathMetadata(**(valid | {"byte_count": -1}))


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
