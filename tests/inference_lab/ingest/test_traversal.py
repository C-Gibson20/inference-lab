"""Tests for deterministic, content-free local source traversal."""

from collections.abc import Callable
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryFailureCategory,
    DiscoveryFileKind,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    LocalTraversalFileSystem,
    TraversalCandidate,
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
        self.after_snapshot: dict[Path, Callable[[], None]] = {}
        self.reverse_snapshots: set[Path] = set()

    def lstat(self, path: Path) -> TraversalPathMetadata:
        if path in self.missing_metadata:
            raise FileNotFoundError("injected missing path")
        if path in self.inaccessible_metadata:
            raise PermissionError("injected inaccessible path")
        return super().lstat(path)

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
    request = _request(_file_input(source, external_identifiers=(identifier,)))

    with capture_logs() as logs:
        inventory = inventory_local_sources(
            request,
            excluded_paths=(),
        )

    assert inventory.corpus_id == request.corpus_id
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
    names = (
        "z.txt",
        "a.txt",
        "nested/b.txt",
        ".hidden/c.txt",
        "A.txt",
        "é.txt",
        "e\N{COMBINING ACUTE ACCENT}.txt",
        "ß.txt",
    )
    for relative in names:
        (root / relative).write_text(relative, encoding="utf-8")
    filesystem = _FaultInjectingFileSystem()
    filesystem.reverse_snapshots.add(root)
    request = _request(_directory_input(root))

    inventory = inventory_local_sources(
        request,
        excluded_paths=(),
        filesystem=filesystem,
    )
    repeated = inventory_local_sources(request, excluded_paths=())

    expected = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.txt"), key=canonical_path_key)
    )
    assert _relative_locations(inventory) == expected
    assert inventory == repeated
    assert inventory.complete_root_ordinals == (0,)


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


@pytest.mark.parametrize(
    "input_kind", [DiscoveryInputKind.FILE, DiscoveryInputKind.DIRECTORY]
)
def test_explicit_root_with_symlinked_ancestor_fails_closed(
    tmp_path: Path,
    input_kind: DiscoveryInputKind,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    if input_kind is DiscoveryInputKind.FILE:
        (target / "source.txt").write_text("content", encoding="utf-8")
        relative = Path("source.txt")
    else:
        (target / "nested").mkdir()
        relative = Path("nested")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")
    root = alias / relative
    discovery_input = DiscoveryInput(root=root, kind=input_kind)

    inventory = inventory_local_sources(
        _request(discovery_input),
        excluded_paths=(),
    )

    assert inventory.candidates == ()
    assert len(inventory.failures) == 1
    assert inventory.failures[0].category is DiscoveryFailureCategory.INVALID_ROOT
    assert inventory.failures[0].file_kind is DiscoveryFileKind.SYMLINK
    assert inventory.complete_root_ordinals == ()


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


@pytest.mark.parametrize(
    ("root_exists", "expected_category", "expected_log_category"),
    (
        (False, DiscoveryFailureCategory.ROOT_NOT_FOUND, "root_not_found"),
        (True, DiscoveryFailureCategory.ROOT_INACCESSIBLE, "root_inaccessible"),
    ),
    ids=("missing", "inaccessible"),
)
def test_unavailable_root_fails_request_without_leaking_path(
    tmp_path: Path,
    root_exists: bool,
    expected_category: DiscoveryFailureCategory,
    expected_log_category: str,
) -> None:
    root = tmp_path / "private-root-marker"
    filesystem = _FaultInjectingFileSystem()
    if root_exists:
        root.mkdir()
        filesystem.inaccessible_metadata.add(root)

    with capture_logs() as logs:
        inventory = inventory_local_sources(
            _request(_directory_input(root)),
            excluded_paths=(),
            filesystem=filesystem,
        )

    assert inventory.candidates == ()
    assert tuple(failure.category for failure in inventory.failures) == (
        expected_category,
    )
    assert inventory.complete_root_ordinals == ()
    assert root.name not in repr(inventory.failures[0])
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("discovery_failed", "error")
    ]
    assert logs[0]["error_category"] == expected_log_category
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


def test_snapshot_handles_removed_and_added_entries_on_next_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    vanished = root / "a-vanished.txt"
    vanished.write_text("temporary", encoding="utf-8")
    (root / "b-safe.txt").write_text("safe", encoding="utf-8")
    added = root / "c-added.txt"
    filesystem = _FaultInjectingFileSystem()

    def change_directory() -> None:
        vanished.unlink()
        added.write_text("added", encoding="utf-8")

    filesystem.after_snapshot[root] = change_directory
    request = _request(_directory_input(root))

    inventory = inventory_local_sources(
        request,
        excluded_paths=(),
        filesystem=filesystem,
    )
    repeated = inventory_local_sources(request, excluded_paths=())

    assert _relative_locations(inventory) == ("b-safe.txt",)
    assert tuple(failure.category for failure in inventory.failures) == (
        DiscoveryFailureCategory.SOURCE_NOT_FOUND,
    )
    assert inventory.complete_root_ordinals == ()
    assert _relative_locations(repeated) == ("b-safe.txt", "c-added.txt")
    assert repeated.failures == ()
    assert repeated.complete_root_ordinals == (0,)


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
