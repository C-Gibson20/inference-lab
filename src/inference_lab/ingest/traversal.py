"""Deterministic, content-free inventory of explicit local discovery roots."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from inference_lab._validation import is_local_path_shape, is_strict_integer
from inference_lab.ingest.discovery import (
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FamilyCandidate,
    FileIdentityHint,
    FileStatEvidence,
    SourceLocator,
    _validate_family_candidates,
)
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from inference_lab.provenance import ExternalIdentifier
from inference_lab.provenance.values import validate_external_identifiers

PathOrderingKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class TraversalPathMetadata:
    """Carry the content-free path metadata required during traversal."""

    mode: int
    byte_count: int
    modified_time_ns: int
    device_id: int
    file_id: int

    def __post_init__(self) -> None:
        """Validate the non-negative integer metadata boundary."""
        for field_name, value in (
            ("mode", self.mode),
            ("byte_count", self.byte_count),
            ("modified_time_ns", self.modified_time_ns),
            ("device_id", self.device_id),
            ("file_id", self.file_id),
        ):
            if not is_strict_integer(value):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@runtime_checkable
class TraversalFileSystem(Protocol):
    """Provide the bounded local-filesystem operations used by traversal."""

    def absolute(self, path: Path) -> Path:
        """Return an absolute lexical path without resolving aliases."""
        ...

    def resolve(self, path: Path) -> Path:
        """Resolve aliases without requiring the final path to exist."""
        ...

    def lstat(self, path: Path) -> TraversalPathMetadata:
        """Return path metadata without following the final component."""
        ...

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        """Return one point-in-time snapshot of immediate entry paths."""
        ...

    def is_junction(self, path: Path) -> bool:
        """Return whether the host identifies the path as a junction."""
        ...


class LocalTraversalFileSystem:
    """Adapt host filesystem operations for content-free local traversal."""

    def absolute(self, path: Path) -> Path:
        """Return an absolute lexical path without resolving aliases."""
        return Path(os.path.abspath(os.fspath(path)))

    def resolve(self, path: Path) -> Path:
        """Resolve aliases without requiring the final path to exist."""
        return path.resolve(strict=False)

    def lstat(self, path: Path) -> TraversalPathMetadata:
        """Return bounded metadata without following the final component."""
        stat_result = path.lstat()
        return TraversalPathMetadata(
            mode=stat_result.st_mode,
            byte_count=stat_result.st_size,
            modified_time_ns=stat_result.st_mtime_ns,
            device_id=stat_result.st_dev,
            file_id=stat_result.st_ino,
        )

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        """Snapshot immediate entry paths without retaining directory handles."""
        with os.scandir(path) as entries:
            return tuple(Path(entry.path) for entry in entries)

    def is_junction(self, path: Path) -> bool:
        """Return whether the host reports a Windows junction at the path."""
        is_junction = getattr(os.path, "isjunction", None)
        return bool(is_junction is not None and is_junction(path))


_LOCAL_FILESYSTEM = LocalTraversalFileSystem()


def canonical_path_key(path: Path) -> PathOrderingKey:
    """Return a documented host-aware key without Unicode normalisation.

    Absolute lexical paths are normalised with the host's ``normcase`` policy.
    The original absolute spelling is retained as a stable tie-breaker. The key
    is deterministic for one filesystem snapshot, not portable across hosts.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a Path.")
    if not is_local_path_shape(path):
        raise ValueError("path must name a local filesystem path.")
    absolute = os.path.abspath(os.fspath(path))
    return os.path.normcase(absolute), absolute


@dataclass(frozen=True, slots=True)
class TraversalCandidate:
    """Describe one regular file found without opening or reading its content."""

    ordinal: int
    locator: SourceLocator
    stat_evidence: FileStatEvidence
    external_identifiers: tuple[ExternalIdentifier, ...] = ()
    family_candidates: tuple[FamilyCandidate, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate types, non-negative ordinals, and bounded evidence."""
        if not is_strict_integer(self.ordinal):
            raise TypeError("ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")
        if not isinstance(self.locator, SourceLocator):
            raise TypeError("locator must be a SourceLocator.")
        if not isinstance(self.stat_evidence, FileStatEvidence):
            raise TypeError("stat_evidence must be FileStatEvidence.")
        validate_external_identifiers(self.external_identifiers)
        _validate_family_candidates(self.family_candidates)


@dataclass(frozen=True, slots=True)
class TraversalInventory:
    """Retain ordered candidates, failures, and complete request-root scans."""

    root_count: int
    candidates: tuple[TraversalCandidate, ...]
    failures: tuple[DiscoveryFailure, ...]
    complete_root_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate types, non-negative counts, and bounded ordinals."""
        if not is_strict_integer(self.root_count):
            raise TypeError("root_count must be an integer.")
        if self.root_count < 1:
            raise ValueError("root_count must be positive.")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple.")
        if any(
            not isinstance(candidate, TraversalCandidate)
            for candidate in self.candidates
        ):
            raise TypeError("candidates must contain TraversalCandidate values.")
        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(failure, DiscoveryFailure) for failure in self.failures):
            raise TypeError("failures must contain DiscoveryFailure values.")
        self._validate_ordinals()
        self._validate_complete_roots()

    def _validate_ordinals(self) -> None:
        candidate_ordinals = tuple(item.ordinal for item in self.candidates)
        failure_ordinals = tuple(item.ordinal for item in self.failures)
        if candidate_ordinals != tuple(sorted(candidate_ordinals)):
            raise ValueError("candidates must be in traversal ordinal order.")
        if failure_ordinals != tuple(sorted(failure_ordinals)):
            raise ValueError("failures must be in traversal ordinal order.")
        combined = candidate_ordinals + failure_ordinals
        if len(combined) != len(set(combined)):
            raise ValueError("traversal ordinals must be unique.")

    def _validate_complete_roots(self) -> None:
        if not isinstance(self.complete_root_ordinals, tuple):
            raise TypeError("complete_root_ordinals must be a tuple.")
        if any(
            not is_strict_integer(ordinal) for ordinal in self.complete_root_ordinals
        ):
            raise TypeError("complete root ordinals must be integers.")
        if self.complete_root_ordinals != tuple(
            sorted(set(self.complete_root_ordinals))
        ):
            raise ValueError("complete root ordinals must be unique and ordered.")
        if any(
            ordinal < 0 or ordinal >= self.root_count
            for ordinal in self.complete_root_ordinals
        ):
            raise ValueError("complete root ordinal is outside the request roots.")


@dataclass(frozen=True, slots=True)
class _PreparedRoot:
    ordinal: int
    discovery_input: DiscoveryInput
    path: Path = field(repr=False)
    metadata: TraversalPathMetadata = field(repr=False)


@dataclass(frozen=True, slots=True)
class _RawCandidate:
    key: PathOrderingKey
    locator: SourceLocator
    stat_evidence: FileStatEvidence
    discovery_input: DiscoveryInput


@dataclass(frozen=True, slots=True)
class _RawFailure:
    key: PathOrderingKey
    category: DiscoveryFailureCategory
    locator: SourceLocator | None
    file_kind: DiscoveryFileKind | None = None


def _resolved_path(path: Path, filesystem: TraversalFileSystem) -> Path:
    """Resolve aliases for path equality without requiring the leaf to exist."""
    return filesystem.resolve(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved local path contains the other."""
    return (
        first == second or first.is_relative_to(second) or second.is_relative_to(first)
    )


def _validate_excluded_paths(
    excluded_paths: object,
    filesystem: TraversalFileSystem,
) -> tuple[Path, ...]:
    """Validate and canonicalise caller-supplied active storage locations."""
    if not isinstance(excluded_paths, tuple):
        raise TypeError("excluded_paths must be a tuple.")
    if any(not isinstance(path, Path) for path in excluded_paths):
        raise TypeError("excluded_paths must contain Path values.")
    if any(not is_local_path_shape(path) for path in excluded_paths):
        raise ValueError("excluded_paths must contain local filesystem paths.")
    resolved = {_resolved_path(path, filesystem) for path in excluded_paths}
    return tuple(sorted(resolved, key=canonical_path_key))


def _root_relative_path(path: Path, root: Path) -> Path:
    """Return a non-sentinel private relative location for a root or child."""
    relative = path.relative_to(root)
    if relative == Path("."):
        return Path(path.name or "__filesystem_root__")
    return relative


def _locator(path: Path, root: _PreparedRoot) -> SourceLocator:
    """Build a private locator for a prepared request root."""
    return SourceLocator(
        path=path,
        request_root_ordinal=root.ordinal,
        relative_path=_root_relative_path(path, root.path),
    )


def _file_kind(mode: int) -> DiscoveryFileKind:
    """Map a stat mode to bounded discovery file-kind metadata."""
    if stat.S_ISREG(mode):
        return DiscoveryFileKind.REGULAR_FILE
    if stat.S_ISDIR(mode):
        return DiscoveryFileKind.DIRECTORY
    if stat.S_ISLNK(mode):
        return DiscoveryFileKind.SYMLINK
    return DiscoveryFileKind.SPECIAL


def _stat_evidence(metadata: TraversalPathMetadata) -> FileStatEvidence:
    """Convert one no-follow stat result into bounded inventory evidence."""
    return FileStatEvidence(
        byte_count=metadata.byte_count,
        modified_time_ns=metadata.modified_time_ns,
        identity_hint=FileIdentityHint(
            device_id=metadata.device_id,
            file_id=metadata.file_id,
        ),
    )


def _root_failure(
    *,
    root_count: int,
    root_ordinal: int,
    path: Path,
    category: DiscoveryFailureCategory,
    file_kind: DiscoveryFileKind | None = None,
) -> TraversalInventory:
    """Build a fail-closed inventory for one invalid request root."""
    relative = Path(path.name or "__filesystem_root__")
    locator = SourceLocator(
        path=path,
        request_root_ordinal=root_ordinal,
        relative_path=relative,
    )
    return TraversalInventory(
        root_count=root_count,
        candidates=(),
        failures=(
            DiscoveryFailure.for_category(
                ordinal=0,
                category=category,
                locator=locator,
                file_kind=file_kind,
            ),
        ),
        complete_root_ordinals=(),
    )


def _same_root_metadata(first: DiscoveryInput, second: DiscoveryInput) -> bool:
    """Compare root semantics while ignoring alias path spelling."""
    return (
        first.kind is second.kind
        and first.external_identifiers == second.external_identifiers
        and first.family_candidates == second.family_candidates
    )


def _prepare_roots(
    request: DiscoveryRequest,
    excluded_paths: tuple[Path, ...],
    filesystem: TraversalFileSystem,
) -> tuple[tuple[_PreparedRoot, ...], TraversalInventory | None]:
    """Validate, sort, deduplicate, and stat all explicit request roots."""
    ordered_inputs = sorted(
        request.inputs,
        key=lambda item: canonical_path_key(item.root),
    )
    unique_inputs: list[tuple[Path, DiscoveryInput, TraversalPathMetadata]] = []
    by_path: dict[PathOrderingKey, DiscoveryInput] = {}
    for input_ordinal, discovery_input in enumerate(ordered_inputs):
        lexical_path = filesystem.absolute(discovery_input.root)
        try:
            metadata = filesystem.lstat(lexical_path)
        except FileNotFoundError:
            return (), _root_failure(
                root_count=len(request.inputs),
                root_ordinal=input_ordinal,
                path=lexical_path,
                category=DiscoveryFailureCategory.ROOT_NOT_FOUND,
            )
        except OSError:
            return (), _root_failure(
                root_count=len(request.inputs),
                root_ordinal=input_ordinal,
                path=lexical_path,
                category=DiscoveryFailureCategory.ROOT_INACCESSIBLE,
            )

        kind = _file_kind(metadata.mode)
        expected_kind = (
            DiscoveryFileKind.REGULAR_FILE
            if discovery_input.kind is DiscoveryInputKind.FILE
            else DiscoveryFileKind.DIRECTORY
        )
        try:
            is_junction = filesystem.is_junction(lexical_path)
        except OSError:
            return (), _root_failure(
                root_count=len(request.inputs),
                root_ordinal=input_ordinal,
                path=lexical_path,
                category=DiscoveryFailureCategory.ROOT_INACCESSIBLE,
            )
        if kind is DiscoveryFileKind.SYMLINK or is_junction:
            return (), _root_failure(
                root_count=len(request.inputs),
                root_ordinal=input_ordinal,
                path=lexical_path,
                category=DiscoveryFailureCategory.INVALID_ROOT,
                file_kind=DiscoveryFileKind.SYMLINK,
            )
        if kind is not expected_kind:
            return (), _root_failure(
                root_count=len(request.inputs),
                root_ordinal=input_ordinal,
                path=lexical_path,
                category=DiscoveryFailureCategory.INVALID_ROOT,
                file_kind=kind,
            )

        try:
            resolved = _resolved_path(lexical_path, filesystem)
        except OSError:
            return (), _root_failure(
                root_count=len(request.inputs),
                root_ordinal=input_ordinal,
                path=lexical_path,
                category=DiscoveryFailureCategory.ROOT_INACCESSIBLE,
            )
        key = canonical_path_key(resolved)
        existing = by_path.get(key)
        if existing is not None:
            if not _same_root_metadata(existing, discovery_input):
                return (), _root_failure(
                    root_count=len(request.inputs),
                    root_ordinal=0,
                    path=resolved,
                    category=DiscoveryFailureCategory.INVALID_REQUEST,
                )
            continue
        by_path[key] = discovery_input
        unique_inputs.append((resolved, discovery_input, metadata))

    unique_inputs.sort(key=lambda item: canonical_path_key(item[0]))
    root_count = len(unique_inputs)
    prepared: list[_PreparedRoot] = []
    for ordinal, (path, discovery_input, metadata) in enumerate(unique_inputs):
        if any(_paths_overlap(path, excluded) for excluded in excluded_paths):
            return (), _root_failure(
                root_count=root_count,
                root_ordinal=ordinal,
                path=path,
                category=DiscoveryFailureCategory.STORAGE_ROOT_OVERLAP,
            )
        prepared.append(_PreparedRoot(ordinal, discovery_input, path, metadata))

    if not prepared:
        raise RuntimeError("validated discovery request produced no roots")
    if root_count != len(prepared):
        raise RuntimeError("prepared root count does not match validated roots")
    return tuple(prepared), None


def _raw_failure(
    path: Path,
    root: _PreparedRoot,
    category: DiscoveryFailureCategory,
    *,
    file_kind: DiscoveryFileKind | None = None,
) -> _RawFailure:
    """Build an ordinal-free traversal failure for final deterministic sorting."""
    return _RawFailure(
        key=canonical_path_key(path),
        category=category,
        locator=_locator(path, root),
        file_kind=file_kind,
    )


def _candidate_from_metadata(
    path: Path,
    root: _PreparedRoot,
    metadata: TraversalPathMetadata,
) -> _RawCandidate:
    """Build an ordinal-free regular-file candidate from no-follow metadata."""
    return _RawCandidate(
        key=canonical_path_key(path),
        locator=_locator(path, root),
        stat_evidence=_stat_evidence(metadata),
        discovery_input=root.discovery_input,
    )


def _prefer_candidate(
    current: _RawCandidate,
    candidate: _RawCandidate,
) -> _RawCandidate:
    """Prefer explicit-file metadata, then the earliest canonical request root."""
    current_is_file = current.discovery_input.kind is DiscoveryInputKind.FILE
    candidate_is_file = candidate.discovery_input.kind is DiscoveryInputKind.FILE
    if candidate_is_file and not current_is_file:
        return candidate
    if current_is_file and not candidate_is_file:
        return current
    if candidate.locator.request_root_ordinal < current.locator.request_root_ordinal:
        return candidate
    return current


def _scan_directory_root(
    root: _PreparedRoot,
    candidates: dict[PathOrderingKey, _RawCandidate],
    failures: list[_RawFailure],
    filesystem: TraversalFileSystem,
) -> bool:
    """Iteratively scan one directory root and return whether it was complete."""
    complete = True
    pending = [root.path]
    root_device = root.metadata.device_id
    while pending:
        directory = pending.pop()
        try:
            snapshot = sorted(
                filesystem.list_directory(directory),
                key=canonical_path_key,
                reverse=True,
            )
        except OSError:
            failures.append(
                _raw_failure(
                    directory,
                    root,
                    DiscoveryFailureCategory.TRAVERSAL_FAILED,
                    file_kind=DiscoveryFileKind.DIRECTORY,
                )
            )
            complete = False
            continue

        for path in snapshot:
            try:
                if filesystem.is_junction(path):
                    failures.append(
                        _raw_failure(
                            path,
                            root,
                            DiscoveryFailureCategory.SOURCE_SYMLINK,
                            file_kind=DiscoveryFileKind.SYMLINK,
                        )
                    )
                    continue
                metadata = filesystem.lstat(path)
            except FileNotFoundError:
                failures.append(
                    _raw_failure(
                        path,
                        root,
                        DiscoveryFailureCategory.SOURCE_NOT_FOUND,
                    )
                )
                complete = False
                continue
            except OSError:
                failures.append(
                    _raw_failure(
                        path,
                        root,
                        DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
                    )
                )
                complete = False
                continue

            kind = _file_kind(metadata.mode)
            if kind is DiscoveryFileKind.SYMLINK:
                failures.append(
                    _raw_failure(
                        path,
                        root,
                        DiscoveryFailureCategory.SOURCE_SYMLINK,
                        file_kind=kind,
                    )
                )
                continue
            if kind is DiscoveryFileKind.DIRECTORY:
                if metadata.device_id != root_device:
                    failures.append(
                        _raw_failure(
                            path,
                            root,
                            DiscoveryFailureCategory.NESTED_MOUNT,
                            file_kind=kind,
                        )
                    )
                    continue
                pending.append(path)
                continue
            if kind is not DiscoveryFileKind.REGULAR_FILE:
                failures.append(
                    _raw_failure(
                        path,
                        root,
                        DiscoveryFailureCategory.SOURCE_SPECIAL_FILE,
                        file_kind=kind,
                    )
                )
                continue

            candidate = _candidate_from_metadata(path, root, metadata)
            existing = candidates.get(candidate.key)
            candidates[candidate.key] = (
                candidate
                if existing is None
                else _prefer_candidate(existing, candidate)
            )
    return complete


def _finalise_inventory(
    *,
    root_count: int,
    raw_candidates: dict[PathOrderingKey, _RawCandidate],
    raw_failures: list[_RawFailure],
    complete_root_ordinals: tuple[int, ...],
) -> TraversalInventory:
    """Assign stable ordinals after the complete inventory pass."""
    unique_failures: dict[PathOrderingKey, _RawFailure] = {}
    for failure in raw_failures:
        unique_failures.setdefault(failure.key, failure)
    entries = [
        (candidate.key, 0, candidate) for candidate in raw_candidates.values()
    ] + [(failure.key, 1, failure) for failure in unique_failures.values()]
    entries.sort(key=lambda item: (item[0], item[1]))

    candidates: list[TraversalCandidate] = []
    failures: list[DiscoveryFailure] = []
    for ordinal, (_, _, entry) in enumerate(entries):
        if isinstance(entry, _RawCandidate):
            candidates.append(
                TraversalCandidate(
                    ordinal=ordinal,
                    locator=entry.locator,
                    stat_evidence=entry.stat_evidence,
                    external_identifiers=entry.discovery_input.external_identifiers,
                    family_candidates=entry.discovery_input.family_candidates,
                )
            )
        else:
            failures.append(
                DiscoveryFailure.for_category(
                    ordinal=ordinal,
                    category=entry.category,
                    locator=entry.locator,
                    file_kind=entry.file_kind,
                )
            )
    return TraversalInventory(
        root_count=root_count,
        candidates=tuple(candidates),
        failures=tuple(failures),
        complete_root_ordinals=complete_root_ordinals,
    )


def inventory_local_sources(
    request: DiscoveryRequest,
    *,
    excluded_paths: tuple[Path, ...],
    filesystem: TraversalFileSystem | None = None,
) -> TraversalInventory:
    """Inventory explicit local roots without following links or reading content."""
    if not isinstance(request, DiscoveryRequest):
        raise TypeError("request must be a DiscoveryRequest.")
    selected_filesystem = _LOCAL_FILESYSTEM if filesystem is None else filesystem
    if not isinstance(selected_filesystem, TraversalFileSystem):
        raise TypeError("filesystem must implement TraversalFileSystem.")
    exclusions = _validate_excluded_paths(excluded_paths, selected_filesystem)
    roots, failure = _prepare_roots(request, exclusions, selected_filesystem)
    if failure is not None:
        return failure

    raw_candidates: dict[PathOrderingKey, _RawCandidate] = {}
    raw_failures: list[_RawFailure] = []
    complete_roots: list[int] = []
    for root in roots:
        if root.discovery_input.kind is DiscoveryInputKind.FILE:
            candidate = _candidate_from_metadata(root.path, root, root.metadata)
            existing = raw_candidates.get(candidate.key)
            raw_candidates[candidate.key] = (
                candidate
                if existing is None
                else _prefer_candidate(existing, candidate)
            )
            complete_roots.append(root.ordinal)
            continue
        if _scan_directory_root(
            root,
            raw_candidates,
            raw_failures,
            selected_filesystem,
        ):
            complete_roots.append(root.ordinal)

    return _finalise_inventory(
        root_count=len(roots),
        raw_candidates=raw_candidates,
        raw_failures=raw_failures,
        complete_root_ordinals=tuple(complete_roots),
    )
