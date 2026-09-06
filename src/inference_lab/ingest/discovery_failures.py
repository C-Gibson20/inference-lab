"""Privacy-safe failure contracts for local source discovery."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.discovery import SourceLocator, _validate_media_type


class DiscoveryFailureCategory(StrEnum):
    """Closed categories for request and per-source discovery failures."""

    INVALID_REQUEST = "invalid_request"
    INVALID_ROOT = "invalid_root"
    STORAGE_ROOT_OVERLAP = "storage_root_overlap"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_INACCESSIBLE = "source_inaccessible"
    SOURCE_SYMLINK = "source_symlink"
    SOURCE_SPECIAL_FILE = "source_special_file"
    NESTED_MOUNT = "nested_mount"
    TRAVERSAL_FAILED = "traversal_failed"
    RECOGNISED_UNSUPPORTED_FORMAT = "recognised_unsupported_format"
    UNKNOWN_FORMAT = "unknown_format"
    SOURCE_TOO_LARGE = "source_too_large"
    SOURCE_CHANGED_DURING_READ = "source_changed_during_read"
    SOURCE_READ_FAILED = "source_read_failed"
    OBSERVATION_AMBIGUOUS = "observation_ambiguous"
    CANDIDATE_METADATA_INVALID = "candidate_metadata_invalid"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    CONCURRENT_WRITE_CONFLICT = "concurrent_write_conflict"
    IDENTITY_WRITE_FAILED = "identity_write_failed"


class DiscoveryFailureScope(StrEnum):
    """Whether a failure invalidates one source or the complete request."""

    SOURCE = "source"
    REQUEST = "request"


class DiscoveryFailureStage(StrEnum):
    """Closed discovery stages at which work may fail."""

    INPUT = "input"
    TRAVERSAL = "traversal"
    CLASSIFICATION = "classification"
    HASHING = "hashing"
    CONTINUITY = "continuity"
    RESOLUTION = "resolution"
    PERSISTENCE = "persistence"


class DiscoveryFileKind(StrEnum):
    """Bounded filesystem kinds safe to include in operational records."""

    REGULAR_FILE = "regular_file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    SPECIAL = "special"
    UNKNOWN = "unknown"


_FailurePolicy = tuple[DiscoveryFailureScope, DiscoveryFailureStage, bool]

_FAILURE_POLICIES: Final[dict[DiscoveryFailureCategory, _FailurePolicy]] = {
    DiscoveryFailureCategory.INVALID_REQUEST: (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.INPUT,
        False,
    ),
    DiscoveryFailureCategory.INVALID_ROOT: (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.INPUT,
        False,
    ),
    DiscoveryFailureCategory.STORAGE_ROOT_OVERLAP: (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.INPUT,
        False,
    ),
    DiscoveryFailureCategory.SOURCE_NOT_FOUND: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.TRAVERSAL,
        True,
    ),
    DiscoveryFailureCategory.SOURCE_INACCESSIBLE: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.TRAVERSAL,
        True,
    ),
    DiscoveryFailureCategory.SOURCE_SYMLINK: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.TRAVERSAL,
        False,
    ),
    DiscoveryFailureCategory.SOURCE_SPECIAL_FILE: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.TRAVERSAL,
        False,
    ),
    DiscoveryFailureCategory.NESTED_MOUNT: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.TRAVERSAL,
        False,
    ),
    DiscoveryFailureCategory.TRAVERSAL_FAILED: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.TRAVERSAL,
        True,
    ),
    DiscoveryFailureCategory.RECOGNISED_UNSUPPORTED_FORMAT: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.CLASSIFICATION,
        False,
    ),
    DiscoveryFailureCategory.UNKNOWN_FORMAT: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.CLASSIFICATION,
        False,
    ),
    DiscoveryFailureCategory.SOURCE_TOO_LARGE: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.HASHING,
        False,
    ),
    DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.HASHING,
        True,
    ),
    DiscoveryFailureCategory.SOURCE_READ_FAILED: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.HASHING,
        True,
    ),
    DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.CONTINUITY,
        False,
    ),
    DiscoveryFailureCategory.CANDIDATE_METADATA_INVALID: (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.RESOLUTION,
        False,
    ),
    DiscoveryFailureCategory.REPOSITORY_UNAVAILABLE: (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.PERSISTENCE,
        True,
    ),
    DiscoveryFailureCategory.CONCURRENT_WRITE_CONFLICT: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.PERSISTENCE,
        True,
    ),
    DiscoveryFailureCategory.IDENTITY_WRITE_FAILED: (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.PERSISTENCE,
        False,
    ),
}

if set(_FAILURE_POLICIES) != set(DiscoveryFailureCategory):
    raise RuntimeError("discovery failure policies must cover every category")


def discovery_failure_policy(category: DiscoveryFailureCategory) -> _FailurePolicy:
    """Return the fixed scope, stage, and retryability for a category."""
    if not isinstance(category, DiscoveryFailureCategory):
        raise TypeError("category must be a DiscoveryFailureCategory.")
    return _FAILURE_POLICIES[category]


@dataclass(frozen=True, slots=True)
class DiscoveryFailure:
    """Describe one bounded failure without retaining exception payloads."""

    ordinal: int
    category: DiscoveryFailureCategory
    scope: DiscoveryFailureScope
    stage: DiscoveryFailureStage
    retryable: bool
    locator: SourceLocator | None = None
    file_kind: DiscoveryFileKind | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate bounded metadata and the category-owned failure policy."""
        if not is_strict_integer(self.ordinal):
            raise TypeError("ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")
        if not isinstance(self.category, DiscoveryFailureCategory):
            raise TypeError("category must be a DiscoveryFailureCategory.")
        if not isinstance(self.scope, DiscoveryFailureScope):
            raise TypeError("scope must be a DiscoveryFailureScope.")
        if not isinstance(self.stage, DiscoveryFailureStage):
            raise TypeError("stage must be a DiscoveryFailureStage.")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean.")

        expected_scope, expected_stage, expected_retryable = discovery_failure_policy(
            self.category
        )
        if self.scope is not expected_scope:
            raise ValueError("failure scope does not match its category.")
        if self.stage is not expected_stage:
            raise ValueError("failure stage does not match its category.")
        if self.retryable is not expected_retryable:
            raise ValueError("failure retryability does not match its category.")

        if self.locator is not None and not isinstance(self.locator, SourceLocator):
            raise TypeError("locator must be a SourceLocator or None.")
        if self.file_kind is not None and not isinstance(
            self.file_kind, DiscoveryFileKind
        ):
            raise TypeError("file_kind must be a DiscoveryFileKind or None.")
        if self.media_type is not None:
            _validate_media_type(self.media_type)

    @classmethod
    def for_category(
        cls,
        *,
        ordinal: int,
        category: DiscoveryFailureCategory,
        locator: SourceLocator | None = None,
        file_kind: DiscoveryFileKind | None = None,
        media_type: str | None = None,
    ) -> "DiscoveryFailure":
        """Construct a failure using category-owned policy values."""
        scope, stage, retryable = discovery_failure_policy(category)
        return cls(
            ordinal=ordinal,
            category=category,
            scope=scope,
            stage=stage,
            retryable=retryable,
            locator=locator,
            file_kind=file_kind,
            media_type=media_type,
        )
