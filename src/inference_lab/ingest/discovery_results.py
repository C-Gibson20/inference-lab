"""Aggregate result contracts for deterministic local source discovery."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.discovery import DiscoveredSource, DiscoveryPolicy
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryFailureStage,
)
from inference_lab.provenance import CorpusId, ResolutionKind

_UNSUPPORTED_CATEGORIES: Final = frozenset(
    {
        DiscoveryFailureCategory.RECOGNISED_UNSUPPORTED_FORMAT,
        DiscoveryFailureCategory.UNKNOWN_FORMAT,
    }
)
_POST_CLASSIFICATION_STAGES: Final = frozenset(
    {
        DiscoveryFailureStage.HASHING,
        DiscoveryFailureStage.CONTINUITY,
        DiscoveryFailureStage.RESOLUTION,
        DiscoveryFailureStage.PERSISTENCE,
    }
)


class DiscoveryStatus(StrEnum):
    """Overall completion state of one discovery request."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


def _validate_sources(sources: object) -> None:
    """Validate an immutable collection of discovered source outcomes."""
    if not isinstance(sources, tuple):
        raise TypeError("sources must be a tuple.")
    if any(not isinstance(source, DiscoveredSource) for source in sources):
        raise TypeError("sources must contain DiscoveredSource values.")


def _validate_failures(failures: object) -> None:
    """Validate an immutable collection of discovery failures."""
    if not isinstance(failures, tuple):
        raise TypeError("failures must be a tuple.")
    if any(not isinstance(failure, DiscoveryFailure) for failure in failures):
        raise TypeError("failures must contain DiscoveryFailure values.")


def _supported_failure_count(failures: tuple[DiscoveryFailure, ...]) -> int:
    """Count source failures reached after successful classification."""
    return sum(
        failure.scope is DiscoveryFailureScope.SOURCE
        and failure.stage in _POST_CLASSIFICATION_STAGES
        for failure in failures
    )


@dataclass(frozen=True, slots=True)
class DiscoveryCounts:
    """Deterministic aggregate counts for one discovery result."""

    files_seen: int
    supported: int
    unsupported: int
    hashed: int
    unchanged: int
    changed: int
    new: int
    duplicate: int
    ambiguous: int
    conflicting: int
    failed: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that counts are non-negative integers and internally consistent."""
        values = (
            self.files_seen,
            self.supported,
            self.unsupported,
            self.hashed,
            self.unchanged,
            self.changed,
            self.new,
            self.duplicate,
            self.ambiguous,
            self.conflicting,
            self.failed,
        )
        if any(not is_strict_integer(value) for value in values):
            raise TypeError("discovery counts must be integers.")
        if any(value < 0 for value in values):
            raise ValueError("discovery counts must be non-negative.")

        identity_outcomes = (
            self.unchanged
            + self.changed
            + self.new
            + self.duplicate
            + self.ambiguous
            + self.conflicting
        )
        if identity_outcomes != self.hashed:
            raise ValueError("identity outcome counts must equal hashed sources.")
        if self.hashed > self.supported:
            raise ValueError("hashed sources cannot exceed supported sources.")
        if self.supported + self.unsupported > self.files_seen:
            raise ValueError("classified sources cannot exceed files seen.")

    @classmethod
    def from_outcomes(
        cls,
        *,
        files_seen: int,
        sources: tuple[DiscoveredSource, ...],
        failures: tuple[DiscoveryFailure, ...],
    ) -> "DiscoveryCounts":
        """Derive all counts except the traversal-owned file inventory total."""
        _validate_sources(sources)
        _validate_failures(failures)
        unsupported = sum(
            failure.category in _UNSUPPORTED_CATEGORIES for failure in failures
        )
        return cls(
            files_seen=files_seen,
            supported=len(sources) + _supported_failure_count(failures),
            unsupported=unsupported,
            hashed=len(sources),
            unchanged=sum(
                source.identity_outcome.kind is ResolutionKind.UNCHANGED
                for source in sources
            ),
            changed=sum(
                source.identity_outcome.kind is ResolutionKind.NEW_VERSION
                for source in sources
            ),
            new=sum(
                source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
                for source in sources
            ),
            duplicate=sum(
                source.identity_outcome.kind is ResolutionKind.DUPLICATE
                for source in sources
            ),
            ambiguous=sum(
                source.identity_outcome.kind is ResolutionKind.AMBIGUOUS
                for source in sources
            ),
            conflicting=sum(
                source.identity_outcome.kind is ResolutionKind.CONFLICT
                for source in sources
            ),
            failed=len(failures) - unsupported,
        )

    @classmethod
    def empty(cls) -> "DiscoveryCounts":
        """Return aggregate counts for a successful empty input scope."""
        return cls.from_outcomes(files_seen=0, sources=(), failures=())


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Represent complete, partial, or failed discovery work for one corpus."""

    corpus_id: CorpusId
    policy: DiscoveryPolicy
    status: DiscoveryStatus
    sources: tuple[DiscoveredSource, ...]
    failures: tuple[DiscoveryFailure, ...]
    counts: DiscoveryCounts

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate ordering, aggregate counts, and status semantics."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.policy, DiscoveryPolicy):
            raise TypeError("policy must be a DiscoveryPolicy.")
        if not isinstance(self.status, DiscoveryStatus):
            raise TypeError("status must be a DiscoveryStatus.")
        _validate_sources(self.sources)
        _validate_failures(self.failures)
        if not isinstance(self.counts, DiscoveryCounts):
            raise TypeError("counts must be DiscoveryCounts.")
        self._validate_ordering()
        self._validate_counts()
        self._validate_status()

    def _validate_ordering(self) -> None:
        """Require stable order within outcomes and unique result positions."""
        source_ordinals = tuple(source.ordinal for source in self.sources)
        failure_ordinals = tuple(failure.ordinal for failure in self.failures)
        if source_ordinals != tuple(sorted(source_ordinals)):
            raise ValueError("sources must be in discovery ordinal order.")
        if failure_ordinals != tuple(sorted(failure_ordinals)):
            raise ValueError("failures must be in discovery ordinal order.")
        combined_ordinals = source_ordinals + failure_ordinals
        if len(combined_ordinals) != len(set(combined_ordinals)):
            raise ValueError("discovery ordinals must be unique across the result.")

    def _validate_counts(self) -> None:
        """Require aggregates to match the retained result exactly."""
        expected = DiscoveryCounts.from_outcomes(
            files_seen=self.counts.files_seen,
            sources=self.sources,
            failures=self.failures,
        )
        if self.counts != expected:
            raise ValueError("discovery counts must match result outcomes.")

    def _validate_status(self) -> None:
        """Validate completed, partial, and failed result invariants."""
        usable_source_count = sum(
            source.identity_outcome.kind
            not in {ResolutionKind.AMBIGUOUS, ResolutionKind.CONFLICT}
            for source in self.sources
        )
        has_request_failure = any(
            failure.scope is DiscoveryFailureScope.REQUEST for failure in self.failures
        )
        if has_request_failure and (
            self.status is not DiscoveryStatus.FAILED or usable_source_count
        ):
            raise ValueError("request failure requires a failed discovery result.")

        if self.status is DiscoveryStatus.COMPLETED:
            if self.failures:
                raise ValueError("completed discovery must not contain failures.")
            return
        if self.status is DiscoveryStatus.PARTIAL:
            if not self.failures or usable_source_count == 0:
                raise ValueError(
                    "partial discovery requires a usable source and a failure."
                )
            return
        if not self.failures:
            raise ValueError("failed discovery requires at least one failure.")
        if usable_source_count:
            raise ValueError("failed discovery must not contain usable sources.")
