"""Versioned, content-free classification of local source candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.discovery import (
    CURRENT_DISCOVERY_POLICY,
    SourceClassification,
    SourceSupport,
    _validate_media_type,
    _validate_semantic_version,
    _validate_slug,
)
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from inference_lab.ingest.traversal import TraversalCandidate, TraversalInventory

_UNKNOWN_MEDIA_TYPE: Final = "application/octet-stream"
_CANONICAL_SUFFIX: Final = re.compile(r"\.[a-z0-9][a-z0-9_+-]*", re.ASCII)


@dataclass(frozen=True, slots=True)
class SourceClassifierRule:
    """Map one canonical ASCII suffix to immutable format metadata."""

    suffix: str
    classification: SourceClassification

    def __post_init__(self) -> None:
        """Validate the canonical suffix and classification contract."""
        if not isinstance(self.suffix, str):
            raise TypeError("suffix must be a string.")
        if _CANONICAL_SUFFIX.fullmatch(self.suffix) is None:
            raise ValueError("suffix must be a canonical lowercase ASCII suffix.")
        if not isinstance(self.classification, SourceClassification):
            raise TypeError("classification must be a SourceClassification.")
        if self.classification.support is SourceSupport.UNKNOWN:
            raise ValueError("classifier rules must describe recognised formats.")


@dataclass(frozen=True, slots=True)
class SourceClassifierRegistry:
    """Hold one immutable, versioned set of suffix-classification rules."""

    classifier_name: str
    classifier_version: str
    rules: tuple[SourceClassifierRule, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate registry identity, rules, and deterministic ordering."""
        _validate_slug(self.classifier_name, field_name="classifier_name")
        _validate_semantic_version(
            self.classifier_version,
            field_name="classifier_version",
        )
        if not isinstance(self.rules, tuple):
            raise TypeError("rules must be a tuple.")
        if not self.rules:
            raise ValueError("rules must contain at least one classifier rule.")
        if any(not isinstance(rule, SourceClassifierRule) for rule in self.rules):
            raise TypeError("rules must contain SourceClassifierRule values.")
        suffixes = tuple(rule.suffix for rule in self.rules)
        if suffixes != tuple(sorted(set(suffixes))):
            raise ValueError("classifier rule suffixes must be unique and ordered.")
        if any(
            rule.classification.classifier_name != self.classifier_name
            or rule.classification.classifier_version != self.classifier_version
            for rule in self.rules
        ):
            raise ValueError("classifier rules must match the registry identity.")

    def classify(self, candidate: TraversalCandidate) -> SourceClassification:
        """Classify one candidate from its final ASCII suffix without reading it."""
        if not isinstance(candidate, TraversalCandidate):
            raise TypeError("candidate must be a TraversalCandidate.")
        suffix = candidate.locator.relative_path.suffix
        if suffix.isascii():
            canonical_suffix = suffix.lower()
            for rule in self.rules:
                if rule.suffix == canonical_suffix:
                    return rule.classification
        return SourceClassification(
            media_type=_UNKNOWN_MEDIA_TYPE,
            support=SourceSupport.UNKNOWN,
            classifier_name=self.classifier_name,
            classifier_version=self.classifier_version,
        )


@dataclass(frozen=True, slots=True)
class ClassifiedSourceCandidate:
    """Pair one traversed source with its auditable format classification."""

    candidate: TraversalCandidate
    classification: SourceClassification

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the candidate and classification types."""
        if not isinstance(self.candidate, TraversalCandidate):
            raise TypeError("candidate must be a TraversalCandidate.")
        if not isinstance(self.classification, SourceClassification):
            raise TypeError("classification must be a SourceClassification.")


@dataclass(frozen=True, slots=True)
class SourceFormatCount:
    """Count files and safe stat bytes for one media-type/support pair."""

    media_type: str
    support: SourceSupport
    file_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate bounded format metadata and non-negative aggregates."""
        _validate_media_type(self.media_type)
        if not isinstance(self.support, SourceSupport):
            raise TypeError("support must be a SourceSupport.")
        for field_name, value in (
            ("file_count", self.file_count),
            ("total_bytes", self.total_bytes),
        ):
            if not is_strict_integer(value):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.file_count == 0:
            raise ValueError("file_count must be positive.")


def _classification_failure(item: ClassifiedSourceCandidate) -> DiscoveryFailure:
    """Build the bounded failure for one non-parseable classification."""
    if item.classification.support is SourceSupport.SUPPORTED:
        raise ValueError("supported candidates do not have classification failures.")
    category = (
        DiscoveryFailureCategory.RECOGNISED_UNSUPPORTED_FORMAT
        if item.classification.support is SourceSupport.RECOGNISED_UNSUPPORTED
        else DiscoveryFailureCategory.UNKNOWN_FORMAT
    )
    return DiscoveryFailure.for_category(
        ordinal=item.candidate.ordinal,
        category=category,
        locator=item.candidate.locator,
        file_kind=DiscoveryFileKind.REGULAR_FILE,
        media_type=item.classification.media_type,
    )


def _format_counts(
    classified_candidates: tuple[ClassifiedSourceCandidate, ...],
) -> tuple[SourceFormatCount, ...]:
    """Aggregate safe stat sizes in deterministic media-type/support order."""
    grouped: dict[tuple[str, SourceSupport], tuple[int, int]] = {}
    for item in classified_candidates:
        key = (item.classification.media_type, item.classification.support)
        file_count, total_bytes = grouped.get(key, (0, 0))
        grouped[key] = (
            file_count + 1,
            total_bytes + item.candidate.stat_evidence.byte_count,
        )
    return tuple(
        SourceFormatCount(
            media_type=media_type,
            support=support,
            file_count=file_count,
            total_bytes=total_bytes,
        )
        for (media_type, support), (file_count, total_bytes) in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], item[0][1].value),
        )
    )


@dataclass(frozen=True, slots=True)
class ClassificationInventory:
    """Retain complete classifications, failures, and hash-eligible sources."""

    classifier_name: str
    classifier_version: str
    traversal: TraversalInventory
    classified_candidates: tuple[ClassifiedSourceCandidate, ...]
    failures: tuple[DiscoveryFailure, ...]
    format_counts: tuple[SourceFormatCount, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate policy identity and exact accounting of traversal entries."""
        _validate_slug(self.classifier_name, field_name="classifier_name")
        _validate_semantic_version(
            self.classifier_version,
            field_name="classifier_version",
        )
        if not isinstance(self.traversal, TraversalInventory):
            raise TypeError("traversal must be a TraversalInventory.")
        if not isinstance(self.classified_candidates, tuple):
            raise TypeError("classified_candidates must be a tuple.")
        if any(
            not isinstance(item, ClassifiedSourceCandidate)
            for item in self.classified_candidates
        ):
            raise TypeError(
                "classified_candidates must contain ClassifiedSourceCandidate values."
            )
        if (
            tuple(item.candidate for item in self.classified_candidates)
            != self.traversal.candidates
        ):
            raise ValueError(
                "classified candidates must exactly match traversal order."
            )
        if any(
            item.classification.classifier_name != self.classifier_name
            or item.classification.classifier_version != self.classifier_version
            for item in self.classified_candidates
        ):
            raise ValueError("classifications must match the inventory identity.")
        self._validate_failures()
        self._validate_format_counts()

    def _validate_failures(self) -> None:
        """Validate ordered traversal and classification failures exactly once."""
        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(failure, DiscoveryFailure) for failure in self.failures):
            raise TypeError("failures must contain DiscoveryFailure values.")
        expected = self.traversal.failures + tuple(
            _classification_failure(item)
            for item in self.classified_candidates
            if item.classification.support is not SourceSupport.SUPPORTED
        )
        expected = tuple(sorted(expected, key=lambda failure: failure.ordinal))
        if self.failures != expected:
            raise ValueError("failures must exactly account for classified traversal.")

    def _validate_format_counts(self) -> None:
        """Validate deterministic safe format aggregates."""
        if not isinstance(self.format_counts, tuple):
            raise TypeError("format_counts must be a tuple.")
        if any(not isinstance(item, SourceFormatCount) for item in self.format_counts):
            raise TypeError("format_counts must contain SourceFormatCount values.")
        if self.format_counts != _format_counts(self.classified_candidates):
            raise ValueError("format_counts must match classified candidates.")

    @property
    def hashable_candidates(self) -> tuple[ClassifiedSourceCandidate, ...]:
        """Return only candidates with an available parser capability."""
        return tuple(
            item
            for item in self.classified_candidates
            if item.classification.support is SourceSupport.SUPPORTED
        )


def _classification(
    *,
    media_type: str,
    support: SourceSupport,
    parser_capability: str | None = None,
    encoding: str | None = None,
) -> SourceClassification:
    """Build one baseline classification under the current policy identity."""
    return SourceClassification(
        media_type=media_type,
        support=support,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability=parser_capability,
        encoding=encoding,
    )


CURRENT_SOURCE_CLASSIFIER: Final = SourceClassifierRegistry(
    classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
    classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
    rules=(
        SourceClassifierRule(
            suffix=".pdf",
            classification=_classification(
                media_type="application/pdf",
                support=SourceSupport.RECOGNISED_UNSUPPORTED,
            ),
        ),
        SourceClassifierRule(
            suffix=".txt",
            classification=_classification(
                media_type="text/plain",
                support=SourceSupport.SUPPORTED,
                parser_capability="text-native",
                encoding="utf-8",
            ),
        ),
    ),
)


def classify_source_candidate(
    candidate: TraversalCandidate,
    *,
    registry: SourceClassifierRegistry = CURRENT_SOURCE_CLASSIFIER,
) -> SourceClassification:
    """Classify one local candidate with an explicit versioned registry."""
    if not isinstance(registry, SourceClassifierRegistry):
        raise TypeError("registry must be a SourceClassifierRegistry.")
    return registry.classify(candidate)


def classify_source_inventory(
    traversal: TraversalInventory,
    *,
    registry: SourceClassifierRegistry = CURRENT_SOURCE_CLASSIFIER,
) -> ClassificationInventory:
    """Classify every traversed file and isolate non-parseable outcomes."""
    if not isinstance(traversal, TraversalInventory):
        raise TypeError("traversal must be a TraversalInventory.")
    if not isinstance(registry, SourceClassifierRegistry):
        raise TypeError("registry must be a SourceClassifierRegistry.")
    classified = tuple(
        ClassifiedSourceCandidate(
            candidate=candidate,
            classification=registry.classify(candidate),
        )
        for candidate in traversal.candidates
    )
    failures = traversal.failures + tuple(
        _classification_failure(item)
        for item in classified
        if item.classification.support is not SourceSupport.SUPPORTED
    )
    failures = tuple(sorted(failures, key=lambda failure: failure.ordinal))
    return ClassificationInventory(
        classifier_name=registry.classifier_name,
        classifier_version=registry.classifier_version,
        traversal=traversal,
        classified_candidates=classified,
        failures=failures,
        format_counts=_format_counts(classified),
    )
