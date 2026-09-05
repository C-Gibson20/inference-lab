"""Versioned, content-free classification of local source candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from inference_lab.ingest.discovery import (
    CURRENT_DISCOVERY_POLICY,
    SourceClassification,
    SourceSupport,
    _validate_semantic_version,
    _validate_slug,
)
from inference_lab.ingest.traversal import TraversalCandidate

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
