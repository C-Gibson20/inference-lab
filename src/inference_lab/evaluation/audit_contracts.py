"""Immutable input contracts for reproducible single-corpus audits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._validation import is_canonical_semantic_version, is_strict_integer
from inference_lab.evaluation.audit_capabilities import (
    ParserCapabilityProfile,
    parser_capability_profile,
    validate_parsed_document_capabilities,
)
from inference_lab.evaluation.audit_errors import (
    AuditContractError,
    AuditErrorCategory,
)
from inference_lab.ingest.classification import (
    ClassificationInventory,
    SourceFormatCount,
)
from inference_lab.ingest.discovery import DiscoveryPolicy, FamilyCandidate
from inference_lab.ingest.discovery_failures import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryFailureStage,
    discovery_failure_policy,
)
from inference_lab.ingest.discovery_results import DiscoveryCounts, DiscoveryResult
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentFamilyMembership,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
)

AUDIT_CONTRACT_VERSION: Final = "1.0.0"
_MEDIA_TYPE = re.compile(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+")
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


class AuditMissingReason(StrEnum):
    """Closed reasons for an expected audit observation being unavailable."""

    NOT_OBSERVED_IN_INTAKE = "not_observed_in_intake"
    PARSE_NOT_ATTEMPTED = "parse_not_attempted"
    PARSE_FAILED = "parse_failed"
    PARSER_FIELD_UNAVAILABLE = "parser_field_unavailable"
    UNSUPPORTED_FORMAT = "unsupported_format"
    SOURCE_FAILURE = "source_failure"


class AcquisitionStatus(StrEnum):
    """Explicit document acquisition assessments."""

    TEXT_NATIVE = "text_native"
    SCANNED = "scanned"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AuditStructureKind(StrEnum):
    """Structure kinds that an audit may report."""

    PAGE = "page"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    FIGURE = "figure"
    CLAIM = "claim"
    EXAMPLE = "example"
    REFERENCE = "reference"
    OTHER = "other"


def _validate_media_type(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("media_type must be a string.")
    if _MEDIA_TYPE.fullmatch(value) is None:
        raise ValueError("media_type must be a canonical lowercase media type.")


def _validate_slug(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _SLUG.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase hyphenated slug.")


def _validate_semver(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise ValueError(f"{field_name} must be a canonical semantic version.")


def _validate_non_negative_int(value: object, *, field_name: str) -> None:
    if (
        not is_strict_integer(value)
        or isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _membership_key(
    membership: CorpusDocumentVersionMembership,
) -> tuple[str, str]:
    return membership.document_id.as_str(), membership.document_version_id.as_str()


@dataclass(frozen=True, slots=True)
class CorpusAuditScope:
    """Assert one complete immutable corpus-version population."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    memberships: tuple[CorpusDocumentVersionMembership, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the corpus scope is complete, ordered, and unique."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.memberships, tuple):
            raise TypeError("memberships must be a tuple.")
        if any(
            not isinstance(item, CorpusDocumentVersionMembership)
            for item in self.memberships
        ):
            raise TypeError(
                "memberships must contain CorpusDocumentVersionMembership values."
            )
        if any(item.corpus_id != self.corpus_id for item in self.memberships):
            raise AuditContractError(AuditErrorCategory.CORPUS_MISMATCH)
        keys = tuple(_membership_key(item) for item in self.memberships)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)
        document_ids = tuple(item.document_id for item in self.memberships)
        if len(document_ids) != len(set(document_ids)):
            raise AuditContractError(AuditErrorCategory.MEMBERSHIP_MISMATCH)

    @property
    def document_version_ids(self) -> frozenset[DocumentVersionId]:
        """Return the selected immutable versions as a membership set."""
        return frozenset(item.document_version_id for item in self.memberships)

    @property
    def document_ids(self) -> frozenset[str]:
        """Return selected logical-document IDs for bounded validation."""
        return frozenset(item.document_id.as_str() for item in self.memberships)


@dataclass(frozen=True, slots=True)
class AuditSourceFailure:
    """Retain bounded discovery-failure metadata without source locators."""

    ordinal: int
    category: DiscoveryFailureCategory
    scope: DiscoveryFailureScope
    stage: DiscoveryFailureStage
    retryable: bool

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the discovery failure is well-formed and consistent."""
        _validate_non_negative_int(self.ordinal, field_name="ordinal")
        if not isinstance(self.category, DiscoveryFailureCategory):
            raise TypeError("category must be a DiscoveryFailureCategory.")
        if not isinstance(self.scope, DiscoveryFailureScope):
            raise TypeError("scope must be a DiscoveryFailureScope.")
        if not isinstance(self.stage, DiscoveryFailureStage):
            raise TypeError("stage must be a DiscoveryFailureStage.")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean.")
        if (self.scope, self.stage, self.retryable) != discovery_failure_policy(
            self.category
        ):
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)

    @classmethod
    def from_discovery_failure(cls, failure: DiscoveryFailure) -> "AuditSourceFailure":
        """Copy only safe fields needed for aggregate failure reporting."""
        if not isinstance(failure, DiscoveryFailure):
            raise TypeError("failure must be a DiscoveryFailure.")
        return cls(
            ordinal=failure.ordinal,
            category=failure.category,
            scope=failure.scope,
            stage=failure.stage,
            retryable=failure.retryable,
        )


@dataclass(frozen=True, slots=True, repr=False)
class AuditSourceInventory:
    """Store sanitised ING-02 intake aggregates for one audit corpus."""

    corpus_id: CorpusId
    policy: DiscoveryPolicy
    format_counts: tuple[SourceFormatCount, ...]
    discovery_counts: DiscoveryCounts
    failures: tuple[AuditSourceFailure, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the inventory is well-formed and consistent."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.policy, DiscoveryPolicy):
            raise TypeError("policy must be a DiscoveryPolicy.")
        if not isinstance(self.format_counts, tuple):
            raise TypeError("format_counts must be a tuple.")
        if any(not isinstance(item, SourceFormatCount) for item in self.format_counts):
            raise TypeError("format_counts must contain SourceFormatCount values.")
        if not isinstance(self.discovery_counts, DiscoveryCounts):
            raise TypeError("discovery_counts must be a DiscoveryCounts.")
        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")
        if any(not isinstance(item, AuditSourceFailure) for item in self.failures):
            raise TypeError("failures must contain AuditSourceFailure values.")
        failure_ordinals = tuple(item.ordinal for item in self.failures)
        if failure_ordinals != tuple(sorted(failure_ordinals)) or len(
            failure_ordinals
        ) != len(set(failure_ordinals)):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)

    @classmethod
    def from_ingest(
        cls,
        classification: ClassificationInventory,
        discovery: DiscoveryResult,
    ) -> "AuditSourceInventory":
        """Adapt ING-02 values and discard locators and source metadata."""
        if not isinstance(classification, ClassificationInventory):
            raise TypeError("classification must be a ClassificationInventory.")
        if not isinstance(discovery, DiscoveryResult):
            raise TypeError("discovery must be a DiscoveryResult.")
        if classification.traversal.corpus_id != discovery.corpus_id:
            raise AuditContractError(AuditErrorCategory.CORPUS_MISMATCH)
        if classification.policy != discovery.policy:
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        expected_ordinals = tuple(
            item.candidate.ordinal for item in classification.classified_candidates
        ) + tuple(item.ordinal for item in classification.traversal.failures)
        observed_ordinals = tuple(item.ordinal for item in discovery.sources) + tuple(
            item.ordinal for item in discovery.failures
        )
        if (
            len(observed_ordinals) != len(expected_ordinals)
            or set(observed_ordinals) != set(expected_ordinals)
            or discovery.counts.files_seen != len(expected_ordinals)
        ):
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        classification_failures = {
            (failure.ordinal, failure.category, failure.stage)
            for failure in classification.failures
        }
        discovery_failures = {
            (failure.ordinal, failure.category, failure.stage)
            for failure in discovery.failures
        }
        if not classification_failures <= discovery_failures:
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        classified_by_ordinal = {
            item.candidate.ordinal: item for item in classification.hashable_candidates
        }
        for source in discovery.sources:
            classified = classified_by_ordinal.get(source.ordinal)
            if classified is None:
                raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
            if (
                source.corpus_id != discovery.corpus_id
                or source.policy != discovery.policy
                or source.classification != classified.classification
                or source.snapshot.locator != classified.candidate.locator
            ):
                raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)

        return cls(
            corpus_id=discovery.corpus_id,
            policy=discovery.policy,
            format_counts=classification.format_counts,
            discovery_counts=discovery.counts,
            failures=tuple(
                AuditSourceFailure.from_discovery_failure(failure)
                for failure in discovery.failures
            ),
        )

    def __repr__(self) -> str:
        """Return aggregate counts without private locators or source metadata."""
        return (
            "AuditSourceInventory("
            f"corpus_id={self.corpus_id.as_str()!r}, "
            f"files_seen={self.discovery_counts.files_seen})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class AuditParseObservation:
    """Record one member's selected parser outcome or explicit absence."""

    membership: CorpusDocumentVersionMembership
    parser_version: ParserVersion | None = None
    media_type: str | None = None
    source_byte_count: int | None = None
    outcome: ParseOutcome | None = None
    missing_reason: AuditMissingReason | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the observation is well-formed and consistent."""
        if not isinstance(self.membership, CorpusDocumentVersionMembership):
            raise TypeError("membership must be a CorpusDocumentVersionMembership.")
        if self.parser_version is not None and not isinstance(
            self.parser_version, ParserVersion
        ):
            raise TypeError("parser_version must be a ParserVersion or None.")
        if self.media_type is not None:
            _validate_media_type(self.media_type)
        if self.source_byte_count is not None:
            _validate_non_negative_int(
                self.source_byte_count,
                field_name="source_byte_count",
            )
        if self.outcome is not None and not isinstance(self.outcome, ParseOutcome):
            raise TypeError("outcome must be a ParseOutcome or None.")
        if self.missing_reason is not None and not isinstance(
            self.missing_reason, AuditMissingReason
        ):
            raise TypeError("missing_reason must be an AuditMissingReason or None.")
        if (self.outcome is None) == (self.missing_reason is None):
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        if self.outcome is not None:
            if self.parser_version is None:
                raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
            document = self.outcome.document
            if document is not None:
                if (
                    document.document_id != self.membership.document_id
                    or document.document_version_id
                    != self.membership.document_version_id
                    or document.parser_version != self.parser_version
                ):
                    raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)

    def __repr__(self) -> str:
        """Return a representation without parser output or source payload."""
        status = self.outcome.status if self.outcome is not None else "missing"
        return (
            "AuditParseObservation("
            f"document_version_id={self.membership.document_version_id.as_str()!r}, "
            f"status={status!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class AuditAcquisitionObservation:
    """Record one explicit, versioned acquisition-status assessment."""

    membership: CorpusDocumentVersionMembership
    status: AcquisitionStatus
    producer_name: str
    producer_version: str
    unknown_reason: AuditMissingReason | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the observation is well-formed and consistent."""
        if not isinstance(self.membership, CorpusDocumentVersionMembership):
            raise TypeError("membership must be a CorpusDocumentVersionMembership.")
        if not isinstance(self.status, AcquisitionStatus):
            raise TypeError("status must be an AcquisitionStatus.")
        _validate_slug(self.producer_name, field_name="producer_name")
        _validate_semver(self.producer_version, field_name="producer_version")
        if self.unknown_reason is not None and not isinstance(
            self.unknown_reason, AuditMissingReason
        ):
            raise TypeError("unknown_reason must be an AuditMissingReason or None.")
        if self.status is AcquisitionStatus.UNKNOWN:
            if self.unknown_reason is None:
                raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        elif self.unknown_reason is not None:
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)

    def __repr__(self) -> str:
        """Return a representation containing only opaque identity and status."""
        return (
            "AuditAcquisitionObservation("
            f"document_version_id={self.membership.document_version_id.as_str()!r}, "
            f"status={self.status.value!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class AuditRelationshipInventory:
    """Retain only canonically ordered in-scope relationship evidence."""

    duplicate_relationships: tuple[DuplicateRelationship, ...] = ()
    family_memberships: tuple[DocumentFamilyMembership, ...] = ()
    family_candidates: tuple[FamilyCandidate, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the relationships are well-formed and consistent."""
        if not isinstance(self.duplicate_relationships, tuple):
            raise TypeError("duplicate_relationships must be a tuple.")
        if not isinstance(self.family_memberships, tuple):
            raise TypeError("family_memberships must be a tuple.")
        if not isinstance(self.family_candidates, tuple):
            raise TypeError("family_candidates must be a tuple.")
        if any(
            not isinstance(item, DuplicateRelationship)
            for item in self.duplicate_relationships
        ):
            raise TypeError("duplicate_relationships must contain valid values.")
        if any(
            item.duplicate_class is not DuplicateClass.BYTE_IDENTICAL
            for item in self.duplicate_relationships
        ):
            raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
        if any(
            not isinstance(item, DocumentFamilyMembership)
            for item in self.family_memberships
        ):
            raise TypeError("family_memberships must contain valid values.")
        if any(
            not isinstance(item, FamilyCandidate) for item in self.family_candidates
        ):
            raise TypeError("family_candidates must contain valid values.")
        duplicate_keys = tuple(item.key() for item in self.duplicate_relationships)
        if duplicate_keys != tuple(sorted(duplicate_keys)) or len(
            duplicate_keys
        ) != len(set(duplicate_keys)):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)
        family_keys = tuple(
            (
                item.family_id.as_str(),
                item.document_id.as_str(),
                item.document_version_id.as_str()
                if item.document_version_id is not None
                else "",
            )
            for item in self.family_memberships
        )
        if family_keys != tuple(sorted(family_keys)) or len(family_keys) != len(
            set(family_keys)
        ):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)
        candidate_keys = tuple(item.key() for item in self.family_candidates)
        if candidate_keys != tuple(sorted(candidate_keys)) or len(
            candidate_keys
        ) != len(set(candidate_keys)):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)

    def __repr__(self) -> str:
        """Return aggregate relationship counts without hidden endpoints."""
        return (
            "AuditRelationshipInventory("
            f"duplicate_count={len(self.duplicate_relationships)}, "
            f"family_count={len(self.family_memberships)}, "
            f"candidate_count={len(self.family_candidates)})"
        )


@dataclass(frozen=True, slots=True)
class AuditCoverage:
    """Reconcile one audit population's observation availability counts."""

    observed: int
    observed_absent: int
    unavailable: int
    missing: int
    failed: int
    invalid: int
    population: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the coverage counts are well-formed and consistent."""
        values = (
            self.observed,
            self.observed_absent,
            self.unavailable,
            self.missing,
            self.failed,
            self.invalid,
            self.population,
        )
        if any(not is_strict_integer(value) for value in values):
            raise TypeError("coverage values must be integers.")
        if any(value < 0 for value in values):
            raise ValueError("coverage values must be non-negative.")
        if (
            self.observed
            + self.observed_absent
            + self.unavailable
            + self.missing
            + self.failed
            + self.invalid
            != self.population
        ):
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)


@dataclass(frozen=True, slots=True)
class AuditRate:
    """Represent an exact integer numerator and denominator."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the rate is well-formed and consistent."""
        _validate_non_negative_int(self.numerator, field_name="numerator")
        _validate_non_negative_int(self.denominator, field_name="denominator")
        if self.numerator > self.denominator:
            raise ValueError("numerator must not exceed denominator.")


@dataclass(frozen=True, slots=True)
class AuditDistribution:
    """Represent a deterministic non-negative integer distribution summary."""

    count: int
    total: int
    minimum: int | None
    maximum: int | None
    zero_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the distribution is well-formed and consistent."""
        _validate_non_negative_int(self.count, field_name="count")
        _validate_non_negative_int(self.total, field_name="total")
        _validate_non_negative_int(self.zero_count, field_name="zero_count")
        if self.zero_count > self.count:
            raise ValueError("zero_count must not exceed count.")
        if self.count == 0:
            if self.minimum is not None or self.maximum is not None or self.total:
                raise ValueError(
                    "empty distributions require null extrema and zero total."
                )
        else:
            if self.minimum is None or self.maximum is None:
                raise ValueError("non-empty distributions require extrema.")
            _validate_non_negative_int(self.minimum, field_name="minimum")
            _validate_non_negative_int(self.maximum, field_name="maximum")
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum.")


@dataclass(frozen=True, slots=True)
class CorpusAuditPolicy:
    """Version every aggregation, rate, distribution, and rendering rule."""

    policy_name: str = "single-corpus-audit"
    policy_version: str = AUDIT_CONTRACT_VERSION
    availability_version: str = "1.0.0"
    rate_decimal_places: int = 2
    rate_rounding: str = "half_up"
    quantile_method: str = "nearest_rank"
    duplicate_grouping_algorithm: str = "connected_components"
    rendering_version: str = "1.0.0"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the policy is well-formed and consistent."""
        _validate_slug(self.policy_name, field_name="policy_name")
        for field_name in (
            "policy_version",
            "availability_version",
            "rendering_version",
        ):
            _validate_semver(getattr(self, field_name), field_name=field_name)
        _validate_non_negative_int(
            self.rate_decimal_places,
            field_name="rate_decimal_places",
        )
        if self.rate_decimal_places > 12:
            raise ValueError("rate_decimal_places must not exceed 12.")
        if self.rate_rounding != "half_up":
            raise ValueError("rate_rounding must be half_up.")
        if self.quantile_method != "nearest_rank":
            raise ValueError("quantile_method must be nearest_rank.")
        if self.duplicate_grouping_algorithm != "connected_components":
            raise ValueError(
                "duplicate_grouping_algorithm must be connected_components."
            )


@dataclass(frozen=True, slots=True, repr=False)
class CorpusAuditRequest:
    """Validate the complete, storage-neutral input to one corpus audit."""

    scope: CorpusAuditScope
    source_inventory: AuditSourceInventory
    parse_observations: tuple[AuditParseObservation, ...]
    acquisition_observations: tuple[AuditAcquisitionObservation, ...] = ()
    relationships: AuditRelationshipInventory = AuditRelationshipInventory()
    capability_profiles: tuple[ParserCapabilityProfile, ...] = ()
    policy: CorpusAuditPolicy = CorpusAuditPolicy()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the request is well-formed and consistent."""
        if not isinstance(self.scope, CorpusAuditScope):
            raise TypeError("scope must be a CorpusAuditScope.")
        if not isinstance(self.source_inventory, AuditSourceInventory):
            raise TypeError("source_inventory must be an AuditSourceInventory.")
        if self.source_inventory.corpus_id != self.scope.corpus_id:
            raise AuditContractError(AuditErrorCategory.CORPUS_MISMATCH)
        self._validate_observations()
        self._validate_relationships()
        self._validate_capabilities()
        if not isinstance(self.policy, CorpusAuditPolicy):
            raise TypeError("policy must be a CorpusAuditPolicy.")

    def _validate_observations(self) -> None:
        if not isinstance(self.parse_observations, tuple):
            raise TypeError("parse_observations must be a tuple.")
        if not isinstance(self.acquisition_observations, tuple):
            raise TypeError("acquisition_observations must be a tuple.")
        if any(
            not isinstance(item, AuditParseObservation)
            for item in self.parse_observations
        ):
            raise TypeError("parse_observations must contain valid values.")
        if any(
            not isinstance(item, AuditAcquisitionObservation)
            for item in self.acquisition_observations
        ):
            raise TypeError("acquisition_observations must contain valid values.")
        expected = self.scope.document_version_ids
        observed = tuple(
            item.membership.document_version_id for item in self.parse_observations
        )
        if frozenset(observed) != expected or len(observed) != len(expected):
            raise AuditContractError(AuditErrorCategory.MEMBERSHIP_MISMATCH)
        if tuple(observed) != tuple(
            item.document_version_id for item in self.scope.memberships
        ):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)
        for item in self.parse_observations:
            if item.membership not in self.scope.memberships:
                raise AuditContractError(AuditErrorCategory.MEMBERSHIP_MISMATCH)
        acquisition_versions = tuple(
            item.membership.document_version_id
            for item in self.acquisition_observations
        )
        if len(acquisition_versions) != len(set(acquisition_versions)):
            raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)
        if any(version not in expected for version in acquisition_versions):
            raise AuditContractError(AuditErrorCategory.MEMBERSHIP_MISMATCH)

    def _validate_relationships(self) -> None:
        if not isinstance(self.relationships, AuditRelationshipInventory):
            raise TypeError("relationships must be an AuditRelationshipInventory.")
        expected_versions = self.scope.document_version_ids
        for relationship in self.relationships.duplicate_relationships:
            if (
                not {
                    relationship.first_version_id,
                    relationship.second_version_id,
                }
                <= expected_versions
            ):
                raise AuditContractError(AuditErrorCategory.RELATIONSHIP_OUTSIDE_SCOPE)
        expected_documents = self.scope.document_ids
        for membership in self.relationships.family_memberships:
            if membership.document_id.as_str() not in expected_documents:
                raise AuditContractError(AuditErrorCategory.RELATIONSHIP_OUTSIDE_SCOPE)
            if membership.document_version_id is not None:
                selected = {
                    item.document_version_id
                    for item in self.scope.memberships
                    if item.document_id == membership.document_id
                }
                if membership.document_version_id not in selected:
                    raise AuditContractError(
                        AuditErrorCategory.RELATIONSHIP_OUTSIDE_SCOPE
                    )
        for candidate in self.relationships.family_candidates:
            if (
                candidate.document_id is not None
                and candidate.document_id.as_str() not in expected_documents
            ):
                raise AuditContractError(AuditErrorCategory.RELATIONSHIP_OUTSIDE_SCOPE)
            if (
                candidate.document_version_id is not None
                and candidate.document_version_id not in expected_versions
            ):
                raise AuditContractError(AuditErrorCategory.RELATIONSHIP_OUTSIDE_SCOPE)

    def _validate_capabilities(self) -> None:
        if not isinstance(self.capability_profiles, tuple):
            raise TypeError("capability_profiles must be a tuple.")
        if any(
            not isinstance(item, ParserCapabilityProfile)
            for item in self.capability_profiles
        ):
            raise TypeError("capability_profiles must contain valid values.")
        profile_keys = tuple(
            item.parser_version.value for item in self.capability_profiles
        )
        if profile_keys != tuple(sorted(profile_keys)) or len(profile_keys) != len(
            set(profile_keys)
        ):
            raise AuditContractError(AuditErrorCategory.INVALID_ORDERING)
        by_version = {
            item.parser_version.value: item for item in self.capability_profiles
        }
        for observation in self.parse_observations:
            if observation.parser_version is None:
                continue
            profile = by_version.get(observation.parser_version.value)
            if profile is None:
                raise AuditContractError(AuditErrorCategory.UNKNOWN_PARSER_CAPABILITY)
            if (
                observation.outcome is not None
                and observation.outcome.document is not None
            ):
                validate_parsed_document_capabilities(
                    observation.outcome.document,
                    profile,
                )
            for acquisition in self.acquisition_observations:
                if acquisition.membership != observation.membership:
                    continue
                if acquisition.status.value not in profile.acquisition_statuses:
                    raise AuditContractError(AuditErrorCategory.UNSUPPORTED_OBSERVATION)
        for profile in self.capability_profiles:
            registered = parser_capability_profile(profile.parser_version)
            if registered is not None and registered != profile:
                raise AuditContractError(AuditErrorCategory.OBSERVATION_MISMATCH)

    def __repr__(self) -> str:
        """Return aggregate identifiers and counts without audit payloads."""
        return (
            "CorpusAuditRequest("
            f"corpus_id={self.scope.corpus_id.as_str()!r}, "
            f"corpus_version={self.scope.corpus_version.as_str()!r}, "
            f"member_count={len(self.scope.memberships)}, "
            f"source_files_seen={self.source_inventory.discovery_counts.files_seen})"
        )
