"""Immutable contracts for reproducible retrieval evaluation runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Final, cast

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab._validation import (
    is_canonical_semantic_version,
    is_lowercase_slug,
    is_strict_integer,
)
from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.contracts import SplitName
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    EvidenceHit,
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalChannel,
)

RETRIEVAL_EVALUATION_SCHEMA_VERSION: Final[str] = "1.0.0"
RETRIEVAL_EVALUATION_IMPLEMENTATION: Final[str] = "retrieval-evaluation"
RETRIEVAL_EVALUATION_IMPLEMENTATION_VERSION: Final[str] = "1.0.0"
RELATIONSHIP_SNAPSHOT_SCHEMA_VERSION: Final[str] = "1.0.0"
RELATIONSHIP_POLICY_VERSION: Final[str] = "1.0.0"
DOCUMENT_RECALL_POLICY_VERSION: Final[str] = "1.0.0"
PASSAGE_RECALL_POLICY_VERSION: Final[str] = "1.0.0"
DUPLICATE_POLICY_VERSION: Final[str] = "1.0.0"
FAMILY_POLICY_VERSION: Final[str] = "1.0.0"
AGGREGATION_POLICY_VERSION: Final[str] = "1.0.0"
PUBLISHED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "run.json",
    "cases.jsonl",
    "report.json",
    "report.md",
)

_CASE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_OPAQUE_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]*$"
)
_ROUNDING_MODES: Final[frozenset[str]] = frozenset(
    {
        "ROUND_CEILING",
        "ROUND_DOWN",
        "ROUND_FLOOR",
        "ROUND_HALF_DOWN",
        "ROUND_HALF_EVEN",
        "ROUND_HALF_UP",
        "ROUND_UP",
        "ROUND_05UP",
    }
)


class RetrievalEvaluationContractError(ValueError):
    """Raised when a retrieval-evaluation contract invariant is violated."""


def _require_type(value: object, expected: type[object], field_name: str) -> None:
    """Require one exact contract type at a runtime boundary."""
    if not isinstance(value, expected):
        raise TypeError(f"{field_name} must be a {expected.__name__}.")


def _require_tuple(value: object, field_name: str) -> tuple[object, ...]:
    """Require an immutable tuple and return it for validation."""
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple.")
    return cast(tuple[object, ...], value)


def _require_semver(value: object, field_name: str) -> str:
    """Validate one canonical Semantic Version value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise RetrievalEvaluationContractError(
            f"{field_name} must be a canonical Semantic Version."
        )
    return value


def _require_slug(value: object, field_name: str) -> str:
    """Validate one project-owned lowercase hyphenated slug."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_lowercase_slug(value):
        raise RetrievalEvaluationContractError(
            f"{field_name} must be a lowercase hyphenated slug."
        )
    return value


def _require_checksum(value: object, field_name: str) -> str:
    """Validate one path-free SHA-256 checksum."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_sha256_checksum(value):
        raise RetrievalEvaluationContractError(
            f"{field_name} must be a SHA-256 checksum."
        )
    return value


def _require_opaque_token(value: object, field_name: str) -> str:
    """Validate one safe identity token without accepting a path."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _OPAQUE_TOKEN_PATTERN.fullmatch(value) is None:
        raise RetrievalEvaluationContractError(
            f"{field_name} must be a path-free identity token."
        )
    return value


def _require_non_negative_integer(value: object, field_name: str) -> int:
    """Validate a non-negative integer while rejecting booleans."""
    if not is_strict_integer(value):
        raise TypeError(f"{field_name} must be an integer.")
    value = cast(int, value)
    if value < 0:
        raise RetrievalEvaluationContractError(f"{field_name} must not be negative.")
    return value


def _require_positive_integer(value: object, field_name: str) -> int:
    """Validate a strictly positive integer while rejecting booleans."""
    value = _require_non_negative_integer(value, field_name)
    if value == 0:
        raise RetrievalEvaluationContractError(f"{field_name} must be positive.")
    return value


def _require_case_id(value: object, field_name: str = "case_id") -> str:
    """Validate the immutable EVAL-01 case identifier shape."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _CASE_ID_PATTERN.fullmatch(value) is None:
        raise RetrievalEvaluationContractError(
            f"{field_name} must use the canonical case identifier shape."
        )
    return value


def _require_unique(values: tuple[object, ...], field_name: str) -> None:
    """Reject duplicate values in an ordered tuple."""
    try:
        unique = len(set(values)) == len(values)
    except TypeError as error:
        raise TypeError(f"{field_name} values must be hashable.") from error
    if not unique:
        raise RetrievalEvaluationContractError(f"{field_name} must be unique.")


def _digest_payload(payload: JsonObject) -> str:
    """Return the canonical SHA-256 identity of one safe JSON payload."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _source_sort_key(source: "SanitisedSourceReference") -> tuple[str, str, str]:
    """Return canonical provenance ordering for a source reference."""
    return (
        source.document_id.as_str(),
        source.document_version_id.as_str(),
        source.source_span_id.as_str(),
    )


class MetricFraction:
    """An exact, applicable metric value represented by integer counts."""

    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator: int, denominator: int) -> None:
        self._validate(numerator, denominator)
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    def _validate(self, numerator: object, denominator: object) -> None:
        """Validate the fraction's exact count semantics."""
        numerator = _require_non_negative_integer(numerator, "numerator")
        denominator = _require_positive_integer(denominator, "denominator")
        if numerator > denominator:
            raise RetrievalEvaluationContractError(
                "numerator must not exceed denominator."
            )

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent mutation after construction."""
        raise AttributeError("MetricFraction is immutable.")

    def __eq__(self, other: object) -> bool:
        """Compare exact numerator and denominator values."""
        return (
            isinstance(other, MetricFraction)
            and self.numerator == other.numerator
            and self.denominator == other.denominator
        )

    def __hash__(self) -> int:
        """Return a hash consistent with exact fraction equality."""
        return hash((self.numerator, self.denominator))

    def as_dict(self) -> dict[str, int]:
        """Return the payload-free authoritative metric counts."""
        return {"numerator": self.numerator, "denominator": self.denominator}

    def decimal_value(self, *, precision: int, rounding_mode: str) -> str:
        """Render the fraction with explicit Decimal arithmetic and rounding."""
        _require_non_negative_integer(precision, "precision")
        if rounding_mode not in _ROUNDING_MODES:
            raise RetrievalEvaluationContractError("Unsupported decimal rounding mode.")
        with localcontext() as context:
            context.prec = max(
                28,
                len(str(self.numerator)) + len(str(self.denominator)) + precision + 4,
            )
            quantum = Decimal(1).scaleb(-precision)
            value = (Decimal(self.numerator) / Decimal(self.denominator)).quantize(
                quantum,
                rounding=rounding_mode,
            )
        return format(value, f".{precision}f")

    def __repr__(self) -> str:
        """Return a safe developer-facing representation."""
        return f"MetricFraction({self.numerator}, {self.denominator})"


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationPolicy:
    """Versioned metric, candidate-depth, and decimal-rendering policy."""

    passage_cutoffs: tuple[int, ...]
    candidate_depth: int
    document_cutoff: int = 10
    document_policy_version: str = DOCUMENT_RECALL_POLICY_VERSION
    passage_policy_version: str = PASSAGE_RECALL_POLICY_VERSION
    duplicate_policy_version: str = DUPLICATE_POLICY_VERSION
    family_policy_version: str = FAMILY_POLICY_VERSION
    aggregation_policy_version: str = AGGREGATION_POLICY_VERSION
    decimal_precision: int = 6
    rounding_mode: str = "ROUND_HALF_UP"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the closed evaluation policy."""
        cutoffs = cast(
            tuple[int, ...],
            _require_tuple(self.passage_cutoffs, "passage_cutoffs"),
        )
        if not cutoffs:
            raise RetrievalEvaluationContractError("passage_cutoffs must not be empty.")
        if any(not is_strict_integer(cutoff) for cutoff in cutoffs):
            raise TypeError("passage_cutoffs must contain only integers.")
        if any(cutoff <= 0 for cutoff in cutoffs):
            raise RetrievalEvaluationContractError(
                "passage_cutoffs must contain positive values."
            )
        if cutoffs != tuple(sorted(set(cutoffs))):
            raise RetrievalEvaluationContractError(
                "passage_cutoffs must be strictly increasing and unique."
            )
        _require_positive_integer(self.candidate_depth, "candidate_depth")
        _require_positive_integer(self.document_cutoff, "document_cutoff")
        if self.document_cutoff != 10:
            raise RetrievalEvaluationContractError("document_cutoff is fixed at 10.")
        if self.candidate_depth < max(self.document_cutoff, max(cutoffs)):
            raise RetrievalEvaluationContractError(
                "candidate_depth must cover the document and passage cut-offs."
            )
        for field_name in (
            "document_policy_version",
            "passage_policy_version",
            "duplicate_policy_version",
            "family_policy_version",
            "aggregation_policy_version",
        ):
            _require_semver(getattr(self, field_name), field_name)
        _require_non_negative_integer(self.decimal_precision, "decimal_precision")
        if self.rounding_mode not in _ROUNDING_MODES:
            raise RetrievalEvaluationContractError(
                "rounding_mode must name a supported Decimal rounding mode."
            )

    def as_dict(self) -> dict[str, object]:
        """Return the complete semantic policy identity."""
        return {
            "passage_cutoffs": list(self.passage_cutoffs),
            "candidate_depth": self.candidate_depth,
            "document_cutoff": self.document_cutoff,
            "document_policy_version": self.document_policy_version,
            "passage_policy_version": self.passage_policy_version,
            "duplicate_policy_version": self.duplicate_policy_version,
            "family_policy_version": self.family_policy_version,
            "aggregation_policy_version": self.aggregation_policy_version,
            "decimal_precision": self.decimal_precision,
            "rounding_mode": self.rounding_mode,
        }


@dataclass(frozen=True, slots=True)
class RetrievalComponentIdentity:
    """Path-free identity for one retrieval index or model component."""

    component: str
    implementation: str
    format_version: str
    contract_version: str
    configuration_digest: str
    generation_id: str | None = None
    source_manifest_digest: str | None = None
    model_identity: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate component identity and compatibility metadata."""
        _require_slug(self.component, "component")
        _require_slug(self.implementation, "implementation")
        _require_semver(self.format_version, "format_version")
        _require_semver(self.contract_version, "contract_version")
        _require_checksum(self.configuration_digest, "configuration_digest")
        self._validate_optional_token(self.generation_id, "generation_id")
        self._validate_optional_checksum(
            self.source_manifest_digest,
            "source_manifest_digest",
        )
        self._validate_optional_token(self.model_identity, "model_identity")
        self._validate_optional_token(self.model_revision, "model_revision")
        if (self.model_identity is None) != (self.model_revision is None):
            raise RetrievalEvaluationContractError(
                "model_identity and model_revision must be supplied together."
            )

    @staticmethod
    def _validate_optional_token(value: str | None, field_name: str) -> None:
        """Validate an optional path-free token."""
        if value is not None:
            _require_opaque_token(value, field_name)

    @staticmethod
    def _validate_optional_checksum(value: str | None, field_name: str) -> None:
        """Validate an optional checksum."""
        if value is not None:
            _require_checksum(value, field_name)

    def as_dict(self) -> dict[str, str | None]:
        """Return the path-free component identity."""
        return {
            "component": self.component,
            "implementation": self.implementation,
            "format_version": self.format_version,
            "contract_version": self.contract_version,
            "configuration_digest": self.configuration_digest,
            "generation_id": self.generation_id,
            "source_manifest_digest": self.source_manifest_digest,
            "model_identity": self.model_identity,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True, slots=True)
class RetrievalSystemIdentity:
    """Complete path-free identity of the retriever invoked by a run."""

    adapter_implementation: str
    adapter_contract_version: str
    query_construction_algorithm: str
    query_construction_version: str
    retrieval_algorithm: str
    retrieval_algorithm_version: str
    configuration_digest: str
    components: tuple[RetrievalComponentIdentity, ...]
    model_identity: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate every identity component before a run starts."""
        _require_slug(self.adapter_implementation, "adapter_implementation")
        _require_semver(self.adapter_contract_version, "adapter_contract_version")
        _require_slug(self.query_construction_algorithm, "query_construction_algorithm")
        _require_semver(self.query_construction_version, "query_construction_version")
        _require_slug(self.retrieval_algorithm, "retrieval_algorithm")
        _require_semver(self.retrieval_algorithm_version, "retrieval_algorithm_version")
        _require_checksum(self.configuration_digest, "configuration_digest")
        components = cast(
            tuple[RetrievalComponentIdentity, ...],
            _require_tuple(self.components, "components"),
        )
        if any(
            not isinstance(component, RetrievalComponentIdentity)
            for component in components
        ):
            raise TypeError(
                "components must contain only RetrievalComponentIdentity values."
            )
        component_names = tuple(component.component for component in components)
        _require_unique(component_names, "component names")
        if (self.model_identity is None) != (self.model_revision is None):
            raise RetrievalEvaluationContractError(
                "model_identity and model_revision must be supplied together."
            )
        if self.model_identity is not None:
            _require_opaque_token(self.model_identity, "model_identity")
            assert self.model_revision is not None
            _require_opaque_token(self.model_revision, "model_revision")

    def as_dict(self) -> dict[str, object]:
        """Return the complete path-free system identity."""
        return {
            "adapter_implementation": self.adapter_implementation,
            "adapter_contract_version": self.adapter_contract_version,
            "query_construction_algorithm": self.query_construction_algorithm,
            "query_construction_version": self.query_construction_version,
            "retrieval_algorithm": self.retrieval_algorithm,
            "retrieval_algorithm_version": self.retrieval_algorithm_version,
            "configuration_digest": self.configuration_digest,
            "components": [component.as_dict() for component in self.components],
            "model_identity": self.model_identity,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True, slots=True)
class DuplicateRelationshipSnapshot:
    """Validated duplicate and family relationships for one corpus version."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    memberships: tuple[CorpusDocumentVersionMembership, ...]
    duplicate_relationships: tuple[DuplicateRelationship, ...] = ()
    family_memberships: tuple[DocumentFamilyMembership, ...] = ()
    schema_version: str = RELATIONSHIP_SNAPSHOT_SCHEMA_VERSION
    policy_version: str = RELATIONSHIP_POLICY_VERSION
    content_digest: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate visibility, ownership, and deterministic relationship shape."""
        _require_semver(self.schema_version, "schema_version")
        _require_semver(self.policy_version, "policy_version")
        _require_type(self.corpus_id, CorpusId, "corpus_id")
        _require_type(self.corpus_version, CorpusVersionId, "corpus_version")
        memberships = cast(
            tuple[CorpusDocumentVersionMembership, ...],
            _require_tuple(self.memberships, "memberships"),
        )
        duplicates = cast(
            tuple[DuplicateRelationship, ...],
            _require_tuple(self.duplicate_relationships, "duplicate_relationships"),
        )
        families = cast(
            tuple[DocumentFamilyMembership, ...],
            _require_tuple(self.family_memberships, "family_memberships"),
        )
        if any(
            not isinstance(item, CorpusDocumentVersionMembership)
            for item in memberships
        ):
            raise TypeError(
                "memberships must contain only CorpusDocumentVersionMembership values."
            )
        if any(not isinstance(item, DuplicateRelationship) for item in duplicates):
            raise TypeError(
                "duplicate_relationships must contain only DuplicateRelationship values."
            )
        if any(not isinstance(item, DocumentFamilyMembership) for item in families):
            raise TypeError(
                "family_memberships must contain only DocumentFamilyMembership values."
            )

        membership_keys = tuple(
            (
                item.document_id.as_str(),
                item.document_version_id.as_str(),
            )
            for item in memberships
        )
        _require_unique(membership_keys, "membership identities")
        version_to_document: dict[DocumentVersionId, DocumentId] = {}
        visible_documents: set[DocumentId] = set()
        visible_versions: set[DocumentVersionId] = set()
        for item in memberships:
            if item.corpus_id != self.corpus_id:
                raise RetrievalEvaluationContractError(
                    "all memberships must belong to the snapshot corpus."
                )
            existing_document = version_to_document.get(item.document_version_id)
            if existing_document is not None and existing_document != item.document_id:
                raise RetrievalEvaluationContractError(
                    "a document version cannot belong to multiple documents."
                )
            version_to_document[item.document_version_id] = item.document_id
            visible_documents.add(item.document_id)
            visible_versions.add(item.document_version_id)

        duplicate_keys: list[tuple[str, str, str]] = []
        for relationship in duplicates:
            if relationship.duplicate_class is not DuplicateClass.BYTE_IDENTICAL:
                raise RetrievalEvaluationContractError(
                    "unsupported duplicate relationship class."
                )
            if (
                not {
                    relationship.first_version_id,
                    relationship.second_version_id,
                }
                <= visible_versions
            ):
                raise RetrievalEvaluationContractError(
                    "duplicate relationships must use visible document versions."
                )
            duplicate_keys.append(relationship.key())
        _require_unique(tuple(duplicate_keys), "duplicate relationships")

        family_keys: list[tuple[str, str, str | None]] = []
        for membership in families:
            if membership.document_id not in visible_documents:
                raise RetrievalEvaluationContractError(
                    "family memberships must use visible documents."
                )
            if membership.document_version_id is not None:
                if membership.document_version_id not in visible_versions:
                    raise RetrievalEvaluationContractError(
                        "family memberships must use visible document versions."
                    )
                if (
                    version_to_document[membership.document_version_id]
                    != membership.document_id
                ):
                    raise RetrievalEvaluationContractError(
                        "family version membership must match its document."
                    )
            family_keys.append(membership.key())
        _require_unique(tuple(family_keys), "family memberships")

        canonical_memberships = tuple(
            sorted(
                memberships,
                key=lambda item: (
                    item.document_id.as_str(),
                    item.document_version_id.as_str(),
                ),
            )
        )
        canonical_duplicates = tuple(sorted(duplicates, key=DuplicateRelationship.key))
        canonical_families = tuple(
            sorted(
                families,
                key=lambda membership: (
                    membership.family_id.as_str(),
                    membership.document_id.as_str(),
                    membership.document_version_id.as_str()
                    if membership.document_version_id is not None
                    else "",
                ),
            )
        )
        object.__setattr__(self, "memberships", canonical_memberships)
        object.__setattr__(self, "duplicate_relationships", canonical_duplicates)
        object.__setattr__(self, "family_memberships", canonical_families)

        calculated_digest = _digest_payload(self._payload_without_digest())
        if self.content_digest is None:
            object.__setattr__(self, "content_digest", calculated_digest)
        else:
            _require_checksum(self.content_digest, "content_digest")
            if self.content_digest != calculated_digest:
                raise RetrievalEvaluationContractError(
                    "content_digest does not match the relationship snapshot."
                )

    def _payload_without_digest(self) -> JsonObject:
        """Return the canonical semantic snapshot payload."""
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "corpus_id": self.corpus_id.as_str(),
            "corpus_version": self.corpus_version.as_str(),
            "memberships": [
                {
                    "corpus_id": membership.corpus_id.as_str(),
                    "document_id": membership.document_id.as_str(),
                    "document_version_id": membership.document_version_id.as_str(),
                }
                for membership in self.memberships
            ],
            "duplicate_relationships": [
                {
                    "first_version_id": relationship.first_version_id.as_str(),
                    "second_version_id": relationship.second_version_id.as_str(),
                    "duplicate_class": relationship.duplicate_class.value,
                }
                for relationship in self.duplicate_relationships
            ],
            "family_memberships": [
                {
                    "family_id": membership.family_id.as_str(),
                    "document_id": membership.document_id.as_str(),
                    "document_version_id": (
                        membership.document_version_id.as_str()
                        if membership.document_version_id is not None
                        else None
                    ),
                }
                for membership in self.family_memberships
            ],
        }

    def as_dict(self) -> dict[str, object]:
        """Return the complete safe relationship snapshot identity."""
        payload = dict(self._payload_without_digest())
        payload["content_digest"] = self.content_digest
        return cast(dict[str, object], payload)


@dataclass(frozen=True, slots=True)
class SanitisedSourceReference:
    """A source link retained in a result without candidate content."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the opaque source identities."""
        _require_type(self.document_id, DocumentId, "document_id")
        _require_type(
            self.document_version_id, DocumentVersionId, "document_version_id"
        )
        _require_type(self.source_span_id, SourceSpanId, "source_span_id")

    def as_dict(self) -> dict[str, str]:
        """Return the safe source triple."""
        return {
            "document_id": self.document_id.as_str(),
            "document_version_id": self.document_version_id.as_str(),
            "source_span_id": self.source_span_id.as_str(),
        }


@dataclass(frozen=True, slots=True)
class SanitisedChannelContribution:
    """A channel-local rank retained without raw scores or candidate text."""

    channel: RetrievalChannel
    rank: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the bounded channel contribution."""
        _require_type(self.channel, RetrievalChannel, "channel")
        _require_positive_integer(self.rank, "rank")

    def as_dict(self) -> dict[str, str | int]:
        """Return the safe channel contribution."""
        return {"channel": self.channel.value, "rank": self.rank}


@dataclass(frozen=True, slots=True)
class SanitisedHitReference:
    """A ranked result reference retaining provenance but not candidate text."""

    rank: int
    record_id: RecordId
    kind: CandidateKind
    sources: tuple[SanitisedSourceReference, ...]
    contributions: tuple[SanitisedChannelContribution, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate result order and source-reference uniqueness."""
        _require_positive_integer(self.rank, "rank")
        _require_type(self.record_id, RecordId, "record_id")
        _require_type(self.kind, CandidateKind, "kind")
        sources = cast(
            tuple[SanitisedSourceReference, ...],
            _require_tuple(self.sources, "sources"),
        )
        contributions = cast(
            tuple[SanitisedChannelContribution, ...],
            _require_tuple(self.contributions, "contributions"),
        )
        if not sources:
            raise RetrievalEvaluationContractError("sources must not be empty.")
        if not contributions:
            raise RetrievalEvaluationContractError("contributions must not be empty.")
        if any(not isinstance(source, SanitisedSourceReference) for source in sources):
            raise TypeError(
                "sources must contain only SanitisedSourceReference values."
            )
        if any(
            not isinstance(contribution, SanitisedChannelContribution)
            for contribution in contributions
        ):
            raise TypeError(
                "contributions must contain only SanitisedChannelContribution values."
            )
        _require_unique(
            tuple(source.source_span_id for source in sources),
            "source spans",
        )
        _require_unique(
            tuple(contribution.channel for contribution in contributions),
            "channels",
        )
        canonical_sources = tuple(sorted(sources, key=_source_sort_key))
        object.__setattr__(self, "sources", canonical_sources)

    @classmethod
    def from_hit(cls, hit: EvidenceHit) -> "SanitisedHitReference":
        """Build a safe reference from one validated RET-01 hit."""
        if not isinstance(hit, EvidenceHit):
            raise TypeError("hit must be an EvidenceHit.")
        return cls(
            rank=hit.rank,
            record_id=hit.candidate.record_id,
            kind=hit.candidate.kind,
            sources=tuple(
                SanitisedSourceReference(
                    source.document_id,
                    source.document_version_id,
                    source.source_span_id,
                )
                for source in hit.candidate.sources
            ),
            contributions=tuple(
                SanitisedChannelContribution(contribution.channel, contribution.rank)
                for contribution in hit.contributions
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the payload-free ranked hit reference."""
        return {
            "rank": self.rank,
            "record_id": self.record_id.as_str(),
            "kind": self.kind.value,
            "sources": [source.as_dict() for source in self.sources],
            "contributions": [
                contribution.as_dict() for contribution in self.contributions
            ],
        }


@dataclass(frozen=True, slots=True)
class SanitisedFailure:
    """A bounded retrieval failure retained without exception payloads."""

    category: FailureCategory
    retryable: bool
    channel: RetrievalChannel | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the safe failure representation."""
        _require_type(self.category, FailureCategory, "category")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a boolean.")
        if self.channel is not None:
            _require_type(self.channel, RetrievalChannel, "channel")

    def as_dict(self) -> dict[str, str | bool | None]:
        """Return the bounded failure fields."""
        return {
            "category": self.category.value,
            "retryable": self.retryable,
            "channel": self.channel.value if self.channel is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RecallMetric:
    """One exact recall fraction labelled by its retrieval cut-off."""

    cutoff: int
    value: MetricFraction

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the cut-off and exact metric value."""
        _require_positive_integer(self.cutoff, "cutoff")
        _require_type(self.value, MetricFraction, "value")

    def as_dict(self) -> dict[str, object]:
        """Return the cut-off and authoritative fraction."""
        return {"cutoff": self.cutoff, "value": self.value.as_dict()}


@dataclass(frozen=True, slots=True)
class RecallMatchRanks:
    """Earliest result rank for each private relevance target."""

    cutoff: int
    ranks: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Validate bounded match-rank diagnostics."""
        _require_positive_integer(self.cutoff, "cutoff")
        ranks = cast(
            tuple[int | None, ...],
            _require_tuple(self.ranks, "ranks"),
        )
        for rank in ranks:
            if rank is not None:
                _require_positive_integer(rank, "match rank")

    def as_dict(self) -> dict[str, object]:
        """Return match ranks without exposing the matched target identities."""
        return {"cutoff": self.cutoff, "ranks": list(self.ranks)}


@dataclass(frozen=True, slots=True)
class DuplicateDiagnostics:
    """Bounded per-case diagnostics for duplicate and family crowding."""

    raw_hit_count: int
    distinct_record_count: int
    distinct_document_count: int
    distinct_duplicate_group_count: int
    distinct_family_count: int
    unknown_family_document_count: int
    exact_document_repeat_excess: int
    byte_identical_group_repeat_excess: int
    family_member_repeat_excess: int
    relevant_non_gold_duplicate_copy_count: int

    def __post_init__(self) -> None:
        """Validate all diagnostic counts as non-negative integers."""
        for field_name in (
            "raw_hit_count",
            "distinct_record_count",
            "distinct_document_count",
            "distinct_duplicate_group_count",
            "distinct_family_count",
            "unknown_family_document_count",
            "exact_document_repeat_excess",
            "byte_identical_group_repeat_excess",
            "family_member_repeat_excess",
            "relevant_non_gold_duplicate_copy_count",
        ):
            _require_non_negative_integer(getattr(self, field_name), field_name)

    def as_dict(self) -> dict[str, int]:
        """Return the bounded duplicate/family diagnostics."""
        return {
            "raw_hit_count": self.raw_hit_count,
            "distinct_record_count": self.distinct_record_count,
            "distinct_document_count": self.distinct_document_count,
            "distinct_duplicate_group_count": self.distinct_duplicate_group_count,
            "distinct_family_count": self.distinct_family_count,
            "unknown_family_document_count": self.unknown_family_document_count,
            "exact_document_repeat_excess": self.exact_document_repeat_excess,
            "byte_identical_group_repeat_excess": self.byte_identical_group_repeat_excess,
            "family_member_repeat_excess": self.family_member_repeat_excess,
            "relevant_non_gold_duplicate_copy_count": (
                self.relevant_non_gold_duplicate_copy_count
            ),
        }


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    """One safe terminal result for one selected evaluation case."""

    case_id: str
    case_ordinal: int
    split: SplitName
    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    requested_limit: int
    status: ResultStatus
    hits: tuple[SanitisedHitReference, ...]
    failures: tuple[SanitisedFailure, ...]
    document_recall: RecallMetric | None = None
    passage_recall: tuple[RecallMetric | None, ...] = ()
    document_match_ranks: tuple[int | None, ...] = ()
    passage_match_ranks: tuple[RecallMatchRanks, ...] = ()
    document_depth_exhausted: bool = False
    duplicate_diagnostics: DuplicateDiagnostics | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the terminal case-result state and safe references."""
        _require_case_id(self.case_id)
        _require_non_negative_integer(self.case_ordinal, "case_ordinal")
        if self.split not in {"train", "development", "held_out"}:
            raise RetrievalEvaluationContractError("split is not a supported split.")
        _require_type(self.corpus_id, CorpusId, "corpus_id")
        _require_type(self.corpus_version, CorpusVersionId, "corpus_version")
        _require_positive_integer(self.requested_limit, "requested_limit")
        _require_type(self.status, ResultStatus, "status")
        hits = cast(
            tuple[SanitisedHitReference, ...],
            _require_tuple(self.hits, "hits"),
        )
        failures = cast(
            tuple[SanitisedFailure, ...],
            _require_tuple(self.failures, "failures"),
        )
        passage_metrics = cast(
            tuple[RecallMetric | None, ...],
            _require_tuple(self.passage_recall, "passage_recall"),
        )
        document_match_ranks = cast(
            tuple[int | None, ...],
            _require_tuple(self.document_match_ranks, "document_match_ranks"),
        )
        passage_match_ranks = cast(
            tuple[RecallMatchRanks, ...],
            _require_tuple(self.passage_match_ranks, "passage_match_ranks"),
        )
        if any(not isinstance(hit, SanitisedHitReference) for hit in hits):
            raise TypeError("hits must contain only SanitisedHitReference values.")
        if any(not isinstance(failure, SanitisedFailure) for failure in failures):
            raise TypeError("failures must contain only SanitisedFailure values.")
        if any(
            metric is not None and not isinstance(metric, RecallMetric)
            for metric in passage_metrics
        ):
            raise TypeError("passage_recall must contain RecallMetric or None values.")
        if any(rank is not None for rank in document_match_ranks):
            for rank in document_match_ranks:
                if rank is not None:
                    _require_positive_integer(rank, "document match rank")
        if any(
            not isinstance(match_ranks, RecallMatchRanks)
            for match_ranks in passage_match_ranks
        ):
            raise TypeError(
                "passage_match_ranks must contain only RecallMatchRanks values."
            )
        if type(self.document_depth_exhausted) is not bool:
            raise TypeError("document_depth_exhausted must be a boolean.")
        if self.duplicate_diagnostics is not None and not isinstance(
            self.duplicate_diagnostics,
            DuplicateDiagnostics,
        ):
            raise TypeError(
                "duplicate_diagnostics must be a DuplicateDiagnostics or None."
            )
        ranks = tuple(hit.rank for hit in hits)
        if ranks != tuple(range(1, len(hits) + 1)):
            raise RetrievalEvaluationContractError(
                "hit ranks must be contiguous and match tuple order."
            )
        if len(hits) > self.requested_limit:
            raise RetrievalEvaluationContractError(
                "hits must not exceed requested_limit."
            )
        if self.status is ResultStatus.COMPLETED and failures:
            raise RetrievalEvaluationContractError(
                "completed case results must not contain failures."
            )
        if self.status is ResultStatus.PARTIAL and (not hits or not failures):
            raise RetrievalEvaluationContractError(
                "partial case results require hits and failures."
            )
        if self.status is ResultStatus.FAILED and (hits or not failures):
            raise RetrievalEvaluationContractError(
                "failed case results require failures and no hits."
            )
        if self.document_recall is not None and self.document_recall.cutoff != 10:
            raise RetrievalEvaluationContractError(
                "document recall must use cutoff 10."
            )
        present_cutoffs = tuple(
            metric.cutoff for metric in passage_metrics if metric is not None
        )
        if present_cutoffs != tuple(sorted(set(present_cutoffs))):
            raise RetrievalEvaluationContractError(
                "passage recall cut-offs must be strictly increasing and unique."
            )
        match_cutoffs = tuple(metric.cutoff for metric in passage_match_ranks)
        if match_cutoffs != tuple(sorted(set(match_cutoffs))):
            raise RetrievalEvaluationContractError(
                "passage match cut-offs must be strictly increasing and unique."
            )

    @property
    def is_empty(self) -> bool:
        """Return whether the case is a completed empty retrieval."""
        return self.status is ResultStatus.COMPLETED and not self.hits

    @classmethod
    def from_result(
        cls,
        *,
        case: EvaluationCase,
        case_ordinal: int,
        corpus_id: CorpusId,
        result: EvidenceResult,
    ) -> "RetrievalCaseResult":
        """Sanitise one validated RET-01 result for later scoring/publication."""
        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase.")
        _require_type(corpus_id, CorpusId, "corpus_id")
        if not isinstance(result, EvidenceResult):
            raise TypeError("result must be an EvidenceResult.")
        return cls(
            case_id=case.case_id,
            case_ordinal=case_ordinal,
            split=case.split,
            corpus_id=corpus_id,
            corpus_version=result.corpus_version,
            requested_limit=result.requested_limit.value,
            status=result.status,
            hits=tuple(SanitisedHitReference.from_hit(hit) for hit in result.hits),
            failures=tuple(
                SanitisedFailure(
                    failure.category,
                    failure.retryable,
                    failure.channel,
                )
                for failure in result.failures
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Return only safe, deterministic case-result fields."""
        return {
            "case_id": self.case_id,
            "case_ordinal": self.case_ordinal,
            "split": self.split,
            "corpus_id": self.corpus_id.as_str(),
            "corpus_version": self.corpus_version.as_str(),
            "requested_limit": self.requested_limit,
            "status": self.status.value,
            "hits": [hit.as_dict() for hit in self.hits],
            "failures": [failure.as_dict() for failure in self.failures],
            "document_recall": (
                self.document_recall.as_dict()
                if self.document_recall is not None
                else None
            ),
            "passage_recall": [
                metric.as_dict() if metric is not None else None
                for metric in self.passage_recall
            ],
            "document_match_ranks": list(self.document_match_ranks),
            "passage_match_ranks": [
                match_ranks.as_dict() for match_ranks in self.passage_match_ranks
            ],
            "document_depth_exhausted": self.document_depth_exhausted,
            "duplicate_diagnostics": (
                self.duplicate_diagnostics.as_dict()
                if self.duplicate_diagnostics is not None
                else None
            ),
        }

    @property
    def result_digest(self) -> str:
        """Return a deterministic digest of this safe case result."""
        return _digest_payload(cast(JsonObject, self.as_dict()))


@dataclass(frozen=True, slots=True)
class AggregateMetric:
    """Exact macro/micro recall and denominator reconciliation for one cut-off."""

    cutoff: int
    macro: MetricFraction | None
    micro: MetricFraction | None
    applicable_case_count: int
    non_applicable_case_count: int
    completed_case_count: int
    partial_case_count: int
    failed_case_count: int
    empty_case_count: int
    matched_target_count: int
    relevant_target_count: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate aggregate counts and fraction applicability."""
        _require_positive_integer(self.cutoff, "cutoff")
        if self.macro is not None:
            _require_type(self.macro, MetricFraction, "macro")
        if self.micro is not None:
            _require_type(self.micro, MetricFraction, "micro")
        count_names = (
            "applicable_case_count",
            "non_applicable_case_count",
            "completed_case_count",
            "partial_case_count",
            "failed_case_count",
            "empty_case_count",
            "matched_target_count",
            "relevant_target_count",
        )
        for field_name in count_names:
            _require_non_negative_integer(getattr(self, field_name), field_name)
        if self.applicable_case_count == 0 and self.macro is not None:
            raise RetrievalEvaluationContractError(
                "macro must be null when there are no applicable cases."
            )
        if self.applicable_case_count > 0 and self.macro is None:
            raise RetrievalEvaluationContractError(
                "macro is required when there are applicable cases."
            )
        if self.relevant_target_count == 0 and self.micro is not None:
            raise RetrievalEvaluationContractError(
                "micro must be null when there are no relevant targets."
            )
        if self.relevant_target_count > 0 and self.micro is None:
            raise RetrievalEvaluationContractError(
                "micro is required when relevant targets exist."
            )
        if self.matched_target_count > self.relevant_target_count:
            raise RetrievalEvaluationContractError(
                "matched_target_count must not exceed relevant_target_count."
            )
        if self.empty_case_count > self.completed_case_count:
            raise RetrievalEvaluationContractError(
                "empty_case_count must not exceed completed_case_count."
            )
        if (
            self.completed_case_count + self.partial_case_count + self.failed_case_count
            != self.applicable_case_count + self.non_applicable_case_count
        ):
            raise RetrievalEvaluationContractError(
                "outcome counts must reconcile to the case population."
            )

    def as_dict(self) -> dict[str, object]:
        """Return exact aggregate counts and optional fractions."""
        return {
            "cutoff": self.cutoff,
            "macro": self.macro.as_dict() if self.macro is not None else None,
            "micro": self.micro.as_dict() if self.micro is not None else None,
            "applicable_case_count": self.applicable_case_count,
            "non_applicable_case_count": self.non_applicable_case_count,
            "completed_case_count": self.completed_case_count,
            "partial_case_count": self.partial_case_count,
            "failed_case_count": self.failed_case_count,
            "empty_case_count": self.empty_case_count,
            "matched_target_count": self.matched_target_count,
            "relevant_target_count": self.relevant_target_count,
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationAggregate:
    """Aggregate document and passage metrics for one completed run."""

    case_count: int
    document: AggregateMetric
    passage: tuple[AggregateMetric, ...]
    duplicate_diagnostics: DuplicateDiagnostics | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate aggregate population and metric cut-off uniqueness."""
        _require_positive_integer(self.case_count, "case_count")
        _require_type(self.document, AggregateMetric, "document")
        passage = cast(
            tuple[AggregateMetric, ...],
            _require_tuple(self.passage, "passage"),
        )
        if any(not isinstance(metric, AggregateMetric) for metric in passage):
            raise TypeError("passage must contain only AggregateMetric values.")
        if self.duplicate_diagnostics is not None and not isinstance(
            self.duplicate_diagnostics,
            DuplicateDiagnostics,
        ):
            raise TypeError(
                "duplicate_diagnostics must be a DuplicateDiagnostics or None."
            )
        cutoffs = tuple(metric.cutoff for metric in passage)
        if cutoffs != tuple(sorted(set(cutoffs))):
            raise RetrievalEvaluationContractError(
                "passage aggregate cut-offs must be strictly increasing and unique."
            )
        if (
            self.document.applicable_case_count
            + self.document.non_applicable_case_count
            != self.case_count
        ):
            raise RetrievalEvaluationContractError(
                "document aggregate must reconcile to case_count."
            )
        for metric in passage:
            if (
                metric.applicable_case_count + metric.non_applicable_case_count
                != self.case_count
            ):
                raise RetrievalEvaluationContractError(
                    "passage aggregates must reconcile to case_count."
                )

    def as_dict(self) -> dict[str, object]:
        """Return the safe aggregate report payload."""
        return {
            "case_count": self.case_count,
            "document": self.document.as_dict(),
            "passage": [metric.as_dict() for metric in self.passage],
            "duplicate_diagnostics": (
                self.duplicate_diagnostics.as_dict()
                if self.duplicate_diagnostics is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRunIdentity:
    """Semantic identity of one single-corpus retrieval evaluation run."""

    dataset_id: str
    dataset_version: str
    dataset_schema_version: str
    case_file_checksum: str
    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    selected_split: SplitName
    selected_case_ids: tuple[str, ...]
    policy: RetrievalEvaluationPolicy
    relationship_snapshot_digest: str
    system_identity: RetrievalSystemIdentity
    result_schema_version: str = RETRIEVAL_EVALUATION_SCHEMA_VERSION
    implementation: str = RETRIEVAL_EVALUATION_IMPLEMENTATION
    implementation_version: str = RETRIEVAL_EVALUATION_IMPLEMENTATION_VERSION
    run_fingerprint: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the complete run identity and compute its fingerprint."""
        _require_case_id(self.dataset_id, "dataset_id")
        _require_semver(self.dataset_version, "dataset_version")
        _require_semver(self.dataset_schema_version, "dataset_schema_version")
        _require_checksum(self.case_file_checksum, "case_file_checksum")
        _require_type(self.corpus_id, CorpusId, "corpus_id")
        _require_type(self.corpus_version, CorpusVersionId, "corpus_version")
        if self.selected_split not in {"train", "development", "held_out"}:
            raise RetrievalEvaluationContractError("selected_split is not supported.")
        case_ids = cast(
            tuple[str, ...],
            _require_tuple(self.selected_case_ids, "selected_case_ids"),
        )
        if not case_ids:
            raise RetrievalEvaluationContractError(
                "selected_case_ids must not be empty."
            )
        for case_id in case_ids:
            _require_case_id(case_id)
        _require_unique(case_ids, "selected_case_ids")
        _require_type(self.policy, RetrievalEvaluationPolicy, "policy")
        _require_checksum(
            self.relationship_snapshot_digest,
            "relationship_snapshot_digest",
        )
        _require_type(self.system_identity, RetrievalSystemIdentity, "system_identity")
        _require_semver(self.result_schema_version, "result_schema_version")
        _require_slug(self.implementation, "implementation")
        _require_semver(self.implementation_version, "implementation_version")

        calculated_fingerprint = _digest_payload(self._semantic_payload())
        if self.run_fingerprint is None:
            object.__setattr__(self, "run_fingerprint", calculated_fingerprint)
        else:
            _require_checksum(self.run_fingerprint, "run_fingerprint")
            if self.run_fingerprint != calculated_fingerprint:
                raise RetrievalEvaluationContractError(
                    "run_fingerprint does not match the semantic run identity."
                )

    def _semantic_payload(self) -> JsonObject:
        """Return run identity fields excluding the derived fingerprint."""
        return cast(
            JsonObject,
            {
                "result_schema_version": self.result_schema_version,
                "implementation": self.implementation,
                "implementation_version": self.implementation_version,
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "dataset_schema_version": self.dataset_schema_version,
                "case_file_checksum": self.case_file_checksum,
                "corpus_id": self.corpus_id.as_str(),
                "corpus_version": self.corpus_version.as_str(),
                "selected_split": self.selected_split,
                "selected_case_ids": list(self.selected_case_ids),
                "policy": self.policy.as_dict(),
                "relationship_snapshot_digest": self.relationship_snapshot_digest,
                "system_identity": self.system_identity.as_dict(),
            },
        )

    def as_dict(self) -> dict[str, object]:
        """Return the complete path-free run identity."""
        payload = dict(self._semantic_payload())
        payload["run_fingerprint"] = self.run_fingerprint
        return cast(dict[str, object], payload)


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """Checksum and size for one required published run artefact."""

    name: str
    size_bytes: int
    checksum: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the bounded artefact inventory entry."""
        if self.name not in PUBLISHED_ARTIFACT_NAMES:
            raise RetrievalEvaluationContractError(
                "name is not a required retrieval-evaluation artefact."
            )
        _require_positive_integer(self.size_bytes, "size_bytes")
        _require_checksum(self.checksum, "checksum")

    def as_dict(self) -> dict[str, str | int]:
        """Return the safe artefact inventory entry."""
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }


@dataclass(frozen=True, slots=True)
class PublishedRetrievalEvaluationManifest:
    """Safe inventory manifest for a published retrieval-evaluation run."""

    run_fingerprint: str
    result_digest: str
    artefacts: tuple[PublishedArtifact, ...]
    schema_version: str = RETRIEVAL_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the exact four-file inventory recorded by manifest.json."""
        _require_checksum(self.run_fingerprint, "run_fingerprint")
        _require_checksum(self.result_digest, "result_digest")
        _require_semver(self.schema_version, "schema_version")
        artefacts = cast(
            tuple[PublishedArtifact, ...],
            _require_tuple(self.artefacts, "artefacts"),
        )
        if any(not isinstance(artefact, PublishedArtifact) for artefact in artefacts):
            raise TypeError("artefacts must contain only PublishedArtifact values.")
        names = tuple(artefact.name for artefact in artefacts)
        if names != PUBLISHED_ARTIFACT_NAMES:
            raise RetrievalEvaluationContractError(
                "artefacts must contain the required files in canonical order."
            )

    def as_dict(self) -> dict[str, object]:
        """Return the safe publication manifest payload."""
        return {
            "schema_version": self.schema_version,
            "run_fingerprint": self.run_fingerprint,
            "result_digest": self.result_digest,
            "artefacts": [artefact.as_dict() for artefact in self.artefacts],
        }
