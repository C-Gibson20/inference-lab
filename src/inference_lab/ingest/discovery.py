"""Immutable contracts for local source discovery and classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from inference_lab._validation import (
    is_canonical_media_type,
    is_canonical_semantic_version,
    is_local_path_shape,
    is_lowercase_slug,
    is_strict_integer,
)
from inference_lab.provenance import (
    CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
    ContentHash,
    CorpusId,
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierResolutionPolicy,
    IdentityResolutionOutcome,
    SourceObservationId,
)
from inference_lab.provenance.values import validate_external_identifiers


def _validate_slug(value: object, *, field_name: str) -> None:
    """Validate a project-owned lowercase slug without exposing its value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_lowercase_slug(value):
        raise ValueError(f"{field_name} must be a lowercase hyphenated slug.")


def _validate_semantic_version(value: object, *, field_name: str) -> None:
    """Validate a canonical semantic version with field-specific errors."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise ValueError(f"{field_name} must be a canonical semantic version.")


def _validate_media_type(value: object) -> None:
    """Validate a canonical, parameter-free media type."""
    if not isinstance(value, str):
        raise TypeError("media_type must be a string.")
    if not is_canonical_media_type(value):
        raise ValueError("media_type must be a canonical lowercase media type.")


def _validate_encoding(value: object) -> None:
    """Validate a safe encoding label without consulting the codec registry."""
    if not isinstance(value, str):
        raise TypeError("encoding must be a string.")
    if not value.strip() or "\x00" in value:
        raise ValueError("encoding must be a non-blank safe label.")


class DiscoveryInputKind(StrEnum):
    """Filesystem object kind declared for one explicit discovery root."""

    FILE = "file"
    DIRECTORY = "directory"


class SymlinkPolicy(StrEnum):
    """Supported symbolic-link traversal policies."""

    EXCLUDE = "exclude"


class NestedMountPolicy(StrEnum):
    """Supported nested-mount traversal policies."""

    EXCLUDE = "exclude"


class HiddenEntryPolicy(StrEnum):
    """Supported hidden-entry traversal policies."""

    INCLUDE = "include"


class DirectorySnapshotPolicy(StrEnum):
    """Supported directory-enumeration policies."""

    SORTED = "sorted"


class StableReadPolicy(StrEnum):
    """Supported stable-source verification policies."""

    PRE_OPEN_POST_STAT = "pre_open_post_stat"


class SourceSupport(StrEnum):
    """Classification of a source against available parser capabilities."""

    SUPPORTED = "supported"
    RECOGNISED_UNSUPPORTED = "recognised_unsupported"
    UNKNOWN = "unknown"


class FamilyEvidenceKind(StrEnum):
    """Bounded evidence kinds accepted for advisory family candidates."""

    EXPLICIT_ASSOCIATION = "explicit_association"


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    """Identify the complete code-owned local discovery behaviour."""

    policy_name: str
    policy_version: str
    symlink_policy: SymlinkPolicy
    nested_mount_policy: NestedMountPolicy
    hidden_entry_policy: HiddenEntryPolicy
    directory_snapshot_policy: DirectorySnapshotPolicy
    stable_read_policy: StableReadPolicy
    classifier_name: str
    classifier_version: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate policy identity and every closed behaviour selector."""
        _validate_slug(self.policy_name, field_name="policy_name")
        _validate_semantic_version(
            self.policy_version,
            field_name="policy_version",
        )
        for field_name, value, expected_type in (
            ("symlink_policy", self.symlink_policy, SymlinkPolicy),
            ("nested_mount_policy", self.nested_mount_policy, NestedMountPolicy),
            ("hidden_entry_policy", self.hidden_entry_policy, HiddenEntryPolicy),
            (
                "directory_snapshot_policy",
                self.directory_snapshot_policy,
                DirectorySnapshotPolicy,
            ),
            ("stable_read_policy", self.stable_read_policy, StableReadPolicy),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{field_name} must use its discovery policy enum.")
        _validate_slug(self.classifier_name, field_name="classifier_name")
        _validate_semantic_version(
            self.classifier_version,
            field_name="classifier_version",
        )

    def with_classifier_identity(
        self,
        *,
        classifier_name: str,
        classifier_version: str,
    ) -> "DiscoveryPolicy":
        """Return this traversal policy with the classifier actually used."""
        return DiscoveryPolicy(
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            symlink_policy=self.symlink_policy,
            nested_mount_policy=self.nested_mount_policy,
            hidden_entry_policy=self.hidden_entry_policy,
            directory_snapshot_policy=self.directory_snapshot_policy,
            stable_read_policy=self.stable_read_policy,
            classifier_name=classifier_name,
            classifier_version=classifier_version,
        )


CURRENT_DISCOVERY_POLICY: Final = DiscoveryPolicy(
    policy_name="local-source-discovery",
    policy_version="1.0.0",
    symlink_policy=SymlinkPolicy.EXCLUDE,
    nested_mount_policy=NestedMountPolicy.EXCLUDE,
    hidden_entry_policy=HiddenEntryPolicy.INCLUDE,
    directory_snapshot_policy=DirectorySnapshotPolicy.SORTED,
    stable_read_policy=StableReadPolicy.PRE_OPEN_POST_STAT,
    classifier_name="text-native-baseline",
    classifier_version="2.0.0",
)


@dataclass(frozen=True, slots=True)
class FamilyCandidate:
    """Retain explicit, advisory evidence for a known document family."""

    document_family_id: DocumentFamilyId
    evidence_kind: FamilyEvidenceKind
    document_id: DocumentId | None = None
    document_version_id: DocumentVersionId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate family and optional candidate identity fields."""
        if not isinstance(self.document_family_id, DocumentFamilyId):
            raise TypeError("document_family_id must be a DocumentFamilyId.")
        if not isinstance(self.evidence_kind, FamilyEvidenceKind):
            raise TypeError("evidence_kind must be a FamilyEvidenceKind.")
        if self.document_id is not None and not isinstance(
            self.document_id, DocumentId
        ):
            raise TypeError("document_id must be a DocumentId or None.")
        if self.document_version_id is not None and not isinstance(
            self.document_version_id, DocumentVersionId
        ):
            raise TypeError("document_version_id must be a DocumentVersionId or None.")
        if self.document_version_id is not None and self.document_id is None:
            raise ValueError("a family version candidate requires a document_id.")

    def key(self) -> tuple[str, str, str, str]:
        """Return a deterministic ordering and duplicate-detection key."""
        return (
            self.document_family_id.as_str(),
            self.document_id.as_str() if self.document_id is not None else "",
            (
                self.document_version_id.as_str()
                if self.document_version_id is not None
                else ""
            ),
            self.evidence_kind.value,
        )


def _validate_family_candidates(candidates: object) -> None:
    """Validate immutable, unique, canonically ordered family evidence."""
    if not isinstance(candidates, tuple):
        raise TypeError("family_candidates must be a tuple.")
    if any(not isinstance(candidate, FamilyCandidate) for candidate in candidates):
        raise TypeError("family_candidates must contain FamilyCandidate values.")
    keys = tuple(candidate.key() for candidate in candidates)
    if len(keys) != len(set(keys)):
        raise ValueError("family_candidates must not contain duplicates.")
    if keys != tuple(sorted(keys)):
        raise ValueError("family_candidates must be in canonical order.")


@dataclass(frozen=True, slots=True, repr=False)
class DiscoveryInput:
    """Describe one private, explicit file or directory discovery root."""

    root: Path = field(repr=False)
    kind: DiscoveryInputKind
    external_identifiers: tuple[ExternalIdentifier, ...] = ()
    family_candidates: tuple[FamilyCandidate, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the local root and file-only source metadata boundary."""
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path.")
        if not is_local_path_shape(self.root):
            raise ValueError("root must name a local filesystem path.")
        if not isinstance(self.kind, DiscoveryInputKind):
            raise TypeError("kind must be a DiscoveryInputKind.")
        validate_external_identifiers(self.external_identifiers)
        _validate_family_candidates(self.family_candidates)
        if self.kind is DiscoveryInputKind.DIRECTORY and (
            self.external_identifiers or self.family_candidates
        ):
            raise ValueError("source metadata is allowed only for a file input.")

    def __repr__(self) -> str:
        """Return a representation that does not disclose the private root."""
        return f"DiscoveryInput(kind={self.kind.value!r}, root=<private>)"


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Select one corpus and immutable local discovery inputs."""

    corpus_id: CorpusId
    inputs: tuple[DiscoveryInput, ...]
    external_identifier_policy: ExternalIdentifierResolutionPolicy = (
        CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate request corpus, inputs, and identity policy."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.inputs, tuple):
            raise TypeError("inputs must be a tuple.")
        if not self.inputs:
            raise ValueError("inputs must contain at least one DiscoveryInput.")
        if any(not isinstance(item, DiscoveryInput) for item in self.inputs):
            raise TypeError("inputs must contain DiscoveryInput values.")
        if not isinstance(
            self.external_identifier_policy,
            ExternalIdentifierResolutionPolicy,
        ):
            raise TypeError(
                "external_identifier_policy must be an "
                "ExternalIdentifierResolutionPolicy."
            )


@dataclass(frozen=True, slots=True, repr=False)
class SourceLocator:
    """Privately locate one source relative to its explicit request root."""

    path: Path = field(repr=False)
    request_root_ordinal: int
    relative_path: Path = field(repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate local private path and root-relative location fields."""
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path.")
        if not is_local_path_shape(self.path):
            raise ValueError("path must name a local filesystem path.")
        if not is_strict_integer(self.request_root_ordinal):
            raise TypeError("request_root_ordinal must be an integer.")
        if self.request_root_ordinal < 0:
            raise ValueError("request_root_ordinal must be non-negative.")
        if not isinstance(self.relative_path, Path):
            raise TypeError("relative_path must be a Path.")
        if not is_local_path_shape(self.relative_path):
            raise ValueError("relative_path must be a local path.")
        if self.relative_path.is_absolute() or self.relative_path == Path("."):
            raise ValueError("relative_path must identify a root-relative entry.")
        if ".." in self.relative_path.parts:
            raise ValueError("relative_path must remain inside its request root.")

    def __repr__(self) -> str:
        """Return a representation that does not disclose source locations."""
        return (
            "SourceLocator("
            f"request_root_ordinal={self.request_root_ordinal}, path=<private>)"
        )


@dataclass(frozen=True, slots=True)
class FileIdentityHint:
    """Carry non-authoritative platform file identity for continuity checks."""

    device_id: int
    file_id: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate non-negative integer device and file identifiers."""
        for field_name, value in (
            ("device_id", self.device_id),
            ("file_id", self.file_id),
        ):
            if not is_strict_integer(value):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class FileStatEvidence:
    """Retain bounded file metadata used to prove a stable source read."""

    byte_count: int
    modified_time_ns: int
    identity_hint: FileIdentityHint | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate non-negative integer file size and modification time."""
        for field_name, value in (
            ("byte_count", self.byte_count),
            ("modified_time_ns", self.modified_time_ns),
        ):
            if not is_strict_integer(value):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.identity_hint is not None and not isinstance(
            self.identity_hint, FileIdentityHint
        ):
            raise TypeError("identity_hint must be a FileIdentityHint or None.")


@dataclass(frozen=True, slots=True)
class StableFileEvidence:
    """Retain pre-open, handle, and post-read metadata for one stable file."""

    pre_open: FileStatEvidence
    opened: FileStatEvidence
    opened_after_read: FileStatEvidence
    path_after_read: FileStatEvidence

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that all file evidence is consistent and stable."""
        observations = (
            self.pre_open,
            self.opened,
            self.opened_after_read,
            self.path_after_read,
        )
        if any(not isinstance(item, FileStatEvidence) for item in observations):
            raise TypeError(
                "stable file evidence must contain FileStatEvidence values."
            )
        first = observations[0]
        if any(item.byte_count != first.byte_count for item in observations[1:]):
            raise ValueError("stable file evidence byte counts must agree.")
        if any(
            item.modified_time_ns != first.modified_time_ns for item in observations[1:]
        ):
            raise ValueError("stable file evidence modification times must agree.")
        if any(item.identity_hint != first.identity_hint for item in observations[1:]):
            raise ValueError("stable file evidence identity hints must agree.")


@dataclass(frozen=True, slots=True)
class SourceClassification:
    """Describe source format and parser-capability support."""

    media_type: str
    support: SourceSupport
    classifier_name: str
    classifier_version: str
    parser_capability: str | None = None
    encoding: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate classification media type, support, and optional fields."""
        _validate_media_type(self.media_type)
        if not isinstance(self.support, SourceSupport):
            raise TypeError("support must be a SourceSupport.")
        _validate_slug(self.classifier_name, field_name="classifier_name")
        _validate_semantic_version(
            self.classifier_version,
            field_name="classifier_version",
        )
        if self.parser_capability is not None:
            _validate_slug(
                self.parser_capability,
                field_name="parser_capability",
            )
        if self.encoding is not None:
            _validate_encoding(self.encoding)
        if self.support is SourceSupport.SUPPORTED:
            if self.parser_capability is None:
                raise ValueError("supported sources require a parser capability.")
        elif self.parser_capability is not None or self.encoding is not None:
            raise ValueError(
                "unsupported sources cannot declare parser capability or encoding."
            )


@dataclass(frozen=True, slots=True)
class VerifiedSourceSnapshot:
    """Identify exact original bytes accepted from one stable local read."""

    locator: SourceLocator
    media_type: str
    encoding: str | None
    byte_count: int
    content_hash: ContentHash
    file_evidence: StableFileEvidence

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the snapshot's locator, media type, encoding, and evidence."""
        if not isinstance(self.locator, SourceLocator):
            raise TypeError("locator must be a SourceLocator.")
        _validate_media_type(self.media_type)
        if self.encoding is not None:
            _validate_encoding(self.encoding)
        if not is_strict_integer(self.byte_count):
            raise TypeError("byte_count must be an integer.")
        if self.byte_count < 0:
            raise ValueError("byte_count must be non-negative.")
        if not isinstance(self.content_hash, ContentHash):
            raise TypeError("content_hash must be a ContentHash.")
        if not isinstance(self.file_evidence, StableFileEvidence):
            raise TypeError("file_evidence must be StableFileEvidence.")
        if self.byte_count != self.file_evidence.opened_after_read.byte_count:
            raise ValueError("snapshot byte count must match stable file evidence.")


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    """Retain one ordered, hashed source and its identity decision."""

    ordinal: int
    corpus_id: CorpusId
    policy: DiscoveryPolicy
    source_observation_id: SourceObservationId
    snapshot: VerifiedSourceSnapshot
    external_identifiers: tuple[ExternalIdentifier, ...]
    family_candidates: tuple[FamilyCandidate, ...]
    classification: SourceClassification
    identity_outcome: IdentityResolutionOutcome

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the discovered source's identity and classification."""
        if not is_strict_integer(self.ordinal):
            raise TypeError("ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.policy, DiscoveryPolicy):
            raise TypeError("policy must be a DiscoveryPolicy.")
        if not isinstance(self.source_observation_id, SourceObservationId):
            raise TypeError("source_observation_id must be a SourceObservationId.")
        if not isinstance(self.snapshot, VerifiedSourceSnapshot):
            raise TypeError("snapshot must be a VerifiedSourceSnapshot.")
        validate_external_identifiers(self.external_identifiers)
        _validate_family_candidates(self.family_candidates)
        if not isinstance(self.classification, SourceClassification):
            raise TypeError("classification must be a SourceClassification.")
        if self.classification.support is not SourceSupport.SUPPORTED:
            raise ValueError("a discovered source must have parser support.")
        if (
            self.classification.classifier_name != self.policy.classifier_name
            or self.classification.classifier_version != self.policy.classifier_version
        ):
            raise ValueError("classification must match the discovery policy.")
        if self.snapshot.media_type != self.classification.media_type:
            raise ValueError("snapshot and classification media types must agree.")
        if self.snapshot.encoding != self.classification.encoding:
            raise ValueError("snapshot and classification encodings must agree.")
        if not isinstance(self.identity_outcome, IdentityResolutionOutcome):
            raise TypeError("identity_outcome must be an IdentityResolutionOutcome.")
