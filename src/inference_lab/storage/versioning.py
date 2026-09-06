"""Immutable contracts for corpus versions and staged publication."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final, Self

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
)

CORPUS_MANIFEST_CONTRACT_VERSION: Final = "1.0.0"

_LOWERCASE_SLUG_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_GENERATION_IDENTIFIER_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_SHA256_HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


class InvalidCorpusVersionContractError(ValueError):
    """Raised when corpus-version values violate semantic invariants."""


def _validate_slug(value: object, field_name: str) -> str:
    """Return a strictly validated lowercase hyphenated slug."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _LOWERCASE_SLUG_PATTERN.fullmatch(value) is None:
        raise InvalidCorpusVersionContractError(
            f"{field_name} must be a lowercase hyphenated slug."
        )
    return value


def _validate_semantic_version(value: object, field_name: str) -> str:
    """Return a strictly validated canonical semantic version."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise InvalidCorpusVersionContractError(
            f"{field_name} must be a canonical semantic version."
        )
    return value


def _validate_sha256_digest(value: object, field_name: str) -> str:
    """Return a canonical lowercase SHA-256 digest."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _SHA256_HEX_PATTERN.fullmatch(value) is None:
        raise InvalidCorpusVersionContractError(
            f"{field_name} must be 64 lowercase hexadecimal characters."
        )
    return value


def _validate_utc_datetime(value: object, field_name: str) -> datetime:
    """Return a timezone-aware datetime whose offset is UTC."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidCorpusVersionContractError(
            f"{field_name} must be timezone-aware UTC."
        )
    if value.utcoffset() != timedelta(0):
        raise InvalidCorpusVersionContractError(f"{field_name} must be in UTC.")
    return value


@dataclass(frozen=True, slots=True)
class ComponentKind:
    """Identify an extensible kind of immutable publication component."""

    value: str

    def __post_init__(self) -> None:
        _validate_slug(self.value, "component_kind")

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Reconstruct a component kind from canonical text."""
        return cls(value)

    def as_str(self) -> str:
        """Return the canonical component-kind value."""
        return self.value


class ComponentGenerationState(StrEnum):
    """Readiness of one prepared immutable component generation."""

    PREPARING = "preparing"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class ComponentGenerationReference:
    """Describe one immutable component generation prepared for publication."""

    component_kind: ComponentKind
    generation_id: str
    implementation: str
    format_version: str
    contract_version: str
    configuration_digest: str
    source_manifest_digest: str
    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    state: ComponentGenerationState

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate generation identity, compatibility, ownership, and state."""
        if not isinstance(self.component_kind, ComponentKind):
            raise TypeError("component_kind must be a ComponentKind.")
        if not isinstance(self.generation_id, str):
            raise TypeError("generation_id must be a string.")
        if _GENERATION_IDENTIFIER_PATTERN.fullmatch(self.generation_id) is None:
            raise InvalidCorpusVersionContractError(
                "generation_id must be a canonical non-path identifier."
            )
        _validate_slug(self.implementation, "implementation")
        _validate_semantic_version(self.format_version, "format_version")
        _validate_semantic_version(self.contract_version, "contract_version")
        _validate_sha256_digest(
            self.configuration_digest,
            "configuration_digest",
        )
        _validate_sha256_digest(
            self.source_manifest_digest,
            "source_manifest_digest",
        )
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.state, ComponentGenerationState):
            raise TypeError("state must be a ComponentGenerationState.")

    def key(self) -> tuple[str, str]:
        """Return the canonical manifest ordering key."""
        return self.component_kind.as_str(), self.generation_id


def canonicalise_component_kinds(kinds: object) -> tuple[ComponentKind, ...]:
    """Validate a unique canonical tuple of required component kinds."""
    if not isinstance(kinds, tuple):
        raise TypeError("required_component_kinds must be a tuple.")
    if any(not isinstance(kind, ComponentKind) for kind in kinds):
        raise TypeError(
            "required_component_kinds must contain only ComponentKind values."
        )
    keys = tuple(kind.as_str() for kind in kinds)
    if len(keys) != len(set(keys)):
        raise InvalidCorpusVersionContractError(
            "required_component_kinds must not contain duplicates."
        )
    return tuple(sorted(kinds, key=ComponentKind.as_str))


def validate_corpus_update_identity(
    corpus_version: CorpusVersionId,
    expected_corpus_version: CorpusVersionId | None,
    restores_corpus_version: CorpusVersionId | None,
) -> None:
    """Validate the relationships between one proposal and its history."""
    if expected_corpus_version == corpus_version:
        raise InvalidCorpusVersionContractError(
            "a corpus update cannot propose its expected active version."
        )
    if restores_corpus_version == corpus_version:
        raise InvalidCorpusVersionContractError(
            "an update cannot restore its own proposed version."
        )
    if restores_corpus_version is not None and expected_corpus_version is None:
        raise InvalidCorpusVersionContractError(
            "a rollback update requires an expected active version."
        )
    if (
        restores_corpus_version is not None
        and restores_corpus_version == expected_corpus_version
    ):
        raise InvalidCorpusVersionContractError(
            "a rollback target must be historical, not the expected active version."
        )


@dataclass(frozen=True, slots=True)
class CorpusVersionManifest:
    """Complete canonical document and component state for one corpus version."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    document_memberships: tuple[CorpusDocumentVersionMembership, ...] = ()
    component_generations: tuple[ComponentGenerationReference, ...] = ()
    manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "manifest_digest",
            hashlib.sha256(self._canonical_semantic_bytes()).hexdigest(),
        )

    def _validate(self) -> None:
        """Validate ownership and canonicalise complete manifest contents."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        memberships = self._canonicalise_memberships()
        components = self._canonicalise_components()
        object.__setattr__(self, "document_memberships", memberships)
        object.__setattr__(self, "component_generations", components)

    def _canonicalise_memberships(
        self,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        """Validate and order materialised document-version membership."""
        if not isinstance(self.document_memberships, tuple):
            raise TypeError("document_memberships must be a tuple.")
        if any(
            not isinstance(item, CorpusDocumentVersionMembership)
            for item in self.document_memberships
        ):
            raise TypeError(
                "document_memberships must contain only "
                "CorpusDocumentVersionMembership values."
            )
        if any(item.corpus_id != self.corpus_id for item in self.document_memberships):
            raise InvalidCorpusVersionContractError(
                "document membership must belong to the manifest corpus."
            )
        document_ids = tuple(item.document_id for item in self.document_memberships)
        if len(document_ids) != len(set(document_ids)):
            raise InvalidCorpusVersionContractError(
                "a manifest cannot contain multiple memberships for one document."
            )
        return tuple(
            sorted(
                self.document_memberships,
                key=lambda item: (
                    item.document_id.as_str(),
                    item.document_version_id.as_str(),
                ),
            )
        )

    def _canonicalise_components(
        self,
    ) -> tuple[ComponentGenerationReference, ...]:
        """Validate and order complete ready component generations."""
        if not isinstance(self.component_generations, tuple):
            raise TypeError("component_generations must be a tuple.")
        if any(
            not isinstance(item, ComponentGenerationReference)
            for item in self.component_generations
        ):
            raise TypeError(
                "component_generations must contain only "
                "ComponentGenerationReference values."
            )
        if any(item.corpus_id != self.corpus_id for item in self.component_generations):
            raise InvalidCorpusVersionContractError(
                "component generations must belong to the manifest corpus."
            )
        if any(
            item.corpus_version != self.corpus_version
            for item in self.component_generations
        ):
            raise InvalidCorpusVersionContractError(
                "component generations must belong to the manifest corpus version."
            )
        if any(
            item.state is not ComponentGenerationState.READY
            for item in self.component_generations
        ):
            raise InvalidCorpusVersionContractError(
                "a manifest can contain only ready component generations."
            )
        component_kinds = tuple(
            item.component_kind for item in self.component_generations
        )
        if len(component_kinds) != len(set(component_kinds)):
            raise InvalidCorpusVersionContractError(
                "a manifest cannot contain conflicting component kinds."
            )
        return tuple(sorted(self.component_generations, key=lambda item: item.key()))

    def _canonical_semantic_bytes(self) -> bytes:
        """Return stable bytes for state equality and manifest hashing.

        The proposed corpus-version identity and readiness state are deliberately
        excluded. They describe publication ownership and staging, not the
        complete corpus state used by the no-change guard.
        """
        payload = {
            "component_generations": [
                {
                    "component_kind": item.component_kind.as_str(),
                    "configuration_digest": item.configuration_digest,
                    "contract_version": item.contract_version,
                    "format_version": item.format_version,
                    "generation_id": item.generation_id,
                    "implementation": item.implementation,
                    "source_manifest_digest": item.source_manifest_digest,
                }
                for item in self.component_generations
            ],
            "contract_version": CORPUS_MANIFEST_CONTRACT_VERSION,
            "corpus_id": self.corpus_id.as_str(),
            "document_memberships": [
                {
                    "document_id": item.document_id.as_str(),
                    "document_version_id": item.document_version_id.as_str(),
                }
                for item in self.document_memberships
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def has_required_components(
        self,
        required_component_kinds: tuple[ComponentKind, ...],
    ) -> bool:
        """Return whether the manifest contains exactly the declared kinds."""
        canonical_required = canonicalise_component_kinds(required_component_kinds)
        actual = tuple(item.component_kind for item in self.component_generations)
        return actual == canonical_required


class CorpusUpdateState(StrEnum):
    """Durable lifecycle states for one proposed corpus version."""

    PREPARING = "preparing"
    READY = "ready"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class CorpusUpdate:
    """Describe one durable proposal to advance a selected corpus."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    expected_corpus_version: CorpusVersionId | None
    required_component_kinds: tuple[ComponentKind, ...]
    state: CorpusUpdateState
    created_at: datetime
    manifest_digest: str | None = None
    restores_corpus_version: CorpusVersionId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate proposal ownership, lifecycle, and sealed-state fields."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if self.expected_corpus_version is not None and not isinstance(
            self.expected_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError(
                "expected_corpus_version must be a CorpusVersionId or None."
            )
        self._validate_restore_target()
        validate_corpus_update_identity(
            self.corpus_version,
            self.expected_corpus_version,
            self.restores_corpus_version,
        )
        canonical_kinds = canonicalise_component_kinds(self.required_component_kinds)
        object.__setattr__(self, "required_component_kinds", canonical_kinds)
        if not isinstance(self.state, CorpusUpdateState):
            raise TypeError("state must be a CorpusUpdateState.")
        _validate_utc_datetime(self.created_at, "created_at")
        self._validate_manifest_digest()

    def _validate_manifest_digest(self) -> None:
        """Require a sealed digest exactly when lifecycle state permits one."""
        if self.manifest_digest is not None:
            _validate_sha256_digest(self.manifest_digest, "manifest_digest")
        if self.state in {CorpusUpdateState.READY, CorpusUpdateState.COMMITTED}:
            if self.manifest_digest is None:
                raise InvalidCorpusVersionContractError(
                    "ready and committed updates require a manifest_digest."
                )
        elif self.state is CorpusUpdateState.PREPARING:
            if self.manifest_digest is not None:
                raise InvalidCorpusVersionContractError(
                    "a preparing update cannot have a manifest_digest."
                )

    def _validate_restore_target(self) -> None:
        """Validate the optional rollback publication audit target."""
        if self.restores_corpus_version is not None and not isinstance(
            self.restores_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError(
                "restores_corpus_version must be a CorpusVersionId or None."
            )


@dataclass(frozen=True, slots=True)
class CorpusVersion:
    """Describe one immutable committed snapshot owned by a corpus."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    sequence: int
    parent_corpus_version: CorpusVersionId | None
    manifest_digest: str
    created_at: datetime
    restores_corpus_version: CorpusVersionId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate local version identity, sequence, and history invariants."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise TypeError("sequence must be an integer.")
        if self.sequence < 1:
            raise InvalidCorpusVersionContractError("sequence must be at least one.")
        if self.parent_corpus_version is not None and not isinstance(
            self.parent_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError("parent_corpus_version must be a CorpusVersionId or None.")
        if self.sequence == 1 and self.parent_corpus_version is not None:
            raise InvalidCorpusVersionContractError(
                "the first corpus version cannot have a parent."
            )
        if self.sequence > 1 and self.parent_corpus_version is None:
            raise InvalidCorpusVersionContractError(
                "a later corpus version requires a parent."
            )
        if self.parent_corpus_version == self.corpus_version:
            raise InvalidCorpusVersionContractError(
                "a corpus version cannot be its own parent."
            )
        _validate_sha256_digest(self.manifest_digest, "manifest_digest")
        _validate_utc_datetime(self.created_at, "created_at")
        self._validate_restore_target()

    def _validate_restore_target(self) -> None:
        """Validate the optional historical version restored by publication."""
        if self.restores_corpus_version is not None and not isinstance(
            self.restores_corpus_version,
            CorpusVersionId,
        ):
            raise TypeError(
                "restores_corpus_version must be a CorpusVersionId or None."
            )
        if self.restores_corpus_version == self.corpus_version:
            raise InvalidCorpusVersionContractError(
                "a corpus version cannot restore itself."
            )
        if self.sequence == 1 and self.restores_corpus_version is not None:
            raise InvalidCorpusVersionContractError(
                "the first corpus version cannot be a rollback publication."
            )
        if (
            self.restores_corpus_version is not None
            and self.restores_corpus_version == self.parent_corpus_version
        ):
            raise InvalidCorpusVersionContractError(
                "a rollback target must be historical, not the version parent."
            )

    def matches_manifest(self, manifest: CorpusVersionManifest) -> bool:
        """Return whether a complete manifest belongs to this version."""
        if not isinstance(manifest, CorpusVersionManifest):
            raise TypeError("manifest must be a CorpusVersionManifest.")
        return (
            manifest.corpus_id == self.corpus_id
            and manifest.corpus_version == self.corpus_version
            and manifest.manifest_digest == self.manifest_digest
        )
