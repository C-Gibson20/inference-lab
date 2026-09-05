"""Immutable contracts for corpus versions and staged publication."""

import hashlib
import json
import re
from dataclasses import dataclass, field
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


def _validate_component_kinds(kinds: object) -> tuple[ComponentKind, ...]:
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
        canonical_required = _validate_component_kinds(required_component_kinds)
        actual = tuple(item.component_kind for item in self.component_generations)
        return actual == canonical_required
