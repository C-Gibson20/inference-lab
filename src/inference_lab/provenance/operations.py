"""Immutable producing-component and operation metadata."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Self

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.provenance._validation import (
    is_lowercase_slug,
    is_sha256_hex_digest,
)
from inference_lab.provenance.errors import InvalidLineageMetadataError

_REVISION_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def _validate_slug(value: object, field_name: str) -> str:
    """Return a canonical project-owned slug after strict validation."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_lowercase_slug(value):
        raise InvalidLineageMetadataError(f"{field_name} must be a lowercase slug.")
    return value


def _validate_semantic_version(value: object, field_name: str) -> str:
    """Return a canonical Semantic Version after strict validation."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise InvalidLineageMetadataError(
            f"{field_name} must be a canonical semantic version."
        )
    return value


@dataclass(frozen=True, slots=True)
class OperationKind:
    """Validated, extensible kind of producing operation."""

    value: str

    def __post_init__(self) -> None:
        _validate_slug(self.value, "operation_kind")

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Reconstruct an operation kind from its canonical string value."""
        return cls(value)

    def as_str(self) -> str:
        """Return the canonical operation-kind value."""
        return self.value


@dataclass(frozen=True, slots=True)
class ComponentRole:
    """Validated role played by one producing component."""

    value: str

    def __post_init__(self) -> None:
        _validate_slug(self.value, "component_role")

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Reconstruct a component role from its canonical string value."""
        return cls(value)

    def as_str(self) -> str:
        """Return the canonical component-role value."""
        return self.value


class DigestAlgorithm(StrEnum):
    """Algorithms supported for privacy-safe lineage digests."""

    SHA256 = "sha256"


@dataclass(frozen=True, slots=True)
class ConfigurationIdentity:
    """Canonical digest of invalidation-relevant producing configuration."""

    algorithm: DigestAlgorithm
    digest: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the algorithm and digest values."""
        if not isinstance(self.algorithm, DigestAlgorithm):
            raise TypeError("configuration algorithm must be a DigestAlgorithm.")
        if not isinstance(self.digest, str):
            raise TypeError("configuration digest must be a string.")
        if self.algorithm is DigestAlgorithm.SHA256 and not is_sha256_hex_digest(
            self.digest
        ):
            raise InvalidLineageMetadataError(
                "configuration digest must be 64 lowercase hexadecimal characters."
            )

    def as_str(self) -> str:
        """Return the algorithm-qualified configuration identity."""
        return f"{self.algorithm.value}:{self.digest}"


@dataclass(frozen=True, slots=True)
class ProducingComponent:
    """Identify one versioned component used by a producing operation."""

    role: ComponentRole
    implementation: str
    contract_version: str
    artefact_identity: str | None = None
    artefact_revision: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the component metadata values."""
        if not isinstance(self.role, ComponentRole):
            raise TypeError("component role must be a ComponentRole.")
        _validate_slug(self.implementation, "component implementation")
        _validate_semantic_version(
            self.contract_version,
            "component contract_version",
        )
        self._validate_artefact()

    def _validate_artefact(self) -> None:
        """Validate optional external artefact identity and revision together."""
        if self.artefact_identity is not None and not isinstance(
            self.artefact_identity, str
        ):
            raise TypeError("artefact_identity must be a string or None.")
        if self.artefact_revision is not None and not isinstance(
            self.artefact_revision, str
        ):
            raise TypeError("artefact_revision must be a string or None.")
        if (self.artefact_identity is None) != (self.artefact_revision is None):
            raise InvalidLineageMetadataError(
                "artefact identity and revision must either both be present or absent."
            )
        if self.artefact_identity is None:
            return
        if _REVISION_PATTERN.fullmatch(self.artefact_identity) is None:
            raise InvalidLineageMetadataError(
                "artefact_identity must be a canonical non-path identifier."
            )
        assert self.artefact_revision is not None
        if _REVISION_PATTERN.fullmatch(self.artefact_revision) is None:
            raise InvalidLineageMetadataError(
                "artefact_revision must be a canonical revision identifier."
            )

    def key(self) -> tuple[str, str, str, str, str]:
        """Return the canonical ordering and uniqueness key."""
        return (
            self.role.as_str(),
            self.implementation,
            self.contract_version,
            self.artefact_identity or "",
            self.artefact_revision or "",
        )


@dataclass(frozen=True, slots=True)
class ProducingOperation:
    """Identify the operation and components that produced a derived record."""

    kind: OperationKind
    implementation: str
    contract_version: str
    configuration_identity: ConfigurationIdentity
    components: tuple[ProducingComponent, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the operation metadata values."""
        if not isinstance(self.kind, OperationKind):
            raise TypeError("operation kind must be an OperationKind.")
        _validate_slug(self.implementation, "operation implementation")
        _validate_semantic_version(
            self.contract_version,
            "operation contract_version",
        )
        if not isinstance(self.configuration_identity, ConfigurationIdentity):
            raise TypeError("configuration_identity must be a ConfigurationIdentity.")
        self._validate_components()

    def _validate_components(self) -> None:
        """Require immutable, unique components in canonical order."""
        if not isinstance(self.components, tuple):
            raise TypeError("components must be a tuple.")
        if any(
            not isinstance(component, ProducingComponent)
            for component in self.components
        ):
            raise TypeError("components must contain only ProducingComponent values.")

        keys = tuple(component.key() for component in self.components)
        if len(keys) != len(set(keys)):
            raise InvalidLineageMetadataError("components must not contain duplicates.")
        if keys != tuple(sorted(keys)):
            raise InvalidLineageMetadataError("components must be in canonical order.")
