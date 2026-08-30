"""Immutable value types for document identity resolution."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

_IDENTIFIER_NAMESPACE: Final = re.compile(r"^[a-z][a-z0-9_-]*$")
_SHA256_HEX_LENGTH: Final = 64
_HEX_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ExternalIdentifier:
    """A stable identifier supplied by a publication or patent source."""

    namespace: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str):
            raise TypeError("namespace must be a string.")

        if not isinstance(self.value, str):
            raise TypeError("value must be a string.")

        if _IDENTIFIER_NAMESPACE.fullmatch(self.namespace) is None:
            raise ValueError("namespace must be a lowercase identifier namespace.")

        if not self.value.strip():
            raise ValueError("value must not be empty or whitespace.")

    def as_tuple(self) -> tuple[str, str]:
        """Return the identifier as a namespace/value tuple."""
        return self.namespace, self.value


class HashAlgorithm(StrEnum):
    """Content-hash algorithms supported by the identity contract."""

    SHA256 = "sha256"


@dataclass(frozen=True, slots=True)
class ContentHash:
    """Hash of the original source bytes."""

    algorithm: HashAlgorithm
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, HashAlgorithm):
            raise TypeError("algorithm must be a HashAlgorithm.")

        if not isinstance(self.digest, str):
            raise TypeError("digest must be a string.")

        if self.algorithm is HashAlgorithm.SHA256 and (
            len(self.digest) != _SHA256_HEX_LENGTH
            or _HEX_DIGEST.fullmatch(self.digest) is None
        ):
            raise ValueError(
                "SHA-256 digest must be 64 lowercase hexadecimal characters."
            )

    def as_str(self) -> str:
        """Return the algorithm-qualified canonical hash."""
        return f"{self.algorithm.value}:{self.digest}"

    @classmethod
    def from_str(cls, value: str) -> "ContentHash":
        """Reconstruct a content hash from canonical text."""
        if not isinstance(value, str):
            raise TypeError("value must be a string.")

        algorithm_value, separator, digest = value.partition(":")
        if not separator:
            raise ValueError("value must be in the format '<algorithm>:<digest>'.")

        try:
            algorithm = HashAlgorithm(algorithm_value)
        except ValueError as error:
            raise ValueError("unsupported content-hash algorithm.") from error

        return cls(algorithm=algorithm, digest=digest)


class DuplicateClass(StrEnum):
    """Supported duplicate relationship classes."""

    BYTE_IDENTICAL = "byte_identical"


class ResolutionKind(StrEnum):
    """Outcomes of deterministic document identity resolution."""

    UNCHANGED = "unchanged"
    NEW_VERSION = "new_version"
    NEW_DOCUMENT = "new_document"
    DUPLICATE = "duplicate"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
