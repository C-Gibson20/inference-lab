"""Immutable, typed identifiers used by the provenance domain."""

import uuid
from typing import Self, final

__all__ = [
    "CorpusId",
    "CorpusVersionId",
    "DocumentId",
    "DocumentVersionId",
    "RecordId",
    "SourceSpanId",
]


class _Identifier:
    """Base value object for opaque, UUID-backed identifiers.

    Concrete subclasses define distinct identity domains. Two identifiers
    with the same serialised value but different concrete types are therefore
    not equal.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """Create an identifier from its canonical string representation."""
        object.__setattr__(self, "_value", self._parse(value))

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Reconstruct an identifier from its canonical string value."""
        return cls(value)

    @classmethod
    def generate(cls) -> Self:
        """Generate a new opaque identifier for this identity domain."""
        return cls(str(uuid.uuid4()))

    @staticmethod
    def _parse(value: str) -> str:
        """Validate and return a canonical UUID string."""
        if not isinstance(value, str):
            raise TypeError("Identifier value must be a string.")

        try:
            parsed = uuid.UUID(value)
        except ValueError as error:
            raise ValueError("Identifier value must be a valid UUID.") from error

        canonical = str(parsed)
        if value != canonical:
            raise ValueError(
                "Identifier value must use the canonical lowercase UUID format."
            )
        return canonical

    def as_str(self) -> str:
        """Return the canonical serialised identifier value."""
        return self._value

    def __str__(self) -> str:
        """Return the canonical serialised identifier value."""
        return self._value

    def __repr__(self) -> str:
        """Return an unambiguous developer-facing representation."""
        return f"{type(self).__name__}({self._value!r})"

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent mutation after construction."""
        raise AttributeError(f"{type(self).__name__} is immutable.")

    def __eq__(self, other: object) -> bool:
        """Compare both identifier domain and value."""
        if not isinstance(other, _Identifier) or type(self) is not type(other):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        """Return a hash consistent with typed identifier equality."""
        return hash((type(self), self._value))


@final
class DocumentId(_Identifier):
    """Identify one logical document across its immutable versions."""

    __slots__ = ()


@final
class DocumentVersionId(_Identifier):
    """Identify one immutable source version of a document."""

    __slots__ = ()


@final
class RecordId(_Identifier):
    """Identify one canonical or derived provenance record."""

    __slots__ = ()


@final
class CorpusId(_Identifier):
    """Identify one persistent corpus independently of its snapshots."""

    __slots__ = ()


@final
class CorpusVersionId(_Identifier):
    """Identify one immutable snapshot of a corpus."""

    __slots__ = ()


@final
class SourceSpanId(_Identifier):
    """Identify one source location within a document version representation."""

    __slots__ = ()
