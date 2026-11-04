"""Errors raised by the in-memory retrieval contract values."""

from enum import StrEnum
from typing import ClassVar, Generic, TypeVar

_CategoryT = TypeVar("_CategoryT", bound=StrEnum)


class InvalidRetrievalContractError(ValueError):
    """Raised when correctly typed retrieval data violates its contract."""


class InvalidRetrievalResultError(InvalidRetrievalContractError):
    """Raised when an evidence result violates its status or ordering rules."""


class InvalidRetrievalSerialisationError(InvalidRetrievalContractError):
    """Raised when serialised retrieval data violates its strict contract."""


class UnsupportedRetrievalSchemaVersionError(InvalidRetrievalContractError):
    """Raised when retrieval data names an unsupported schema version."""


class RetrievalOperationalError(Exception, Generic[_CategoryT]):
    """Base class for payload-free failures crossing a retrieval boundary."""

    category_type: ClassVar[type[StrEnum]]

    def __init__(self, category: _CategoryT, *, retryable: bool = False) -> None:
        """Create an error carrying only a bounded category and a retry hint."""
        if not isinstance(category, self.category_type):
            raise TypeError(f"category must be a {self.category_type.__name__}.")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean.")
        super().__init__(category.value)
        self.category: _CategoryT = category
        self.retryable = retryable
