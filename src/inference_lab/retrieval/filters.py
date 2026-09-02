"""Typed, immutable retrieval filter expressions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, TypeAlias, cast

from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.values import validate_field_name, validate_filter_scalar

FilterScalar: TypeAlias = str | int | float | bool | date


class _ComparableScalar(Protocol):
    def __gt__(self, other: object, /) -> bool: ...


def _validate_field_and_value(field: object, value: object) -> None:
    validate_field_name(field)
    validate_filter_scalar(value)


@dataclass(frozen=True, slots=True)
class ExactFilter:
    """Match a field against one exactly typed value."""

    field: str
    value: FilterScalar

    def __post_init__(self) -> None:
        _validate_field_and_value(self.field, self.value)


@dataclass(frozen=True, slots=True)
class OneOfFilter:
    """Match a field against one of several uniquely typed values."""

    field: str
    values: tuple[FilterScalar, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the filter's field and values."""
        validate_field_name(self.field)
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple.")
        if not self.values:
            raise InvalidRetrievalContractError("values must not be empty.")
        for value in self.values:
            validate_filter_scalar(value)
        if len(self.values) != len(set(self.values)):
            raise InvalidRetrievalContractError("values must not contain duplicates.")
        if len({type(value) for value in self.values}) != 1:
            raise InvalidRetrievalContractError("values must have one scalar type.")


@dataclass(frozen=True, slots=True)
class RangeFilter:
    """Match a field within one or two inclusive or exclusive bounds."""

    field: str
    lower: FilterScalar | None = None
    upper: FilterScalar | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the filter's field and bounds."""
        validate_field_name(self.field)
        if self.lower is None and self.upper is None:
            raise InvalidRetrievalContractError(
                "a range must contain at least one bound."
            )
        if self.lower is not None:
            validate_filter_scalar(self.lower, name="lower")
        if self.upper is not None:
            validate_filter_scalar(self.upper, name="upper")
        if type(self.lower_inclusive) is not bool:
            raise TypeError("lower_inclusive must be a boolean.")
        if type(self.upper_inclusive) is not bool:
            raise TypeError("upper_inclusive must be a boolean.")
        if self.lower is not None and self.upper is not None:
            if type(self.lower) is not type(self.upper):
                raise InvalidRetrievalContractError(
                    "range bounds must have one scalar type."
                )
            lower = cast(_ComparableScalar, self.lower)
            upper = cast(_ComparableScalar, self.upper)
            if lower > upper:
                raise InvalidRetrievalContractError(
                    "range lower bound must not exceed upper bound."
                )
            if self.lower == self.upper and not (
                self.lower_inclusive and self.upper_inclusive
            ):
                raise InvalidRetrievalContractError(
                    "an empty range is not a valid filter."
                )


@dataclass(frozen=True, slots=True)
class ExistsFilter:
    """Match records where a field is present."""

    field: str

    def __post_init__(self) -> None:
        validate_field_name(self.field)


@dataclass(frozen=True, slots=True)
class MissingFilter:
    """Match records where a field is missing."""

    field: str

    def __post_init__(self) -> None:
        validate_field_name(self.field)


@dataclass(frozen=True, slots=True)
class AllFilter:
    """Match records satisfying every child expression."""

    children: tuple[FilterExpression, ...]

    def __post_init__(self) -> None:
        _validate_children(self.children, "children")


@dataclass(frozen=True, slots=True)
class AnyFilter:
    """Match records satisfying at least one child expression."""

    children: tuple[FilterExpression, ...]

    def __post_init__(self) -> None:
        _validate_children(self.children, "children")


@dataclass(frozen=True, slots=True)
class NotFilter:
    """Match records that do not satisfy the child expression."""

    child: FilterExpression

    def __post_init__(self) -> None:
        if not isinstance(self.child, _FILTER_TYPES):
            raise TypeError("child must be a filter expression.")


FilterExpression: TypeAlias = (
    ExactFilter
    | OneOfFilter
    | RangeFilter
    | ExistsFilter
    | MissingFilter
    | AllFilter
    | AnyFilter
    | NotFilter
)


def _validate_children(children: object, name: str) -> None:
    if not isinstance(children, tuple):
        raise TypeError(f"{name} must be a tuple.")
    if not children:
        raise InvalidRetrievalContractError(f"{name} must not be empty.")
    if any(not isinstance(child, _FILTER_TYPES) for child in children):
        raise TypeError(f"{name} must contain only filter expressions.")


_FILTER_TYPES = (
    ExactFilter,
    OneOfFilter,
    RangeFilter,
    ExistsFilter,
    MissingFilter,
    AllFilter,
    AnyFilter,
    NotFilter,
)
