"""Tests for the retrieval filter contract."""

from datetime import date

import pytest

from inference_lab.retrieval import (
    AllFilter,
    AnyFilter,
    ExactFilter,
    ExistsFilter,
    InvalidRetrievalContractError,
    MissingFilter,
    NotFilter,
    OneOfFilter,
    RangeFilter,
    RetrievalLimit,
)


@pytest.mark.parametrize("value", [True, 1.0, "10"])
def test_limit_requires_a_strict_integer(value: object) -> None:
    with pytest.raises(TypeError):
        RetrievalLimit(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_limit_must_be_positive(value: int) -> None:
    with pytest.raises(InvalidRetrievalContractError):
        RetrievalLimit(value)


def test_filter_algebra_covers_all_predicates_and_combinators() -> None:
    exact = ExactFilter("assignee", "Acme")
    one_of = OneOfFilter("document-type", ("paper", "patent"))
    bounded = RangeFilter("publication-year", 2018, 2024, False, True)
    present = ExistsFilter("priority-date")
    missing = MissingFilter("inventor")
    expression = AllFilter((exact, AnyFilter((one_of, bounded)), NotFilter(missing)))

    nested = expression.children[1]
    assert isinstance(nested, AnyFilter)
    assert nested.children == (one_of, bounded)
    assert present.field == "priority-date"


@pytest.mark.parametrize(
    ("constructor", "arguments"),
    [
        (ExactFilter, ("Bad Field", "value")),
        (OneOfFilter, ("field", ("a", "a"))),
        (OneOfFilter, ("field", ("a", 1))),
        (RangeFilter, ("field", None, None)),
        (RangeFilter, ("field", 3, 2)),
        (AllFilter, ((),)),
        (AnyFilter, ((),)),
    ],
)
def test_filter_invariants_are_rejected(
    constructor: type[object], arguments: tuple[object, ...]
) -> None:
    with pytest.raises((InvalidRetrievalContractError, TypeError)):
        constructor(*arguments)  # type: ignore[operator]


def test_filter_scalars_distinguish_dates_booleans_and_finite_numbers() -> None:
    assert ExactFilter("priority-date", date(2024, 1, 2)).value == date(2024, 1, 2)
    assert ExactFilter("is-public", True).value is True
    with pytest.raises(InvalidRetrievalContractError):
        ExactFilter("score", float("inf"))
    with pytest.raises(TypeError):
        ExactFilter("score", object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("patent", "paper"), ("paper", "patent")),
        ((10, -2, 3), (-2, 3, 10)),
        ((1.5, -2.0, 0.0), (-2.0, 0.0, 1.5)),
        ((True, False), (False, True)),
        (
            (date(2024, 3, 1), date(2023, 12, 1)),
            (date(2023, 12, 1), date(2024, 3, 1)),
        ),
    ],
)
def test_one_of_values_are_canonical_within_their_scalar_type(
    values: tuple[object, ...],
    expected: tuple[object, ...],
) -> None:
    assert OneOfFilter("field", values).values == expected  # type: ignore[arg-type]
