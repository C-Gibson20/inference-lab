"""Tests for the RET-03 executable metadata-filter contract."""

import pytest

from inference_lab.retrieval import (
    CURRENT_METADATA_FILTER_REGISTRY,
    MAX_METADATA_FILTER_DEPTH,
    MAX_METADATA_FILTER_NODES,
    AllFilter,
    AnyFilter,
    ExactFilter,
    ExistsFilter,
    InvalidMetadataFilterError,
    InvalidRetrievalContractError,
    MetadataCardinality,
    MetadataFieldType,
    MetadataFilterErrorCategory,
    MetadataFilterOperator,
    MetadataFilterRegistry,
    MetadataMissingPolicy,
    MissingFilter,
    NotFilter,
    OneOfFilter,
    RangeFilter,
    evaluate_metadata_filter,
    validate_metadata_filter,
)


def test_current_registry_is_closed_versioned_and_durably_described() -> None:
    assert CURRENT_METADATA_FILTER_REGISTRY.registry_version == "1.0.0"
    assert tuple(field.name for field in CURRENT_METADATA_FILTER_REGISTRY.fields) == (
        "character-count",
        "document-type",
        "language",
        "page-count",
        "source-byte-count",
        "token-count",
    )
    assert all(
        field.cardinality is MetadataCardinality.SCALAR
        for field in CURRENT_METADATA_FILTER_REGISTRY.fields
    )
    document_type = CURRENT_METADATA_FILTER_REGISTRY.descriptor("document-type")
    language = CURRENT_METADATA_FILTER_REGISTRY.descriptor("language")
    assert document_type is not None
    assert language is not None
    assert document_type.field_type is MetadataFieldType.STRING
    assert language.missing_policy is MetadataMissingPolicy.NULLABLE
    assert MetadataFilterOperator.RANGE not in (language.supported_operators)


@pytest.mark.parametrize(
    ("field", "value", "expected_operator_values"),
    [
        (
            "document-type",
            "text/plain",
            {"exact", "one_of", "exists", "missing"},
        ),
        (
            "language",
            "en",
            {"exact", "one_of", "exists", "missing"},
        ),
        (
            "source-byte-count",
            100,
            {"exact", "one_of", "range", "exists", "missing"},
        ),
        (
            "page-count",
            3,
            {"exact", "one_of", "range", "exists", "missing"},
        ),
    ],
)
def test_each_initial_field_supports_its_declared_predicates(
    field: str,
    value: str | int,
    expected_operator_values: set[str],
) -> None:
    descriptor = CURRENT_METADATA_FILTER_REGISTRY.descriptor(field)
    assert descriptor is not None
    assert {operator.value for operator in descriptor.supported_operators} == (
        expected_operator_values
    )
    assert validate_metadata_filter(ExactFilter(field, value)).node_count == 1
    assert validate_metadata_filter(OneOfFilter(field, (value,))).node_count == 1
    assert validate_metadata_filter(ExistsFilter(field)).node_count == 1
    assert validate_metadata_filter(MissingFilter(field)).node_count == 1
    if MetadataFilterOperator.RANGE in descriptor.supported_operators:
        assert (
            validate_metadata_filter(RangeFilter(field, value, value)).node_count == 1
        )


@pytest.mark.parametrize(
    "expression",
    [
        ExactFilter("publication-date", "2024-01-01"),
        ExactFilter("priority-date", "2024-01-01"),
        ExactFilter("assignee", "Acme"),
    ],
)
def test_reserved_phase_9_fields_are_unsupported_not_missing(
    expression: ExactFilter,
) -> None:
    with pytest.raises(InvalidMetadataFilterError) as error:
        validate_metadata_filter(expression)
    assert error.value.category is MetadataFilterErrorCategory.UNSUPPORTED_FIELD
    assert str(error.value) == "unsupported_field"
    assert "2024" not in repr(error.value)
    assert "Acme" not in repr(error.value)


@pytest.mark.parametrize(
    ("expression", "category"),
    [
        (
            RangeFilter("language", "a", "z"),
            MetadataFilterErrorCategory.UNSUPPORTED_OPERATOR,
        ),
        (
            ExactFilter("page-count", "3"),
            MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH,
        ),
        (
            OneOfFilter("source-byte-count", (True,)),
            MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH,
        ),
        (
            ExactFilter("document-type", 1),
            MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH,
        ),
    ],
)
def test_field_operator_and_scalar_types_are_validated(
    expression: ExactFilter | OneOfFilter | RangeFilter,
    category: MetadataFilterErrorCategory,
) -> None:
    with pytest.raises(InvalidMetadataFilterError) as error:
        validate_metadata_filter(expression)
    assert error.value.category is category


@pytest.mark.parametrize("field", ["document-type", "language"])
def test_range_is_rejected_for_every_string_field(field: str) -> None:
    with pytest.raises(InvalidMetadataFilterError) as error:
        validate_metadata_filter(RangeFilter(field, "a", "z"))
    assert error.value.category is MetadataFilterErrorCategory.UNSUPPORTED_OPERATOR


@pytest.mark.parametrize(
    "field", ["character-count", "page-count", "source-byte-count", "token-count"]
)
def test_string_values_are_rejected_for_every_integer_field(field: str) -> None:
    with pytest.raises(InvalidMetadataFilterError) as error:
        validate_metadata_filter(ExactFilter(field, "1"))
    assert error.value.category is MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH


def test_all_initial_fields_accept_exact_one_of_exists_missing_and_ranges() -> None:
    values = {
        "document-type": "text/plain",
        "source-byte-count": 100,
        "language": "en",
        "page-count": 2,
        "character-count": 80,
        "token-count": 20,
    }
    for field, value in values.items():
        assert evaluate_metadata_filter(ExactFilter(field, value), values)
        assert evaluate_metadata_filter(OneOfFilter(field, (value,)), values)
        assert evaluate_metadata_filter(ExistsFilter(field), values)
        assert not evaluate_metadata_filter(MissingFilter(field), values)
        if type(value) is int:
            assert evaluate_metadata_filter(RangeFilter(field, value, value), values)


@pytest.mark.parametrize(
    ("expression", "values", "expected"),
    [
        (ExactFilter("language", "en"), {"language": "en"}, True),
        (ExactFilter("language", "en"), {"language": None}, False),
        (
            OneOfFilter("document-type", ("text/plain", "application/pdf")),
            {"document-type": "text/plain"},
            True,
        ),
        (RangeFilter("page-count", 2, 4, False, True), {"page-count": 2}, False),
        (RangeFilter("page-count", 2, 4, False, True), {"page-count": 3}, True),
        (RangeFilter("page-count", None, 4), {"page-count": 4}, True),
        (RangeFilter("page-count", 2, None, True, False), {"page-count": 1}, False),
        (ExistsFilter("language"), {}, False),
        (MissingFilter("language"), {}, True),
        (NotFilter(ExactFilter("language", "en")), {}, True),
        (
            AllFilter(
                (
                    ExistsFilter("language"),
                    AnyFilter(
                        (
                            ExactFilter("language", "en"),
                            ExactFilter("language", "fr"),
                        )
                    ),
                )
            ),
            {"language": "en"},
            True,
        ),
    ],
)
def test_predicates_use_strict_presence_and_boolean_semantics(
    expression: object,
    values: dict[str, object],
    expected: bool,
) -> None:
    assert evaluate_metadata_filter(expression, values) is expected  # type: ignore[arg-type]


def test_missing_is_complement_of_presence_and_zero_is_not_missing() -> None:
    assert evaluate_metadata_filter(ExistsFilter("page-count"), {"page-count": 0})
    assert not evaluate_metadata_filter(MissingFilter("page-count"), {"page-count": 0})
    assert evaluate_metadata_filter(
        NotFilter(ExactFilter("page-count", 2)), {"page-count": None}
    )
    assert not evaluate_metadata_filter(
        NotFilter(MissingFilter("page-count")), {"page-count": None}
    )


def test_validation_reports_bounded_counts_and_registry_identity() -> None:
    validation = validate_metadata_filter(
        AllFilter(
            (
                ExactFilter("language", "en"),
                NotFilter(ExistsFilter("page-count")),
            )
        )
    )
    assert validation.node_count == 4
    assert validation.maximum_depth == 3
    assert validation.registry_version == "1.0.0"


def test_evaluation_rechecks_a_validation_token_before_use() -> None:
    expression = ExactFilter("language", "en")
    validation = validate_metadata_filter(expression)
    forged = type(validation)(
        expression=expression,
        registry_version=validation.registry_version,
        node_count=validation.node_count + 1,
        maximum_depth=validation.maximum_depth,
    )
    with pytest.raises(InvalidMetadataFilterError) as error:
        evaluate_metadata_filter(forged, {"language": "en"})
    assert error.value.category is MetadataFilterErrorCategory.INVALID_METADATA


def test_validation_rejects_depth_and_node_budget_overruns() -> None:
    deep: object = ExactFilter("language", "en")
    for _ in range(MAX_METADATA_FILTER_DEPTH):
        deep = NotFilter(deep)  # type: ignore[arg-type]
    with pytest.raises(InvalidMetadataFilterError) as depth_error:
        validate_metadata_filter(deep)  # type: ignore[arg-type]
    assert depth_error.value.category is MetadataFilterErrorCategory.DEPTH_EXCEEDED

    wide: object = AllFilter(
        tuple(ExactFilter("language", "en") for _ in range(MAX_METADATA_FILTER_NODES))
    )
    with pytest.raises(InvalidMetadataFilterError) as node_error:
        validate_metadata_filter(wide)  # type: ignore[arg-type]
    assert node_error.value.category is MetadataFilterErrorCategory.NODE_COUNT_EXCEEDED


@pytest.mark.parametrize(
    "values",
    [
        {"page-count": True},
        {"language": 1},
        {"unsupported": "value"},
        {1: "value"},
    ],
)
def test_metadata_views_fail_closed_without_echoing_values(values: object) -> None:
    with pytest.raises((InvalidMetadataFilterError, TypeError)) as error:
        evaluate_metadata_filter(ExistsFilter("language"), values)  # type: ignore[arg-type]
    assert "value" not in repr(error.value)


def test_evaluation_does_not_mutate_the_metadata_view() -> None:
    values = {"language": "en", "page-count": 0}
    original = values.copy()
    evaluate_metadata_filter(
        AllFilter((ExistsFilter("language"), ExistsFilter("page-count"))),
        values,
    )
    assert values == original


def test_registry_rejects_noncanonical_versions_and_unordered_fields() -> None:
    with pytest.raises(ValueError):
        MetadataFilterRegistry("1.0", CURRENT_METADATA_FILTER_REGISTRY.fields)
    with pytest.raises(InvalidRetrievalContractError):
        MetadataFilterRegistry(
            "1.0.0", tuple(reversed(CURRENT_METADATA_FILTER_REGISTRY.fields))
        )
