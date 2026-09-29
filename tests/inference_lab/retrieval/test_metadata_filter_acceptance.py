"""Acceptance tests for supported metadata filter fields and predicates."""

import pytest

from inference_lab.retrieval import (
    CURRENT_METADATA_FILTER_REGISTRY,
    MAX_METADATA_FILTER_DEPTH,
    MAX_METADATA_FILTER_NODES,
    AllFilter,
    AnyFilter,
    ExactFilter,
    ExistsFilter,
    FilterScalar,
    InvalidMetadataFilterError,
    MetadataFieldType,
    MetadataFilterErrorCategory,
    MetadataFilterOperator,
    MetadataMissingPolicy,
    MetadataValues,
    MissingFilter,
    NotFilter,
    OneOfFilter,
    RangeFilter,
    evaluate_metadata_filter,
    validate_metadata_filter,
)


def test_ret03_registry_accepts_predicates_for_every_supported_field() -> None:
    values: dict[str, str | int] = {
        "character-count": 80,
        "document-type": "text/plain",
        "language": "en",
        "page-count": 2,
        "source-byte-count": 100,
        "token-count": 20,
    }
    assert CURRENT_METADATA_FILTER_REGISTRY.registry_version == "1.0.0"
    assert tuple(field.name for field in CURRENT_METADATA_FILTER_REGISTRY.fields) == (
        "character-count",
        "document-type",
        "language",
        "page-count",
        "source-byte-count",
        "token-count",
    )

    for name, value in values.items():
        field = CURRENT_METADATA_FILTER_REGISTRY.descriptor(name)
        assert field is not None
        expected_type = (
            MetadataFieldType.STRING
            if isinstance(value, str)
            else MetadataFieldType.INTEGER
        )
        assert field.field_type is expected_type
        assert validate_metadata_filter(ExactFilter(name, value)).node_count == 1
        assert validate_metadata_filter(OneOfFilter(name, (value,))).node_count == 1
        assert validate_metadata_filter(ExistsFilter(name)).node_count == 1
        assert validate_metadata_filter(MissingFilter(name)).node_count == 1
        if type(value) is int:
            assert (
                validate_metadata_filter(RangeFilter(name, value, value)).node_count
                == 1
            )

    language = CURRENT_METADATA_FILTER_REGISTRY.descriptor("language")
    assert language is not None
    assert language.missing_policy is MetadataMissingPolicy.NULLABLE
    assert MetadataFilterOperator.RANGE not in language.supported_operators


def test_ret03_predicates_and_boolean_combinations_match_real_field_values() -> None:
    values: MetadataValues = {
        "character-count": 80,
        "document-type": "text/plain",
        "language": "en",
        "page-count": 2,
        "source-byte-count": 100,
        "token-count": 20,
    }
    for field, value in values.items():
        assert value is not None
        scalar: FilterScalar = value
        assert evaluate_metadata_filter(ExactFilter(field, scalar), values)
        assert evaluate_metadata_filter(OneOfFilter(field, (scalar,)), values)
        assert evaluate_metadata_filter(ExistsFilter(field), values)
        assert not evaluate_metadata_filter(MissingFilter(field), values)
        if type(value) is int:
            assert evaluate_metadata_filter(RangeFilter(field, value, value), values)

    combined = AllFilter(
        (
            ExistsFilter("language"),
            AnyFilter(
                (
                    ExactFilter("language", "en"),
                    ExactFilter("language", "fr"),
                )
            ),
            RangeFilter("page-count", 1, 3, lower_inclusive=False),
        )
    )
    assert evaluate_metadata_filter(combined, values)
    assert evaluate_metadata_filter(RangeFilter("source-byte-count", 100, None), values)
    assert not evaluate_metadata_filter(
        RangeFilter("page-count", 2, 4, lower_inclusive=False), values
    )
    assert evaluate_metadata_filter(ExistsFilter("page-count"), {"page-count": 0})
    assert evaluate_metadata_filter(MissingFilter("language"), {"language": None})
    assert not evaluate_metadata_filter(ExistsFilter("language"), {"language": None})


def test_ret03_rejects_unsupported_fields_and_invalid_field_operator_pairs() -> None:
    unsupported = (
        ExactFilter("publication-date", "2024-01-01"),
        ExactFilter("priority-date", "2024-01-01"),
        ExactFilter("assignee", "Acme"),
    )
    for expression in unsupported:
        with pytest.raises(InvalidMetadataFilterError) as error:
            validate_metadata_filter(expression)
        assert error.value.category is MetadataFilterErrorCategory.UNSUPPORTED_FIELD

    invalid = (
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
    )
    for expression, category in invalid:
        with pytest.raises(InvalidMetadataFilterError) as error:
            validate_metadata_filter(expression)
        assert error.value.category is category

    for field in ("character-count", "page-count", "source-byte-count", "token-count"):
        with pytest.raises(InvalidMetadataFilterError) as error:
            validate_metadata_filter(ExactFilter(field, "1"))
        assert error.value.category is MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH


def test_ret03_validation_is_bounded_private_and_rechecks_its_token() -> None:
    expression = ExactFilter("language", "en")
    validation = validate_metadata_filter(expression)
    forged = type(validation)(
        expression=expression,
        registry_version=validation.registry_version,
        node_count=validation.node_count + 1,
        maximum_depth=validation.maximum_depth,
    )
    with pytest.raises(InvalidMetadataFilterError) as token_error:
        evaluate_metadata_filter(forged, {"language": "en"})
    assert token_error.value.category is MetadataFilterErrorCategory.INVALID_METADATA

    deep: object = expression
    for _ in range(MAX_METADATA_FILTER_DEPTH):
        deep = NotFilter(deep)  # type: ignore[arg-type]
    wide = AllFilter(
        tuple(ExactFilter("language", "en") for _ in range(MAX_METADATA_FILTER_NODES))
    )
    for candidate, category in (
        (deep, MetadataFilterErrorCategory.DEPTH_EXCEEDED),
        (wide, MetadataFilterErrorCategory.NODE_COUNT_EXCEEDED),
    ):
        with pytest.raises(InvalidMetadataFilterError) as error:
            validate_metadata_filter(candidate)  # type: ignore[arg-type]
        assert error.value.category is category

    for view in (
        {"page-count": True},
        {"language": 1},
        {"unsupported": "private-value"},
        {1: "private-value"},
    ):
        with pytest.raises((InvalidMetadataFilterError, TypeError)) as error:
            evaluate_metadata_filter(ExistsFilter("language"), view)  # type: ignore[arg-type]
        assert "private-value" not in repr(error.value)
