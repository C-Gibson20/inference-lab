"""Executable, corpus-metadata filter contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, NoReturn, TypeAlias, cast

from inference_lab._validation import is_canonical_semantic_version
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.filters import (
    AllFilter,
    AnyFilter,
    ExactFilter,
    ExistsFilter,
    FilterExpression,
    FilterScalar,
    MissingFilter,
    NotFilter,
    OneOfFilter,
    RangeFilter,
)
from inference_lab.retrieval.values import validate_field_name

METADATA_FILTER_REGISTRY_VERSION: Final[str] = "1.0.0"
MAX_METADATA_FILTER_DEPTH: Final[int] = 16
MAX_METADATA_FILTER_NODES: Final[int] = 128


class MetadataFieldType(StrEnum):
    """Scalar types available in the executable metadata registry."""

    STRING = "string"
    INTEGER = "integer"


class MetadataCardinality(StrEnum):
    """Cardinality of one executable metadata field."""

    SCALAR = "scalar"


class MetadataFilterOperator(StrEnum):
    """Predicate operators that a metadata field may support."""

    EXACT = "exact"
    ONE_OF = "one_of"
    RANGE = "range"
    EXISTS = "exists"
    MISSING = "missing"


class MetadataMissingPolicy(StrEnum):
    """Whether a durable source normally requires a value to be present."""

    REQUIRED = "required"
    NULLABLE = "nullable"


class MetadataFilterErrorCategory(StrEnum):
    """Bounded reasons why a metadata filter cannot execute."""

    UNSUPPORTED_FIELD = "unsupported_field"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    SCALAR_TYPE_MISMATCH = "scalar_type_mismatch"
    DEPTH_EXCEEDED = "depth_exceeded"
    NODE_COUNT_EXCEEDED = "node_count_exceeded"
    REGISTRY_VERSION_MISMATCH = "registry_version_mismatch"
    INVALID_METADATA = "invalid_metadata"


class InvalidMetadataFilterError(InvalidRetrievalContractError):
    """A privacy-safe, bounded failure in executable metadata filtering."""

    def __init__(
        self,
        category: MetadataFilterErrorCategory,
        *,
        position: tuple[str | int, ...] = (),
    ) -> None:
        if not isinstance(category, MetadataFilterErrorCategory):
            raise TypeError("category must be a MetadataFilterErrorCategory.")
        if not isinstance(position, tuple):
            raise TypeError("position must be a tuple.")
        if any(type(item) not in {str, int} for item in position):
            raise TypeError("position must contain only strings and integers.")
        super().__init__(category.value)
        self.category = category
        self.position = position


@dataclass(frozen=True, slots=True)
class MetadataFieldDescriptor:
    """Describe one canonical, durably sourced metadata field."""

    name: str
    field_type: MetadataFieldType
    cardinality: MetadataCardinality
    supported_operators: frozenset[MetadataFilterOperator]
    durable_source: str
    missing_policy: MetadataMissingPolicy

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the immutable field descriptor."""
        validate_field_name(self.name)
        if not isinstance(self.field_type, MetadataFieldType):
            raise TypeError("field_type must be a MetadataFieldType.")
        if not isinstance(self.cardinality, MetadataCardinality):
            raise TypeError("cardinality must be a MetadataCardinality.")
        if self.cardinality is not MetadataCardinality.SCALAR:
            raise InvalidRetrievalContractError(
                "the executable metadata registry supports scalar fields only."
            )
        if not isinstance(self.supported_operators, frozenset):
            raise TypeError("supported_operators must be a frozenset.")
        if not self.supported_operators:
            raise InvalidRetrievalContractError(
                "supported_operators must not be empty."
            )
        if any(
            not isinstance(operator, MetadataFilterOperator)
            for operator in self.supported_operators
        ):
            raise TypeError(
                "supported_operators must contain MetadataFilterOperator values."
            )
        if not isinstance(self.durable_source, str) or not self.durable_source:
            raise InvalidRetrievalContractError("durable_source must be non-empty.")
        if "\x00" in self.durable_source:
            raise InvalidRetrievalContractError("durable_source must not contain NUL.")
        if not isinstance(self.missing_policy, MetadataMissingPolicy):
            raise TypeError("missing_policy must be a MetadataMissingPolicy.")


@dataclass(frozen=True, slots=True)
class MetadataFilterRegistry:
    """Hold one immutable, versioned executable metadata field registry."""

    registry_version: str
    fields: tuple[MetadataFieldDescriptor, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate registry identity, ordering, and field uniqueness."""
        if not isinstance(self.registry_version, str):
            raise TypeError("registry_version must be a string.")
        if not is_canonical_semantic_version(self.registry_version):
            raise ValueError("registry_version must be a canonical semantic version.")
        if not isinstance(self.fields, tuple):
            raise TypeError("fields must be a tuple.")
        if not self.fields:
            raise InvalidRetrievalContractError("fields must not be empty.")
        if any(not isinstance(field, MetadataFieldDescriptor) for field in self.fields):
            raise TypeError("fields must contain MetadataFieldDescriptor values.")
        names = tuple(field.name for field in self.fields)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise InvalidRetrievalContractError(
                "metadata fields must be unique and canonically ordered."
            )

    def descriptor(self, field: str) -> MetadataFieldDescriptor | None:
        """Return a descriptor, or ``None`` for a non-registry field."""
        if not isinstance(field, str):
            raise TypeError("field must be a string.")
        for descriptor in self.fields:
            if descriptor.name == field:
                return descriptor
        return None


@dataclass(frozen=True, slots=True)
class MetadataFilterValidation:
    """Validated expression metadata reused by the execution boundary."""

    expression: FilterExpression
    registry_version: str
    node_count: int
    maximum_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.expression, _FILTER_TYPES):
            raise TypeError("expression must be a filter expression.")
        if not isinstance(self.registry_version, str):
            raise TypeError("registry_version must be a string.")
        if not is_canonical_semantic_version(self.registry_version):
            raise ValueError("registry_version must be a canonical semantic version.")
        if type(self.node_count) is not int or self.node_count <= 0:
            raise ValueError("node_count must be a positive integer.")
        if type(self.maximum_depth) is not int or self.maximum_depth <= 0:
            raise ValueError("maximum_depth must be a positive integer.")


MetadataValue: TypeAlias = FilterScalar | None
MetadataValues: TypeAlias = Mapping[str, MetadataValue]


def _descriptor(
    name: str,
    field_type: MetadataFieldType,
    durable_source: str,
    missing_policy: MetadataMissingPolicy,
    *,
    supports_range: bool = False,
) -> MetadataFieldDescriptor:
    """Build one descriptor using the closed operator policy."""
    operators = {
        MetadataFilterOperator.EXACT,
        MetadataFilterOperator.ONE_OF,
        MetadataFilterOperator.EXISTS,
        MetadataFilterOperator.MISSING,
    }
    if supports_range:
        operators.add(MetadataFilterOperator.RANGE)
    return MetadataFieldDescriptor(
        name=name,
        field_type=field_type,
        cardinality=MetadataCardinality.SCALAR,
        supported_operators=frozenset(operators),
        durable_source=durable_source,
        missing_policy=missing_policy,
    )


CURRENT_METADATA_FILTER_REGISTRY: Final[MetadataFilterRegistry] = (
    MetadataFilterRegistry(
        registry_version=METADATA_FILTER_REGISTRY_VERSION,
        fields=(
            _descriptor(
                "character-count",
                MetadataFieldType.INTEGER,
                "ParsedRepresentation.character_count",
                MetadataMissingPolicy.REQUIRED,
                supports_range=True,
            ),
            _descriptor(
                "document-type",
                MetadataFieldType.STRING,
                "SourceArtefact.media_type",
                MetadataMissingPolicy.REQUIRED,
            ),
            _descriptor(
                "language",
                MetadataFieldType.STRING,
                "ParsedRepresentation.language",
                MetadataMissingPolicy.NULLABLE,
            ),
            _descriptor(
                "page-count",
                MetadataFieldType.INTEGER,
                "ParsedRepresentation.page_count",
                MetadataMissingPolicy.NULLABLE,
                supports_range=True,
            ),
            _descriptor(
                "source-byte-count",
                MetadataFieldType.INTEGER,
                "SourceArtefact.byte_count",
                MetadataMissingPolicy.REQUIRED,
                supports_range=True,
            ),
            _descriptor(
                "token-count",
                MetadataFieldType.INTEGER,
                "ParsedRepresentation.token_count",
                MetadataMissingPolicy.NULLABLE,
                supports_range=True,
            ),
        ),
    )
)


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


def _raise(
    category: MetadataFilterErrorCategory,
    position: tuple[str | int, ...],
) -> NoReturn:
    """Raise one bounded error without echoing private filter data."""
    raise InvalidMetadataFilterError(category, position=position)


def _matches_type(value: object, field_type: MetadataFieldType) -> bool:
    """Return whether a value has the field's exact scalar type."""
    if field_type is MetadataFieldType.STRING:
        return type(value) is str
    return type(value) is int


def _operator_for(expression: FilterExpression) -> MetadataFilterOperator:
    """Return the executable operator represented by one filter node."""
    if isinstance(expression, ExactFilter):
        return MetadataFilterOperator.EXACT
    if isinstance(expression, OneOfFilter):
        return MetadataFilterOperator.ONE_OF
    if isinstance(expression, RangeFilter):
        return MetadataFilterOperator.RANGE
    if isinstance(expression, ExistsFilter):
        return MetadataFilterOperator.EXISTS
    if isinstance(expression, MissingFilter):
        return MetadataFilterOperator.MISSING
    raise TypeError("expression must be a leaf filter.")


def _validate_leaf(
    expression: ExactFilter | OneOfFilter | RangeFilter | ExistsFilter | MissingFilter,
    *,
    registry: MetadataFilterRegistry,
    position: tuple[str | int, ...],
) -> None:
    """Validate one leaf against the closed registry."""
    descriptor = registry.descriptor(expression.field)
    if descriptor is None:
        _raise(MetadataFilterErrorCategory.UNSUPPORTED_FIELD, position + ("field",))
    operator = _operator_for(expression)
    if operator not in descriptor.supported_operators:
        _raise(
            MetadataFilterErrorCategory.UNSUPPORTED_OPERATOR,
            position + ("operator",),
        )
    if (
        operator is MetadataFilterOperator.RANGE
        and descriptor.field_type is not MetadataFieldType.INTEGER
    ):
        _raise(
            MetadataFilterErrorCategory.UNSUPPORTED_OPERATOR,
            position + ("operator",),
        )
    if isinstance(expression, ExactFilter):
        values = (expression.value,)
    elif isinstance(expression, OneOfFilter):
        values = expression.values
    elif isinstance(expression, RangeFilter):
        values = tuple(
            value for value in (expression.lower, expression.upper) if value is not None
        )
    else:
        values = ()
    if any(not _matches_type(value, descriptor.field_type) for value in values):
        _raise(
            MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH,
            position + ("value",),
        )


def _walk_expression(
    expression: FilterExpression,
    *,
    registry: MetadataFilterRegistry,
    position: tuple[str | int, ...],
    depth: int,
    state: list[int],
) -> int:
    """Validate one expression tree and return its maximum depth."""
    state[0] += 1
    if state[0] > MAX_METADATA_FILTER_NODES:
        _raise(MetadataFilterErrorCategory.NODE_COUNT_EXCEEDED, position)
    if depth > MAX_METADATA_FILTER_DEPTH:
        _raise(MetadataFilterErrorCategory.DEPTH_EXCEEDED, position)
    if isinstance(
        expression,
        (ExactFilter, OneOfFilter, RangeFilter, ExistsFilter, MissingFilter),
    ):
        _validate_leaf(expression, registry=registry, position=position)
        return depth
    if isinstance(expression, (AllFilter, AnyFilter)):
        return max(
            _walk_expression(
                child,
                registry=registry,
                position=position + ("children", index),
                depth=depth + 1,
                state=state,
            )
            for index, child in enumerate(expression.children)
        )
    if isinstance(expression, NotFilter):
        return _walk_expression(
            expression.child,
            registry=registry,
            position=position + ("child",),
            depth=depth + 1,
            state=state,
        )
    raise TypeError("expression must be a filter expression.")


def validate_metadata_filter(
    expression: FilterExpression,
    *,
    registry: MetadataFilterRegistry = CURRENT_METADATA_FILTER_REGISTRY,
) -> MetadataFilterValidation:
    """Validate one complete filter tree against an executable registry."""
    if not isinstance(expression, _FILTER_TYPES):
        raise TypeError("expression must be a filter expression.")
    if not isinstance(registry, MetadataFilterRegistry):
        raise TypeError("registry must be a MetadataFilterRegistry.")
    state = [0]
    maximum_depth = _walk_expression(
        expression,
        registry=registry,
        position=(),
        depth=1,
        state=state,
    )
    return MetadataFilterValidation(
        expression=expression,
        registry_version=registry.registry_version,
        node_count=state[0],
        maximum_depth=maximum_depth,
    )


def _validate_metadata_values(
    values: MetadataValues,
    *,
    registry: MetadataFilterRegistry,
) -> None:
    """Validate the private metadata view supplied by an execution adapter."""
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping.")
    for index, (field, value) in enumerate(values.items()):
        if not isinstance(field, str):
            raise TypeError("metadata field names must be strings.")
        descriptor = registry.descriptor(field)
        if descriptor is None:
            _raise(
                MetadataFilterErrorCategory.UNSUPPORTED_FIELD,
                ("metadata", index),
            )
        if value is not None and not _matches_type(value, descriptor.field_type):
            _raise(
                MetadataFilterErrorCategory.SCALAR_TYPE_MISMATCH,
                ("metadata", index),
            )


def _evaluate(
    expression: FilterExpression,
    values: MetadataValues,
) -> bool:
    """Evaluate a validated expression with explicit presence semantics."""
    if isinstance(expression, ExactFilter):
        value = values.get(expression.field)
        return value is not None and value == expression.value
    if isinstance(expression, OneOfFilter):
        value = values.get(expression.field)
        return value is not None and value in expression.values
    if isinstance(expression, RangeFilter):
        value = values.get(expression.field)
        if value is None:
            return False
        comparable_value = cast(int, value)
        if expression.lower is not None:
            lower = cast(int, expression.lower)
            lower_matches = (
                comparable_value >= lower
                if expression.lower_inclusive
                else comparable_value > lower
            )
            if not lower_matches:
                return False
        if expression.upper is not None:
            upper = cast(int, expression.upper)
            upper_matches = (
                comparable_value <= upper
                if expression.upper_inclusive
                else comparable_value < upper
            )
            if not upper_matches:
                return False
        return True
    if isinstance(expression, ExistsFilter):
        return values.get(expression.field) is not None
    if isinstance(expression, MissingFilter):
        return values.get(expression.field) is None
    if isinstance(expression, AllFilter):
        return all(_evaluate(child, values) for child in expression.children)
    if isinstance(expression, AnyFilter):
        return any(_evaluate(child, values) for child in expression.children)
    if isinstance(expression, NotFilter):
        return not _evaluate(expression.child, values)
    raise TypeError("expression must be a filter expression.")


def evaluate_metadata_filter(
    expression: FilterExpression | MetadataFilterValidation,
    values: MetadataValues,
    *,
    registry: MetadataFilterRegistry = CURRENT_METADATA_FILTER_REGISTRY,
) -> bool:
    """Evaluate one executable filter against one immutable metadata view."""
    if not isinstance(registry, MetadataFilterRegistry):
        raise TypeError("registry must be a MetadataFilterRegistry.")
    if isinstance(expression, MetadataFilterValidation):
        if expression.registry_version != registry.registry_version:
            _raise(
                MetadataFilterErrorCategory.REGISTRY_VERSION_MISMATCH,
                (),
            )
        checked = validate_metadata_filter(expression.expression, registry=registry)
        if (
            checked.node_count != expression.node_count
            or checked.maximum_depth != expression.maximum_depth
        ):
            _raise(MetadataFilterErrorCategory.INVALID_METADATA, ())
        validation = checked
    else:
        validation = validate_metadata_filter(expression, registry=registry)
    _validate_metadata_values(values, registry=registry)
    return _evaluate(validation.expression, values)
