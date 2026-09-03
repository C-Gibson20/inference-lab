"""Canonical serialisation for retrieval query contracts."""

import math
from datetime import date
from typing import cast

from inference_lab.json_types import JsonObject, JsonValue
from inference_lab.provenance import CorpusVersionId
from inference_lab.retrieval._serialisation_json import (
    canonical_json_bytes,
    load_json,
    parse_identifier,
    require_array,
    require_boolean,
    require_integer,
    require_object,
    require_string,
    validate_schema,
)
from inference_lab.retrieval.errors import (
    InvalidRetrievalContractError,
    InvalidRetrievalSerialisationError,
)
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
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.values import RetrievalLimit
from inference_lab.retrieval.versioning import CURRENT_RETRIEVAL_SCHEMA_VERSION


def _scalar_payload(value: object) -> JsonObject:
    """Tag one filter scalar so JSON cannot erase its type."""
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": value}
    if type(value) is float:
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if type(value) is date:
        return {"type": "date", "value": value.isoformat()}
    raise TypeError("filter scalar has an unsupported type.")


def _filter_payload(expression: FilterExpression) -> JsonObject:
    """Return canonical JSON-compatible data for one filter expression."""
    if isinstance(expression, ExactFilter):
        return {
            "kind": "exact",
            "field": expression.field,
            "value": _scalar_payload(expression.value),
        }
    if isinstance(expression, OneOfFilter):
        return {
            "kind": "one_of",
            "field": expression.field,
            "values": [_scalar_payload(value) for value in expression.values],
        }
    if isinstance(expression, RangeFilter):
        return {
            "kind": "range",
            "field": expression.field,
            "lower": (
                _scalar_payload(expression.lower)
                if expression.lower is not None
                else None
            ),
            "upper": (
                _scalar_payload(expression.upper)
                if expression.upper is not None
                else None
            ),
            "lower_inclusive": expression.lower_inclusive,
            "upper_inclusive": expression.upper_inclusive,
        }
    if isinstance(expression, ExistsFilter):
        return {"kind": "exists", "field": expression.field}
    if isinstance(expression, MissingFilter):
        return {"kind": "missing", "field": expression.field}
    if isinstance(expression, AllFilter):
        return {
            "kind": "all",
            "children": [_filter_payload(child) for child in expression.children],
        }
    if isinstance(expression, AnyFilter):
        return {
            "kind": "any",
            "children": [_filter_payload(child) for child in expression.children],
        }
    if isinstance(expression, NotFilter):
        return {"kind": "not", "child": _filter_payload(expression.child)}
    raise TypeError("expression must be a retrieval filter.")


def _query_payload(query: RetrievalQuery) -> JsonObject:
    """Return canonical JSON-compatible data for one query."""
    if not isinstance(query, RetrievalQuery):
        raise TypeError("query must be a RetrievalQuery.")
    return {
        "retrieval_schema_version": CURRENT_RETRIEVAL_SCHEMA_VERSION,
        "text": query.text,
        "corpus_version": query.corpus_version.as_str(),
        "limit": query.limit.value,
        "filter": _filter_payload(query.filter) if query.filter is not None else None,
    }


def serialise_query(query: RetrievalQuery) -> bytes:
    """Serialise one query as canonical UTF-8 JSON."""
    return canonical_json_bytes(_query_payload(query))


def _scalar(value: JsonValue) -> FilterScalar:
    """Decode one strictly tagged filter scalar."""
    payload = require_object(
        value,
        name="scalar",
        keys=frozenset({"type", "value"}),
    )
    kind = require_string(payload["type"], name="scalar.type")
    scalar = payload["value"]
    if kind == "string":
        return require_string(scalar, name="scalar.value")
    if kind == "integer":
        return require_integer(scalar, name="scalar.value")
    if kind == "float":
        if isinstance(scalar, bool) or not isinstance(scalar, float):
            raise InvalidRetrievalSerialisationError("scalar.value must be a float.")
        if not math.isfinite(scalar):
            raise InvalidRetrievalSerialisationError("scalar.value must be finite.")
        return scalar
    if kind == "boolean":
        return require_boolean(scalar, name="scalar.value")
    if kind == "date":
        text = require_string(scalar, name="scalar.value")
        try:
            parsed = date.fromisoformat(text)
        except ValueError:
            raise InvalidRetrievalSerialisationError(
                "scalar.value must be a date."
            ) from None
        if parsed.isoformat() != text:
            raise InvalidRetrievalSerialisationError("scalar.value must be canonical.")
        return parsed
    raise InvalidRetrievalSerialisationError("scalar.type is not supported.")


def _filter(value: JsonValue) -> FilterExpression:
    """Decode one tagged filter expression and reuse value validation."""
    if not isinstance(value, dict):
        raise InvalidRetrievalSerialisationError("filter must be an object.")
    payload = cast(JsonObject, value)
    if "kind" not in payload:
        raise InvalidRetrievalSerialisationError("filter requires a kind.")
    kind = require_string(payload["kind"], name="filter.kind")
    schemas: dict[str, frozenset[str]] = {
        "exact": frozenset({"kind", "field", "value"}),
        "one_of": frozenset({"kind", "field", "values"}),
        "range": frozenset(
            {"kind", "field", "lower", "upper", "lower_inclusive", "upper_inclusive"}
        ),
        "exists": frozenset({"kind", "field"}),
        "missing": frozenset({"kind", "field"}),
        "all": frozenset({"kind", "children"}),
        "any": frozenset({"kind", "children"}),
        "not": frozenset({"kind", "child"}),
    }
    keys = schemas.get(kind)
    if keys is None:
        raise InvalidRetrievalSerialisationError("filter.kind is not supported.")
    payload = require_object(value, name="filter", keys=keys)
    field = payload.get("field")
    if kind == "exact":
        return ExactFilter(
            require_string(field, name="filter.field"),
            _scalar(payload["value"]),
        )
    if kind == "one_of":
        return OneOfFilter(
            require_string(field, name="filter.field"),
            tuple(
                _scalar(item)
                for item in require_array(payload["values"], name="filter.values")
            ),
        )
    if kind == "range":
        lower = None if payload["lower"] is None else _scalar(payload["lower"])
        upper = None if payload["upper"] is None else _scalar(payload["upper"])
        return RangeFilter(
            require_string(field, name="filter.field"),
            lower,
            upper,
            require_boolean(
                payload["lower_inclusive"],
                name="filter.lower_inclusive",
            ),
            require_boolean(
                payload["upper_inclusive"],
                name="filter.upper_inclusive",
            ),
        )
    if kind == "exists":
        return ExistsFilter(require_string(field, name="filter.field"))
    if kind == "missing":
        return MissingFilter(require_string(field, name="filter.field"))
    if kind in {"all", "any"}:
        children = tuple(
            _filter(item)
            for item in require_array(payload["children"], name="filter.children")
        )
        return AllFilter(children) if kind == "all" else AnyFilter(children)
    return NotFilter(_filter(payload["child"]))


def deserialise_query(data: bytes) -> RetrievalQuery:
    """Deserialise one strict canonical query payload."""
    payload = load_json(data)
    validate_schema(payload)
    payload = require_object(
        payload,
        name="query",
        keys=frozenset(
            {"retrieval_schema_version", "text", "corpus_version", "limit", "filter"}
        ),
    )
    try:
        return RetrievalQuery(
            require_string(payload["text"], name="query.text"),
            cast(
                CorpusVersionId,
                parse_identifier(
                    payload["corpus_version"],
                    name="query.corpus_version",
                    constructor=CorpusVersionId.from_str,
                ),
            ),
            RetrievalLimit(require_integer(payload["limit"], name="query.limit")),
            None if payload["filter"] is None else _filter(payload["filter"]),
        )
    except InvalidRetrievalSerialisationError:
        raise
    except (InvalidRetrievalContractError, TypeError, ValueError):
        raise InvalidRetrievalSerialisationError(
            "query violates its retrieval contract."
        ) from None
