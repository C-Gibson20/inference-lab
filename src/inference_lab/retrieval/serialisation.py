"""Strict canonical JSON serialisation for retrieval contract values."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import date
from typing import cast

from inference_lab.json_types import JsonObject, JsonValue
from inference_lab.provenance import (
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.errors import (
    InvalidRetrievalContractError,
    InvalidRetrievalSerialisationError,
)
from inference_lab.retrieval.evidence import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceSource,
    RetrievalCandidate,
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
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.values import RetrievalChannel, RetrievalLimit
from inference_lab.retrieval.versioning import (
    CURRENT_RETRIEVAL_SCHEMA_VERSION,
    validate_retrieval_schema_version,
)


def _canonical_json_bytes(payload: JsonObject) -> bytes:
    """Encode one canonical newline-terminated JSON object."""
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


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
    raise TypeError("expression must be a FilterExpression.")


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
    return _canonical_json_bytes(_query_payload(query))


def _source_payload(source: EvidenceSource) -> JsonObject:
    """Return canonical JSON-compatible data for one evidence source."""
    return {
        "document_id": source.document_id.as_str(),
        "document_version_id": source.document_version_id.as_str(),
        "source_span_id": source.source_span_id.as_str(),
    }


def _candidate_payload(candidate: RetrievalCandidate) -> JsonObject:
    """Return canonical JSON-compatible data for one candidate."""
    return {
        "record_id": candidate.record_id.as_str(),
        "kind": candidate.kind.value,
        "content": candidate.content,
        "sources": [_source_payload(source) for source in candidate.sources],
    }


def _score_payload(score: ChannelScore) -> JsonObject:
    """Return canonical JSON-compatible data for one channel contribution."""
    return {
        "channel": score.channel.value,
        "rank": score.rank,
        "raw_score": score.raw_score,
        "score_label": score.score_label,
    }


def _hit_payload(hit: EvidenceHit) -> JsonObject:
    """Return canonical JSON-compatible data for one ranked hit."""
    return {
        "candidate": _candidate_payload(hit.candidate),
        "contributions": [_score_payload(score) for score in hit.contributions],
        "rank": hit.rank,
    }


def _failure_payload(failure: RetrievalFailure) -> JsonObject:
    """Return canonical JSON-compatible data for one typed failure."""
    return {
        "category": failure.category.value,
        "channel": failure.channel.value if failure.channel is not None else None,
        "retryable": failure.retryable,
    }


def _result_payload(result: EvidenceResult) -> JsonObject:
    """Return canonical JSON-compatible data for one evidence result."""
    if not isinstance(result, EvidenceResult):
        raise TypeError("result must be an EvidenceResult.")
    return {
        "retrieval_schema_version": CURRENT_RETRIEVAL_SCHEMA_VERSION,
        "corpus_version": result.corpus_version.as_str(),
        "requested_limit": result.requested_limit.value,
        "hits": [_hit_payload(hit) for hit in result.hits],
        "status": result.status.value,
        "failures": [_failure_payload(failure) for failure in result.failures],
    }


def serialise_evidence_result(result: EvidenceResult) -> bytes:
    """Serialise one evidence result as canonical UTF-8 JSON."""
    return _canonical_json_bytes(_result_payload(result))


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Build one JSON object while rejecting duplicate keys."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise InvalidRetrievalSerialisationError(
                "retrieval JSON must not contain duplicate object keys."
            )
        result[key] = value
    return result


def _reject_constant(_: str) -> JsonValue:
    """Reject non-finite JSON constants."""
    raise InvalidRetrievalSerialisationError(
        "retrieval JSON must not contain non-finite values."
    )


def _load_json(data: bytes) -> JsonObject:
    """Decode strict UTF-8 JSON without exposing rejected payload content."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InvalidRetrievalSerialisationError(
            "retrieval data must be valid UTF-8 JSON."
        ) from None
    if not isinstance(payload, dict):
        raise InvalidRetrievalSerialisationError(
            "retrieval JSON payload must be an object."
        )
    return cast(JsonObject, payload)


def _object(value: JsonValue, *, name: str, keys: frozenset[str]) -> JsonObject:
    """Require exactly the fields belonging to one decoded object."""
    if not isinstance(value, dict):
        raise InvalidRetrievalSerialisationError(f"{name} must be an object.")
    if frozenset(value) != keys:
        raise InvalidRetrievalSerialisationError(
            f"{name} must contain exactly the canonical fields."
        )
    return cast(JsonObject, value)


def _array(value: JsonValue, *, name: str) -> list[JsonValue]:
    """Require one JSON array."""
    if not isinstance(value, list):
        raise InvalidRetrievalSerialisationError(f"{name} must be an array.")
    return value


def _string(value: JsonValue, *, name: str) -> str:
    """Require one JSON string."""
    if not isinstance(value, str):
        raise InvalidRetrievalSerialisationError(f"{name} must be a string.")
    return value


def _boolean(value: JsonValue, *, name: str) -> bool:
    """Require one JSON boolean."""
    if not isinstance(value, bool):
        raise InvalidRetrievalSerialisationError(f"{name} must be a boolean.")
    return value


def _integer(value: JsonValue, *, name: str) -> int:
    """Require one strict JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRetrievalSerialisationError(f"{name} must be an integer.")
    return value


def _enum(value: JsonValue, *, name: str, enum_type: type[object]) -> object:
    """Parse one closed string enum without leaking rejected values."""
    text = _string(value, name=name)
    try:
        return enum_type(text)  # type: ignore[call-arg]
    except ValueError:
        raise InvalidRetrievalSerialisationError(f"{name} is not supported.") from None


def _identifier(
    value: JsonValue,
    *,
    name: str,
    constructor: Callable[[str], object],
) -> object:
    """Parse one canonical provenance identifier."""
    try:
        return constructor(_string(value, name=name))
    except (InvalidRetrievalSerialisationError, ValueError, TypeError):
        raise InvalidRetrievalSerialisationError(
            f"{name} must be a canonical provenance identifier."
        ) from None


def _schema(payload: JsonObject) -> None:
    """Validate the schema field before interpreting nested data."""
    if "retrieval_schema_version" not in payload:
        raise InvalidRetrievalSerialisationError(
            "retrieval payload requires retrieval_schema_version."
        )
    try:
        validate_retrieval_schema_version(payload["retrieval_schema_version"])
    except TypeError as error:
        raise InvalidRetrievalSerialisationError(str(error)) from None


def _scalar(value: JsonValue) -> FilterScalar:
    """Decode one strictly tagged filter scalar."""
    payload = _object(value, name="scalar", keys=frozenset({"type", "value"}))
    kind = _string(payload["type"], name="scalar.type")
    scalar = payload["value"]
    if kind == "string":
        return _string(scalar, name="scalar.value")
    if kind == "integer":
        return _integer(scalar, name="scalar.value")
    if kind == "float":
        if isinstance(scalar, bool) or not isinstance(scalar, float):
            raise InvalidRetrievalSerialisationError("scalar.value must be a float.")
        if not math.isfinite(scalar):
            raise InvalidRetrievalSerialisationError("scalar.value must be finite.")
        return scalar
    if kind == "boolean":
        return _boolean(scalar, name="scalar.value")
    if kind == "date":
        text = _string(scalar, name="scalar.value")
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
    kind = _string(payload["kind"], name="filter.kind")
    # The kind is used to select the exact object schema before parsing fields.
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
    payload = _object(value, name="filter", keys=keys)
    field = payload.get("field")
    if kind == "exact":
        return ExactFilter(
            _string(field, name="filter.field"), _scalar(payload["value"])
        )
    if kind == "one_of":
        return OneOfFilter(
            _string(field, name="filter.field"),
            tuple(
                _scalar(item)
                for item in _array(payload["values"], name="filter.values")
            ),
        )
    if kind == "range":
        lower = None if payload["lower"] is None else _scalar(payload["lower"])
        upper = None if payload["upper"] is None else _scalar(payload["upper"])
        return RangeFilter(
            _string(field, name="filter.field"),
            lower,
            upper,
            _boolean(payload["lower_inclusive"], name="filter.lower_inclusive"),
            _boolean(payload["upper_inclusive"], name="filter.upper_inclusive"),
        )
    if kind == "exists":
        return ExistsFilter(_string(field, name="filter.field"))
    if kind == "missing":
        return MissingFilter(_string(field, name="filter.field"))
    if kind in {"all", "any"}:
        children = tuple(
            _filter(item)
            for item in _array(payload["children"], name="filter.children")
        )
        return AllFilter(children) if kind == "all" else AnyFilter(children)
    return NotFilter(_filter(payload["child"]))


def deserialise_query(data: bytes) -> RetrievalQuery:
    """Deserialise one strict canonical query payload."""
    payload = _load_json(data)
    _schema(payload)
    payload = _object(
        payload,
        name="query",
        keys=frozenset(
            {"retrieval_schema_version", "text", "corpus_version", "limit", "filter"}
        ),
    )
    try:
        return RetrievalQuery(
            _string(payload["text"], name="query.text"),
            cast(
                CorpusVersionId,
                _identifier(
                    payload["corpus_version"],
                    name="query.corpus_version",
                    constructor=CorpusVersionId.from_str,
                ),
            ),
            RetrievalLimit(_integer(payload["limit"], name="query.limit")),
            None if payload["filter"] is None else _filter(payload["filter"]),
        )
    except InvalidRetrievalSerialisationError:
        raise
    except (InvalidRetrievalContractError, TypeError, ValueError):
        raise InvalidRetrievalSerialisationError(
            "query violates its retrieval contract."
        ) from None


def _source(value: JsonValue) -> EvidenceSource:
    """Decode one strict evidence source."""
    payload = _object(
        value,
        name="source",
        keys=frozenset({"document_id", "document_version_id", "source_span_id"}),
    )
    return EvidenceSource(
        cast(
            DocumentId,
            _identifier(
                payload["document_id"],
                name="source.document_id",
                constructor=DocumentId.from_str,
            ),
        ),
        cast(
            DocumentVersionId,
            _identifier(
                payload["document_version_id"],
                name="source.document_version_id",
                constructor=DocumentVersionId.from_str,
            ),
        ),
        cast(
            SourceSpanId,
            _identifier(
                payload["source_span_id"],
                name="source.source_span_id",
                constructor=SourceSpanId.from_str,
            ),
        ),
    )


def _candidate(value: JsonValue) -> RetrievalCandidate:
    """Decode one strict candidate."""
    payload = _object(
        value,
        name="candidate",
        keys=frozenset({"record_id", "kind", "content", "sources"}),
    )
    return RetrievalCandidate(
        cast(
            RecordId,
            _identifier(
                payload["record_id"],
                name="candidate.record_id",
                constructor=RecordId.from_str,
            ),
        ),
        cast(
            CandidateKind,
            _enum(payload["kind"], name="candidate.kind", enum_type=CandidateKind),
        ),
        _string(payload["content"], name="candidate.content"),
        tuple(
            _source(item)
            for item in _array(payload["sources"], name="candidate.sources")
        ),
    )


def _score(value: JsonValue) -> ChannelScore:
    """Decode one strict channel contribution."""
    payload = _object(
        value,
        name="score",
        keys=frozenset({"channel", "rank", "raw_score", "score_label"}),
    )
    raw_score = payload["raw_score"]
    if raw_score is not None and (
        isinstance(raw_score, bool) or type(raw_score) not in (int, float)
    ):
        raise InvalidRetrievalSerialisationError(
            "score.raw_score must be numeric or null."
        )
    if isinstance(raw_score, float) and not math.isfinite(raw_score):
        raise InvalidRetrievalSerialisationError("score.raw_score must be finite.")
    score_label = payload["score_label"]
    if score_label is not None and not isinstance(score_label, str):
        raise InvalidRetrievalSerialisationError(
            "score.score_label must be a string or null."
        )
    return ChannelScore(
        cast(
            RetrievalChannel,
            _enum(payload["channel"], name="score.channel", enum_type=RetrievalChannel),
        ),
        _integer(payload["rank"], name="score.rank"),
        cast(int | float | None, raw_score),
        score_label,
    )


def _hit(value: JsonValue) -> EvidenceHit:
    """Decode one strict ranked hit."""
    payload = _object(
        value, name="hit", keys=frozenset({"candidate", "contributions", "rank"})
    )
    return EvidenceHit(
        _candidate(payload["candidate"]),
        tuple(
            _score(item)
            for item in _array(payload["contributions"], name="hit.contributions")
        ),
        _integer(payload["rank"], name="hit.rank"),
    )


def _failure(value: JsonValue) -> RetrievalFailure:
    """Decode one strict typed failure."""
    payload = _object(
        value, name="failure", keys=frozenset({"category", "channel", "retryable"})
    )
    channel = payload["channel"]
    return RetrievalFailure(
        cast(
            FailureCategory,
            _enum(
                payload["category"], name="failure.category", enum_type=FailureCategory
            ),
        ),
        _boolean(payload["retryable"], name="failure.retryable"),
        None
        if channel is None
        else cast(
            RetrievalChannel,
            _enum(channel, name="failure.channel", enum_type=RetrievalChannel),
        ),
    )


def deserialise_evidence_result(data: bytes) -> EvidenceResult:
    """Deserialise one strict canonical evidence result payload."""
    payload = _load_json(data)
    _schema(payload)
    payload = _object(
        payload,
        name="result",
        keys=frozenset(
            {
                "retrieval_schema_version",
                "corpus_version",
                "requested_limit",
                "hits",
                "status",
                "failures",
            }
        ),
    )
    try:
        return EvidenceResult(
            cast(
                CorpusVersionId,
                _identifier(
                    payload["corpus_version"],
                    name="result.corpus_version",
                    constructor=CorpusVersionId.from_str,
                ),
            ),
            RetrievalLimit(
                _integer(payload["requested_limit"], name="result.requested_limit")
            ),
            tuple(_hit(item) for item in _array(payload["hits"], name="result.hits")),
            cast(
                ResultStatus,
                _enum(payload["status"], name="result.status", enum_type=ResultStatus),
            ),
            tuple(
                _failure(item)
                for item in _array(payload["failures"], name="result.failures")
            ),
        )
    except InvalidRetrievalSerialisationError:
        raise
    except (InvalidRetrievalContractError, TypeError, ValueError):
        raise InvalidRetrievalSerialisationError(
            "result violates its retrieval contract."
        ) from None
