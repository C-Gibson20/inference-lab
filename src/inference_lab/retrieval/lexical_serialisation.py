"""Strict canonical serialisation for lexical generation manifests."""

from typing import cast

from inference_lab._json import JsonValue, canonical_json_bytes, load_json_object
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval.lexical import (
    _CONFIGURATION_FIELDS,
    LEXICAL_MANIFEST_SCHEMA_VERSION,
    InvalidLexicalSerialisationError,
    LexicalIndexConfiguration,
    LexicalIndexManifest,
)


def _require_object(value: object, field_name: str) -> dict[str, JsonValue]:
    """Require a JSON object without retaining rejected payload values."""
    if not isinstance(value, dict):
        raise InvalidLexicalSerialisationError(f"{field_name} must be an object.")
    return cast(dict[str, JsonValue], value)


def _require_fields(
    payload: dict[str, JsonValue],
    expected: frozenset[str],
    field_name: str,
) -> None:
    """Require an exact set of canonical object fields."""
    if frozenset(payload) != expected:
        raise InvalidLexicalSerialisationError(
            f"{field_name} must contain exactly the canonical fields."
        )


def _require_string(payload: dict[str, JsonValue], field_name: str) -> str:
    """Read one required JSON string."""
    value = payload[field_name]
    if not isinstance(value, str):
        raise InvalidLexicalSerialisationError(f"{field_name} must be a string.")
    return value


def _require_integer(payload: dict[str, JsonValue], field_name: str) -> int:
    """Read one required JSON integer, excluding booleans."""
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidLexicalSerialisationError(f"{field_name} must be an integer.")
    return value


def _configuration_payload(
    value: JsonValue,
) -> LexicalIndexConfiguration:
    """Decode one complete lexical configuration."""
    payload = _require_object(value, "configuration")
    _require_fields(payload, _CONFIGURATION_FIELDS, "configuration")
    weights = payload["column_weights"]
    if not isinstance(weights, list):
        raise InvalidLexicalSerialisationError(
            "configuration.column_weights must be an array."
        )
    if any(isinstance(item, bool) or not isinstance(item, float) for item in weights):
        raise InvalidLexicalSerialisationError(
            "configuration.column_weights must contain floats."
        )
    try:
        return LexicalIndexConfiguration(
            engine=_require_string(payload, "engine"),
            engine_revision=_require_string(payload, "engine_revision"),
            index_schema_version=_require_string(payload, "index_schema_version"),
            fts_tokenizer=_require_string(payload, "fts_tokenizer"),
            unicode_normalisation=_require_string(payload, "unicode_normalisation"),
            term_case_policy=_require_string(payload, "term_case_policy"),
            identifier_grammar_version=_require_string(
                payload, "identifier_grammar_version"
            ),
            identifier_case_policy=_require_string(payload, "identifier_case_policy"),
            query_escaping_version=_require_string(payload, "query_escaping_version"),
            bm25_k1=payload["bm25_k1"],  # type: ignore[arg-type]
            bm25_b=payload["bm25_b"],  # type: ignore[arg-type]
            column_weights=tuple(weights),  # type: ignore[arg-type]
            score_label=_require_string(payload, "score_label"),
            identifier_score_label=_require_string(payload, "identifier_score_label"),
            tie_break_policy=_require_string(payload, "tie_break_policy"),
        )
    except (TypeError, ValueError):
        raise InvalidLexicalSerialisationError(
            "configuration violates its lexical contract."
        ) from None


def serialise_lexical_manifest(manifest: LexicalIndexManifest) -> bytes:
    """Serialise one lexical manifest as canonical UTF-8 JSON."""
    if not isinstance(manifest, LexicalIndexManifest):
        raise TypeError("manifest must be a LexicalIndexManifest.")
    return canonical_json_bytes(manifest.as_payload())


def deserialise_lexical_manifest(data: bytes) -> LexicalIndexManifest:
    """Deserialise one strict lexical manifest without exposing its payload."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        payload = load_json_object(data)
        expected = frozenset(
            {
                "component_kind",
                "configuration",
                "configuration_digest",
                "contract_version",
                "corpus_id",
                "corpus_version",
                "database_checksum",
                "format_version",
                "generation_id",
                "identifier_count",
                "implementation",
                "member_count",
                "schema_version",
                "source_manifest_digest",
                "chunk_count",
            }
        )
        _require_fields(payload, expected, "manifest")
        schema_version = _require_string(payload, "schema_version")
        if schema_version != LEXICAL_MANIFEST_SCHEMA_VERSION:
            raise InvalidLexicalSerialisationError(
                "unsupported lexical manifest schema version."
            )
        configuration = _configuration_payload(payload["configuration"])
        return LexicalIndexManifest(
            corpus_id=CorpusId.from_str(_require_string(payload, "corpus_id")),
            corpus_version=CorpusVersionId.from_str(
                _require_string(payload, "corpus_version")
            ),
            generation_id=_require_string(payload, "generation_id"),
            configuration=configuration,
            configuration_digest=_require_string(payload, "configuration_digest"),
            source_manifest_digest=_require_string(payload, "source_manifest_digest"),
            database_checksum=_require_string(payload, "database_checksum"),
            member_count=_require_integer(payload, "member_count"),
            chunk_count=_require_integer(payload, "chunk_count"),
            identifier_count=_require_integer(payload, "identifier_count"),
            schema_version=schema_version,
            component_kind=_require_string(payload, "component_kind"),
            implementation=_require_string(payload, "implementation"),
            format_version=_require_string(payload, "format_version"),
            contract_version=_require_string(payload, "contract_version"),
        )
    except InvalidLexicalSerialisationError:
        raise
    except (TypeError, ValueError):
        raise InvalidLexicalSerialisationError(
            "lexical manifest violates its serialisation contract."
        ) from None
