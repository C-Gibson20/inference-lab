"""Canonical serialisation and strict reconstruction for chunk records."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from inference_lab._json import (
    JsonDecodingError,
    JsonObject,
    JsonStructureError,
    JsonValue,
    canonical_json_bytes,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_integer,
    require_json_object,
    require_json_string,
)
from inference_lab.ingest._chunk_records import (
    CHUNK_PAYLOAD_SCHEMA_VERSION,
    Chunk,
    ChunkMinimumException,
    ChunkSet,
    ChunkSourceReference,
    ChunkSplitReason,
)
from inference_lab.ingest.chunking import (
    ChunkerVersion,
    ChunkingPolicy,
    OversizeUnitPolicy,
    StructuralBoundaryPolicy,
    TokeniserIdentity,
)
from inference_lab.ingest.sources import StructuralBlockKind
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    InvalidLineageError,
    SourceSpanId,
    deserialise_derived_record,
    deserialise_lineage_bundle,
)
from inference_lab.provenance.serialisation import (
    derived_record_payload,
    lineage_bundle_payload,
)


class InvalidChunkSerialisationError(ValueError):
    """Report strict chunk wire-format failures without retaining payloads."""


_CHUNK_KEYS = frozenset(
    {
        "payload_schema_version",
        "document_id",
        "document_version_id",
        "ordinal",
        "text",
        "token_count",
        "source_range",
        "overlap",
        "source_references",
        "structural_kinds",
        "structural_block_ordinals",
        "minimum_exception",
        "split_reason",
        "chunker_version",
        "tokeniser_identity",
        "configuration_identity",
        "metadata",
    }
)
_CHUNK_SET_KEYS = frozenset(
    {
        "payload_schema_version",
        "corpus_id",
        "document_id",
        "document_version_id",
        "parser_version",
        "chunker_version",
        "tokeniser_identity",
        "policy",
        "configuration_identity",
        "chunks",
        "lineage",
    }
)
_IdentifierT = TypeVar("_IdentifierT")


def chunk_payload(chunk: Chunk) -> JsonObject:
    """Return the canonical JSON-compatible payload for one chunk."""
    if not isinstance(chunk, Chunk):
        raise TypeError("chunk must be a Chunk.")
    return {
        "payload_schema_version": chunk.payload_schema_version,
        "document_id": chunk.document_id.as_str(),
        "document_version_id": chunk.document_version_id.as_str(),
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "source_range": {
            "start": chunk.source_start,
            "end": chunk.source_end,
        },
        "overlap": {
            "target_tokens": chunk.target_overlap_tokens,
            "source_start": chunk.overlap_source_start,
            "source_end": chunk.overlap_source_end,
            "token_count": chunk.overlap_token_count,
        },
        "source_references": [
            {
                "source_span_id": reference.source_span_id.as_str(),
                "document_version_id": reference.document_version_id.as_str(),
                "block_ordinal": reference.block_ordinal,
                "block_kind": reference.block_kind,
                "source_start": reference.source_start,
                "source_end": reference.source_end,
            }
            for reference in chunk.source_references
        ],
        "structural_kinds": list(chunk.structural_kinds),
        "structural_block_ordinals": list(chunk.structural_block_ordinals),
        "minimum_exception": (
            chunk.minimum_exception.value
            if chunk.minimum_exception is not None
            else None
        ),
        "split_reason": (
            chunk.split_reason.value if chunk.split_reason is not None else None
        ),
        "chunker_version": {
            "implementation": chunk.chunker_version.implementation,
            "contract_version": chunk.chunker_version.contract_version,
        },
        "tokeniser_identity": _tokeniser_payload(chunk.tokeniser_identity),
        "configuration_identity": _configuration_payload(chunk.configuration_identity),
        "metadata": derived_record_payload(chunk.metadata),
    }


def chunk_set_payload(chunk_set: ChunkSet) -> JsonObject:
    """Return the canonical JSON-compatible payload for a complete chunk set."""
    if not isinstance(chunk_set, ChunkSet):
        raise TypeError("chunk_set must be a ChunkSet.")
    return {
        "payload_schema_version": chunk_set.payload_schema_version,
        "corpus_id": chunk_set.corpus_id.as_str(),
        "document_id": chunk_set.document_id.as_str(),
        "document_version_id": chunk_set.document_version_id.as_str(),
        "parser_version": _parser_version_payload(chunk_set.parser_version),
        "chunker_version": _chunker_payload(chunk_set.chunker_version),
        "tokeniser_identity": _tokeniser_payload(chunk_set.tokeniser_identity),
        "policy": _policy_payload(chunk_set.policy),
        "configuration_identity": _configuration_payload(
            chunk_set.configuration_identity
        ),
        "chunks": [chunk_payload(chunk) for chunk in chunk_set.chunks],
        "lineage": lineage_bundle_payload(chunk_set.lineage),
    }


def serialise_chunk(chunk: Chunk) -> bytes:
    """Serialise one chunk as canonical UTF-8 JSON."""
    return canonical_json_bytes(chunk_payload(chunk))


def serialise_chunk_set(chunk_set: ChunkSet) -> bytes:
    """Serialise one complete chunk set as canonical UTF-8 JSON."""
    return canonical_json_bytes(chunk_set_payload(chunk_set))


def deserialise_chunk(data: bytes) -> Chunk:
    """Strictly reconstruct one immutable chunk from canonical JSON data."""
    payload = _load_payload(data)
    try:
        return _parse_chunk(payload)
    except InvalidChunkSerialisationError:
        raise
    except (TypeError, ValueError, InvalidLineageError):
        raise InvalidChunkSerialisationError(
            "chunk payload violates its immutable contract."
        ) from None


def deserialise_chunk_set(data: bytes) -> ChunkSet:
    """Strictly reconstruct one complete chunk set from canonical JSON data."""
    payload = _load_payload(data)
    try:
        parsed = _require_object(payload, field_name="chunk_set", keys=_CHUNK_SET_KEYS)
        _require_schema_version(parsed)
        chunks = tuple(
            _parse_chunk(value)
            for value in _require_array(parsed["chunks"], field_name="chunks")
        )
        lineage_payload = _require_object(parsed["lineage"], field_name="lineage")
        lineage = deserialise_lineage_bundle(canonical_json_bytes(lineage_payload))
        return ChunkSet(
            corpus_id=_parse_identifier(
                parsed["corpus_id"],
                field_name="corpus_id",
                constructor=CorpusId.from_str,
            ),
            document_id=_parse_identifier(
                parsed["document_id"],
                field_name="document_id",
                constructor=DocumentId.from_str,
            ),
            document_version_id=_parse_identifier(
                parsed["document_version_id"],
                field_name="document_version_id",
                constructor=DocumentVersionId.from_str,
            ),
            parser_version=_parse_parser_version(parsed["parser_version"]),
            chunker_version=_parse_chunker_version(parsed["chunker_version"]),
            tokeniser_identity=_parse_tokeniser_identity(parsed["tokeniser_identity"]),
            policy=_parse_policy(parsed["policy"]),
            configuration_identity=_parse_configuration_identity(
                parsed["configuration_identity"]
            ),
            chunks=chunks,
            lineage=lineage,
        )
    except InvalidChunkSerialisationError:
        raise
    except (TypeError, ValueError, InvalidLineageError):
        raise InvalidChunkSerialisationError(
            "chunk-set payload violates its immutable contract."
        ) from None


def deserialise_chunk_set_selection(
    data: bytes,
    *,
    corpus_id: CorpusId,
) -> ChunkSet:
    """Reconstruct a corpus-scoped chunk set from its detached selection payload."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    payload = _load_payload(data)
    if "corpus_id" in payload:
        raise InvalidChunkSerialisationError(
            "selection payload must not contain corpus identity."
        )
    payload_with_corpus = dict(payload)
    payload_with_corpus["corpus_id"] = corpus_id.as_str()
    return deserialise_chunk_set(canonical_json_bytes(payload_with_corpus))


def _parse_chunk(value: JsonValue) -> Chunk:
    """Parse one complete chunk object with exact field validation."""
    payload = _require_object(value, field_name="chunk", keys=_CHUNK_KEYS)
    _require_schema_version(payload)
    source_range = _require_object(
        payload["source_range"],
        field_name="source_range",
        keys=frozenset({"start", "end"}),
    )
    overlap = _require_object(
        payload["overlap"],
        field_name="overlap",
        keys=frozenset({"target_tokens", "source_start", "source_end", "token_count"}),
    )
    references = tuple(
        _parse_source_reference(value)
        for value in _require_array(
            payload["source_references"], field_name="source_references"
        )
    )
    structural_kinds = tuple(
        _require_string(value, field_name="structural_kind")
        for value in _require_array(
            payload["structural_kinds"], field_name="structural_kinds"
        )
    )
    structural_ordinals = tuple(
        _require_integer(value, field_name="structural_block_ordinal")
        for value in _require_array(
            payload["structural_block_ordinals"],
            field_name="structural_block_ordinals",
        )
    )
    minimum_exception = cast(
        ChunkMinimumException | None,
        _parse_optional_enum(
            payload["minimum_exception"],
            field_name="minimum_exception",
            enum_type=ChunkMinimumException,
        ),
    )
    split_reason = cast(
        ChunkSplitReason | None,
        _parse_optional_enum(
            payload["split_reason"],
            field_name="split_reason",
            enum_type=ChunkSplitReason,
        ),
    )
    metadata_payload = _require_object(payload["metadata"], field_name="metadata")
    metadata = deserialise_derived_record(canonical_json_bytes(metadata_payload))
    return Chunk(
        document_id=_parse_identifier(
            payload["document_id"],
            field_name="document_id",
            constructor=DocumentId.from_str,
        ),
        document_version_id=_parse_identifier(
            payload["document_version_id"],
            field_name="document_version_id",
            constructor=DocumentVersionId.from_str,
        ),
        ordinal=_require_integer(payload["ordinal"], field_name="ordinal"),
        text=_require_string(payload["text"], field_name="text"),
        token_count=_require_integer(payload["token_count"], field_name="token_count"),
        source_start=_require_integer(
            source_range["start"], field_name="source_range.start"
        ),
        source_end=_require_integer(source_range["end"], field_name="source_range.end"),
        target_overlap_tokens=_require_integer(
            overlap["target_tokens"], field_name="overlap.target_tokens"
        ),
        overlap_source_start=_require_integer(
            overlap["source_start"], field_name="overlap.source_start"
        ),
        overlap_source_end=_require_integer(
            overlap["source_end"], field_name="overlap.source_end"
        ),
        overlap_token_count=_require_integer(
            overlap["token_count"], field_name="overlap.token_count"
        ),
        source_references=references,
        structural_kinds=structural_kinds,
        structural_block_ordinals=structural_ordinals,
        minimum_exception=minimum_exception,
        split_reason=split_reason,
        chunker_version=_parse_chunker_version(payload["chunker_version"]),
        tokeniser_identity=_parse_tokeniser_identity(payload["tokeniser_identity"]),
        configuration_identity=_parse_configuration_identity(
            payload["configuration_identity"]
        ),
        metadata=metadata,
    )


def _parse_source_reference(value: JsonValue) -> ChunkSourceReference:
    """Parse one clipped source reference."""
    payload = _require_object(
        value,
        field_name="source_reference",
        keys=frozenset(
            {
                "source_span_id",
                "document_version_id",
                "block_ordinal",
                "block_kind",
                "source_start",
                "source_end",
            }
        ),
    )
    return ChunkSourceReference(
        source_span_id=_parse_identifier(
            payload["source_span_id"],
            field_name="source_span_id",
            constructor=SourceSpanId.from_str,
        ),
        document_version_id=_parse_identifier(
            payload["document_version_id"],
            field_name="document_version_id",
            constructor=DocumentVersionId.from_str,
        ),
        block_ordinal=_require_integer(
            payload["block_ordinal"], field_name="block_ordinal"
        ),
        block_kind=cast(
            StructuralBlockKind,
            _require_string(payload["block_kind"], field_name="block_kind"),
        ),
        source_start=_require_integer(
            payload["source_start"], field_name="source_start"
        ),
        source_end=_require_integer(payload["source_end"], field_name="source_end"),
    )


def _parse_parser_version(value: JsonValue) -> ParserVersion:
    """Parse one exact parser-version object."""
    payload = _require_object(
        value,
        field_name="parser_version",
        keys=frozenset({"implementation", "contract_version", "normalisation_version"}),
    )
    return ParserVersion(
        implementation=_require_string(
            payload["implementation"], field_name="parser_version.implementation"
        ),
        contract_version=_require_string(
            payload["contract_version"],
            field_name="parser_version.contract_version",
        ),
        normalisation_version=_require_string(
            payload["normalisation_version"],
            field_name="parser_version.normalisation_version",
        ),
    )


def _parse_chunker_version(value: JsonValue) -> ChunkerVersion:
    """Parse one exact chunker-version object."""
    payload = _require_object(
        value,
        field_name="chunker_version",
        keys=frozenset({"implementation", "contract_version"}),
    )
    return ChunkerVersion(
        implementation=_require_string(
            payload["implementation"], field_name="chunker_version.implementation"
        ),
        contract_version=_require_string(
            payload["contract_version"],
            field_name="chunker_version.contract_version",
        ),
    )


def _parse_tokeniser_identity(value: JsonValue) -> TokeniserIdentity:
    """Parse one exact path-free tokeniser identity."""
    payload = _require_object(
        value,
        field_name="tokeniser_identity",
        keys=frozenset(
            {
                "implementation",
                "contract_version",
                "artefact_identity",
                "artefact_revision",
                "options",
            }
        ),
    )
    options = tuple(
        (
            _require_string(option["key"], field_name="tokeniser option.key"),
            _require_string(option["value"], field_name="tokeniser option.value"),
        )
        for option in (
            _require_object(
                item, field_name="tokeniser option", keys=frozenset({"key", "value"})
            )
            for item in _require_array(
                payload["options"], field_name="tokeniser_identity.options"
            )
        )
    )
    return TokeniserIdentity(
        implementation=_require_string(
            payload["implementation"], field_name="tokeniser_identity.implementation"
        ),
        contract_version=_require_string(
            payload["contract_version"],
            field_name="tokeniser_identity.contract_version",
        ),
        artefact_identity=_require_string(
            payload["artefact_identity"],
            field_name="tokeniser_identity.artefact_identity",
        ),
        artefact_revision=_require_string(
            payload["artefact_revision"],
            field_name="tokeniser_identity.artefact_revision",
        ),
        options=options,
    )


def _parse_policy(value: JsonValue) -> ChunkingPolicy:
    """Parse one exact validated chunking policy."""
    payload = _require_object(
        value,
        field_name="policy",
        keys=frozenset(
            {
                "min_tokens",
                "max_tokens",
                "overlap_numerator",
                "overlap_denominator",
                "structural_boundary_policy",
                "oversize_unit_policy",
            }
        ),
    )
    try:
        structural_policy = StructuralBoundaryPolicy(
            _require_string(
                payload["structural_boundary_policy"],
                field_name="policy.structural_boundary_policy",
            )
        )
        oversize_policy = OversizeUnitPolicy(
            _require_string(
                payload["oversize_unit_policy"],
                field_name="policy.oversize_unit_policy",
            )
        )
        return ChunkingPolicy(
            min_tokens=_require_integer(
                payload["min_tokens"], field_name="policy.min_tokens"
            ),
            max_tokens=_require_integer(
                payload["max_tokens"], field_name="policy.max_tokens"
            ),
            overlap_numerator=_require_integer(
                payload["overlap_numerator"], field_name="policy.overlap_numerator"
            ),
            overlap_denominator=_require_integer(
                payload["overlap_denominator"], field_name="policy.overlap_denominator"
            ),
            structural_boundary_policy=structural_policy,
            oversize_unit_policy=oversize_policy,
        )
    except ValueError:
        raise InvalidChunkSerialisationError("policy is not supported.") from None


def _parse_configuration_identity(value: JsonValue) -> ConfigurationIdentity:
    """Parse one exact configuration identity."""
    payload = _require_object(
        value,
        field_name="configuration_identity",
        keys=frozenset({"algorithm", "digest"}),
    )
    try:
        algorithm = DigestAlgorithm(
            _require_string(
                payload["algorithm"], field_name="configuration_identity.algorithm"
            )
        )
        return ConfigurationIdentity(
            algorithm=algorithm,
            digest=_require_string(
                payload["digest"], field_name="configuration_identity.digest"
            ),
        )
    except ValueError:
        raise InvalidChunkSerialisationError(
            "configuration_identity is not supported."
        ) from None


def _tokeniser_payload(identity: TokeniserIdentity) -> JsonObject:
    """Return the canonical tokeniser identity payload."""
    return {
        "implementation": identity.implementation,
        "contract_version": identity.contract_version,
        "artefact_identity": identity.artefact_identity,
        "artefact_revision": identity.artefact_revision,
        "options": [{"key": key, "value": value} for key, value in identity.options],
    }


def _parser_version_payload(version: ParserVersion) -> JsonObject:
    """Return the canonical parser-version payload."""
    return {
        "implementation": version.implementation,
        "contract_version": version.contract_version,
        "normalisation_version": version.normalisation_version,
    }


def _chunker_payload(version: ChunkerVersion) -> JsonObject:
    """Return the canonical chunker-version payload."""
    return {
        "implementation": version.implementation,
        "contract_version": version.contract_version,
    }


def _policy_payload(policy: ChunkingPolicy) -> JsonObject:
    """Return the canonical policy payload."""
    return {
        "min_tokens": policy.min_tokens,
        "max_tokens": policy.max_tokens,
        "overlap_numerator": policy.overlap_numerator,
        "overlap_denominator": policy.overlap_denominator,
        "structural_boundary_policy": policy.structural_boundary_policy.value,
        "oversize_unit_policy": policy.oversize_unit_policy.value,
    }


def _configuration_payload(identity: ConfigurationIdentity) -> JsonObject:
    """Return the canonical configuration identity payload."""
    return {"algorithm": identity.algorithm.value, "digest": identity.digest}


def _load_payload(data: bytes) -> JsonObject:
    """Load strict UTF-8 JSON without retaining rejected payloads."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    try:
        return load_json_object(data)
    except JsonDecodingError:
        raise InvalidChunkSerialisationError(
            "chunk JSON must be valid UTF-8 with a unique object structure."
        ) from None


def _require_schema_version(payload: JsonObject) -> None:
    """Require the one supported chunk payload schema version."""
    value = payload.get("payload_schema_version")
    if value != CHUNK_PAYLOAD_SCHEMA_VERSION:
        raise InvalidChunkSerialisationError(
            "unsupported chunk payload schema version."
        )


def _require_object(
    value: JsonValue,
    *,
    field_name: str,
    keys: frozenset[str] | None = None,
) -> JsonObject:
    """Require an object and, when requested, its exact fields."""
    try:
        result = require_json_object(value)
    except JsonStructureError:
        raise InvalidChunkSerialisationError(
            f"{field_name} must be an object."
        ) from None
    if keys is not None:
        try:
            require_exact_json_fields(result, keys)
        except JsonStructureError:
            raise InvalidChunkSerialisationError(
                f"{field_name} must contain exactly the canonical fields."
            ) from None
    return result


def _require_array(value: JsonValue, *, field_name: str) -> list[JsonValue]:
    """Require an array without scalar coercion."""
    try:
        return require_json_array(value)
    except JsonStructureError:
        raise InvalidChunkSerialisationError(
            f"{field_name} must be an array."
        ) from None


def _require_string(value: JsonValue, *, field_name: str) -> str:
    """Require a string without scalar coercion."""
    try:
        return require_json_string(value)
    except JsonStructureError:
        raise InvalidChunkSerialisationError(
            f"{field_name} must be a string."
        ) from None


def _require_integer(value: JsonValue, *, field_name: str) -> int:
    """Require an integer without boolean coercion."""
    try:
        return require_json_integer(value)
    except JsonStructureError:
        raise InvalidChunkSerialisationError(
            f"{field_name} must be an integer."
        ) from None


def _parse_identifier(
    value: JsonValue,
    *,
    field_name: str,
    constructor: Callable[[str], _IdentifierT],
) -> _IdentifierT:
    """Parse one canonical typed UUID identifier."""
    text = _require_string(value, field_name=field_name)
    try:
        return constructor(text)
    except (TypeError, ValueError):
        raise InvalidChunkSerialisationError(
            f"{field_name} must be a canonical provenance identifier."
        ) from None


def _parse_optional_enum(
    value: JsonValue,
    *,
    field_name: str,
    enum_type: type[ChunkMinimumException] | type[ChunkSplitReason],
) -> ChunkMinimumException | ChunkSplitReason | None:
    """Parse one nullable closed enum."""
    if value is None:
        return None
    text = _require_string(value, field_name=field_name)
    try:
        return enum_type(text)
    except ValueError:
        raise InvalidChunkSerialisationError(
            f"{field_name} is not supported."
        ) from None
