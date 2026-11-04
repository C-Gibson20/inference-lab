"""Strict canonical serialisation for dense generation manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from inference_lab._json import (
    JsonObject,
    canonical_json_bytes,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_boolean,
    require_json_integer,
    require_json_string,
    require_optional_json_integer,
    require_optional_json_string,
)
from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
)
from inference_lab.backends.runtime.types import EmbeddingOverLimitPolicy
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.retrieval.dense import (
    DENSE_FORMAT_VERSION,
    DenseDistanceMetric,
    DenseIndexCompatibility,
    DenseIndexConfiguration,
)
from inference_lab.retrieval.dense_store_models import (
    DenseError,
    DenseFailureCategory,
    DenseIndexManifest,
    DenseMemberSelection,
)


def _manifest_json(manifest: DenseIndexManifest) -> bytes:
    return canonical_json_bytes(manifest.as_payload())


def _read_manifest(path: Path) -> DenseIndexManifest:
    try:
        payload = load_json_object(path.read_bytes())
        require_exact_json_fields(
            payload,
            frozenset(
                {
                    "compatibility",
                    "corpus_id",
                    "corpus_version",
                    "database_digest",
                    "entry_count",
                    "format_version",
                    "generation_id",
                    "library_version",
                    "members",
                    "source_manifest_digest",
                }
            ),
        )
        if payload.get("format_version") != DENSE_FORMAT_VERSION:
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        raw_compatibility = payload["compatibility"]
        if not isinstance(raw_compatibility, dict):
            raise ValueError("manifest compatibility must be an object.")
        compatibility = _compatibility_from_payload(raw_compatibility)
        raw_members = payload["members"]
        if not isinstance(raw_members, list):
            raise ValueError("manifest members must be an array.")
        members = tuple(
            _member_from_payload(item) for item in raw_members if isinstance(item, dict)
        )
        if len(members) != len(raw_members):
            raise ValueError("every manifest member must be an object.")
        return DenseIndexManifest(
            generation_id=require_json_string(payload["generation_id"]),
            corpus_id=CorpusId.from_str(require_json_string(payload["corpus_id"])),
            corpus_version=CorpusVersionId.from_str(
                require_json_string(payload["corpus_version"])
            ),
            source_manifest_digest=require_json_string(
                payload["source_manifest_digest"]
            ),
            compatibility=compatibility,
            members=members,
            entry_count=require_json_integer(payload["entry_count"]),
            library_version=require_json_string(payload["library_version"]),
            database_digest=require_json_string(payload["database_digest"]),
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


def _member_from_payload(payload: JsonObject) -> DenseMemberSelection:
    require_exact_json_fields(
        payload,
        frozenset(
            {"chunk_count", "document_id", "document_version_id", "selection_digest"}
        ),
    )
    return DenseMemberSelection(
        document_id=DocumentId.from_str(require_json_string(payload["document_id"])),
        document_version_id=DocumentVersionId.from_str(
            require_json_string(payload["document_version_id"])
        ),
        selection_digest=require_json_string(payload["selection_digest"]),
        chunk_count=require_json_integer(payload["chunk_count"]),
    )


_COMPATIBILITY_FIELDS: Final = frozenset(
    {"configuration", "dimensions", "embedding_profile", "model"}
)
_MODEL_FIELDS: Final = frozenset(
    {
        "backend_name",
        "capabilities",
        "model_name",
        "model_revision",
        "preprocessing_profile_version",
        "supported_operations",
    }
)
_PROFILE_FIELDS: Final = frozenset(
    {
        "device_type",
        "document_prefix",
        "inference_dtype",
        "input_normalisation",
        "l2_normalised",
        "max_input_tokens",
        "over_limit_policy",
        "pooling_strategy",
        "profile_version",
        "query_prefix",
        "tokenizer_id",
        "tokenizer_revision",
    }
)
_CONFIGURATION_FIELDS: Final = frozenset(
    {
        "contract_version",
        "distance_metric",
        "format_version",
        "implementation",
        "tie_break_policy",
    }
)


def _decode_backend_metadata(model: JsonObject, dimensions: int) -> BackendMetadata:
    """Decode the backend identity that produced these vectors."""
    require_exact_json_fields(model, _MODEL_FIELDS)
    return BackendMetadata(
        backend_name=require_json_string(model["backend_name"]),
        model_name=require_json_string(model["model_name"]),
        model_revision=require_json_string(model["model_revision"]),
        preprocessing_profile_version=require_optional_json_string(
            model["preprocessing_profile_version"]
        ),
        supported_operations=tuple(
            BackendOperation(require_json_string(value))
            for value in require_json_array(model["supported_operations"])
        ),
        capabilities=tuple(
            BackendCapability(require_json_string(value))
            for value in require_json_array(model["capabilities"])
        ),
        embedding_dimensions=dimensions,
    )


def _decode_embedding_profile(profile: JsonObject) -> EmbeddingProfile:
    """Decode the runtime preprocessing identity recorded for these vectors."""
    require_exact_json_fields(profile, _PROFILE_FIELDS)
    policy = require_optional_json_string(profile["over_limit_policy"])
    if policy not in (None, "truncate_tail", "reject"):
        raise ValueError("over_limit_policy names an unsupported policy.")
    return EmbeddingProfile(
        profile_version=require_json_string(profile["profile_version"]),
        tokenizer_id=require_json_string(profile["tokenizer_id"]),
        tokenizer_revision=require_json_string(profile["tokenizer_revision"]),
        query_prefix=require_json_string(profile["query_prefix"]),
        document_prefix=require_json_string(profile["document_prefix"]),
        pooling_strategy=require_json_string(profile["pooling_strategy"]),
        input_normalisation=require_json_string(profile["input_normalisation"]),
        max_input_tokens=require_optional_json_integer(profile["max_input_tokens"]),
        over_limit_policy=cast(EmbeddingOverLimitPolicy | None, policy),
        l2_normalised=require_json_boolean(profile["l2_normalised"]),
        inference_dtype=require_json_string(profile["inference_dtype"]),
        device_type=require_json_string(profile["device_type"]),
    )


def _decode_index_configuration(config: JsonObject) -> DenseIndexConfiguration:
    """Decode the versioned settings that govern index interpretation."""
    require_exact_json_fields(config, _CONFIGURATION_FIELDS)
    return DenseIndexConfiguration(
        implementation=require_json_string(config["implementation"]),
        format_version=require_json_string(config["format_version"]),
        contract_version=require_json_string(config["contract_version"]),
        distance_metric=DenseDistanceMetric(
            require_json_string(config["distance_metric"])
        ),
        tie_break_policy=require_json_string(config["tie_break_policy"]),
    )


def _compatibility_from_payload(payload: JsonObject) -> DenseIndexCompatibility:
    try:
        model = payload["model"]
        profile = payload["embedding_profile"]
        config = payload["configuration"]
        if (
            not isinstance(model, dict)
            or not isinstance(profile, dict)
            or not isinstance(config, dict)
        ):
            raise ValueError(
                "compatibility must hold model, profile, and configuration objects."
            )
        require_exact_json_fields(payload, _COMPATIBILITY_FIELDS)
        dimensions = require_json_integer(payload["dimensions"])
        return DenseIndexCompatibility(
            model_metadata=_decode_backend_metadata(model, dimensions),
            dimensions=dimensions,
            embedding_profile=_decode_embedding_profile(profile),
            configuration=_decode_index_configuration(config),
        )
    except (KeyError, TypeError, ValueError):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None
