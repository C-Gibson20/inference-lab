"""Canonical serialisation for retrieval result contracts."""

import math
from typing import cast

from inference_lab._json import JsonObject, JsonValue, canonical_json_bytes
from inference_lab.provenance import (
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval._serialisation_json import (
    load_json,
    parse_enum,
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
from inference_lab.retrieval.evidence import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceSource,
    RetrievalCandidate,
)
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.values import RetrievalChannel, RetrievalLimit
from inference_lab.retrieval.versioning import CURRENT_RETRIEVAL_SCHEMA_VERSION


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
    return canonical_json_bytes(_result_payload(result))


def _source(value: JsonValue) -> EvidenceSource:
    """Decode one strict evidence source."""
    payload = require_object(
        value,
        name="source",
        keys=frozenset({"document_id", "document_version_id", "source_span_id"}),
    )
    return EvidenceSource(
        cast(
            DocumentId,
            parse_identifier(
                payload["document_id"],
                name="source.document_id",
                constructor=DocumentId.from_str,
            ),
        ),
        cast(
            DocumentVersionId,
            parse_identifier(
                payload["document_version_id"],
                name="source.document_version_id",
                constructor=DocumentVersionId.from_str,
            ),
        ),
        cast(
            SourceSpanId,
            parse_identifier(
                payload["source_span_id"],
                name="source.source_span_id",
                constructor=SourceSpanId.from_str,
            ),
        ),
    )


def _candidate(value: JsonValue) -> RetrievalCandidate:
    """Decode one strict candidate."""
    payload = require_object(
        value,
        name="candidate",
        keys=frozenset({"record_id", "kind", "content", "sources"}),
    )
    return RetrievalCandidate(
        cast(
            RecordId,
            parse_identifier(
                payload["record_id"],
                name="candidate.record_id",
                constructor=RecordId.from_str,
            ),
        ),
        cast(
            CandidateKind,
            parse_enum(payload["kind"], name="candidate.kind", enum_type=CandidateKind),
        ),
        require_string(payload["content"], name="candidate.content"),
        tuple(
            _source(item)
            for item in require_array(payload["sources"], name="candidate.sources")
        ),
    )


def _score(value: JsonValue) -> ChannelScore:
    """Decode one strict channel contribution."""
    payload = require_object(
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
            parse_enum(
                payload["channel"],
                name="score.channel",
                enum_type=RetrievalChannel,
            ),
        ),
        require_integer(payload["rank"], name="score.rank"),
        cast(int | float | None, raw_score),
        score_label,
    )


def _hit(value: JsonValue) -> EvidenceHit:
    """Decode one strict ranked hit."""
    payload = require_object(
        value,
        name="hit",
        keys=frozenset({"candidate", "contributions", "rank"}),
    )
    return EvidenceHit(
        _candidate(payload["candidate"]),
        tuple(
            _score(item)
            for item in require_array(
                payload["contributions"],
                name="hit.contributions",
            )
        ),
        require_integer(payload["rank"], name="hit.rank"),
    )


def _failure(value: JsonValue) -> RetrievalFailure:
    """Decode one strict typed failure."""
    payload = require_object(
        value,
        name="failure",
        keys=frozenset({"category", "channel", "retryable"}),
    )
    channel = payload["channel"]
    return RetrievalFailure(
        cast(
            FailureCategory,
            parse_enum(
                payload["category"],
                name="failure.category",
                enum_type=FailureCategory,
            ),
        ),
        require_boolean(payload["retryable"], name="failure.retryable"),
        None
        if channel is None
        else cast(
            RetrievalChannel,
            parse_enum(
                channel,
                name="failure.channel",
                enum_type=RetrievalChannel,
            ),
        ),
    )


def deserialise_evidence_result(data: bytes) -> EvidenceResult:
    """Deserialise one strict canonical evidence result payload."""
    payload = load_json(data)
    validate_schema(payload)
    payload = require_object(
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
                parse_identifier(
                    payload["corpus_version"],
                    name="result.corpus_version",
                    constructor=CorpusVersionId.from_str,
                ),
            ),
            RetrievalLimit(
                require_integer(
                    payload["requested_limit"],
                    name="result.requested_limit",
                )
            ),
            tuple(
                _hit(item)
                for item in require_array(payload["hits"], name="result.hits")
            ),
            cast(
                ResultStatus,
                parse_enum(
                    payload["status"],
                    name="result.status",
                    enum_type=ResultStatus,
                ),
            ),
            tuple(
                _failure(item)
                for item in require_array(
                    payload["failures"],
                    name="result.failures",
                )
            ),
        )
    except InvalidRetrievalSerialisationError:
        raise
    except (InvalidRetrievalContractError, TypeError, ValueError):
        raise InvalidRetrievalSerialisationError(
            "result violates its retrieval contract."
        ) from None
