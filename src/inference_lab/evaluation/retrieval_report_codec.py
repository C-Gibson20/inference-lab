"""Canonical bytes and strict readers for retrieval-evaluation artefacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

from inference_lab._json import (
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    load_json_object,
    require_exact_json_fields,
    require_json_array,
    require_json_boolean,
    require_json_integer,
    require_json_object,
    require_json_string,
)
from inference_lab.evaluation._validation import is_sha256_checksum
from inference_lab.evaluation.retrieval_contracts import (
    AggregateMetric,
    DuplicateDiagnostics,
    MetricFraction,
    PublishedArtifact,
    PublishedRetrievalEvaluationManifest,
    RecallMatchRanks,
    RecallMetric,
    RetrievalCaseResult,
    RetrievalComponentIdentity,
    RetrievalEvaluationAggregate,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRunIdentity,
    RetrievalSystemIdentity,
    SanitisedChannelContribution,
    SanitisedFailure,
    SanitisedHitReference,
    SanitisedSourceReference,
)
from inference_lab.evaluation.retrieval_execution import RetrievalEvaluationRunResult
from inference_lab.evaluation.retrieval_report_contracts import (
    RetrievalEvaluationReport,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    FailureCategory,
    ResultStatus,
    RetrievalChannel,
)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRunArtifact:
    """Decoded `run.json` identity and result linkage."""

    identity: RetrievalEvaluationRunIdentity
    result_digest: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the run artifact fields."""
        if not isinstance(self.identity, RetrievalEvaluationRunIdentity):
            raise TypeError("identity must be a RetrievalEvaluationRunIdentity.")
        if not isinstance(self.result_digest, str) or not is_sha256_checksum(
            self.result_digest
        ):
            raise ValueError("result_digest must be a sha256 checksum.")
        if not isinstance(self.limitations, tuple):
            raise TypeError("limitations must be a tuple.")
        if any(
            not isinstance(item, str) or not item.strip() for item in self.limitations
        ):
            raise ValueError("limitations must contain non-blank strings.")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("limitations must be unique.")


def _object(value: JsonValue) -> JsonObject:
    """Require one decoded JSON object."""
    return require_json_object(value)


def _array(value: JsonValue) -> list[JsonValue]:
    """Require one decoded JSON array."""
    return require_json_array(value)


def _string(value: JsonValue) -> str:
    """Require one decoded JSON string."""
    return require_json_string(value)


def _integer(value: JsonValue) -> int:
    """Require one decoded JSON integer."""
    return require_json_integer(value)


def _boolean(value: JsonValue) -> bool:
    """Require one decoded JSON boolean."""
    return require_json_boolean(value)


def _nullable_string(value: JsonValue) -> str | None:
    """Require one decoded string or JSON null."""
    if value is None:
        return None
    return _string(value)


def _canonical_object(data: bytes) -> JsonObject:
    """Decode one canonical JSON object and reject non-canonical bytes."""
    payload = load_json_object(data)
    if canonical_json_bytes(payload) != data:
        raise ValueError("retrieval evaluation JSON is not canonical.")
    return payload


def _decode_metric_fraction(payload: JsonObject) -> MetricFraction:
    """Decode one exact metric fraction."""
    require_exact_json_fields(payload, frozenset({"numerator", "denominator"}))
    return MetricFraction(
        _integer(payload["numerator"]), _integer(payload["denominator"])
    )


def _decode_optional_metric(value: JsonValue) -> MetricFraction | None:
    """Decode an optional metric fraction."""
    if value is None:
        return None
    return _decode_metric_fraction(_object(value))


def _decode_policy(payload: JsonObject) -> RetrievalEvaluationPolicy:
    """Decode one complete metric and aggregation policy."""
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "passage_cutoffs",
                "candidate_depth",
                "document_cutoff",
                "document_policy_version",
                "passage_policy_version",
                "duplicate_policy_version",
                "family_policy_version",
                "aggregation_policy_version",
                "decimal_precision",
                "rounding_mode",
            }
        ),
    )
    return RetrievalEvaluationPolicy(
        passage_cutoffs=tuple(
            _integer(item) for item in _array(payload["passage_cutoffs"])
        ),
        candidate_depth=_integer(payload["candidate_depth"]),
        document_cutoff=_integer(payload["document_cutoff"]),
        document_policy_version=_string(payload["document_policy_version"]),
        passage_policy_version=_string(payload["passage_policy_version"]),
        duplicate_policy_version=_string(payload["duplicate_policy_version"]),
        family_policy_version=_string(payload["family_policy_version"]),
        aggregation_policy_version=_string(payload["aggregation_policy_version"]),
        decimal_precision=_integer(payload["decimal_precision"]),
        rounding_mode=_string(payload["rounding_mode"]),
    )


def _decode_component(payload: JsonObject) -> RetrievalComponentIdentity:
    """Decode one retrieval component identity."""
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "component",
                "implementation",
                "format_version",
                "contract_version",
                "configuration_digest",
                "generation_id",
                "source_manifest_digest",
                "model_identity",
                "model_revision",
            }
        ),
    )
    return RetrievalComponentIdentity(
        component=_string(payload["component"]),
        implementation=_string(payload["implementation"]),
        format_version=_string(payload["format_version"]),
        contract_version=_string(payload["contract_version"]),
        configuration_digest=_string(payload["configuration_digest"]),
        generation_id=_nullable_string(payload["generation_id"]),
        source_manifest_digest=_nullable_string(payload["source_manifest_digest"]),
        model_identity=_nullable_string(payload["model_identity"]),
        model_revision=_nullable_string(payload["model_revision"]),
    )


def _decode_system_identity(payload: JsonObject) -> RetrievalSystemIdentity:
    """Decode one complete retrieval system identity."""
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "adapter_implementation",
                "adapter_contract_version",
                "query_construction_algorithm",
                "query_construction_version",
                "retrieval_algorithm",
                "retrieval_algorithm_version",
                "configuration_digest",
                "components",
                "model_identity",
                "model_revision",
            }
        ),
    )
    return RetrievalSystemIdentity(
        adapter_implementation=_string(payload["adapter_implementation"]),
        adapter_contract_version=_string(payload["adapter_contract_version"]),
        query_construction_algorithm=_string(payload["query_construction_algorithm"]),
        query_construction_version=_string(payload["query_construction_version"]),
        retrieval_algorithm=_string(payload["retrieval_algorithm"]),
        retrieval_algorithm_version=_string(payload["retrieval_algorithm_version"]),
        configuration_digest=_string(payload["configuration_digest"]),
        components=tuple(
            _decode_component(_object(item)) for item in _array(payload["components"])
        ),
        model_identity=_nullable_string(payload["model_identity"]),
        model_revision=_nullable_string(payload["model_revision"]),
    )


def _decode_run_identity(payload: JsonObject) -> RetrievalEvaluationRunIdentity:
    """Decode one semantic run identity."""
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "result_schema_version",
                "implementation",
                "implementation_version",
                "dataset_id",
                "dataset_version",
                "dataset_schema_version",
                "case_file_checksum",
                "corpus_id",
                "corpus_version",
                "selected_split",
                "selected_case_ids",
                "policy",
                "relationship_snapshot_digest",
                "system_identity",
                "run_fingerprint",
            }
        ),
    )
    return RetrievalEvaluationRunIdentity(
        dataset_id=_string(payload["dataset_id"]),
        dataset_version=_string(payload["dataset_version"]),
        dataset_schema_version=_string(payload["dataset_schema_version"]),
        case_file_checksum=_string(payload["case_file_checksum"]),
        corpus_id=CorpusId.from_str(_string(payload["corpus_id"])),
        corpus_version=CorpusVersionId.from_str(_string(payload["corpus_version"])),
        selected_split=_string(payload["selected_split"]),  # type: ignore[arg-type]
        selected_case_ids=tuple(
            _string(item) for item in _array(payload["selected_case_ids"])
        ),
        policy=_decode_policy(_object(payload["policy"])),
        relationship_snapshot_digest=_string(payload["relationship_snapshot_digest"]),
        system_identity=_decode_system_identity(_object(payload["system_identity"])),
        result_schema_version=_string(payload["result_schema_version"]),
        implementation=_string(payload["implementation"]),
        implementation_version=_string(payload["implementation_version"]),
        run_fingerprint=_string(payload["run_fingerprint"]),
    )


def _decode_source(payload: JsonObject) -> SanitisedSourceReference:
    """Decode one sanitised source reference."""
    require_exact_json_fields(
        payload,
        frozenset({"document_id", "document_version_id", "source_span_id"}),
    )
    return SanitisedSourceReference(
        DocumentId.from_str(_string(payload["document_id"])),
        DocumentVersionId.from_str(_string(payload["document_version_id"])),
        SourceSpanId.from_str(_string(payload["source_span_id"])),
    )


def _decode_contribution(payload: JsonObject) -> SanitisedChannelContribution:
    """Decode one sanitised channel contribution."""
    require_exact_json_fields(payload, frozenset({"channel", "rank"}))
    return SanitisedChannelContribution(
        RetrievalChannel(_string(payload["channel"])),
        _integer(payload["rank"]),
    )


def _decode_hit(payload: JsonObject) -> SanitisedHitReference:
    """Decode one sanitised ranked hit."""
    require_exact_json_fields(
        payload,
        frozenset({"rank", "record_id", "kind", "sources", "contributions"}),
    )
    return SanitisedHitReference(
        rank=_integer(payload["rank"]),
        record_id=RecordId.from_str(_string(payload["record_id"])),
        kind=CandidateKind(_string(payload["kind"])),
        sources=tuple(
            _decode_source(_object(item)) for item in _array(payload["sources"])
        ),
        contributions=tuple(
            _decode_contribution(_object(item))
            for item in _array(payload["contributions"])
        ),
    )


def _decode_failure(payload: JsonObject) -> SanitisedFailure:
    """Decode one bounded retrieval failure."""
    require_exact_json_fields(payload, frozenset({"category", "retryable", "channel"}))
    channel_value = payload["channel"]
    return SanitisedFailure(
        category=FailureCategory(_string(payload["category"])),
        retryable=_boolean(payload["retryable"]),
        channel=(
            None if channel_value is None else RetrievalChannel(_string(channel_value))
        ),
    )


def _decode_recall_metric(payload: JsonObject) -> RecallMetric:
    """Decode one recall metric."""
    require_exact_json_fields(payload, frozenset({"cutoff", "value"}))
    return RecallMetric(
        cutoff=_integer(payload["cutoff"]),
        value=_decode_metric_fraction(_object(payload["value"])),
    )


def _decode_match_ranks(payload: JsonObject) -> RecallMatchRanks:
    """Decode one exact match-rank diagnostic."""
    require_exact_json_fields(payload, frozenset({"cutoff", "ranks"}))
    ranks = tuple(
        None if item is None else _integer(item) for item in _array(payload["ranks"])
    )
    return RecallMatchRanks(cutoff=_integer(payload["cutoff"]), ranks=ranks)


def _decode_diagnostics(value: JsonValue) -> DuplicateDiagnostics | None:
    """Decode optional duplicate and family diagnostics."""
    if value is None:
        return None
    payload = _object(value)
    fields = frozenset(
        {
            "raw_hit_count",
            "distinct_record_count",
            "distinct_document_count",
            "distinct_duplicate_group_count",
            "distinct_family_count",
            "unknown_family_document_count",
            "exact_document_repeat_excess",
            "byte_identical_group_repeat_excess",
            "family_member_repeat_excess",
            "relevant_non_gold_duplicate_copy_count",
        }
    )
    require_exact_json_fields(payload, fields)
    return DuplicateDiagnostics(
        **{field: _integer(payload[field]) for field in fields}  # type: ignore[arg-type]
    )


def _decode_case_result(payload: JsonObject) -> RetrievalCaseResult:
    """Decode one sanitised case result."""
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "case_id",
                "case_ordinal",
                "split",
                "corpus_id",
                "corpus_version",
                "requested_limit",
                "status",
                "hits",
                "failures",
                "document_recall",
                "passage_recall",
                "document_match_ranks",
                "passage_match_ranks",
                "document_depth_exhausted",
                "duplicate_diagnostics",
            }
        ),
    )
    document_recall = payload["document_recall"]
    passage_recall = tuple(
        None if item is None else _decode_recall_metric(_object(item))
        for item in _array(payload["passage_recall"])
    )
    document_match_ranks = tuple(
        None if item is None else _integer(item)
        for item in _array(payload["document_match_ranks"])
    )
    return RetrievalCaseResult(
        case_id=_string(payload["case_id"]),
        case_ordinal=_integer(payload["case_ordinal"]),
        split=_string(payload["split"]),  # type: ignore[arg-type]
        corpus_id=CorpusId.from_str(_string(payload["corpus_id"])),
        corpus_version=CorpusVersionId.from_str(_string(payload["corpus_version"])),
        requested_limit=_integer(payload["requested_limit"]),
        status=ResultStatus(_string(payload["status"])),
        hits=tuple(_decode_hit(_object(item)) for item in _array(payload["hits"])),
        failures=tuple(
            _decode_failure(_object(item)) for item in _array(payload["failures"])
        ),
        document_recall=(
            None
            if document_recall is None
            else _decode_recall_metric(_object(document_recall))
        ),
        passage_recall=passage_recall,
        document_match_ranks=document_match_ranks,
        passage_match_ranks=tuple(
            _decode_match_ranks(_object(item))
            for item in _array(payload["passage_match_ranks"])
        ),
        document_depth_exhausted=_boolean(payload["document_depth_exhausted"]),
        duplicate_diagnostics=_decode_diagnostics(payload["duplicate_diagnostics"]),
    )


def _decode_aggregate_metric(payload: JsonObject) -> AggregateMetric:
    """Decode one exact aggregate metric."""
    require_exact_json_fields(
        payload,
        frozenset(
            {
                "cutoff",
                "macro",
                "micro",
                "applicable_case_count",
                "non_applicable_case_count",
                "completed_case_count",
                "partial_case_count",
                "failed_case_count",
                "empty_case_count",
                "matched_target_count",
                "relevant_target_count",
            }
        ),
    )
    return AggregateMetric(
        cutoff=_integer(payload["cutoff"]),
        macro=_decode_optional_metric(payload["macro"]),
        micro=_decode_optional_metric(payload["micro"]),
        applicable_case_count=_integer(payload["applicable_case_count"]),
        non_applicable_case_count=_integer(payload["non_applicable_case_count"]),
        completed_case_count=_integer(payload["completed_case_count"]),
        partial_case_count=_integer(payload["partial_case_count"]),
        failed_case_count=_integer(payload["failed_case_count"]),
        empty_case_count=_integer(payload["empty_case_count"]),
        matched_target_count=_integer(payload["matched_target_count"]),
        relevant_target_count=_integer(payload["relevant_target_count"]),
    )


def _decode_aggregate(payload: JsonObject) -> RetrievalEvaluationAggregate:
    """Decode one complete retrieval aggregate."""
    require_exact_json_fields(
        payload,
        frozenset({"case_count", "document", "passage", "duplicate_diagnostics"}),
    )
    return RetrievalEvaluationAggregate(
        case_count=_integer(payload["case_count"]),
        document=_decode_aggregate_metric(_object(payload["document"])),
        passage=tuple(
            _decode_aggregate_metric(_object(item))
            for item in _array(payload["passage"])
        ),
        duplicate_diagnostics=_decode_diagnostics(payload["duplicate_diagnostics"]),
    )


def _validate_limitations(value: JsonValue) -> tuple[str, ...]:
    """Decode a unique ordered limitation list."""
    limitations = tuple(_string(item) for item in _array(value))
    if any(not limitation.strip() for limitation in limitations):
        raise ValueError("limitations must not contain blank values.")
    if len(limitations) != len(set(limitations)):
        raise ValueError("limitations must be unique.")
    return limitations


def retrieval_result_digest(run: RetrievalEvaluationRunResult) -> str:
    """Return the canonical digest of ordered case results and aggregate."""
    if not isinstance(run, RetrievalEvaluationRunResult):
        raise TypeError("run must be a RetrievalEvaluationRunResult.")
    payload = {
        "run_fingerprint": run.identity.run_fingerprint,
        "case_results": [result.as_dict() for result in run.case_results],
        "aggregate": run.aggregate.as_dict(),
    }
    return (
        "sha256:"
        + hashlib.sha256(canonical_json_bytes(cast(JsonObject, payload))).hexdigest()
    )


def serialise_run_artifact(
    identity: RetrievalEvaluationRunIdentity,
    result_digest: str,
    limitations: tuple[str, ...] = (),
) -> bytes:
    """Return canonical UTF-8 bytes for `run.json`."""
    artifact = RetrievalEvaluationRunArtifact(identity, result_digest, limitations)
    payload = {
        "identity": artifact.identity.as_dict(),
        "result_digest": artifact.result_digest,
        "limitations": list(artifact.limitations),
    }
    return canonical_json_bytes(cast(JsonObject, payload))


def deserialise_run_artifact(data: bytes) -> RetrievalEvaluationRunArtifact:
    """Decode and strictly validate canonical `run.json` bytes."""
    payload = _canonical_object(data)
    require_exact_json_fields(
        payload,
        frozenset({"identity", "result_digest", "limitations"}),
    )
    artifact = RetrievalEvaluationRunArtifact(
        identity=_decode_run_identity(_object(payload["identity"])),
        result_digest=_string(payload["result_digest"]),
        limitations=_validate_limitations(payload["limitations"]),
    )
    if (
        serialise_run_artifact(
            artifact.identity,
            artifact.result_digest,
            artifact.limitations,
        )
        != data
    ):
        raise ValueError("run artifact is not canonical.")
    return artifact


def serialise_case_results(run: RetrievalEvaluationRunResult) -> bytes:
    """Return canonical JSONL bytes for `cases.jsonl`."""
    if not isinstance(run, RetrievalEvaluationRunResult):
        raise TypeError("run must be a RetrievalEvaluationRunResult.")
    lines = []
    for result in run.case_results:
        payload = dict(result.as_dict())
        payload["run_fingerprint"] = run.identity.run_fingerprint
        lines.append(
            canonical_json_bytes(cast(JsonObject, payload)).decode("utf-8").rstrip("\n")
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def deserialise_case_results(
    data: bytes,
    identity: RetrievalEvaluationRunIdentity,
) -> tuple[RetrievalCaseResult, ...]:
    """Decode and strictly validate canonical `cases.jsonl` bytes."""
    if not isinstance(identity, RetrievalEvaluationRunIdentity):
        raise TypeError("identity must be a RetrievalEvaluationRunIdentity.")
    if (
        not isinstance(data, bytes)
        or not data.endswith(b"\n")
        or data.endswith(b"\n\n")
    ):
        raise ValueError("cases JSONL must have exactly one trailing newline.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("cases JSONL must contain valid UTF-8.") from None
    lines = text[:-1].split("\n")
    if not lines or any(not line for line in lines):
        raise ValueError("cases JSONL must not contain blank records.")
    results: list[RetrievalCaseResult] = []
    expected_fields = frozenset(
        {
            "run_fingerprint",
            "case_id",
            "case_ordinal",
            "split",
            "corpus_id",
            "corpus_version",
            "requested_limit",
            "status",
            "hits",
            "failures",
            "document_recall",
            "passage_recall",
            "document_match_ranks",
            "passage_match_ranks",
            "document_depth_exhausted",
            "duplicate_diagnostics",
        }
    )
    for line in lines:
        payload = load_json_object(line.encode("utf-8"))
        if canonical_json_bytes(payload).decode("utf-8").rstrip("\n") != line:
            raise ValueError("cases JSONL contains a non-canonical record.")
        require_exact_json_fields(payload, expected_fields)
        if _string(payload["run_fingerprint"]) != identity.run_fingerprint:
            raise ValueError("cases JSONL run identity does not match.")
        result_payload = dict(payload)
        result_payload.pop("run_fingerprint")
        results.append(_decode_case_result(cast(JsonObject, result_payload)))
    decoded = tuple(results)
    if tuple(result.case_id for result in decoded) != identity.selected_case_ids:
        raise ValueError("cases JSONL case order does not match the run identity.")
    if any(result.case_ordinal != ordinal for ordinal, result in enumerate(decoded)):
        raise ValueError("cases JSONL ordinals are not canonical.")
    if any(
        result.corpus_id != identity.corpus_id
        or result.corpus_version != identity.corpus_version
        or result.split != identity.selected_split
        or result.requested_limit != identity.policy.candidate_depth
        for result in decoded
    ):
        raise ValueError("cases JSONL case identity does not match the run identity.")
    return decoded


def serialise_retrieval_report(report: RetrievalEvaluationReport) -> bytes:
    """Return canonical UTF-8 bytes for `report.json`."""
    if not isinstance(report, RetrievalEvaluationReport):
        raise TypeError("report must be a RetrievalEvaluationReport.")
    return canonical_json_bytes(cast(JsonObject, report.as_dict()))


def deserialise_retrieval_report(data: bytes) -> RetrievalEvaluationReport:
    """Decode and strictly validate canonical `report.json` bytes."""
    payload = _canonical_object(data)
    require_exact_json_fields(
        payload,
        frozenset({"run_fingerprint", "result_digest", "aggregate", "limitations"}),
    )
    report = RetrievalEvaluationReport(
        run_fingerprint=_string(payload["run_fingerprint"]),
        result_digest=_string(payload["result_digest"]),
        aggregate=_decode_aggregate(_object(payload["aggregate"])),
        limitations=_validate_limitations(payload["limitations"]),
    )
    if serialise_retrieval_report(report) != data:
        raise ValueError("retrieval report is not canonical.")
    return report


def serialise_retrieval_manifest(
    manifest: PublishedRetrievalEvaluationManifest,
) -> bytes:
    """Return canonical UTF-8 bytes for `manifest.json`."""
    if not isinstance(manifest, PublishedRetrievalEvaluationManifest):
        raise TypeError("manifest must be a PublishedRetrievalEvaluationManifest.")
    return canonical_json_bytes(cast(JsonObject, manifest.as_dict()))


def deserialise_retrieval_manifest(
    data: bytes,
) -> PublishedRetrievalEvaluationManifest:
    """Decode and strictly validate canonical `manifest.json` bytes."""
    payload = _canonical_object(data)
    require_exact_json_fields(
        payload,
        frozenset({"schema_version", "run_fingerprint", "result_digest", "artefacts"}),
    )
    artefacts = []
    for item in _array(payload["artefacts"]):
        artefact = _object(item)
        require_exact_json_fields(
            artefact, frozenset({"name", "size_bytes", "checksum"})
        )
        artefacts.append(
            PublishedArtifact(
                name=_string(artefact["name"]),
                size_bytes=_integer(artefact["size_bytes"]),
                checksum=_string(artefact["checksum"]),
            )
        )
    manifest = PublishedRetrievalEvaluationManifest(
        schema_version=_string(payload["schema_version"]),
        run_fingerprint=_string(payload["run_fingerprint"]),
        result_digest=_string(payload["result_digest"]),
        artefacts=tuple(artefacts),
    )
    if serialise_retrieval_manifest(manifest) != data:
        raise ValueError("retrieval manifest is not canonical.")
    return manifest
