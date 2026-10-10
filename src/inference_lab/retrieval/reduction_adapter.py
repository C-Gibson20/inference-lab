"""Snapshot-pinned evidence reduction with exact lexical source mapping."""

import time
from enum import StrEnum
from typing import Literal

import structlog

from inference_lab.ingest import (
    NormalisedText,
    build_canonical_source_boundary_map,
)
from inference_lab.provenance import CorpusId, CorpusVersionId, RecordId
from inference_lab.retrieval.evidence import EvidenceSource
from inference_lab.retrieval.lexical import LEXICAL_COMPONENT_KIND
from inference_lab.retrieval.lexical_store import (
    LexicalGeneration,
    LexicalIndexedChunk,
)
from inference_lab.retrieval.metadata_adapter import resolve_lexical_candidate_metadata
from inference_lab.retrieval.reduction import (
    EvidenceReductionOutcome,
    EvidenceReductionRequest,
    EvidenceTextSourceRange,
    EvidenceTokeniser,
    EvidenceTokeniserError,
    EvidenceTokeniserIdentity,
    TextSourceBoundary,
    reduce_evidence,
)
from inference_lab.retrieval.results import EvidenceResult
from inference_lab.settings import RetrievalConfig
from inference_lab.storage import ComponentGenerationState, CorpusSnapshot

_LOGGER = structlog.get_logger("inference_lab.retrieval")


class EvidenceReductionFailureCategory(StrEnum):
    """Bounded reasons the evidence-reduction boundary could not complete."""

    CORPUS_VERSION_MISMATCH = "corpus_version_mismatch"
    TOKENISER_MISMATCH = "tokeniser_mismatch"
    TOKENISER_UNAVAILABLE = "tokeniser_unavailable"
    INVALID_TOKENISER_OUTPUT = "invalid_tokeniser_output"
    REDUCTION_FAILURE = "reduction_failure"


class EvidenceReductionExecutionError(RuntimeError):
    """A privacy-safe failure raised by the evidence-reduction execution boundary."""

    def __init__(self, category: EvidenceReductionFailureCategory) -> None:
        self.category = category
        super().__init__(category.value)


def reduce_retrieval_evidence(
    corpus_id: CorpusId,
    snapshot: CorpusSnapshot,
    generation: LexicalGeneration,
    result: EvidenceResult,
    tokeniser: EvidenceTokeniser,
    configuration: RetrievalConfig,
    *,
    token_budget_override: int | None = None,
    item_budget_override: int | None = None,
    diversity_policy_override: Literal["none", "per_document"] | None = None,
    expected_tokeniser_identity: EvidenceTokeniserIdentity | None = None,
) -> EvidenceReductionOutcome:
    """Reduce one reranked result against its exact pinned corpus snapshot."""
    started = time.perf_counter()
    corpus_version: CorpusVersionId | None = None
    try:
        if not isinstance(corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if any(
            not callable(getattr(snapshot, method_name, None))
            for method_name in (
                "get_source_artefact",
                "get_parsed_representation",
                "resolve_source_span",
                "get_document_version",
            )
        ):
            raise TypeError("snapshot must implement CorpusSnapshot.")
        if (
            not hasattr(snapshot, "corpus_id")
            or not hasattr(snapshot, "version")
            or not hasattr(snapshot, "manifest")
        ):
            raise TypeError("snapshot must expose its selected corpus version.")
        if not isinstance(generation, LexicalGeneration):
            raise TypeError("generation must be a LexicalGeneration.")
        if not isinstance(result, EvidenceResult):
            raise TypeError("result must be an EvidenceResult.")
        if not isinstance(configuration, RetrievalConfig):
            raise TypeError("configuration must be a RetrievalConfig.")
        corpus_version = snapshot.version.corpus_version
        if (
            snapshot.corpus_id != corpus_id
            or result.corpus_version != corpus_version
            or generation.manifest.corpus_id != corpus_id
            or generation.manifest.corpus_version != corpus_version
        ):
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.CORPUS_VERSION_MISMATCH
            )
        _validate_result_source_versions(snapshot, result)
        references = tuple(
            reference
            for reference in snapshot.manifest.component_generations
            if reference.component_kind.as_str() == LEXICAL_COMPONENT_KIND
        )
        if len(references) != 1:
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.CORPUS_VERSION_MISMATCH
            )
        reference = references[0]
        manifest = generation.manifest
        if (
            reference.state is not ComponentGenerationState.READY
            or reference.corpus_id != corpus_id
            or reference.corpus_version != corpus_version
            or reference.generation_id != manifest.generation_id
            or reference.implementation != manifest.implementation
            or reference.format_version != manifest.format_version
            or reference.contract_version != manifest.contract_version
            or reference.configuration_digest != manifest.configuration_digest
            or reference.source_manifest_digest != manifest.source_manifest_digest
        ):
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.CORPUS_VERSION_MISMATCH
            )
        if not callable(getattr(tokeniser, "tokenise", None)) or not callable(
            getattr(tokeniser, "count_tokens", None)
        ):
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.TOKENISER_UNAVAILABLE
            )
        if expected_tokeniser_identity is not None and not isinstance(
            expected_tokeniser_identity, EvidenceTokeniserIdentity
        ):
            raise TypeError(
                "expected_tokeniser_identity must be an EvidenceTokeniserIdentity."
            )
        try:
            identity = tokeniser.identity
        except Exception:
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.TOKENISER_UNAVAILABLE
            ) from None
        if not isinstance(identity, EvidenceTokeniserIdentity):
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.INVALID_TOKENISER_OUTPUT
            )
        if (
            expected_tokeniser_identity is not None
            and identity != expected_tokeniser_identity
        ):
            raise EvidenceReductionExecutionError(
                EvidenceReductionFailureCategory.TOKENISER_MISMATCH
            )

        if token_budget_override is not None and (
            type(token_budget_override) is not int or token_budget_override <= 0
        ):
            raise ValueError("token_budget_override must be a positive integer.")

        token_budget = (
            configuration.evidence_token_budget
            if token_budget_override is None
            else token_budget_override
        )
        item_budget = min(result.requested_limit.value, configuration.evidence_count)
        if item_budget_override is not None:
            if type(item_budget_override) is not int or item_budget_override <= 0:
                raise ValueError("item_budget_override must be a positive integer.")
            item_budget = min(item_budget, item_budget_override)
        diversity_policy = (
            configuration.diversity_policy
            if diversity_policy_override is None
            else diversity_policy_override
        )
        if not isinstance(diversity_policy, str):
            raise TypeError("diversity policy override must be a string.")
        if diversity_policy not in {"none", "per_document"}:
            raise ValueError("unsupported diversity policy.")
        source_ranges: dict[RecordId, tuple[EvidenceTextSourceRange, ...]] = {}
        for hit in result.hits[:item_budget]:
            mapped = _source_ranges_for_candidate(
                snapshot,
                generation,
                hit.candidate.record_id,
                hit.candidate.content,
                hit.candidate.sources,
            )
            if mapped:
                source_ranges[hit.candidate.record_id] = mapped
        request = EvidenceReductionRequest(
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            result=result,
            tokeniser=tokeniser,
            token_budget=token_budget,
            item_budget=item_budget,
            source_ranges=source_ranges,
            diversity_policy=diversity_policy,
        )
        outcome = reduce_evidence(request)
        _log_outcome(outcome, started=started)
        return outcome
    except EvidenceReductionExecutionError as error:
        _log_failure(
            corpus_id,
            corpus_version,
            error.category,
            started=started,
        )
        raise
    except EvidenceTokeniserError as error:
        category = (
            EvidenceReductionFailureCategory.INVALID_TOKENISER_OUTPUT
            if error.category
            in {
                "invalid_tokeniser_output",
                "invalid_token_offsets",
                "inconsistent_token_count",
                "unmeasurable_text",
            }
            else EvidenceReductionFailureCategory.TOKENISER_UNAVAILABLE
        )
        _log_failure(corpus_id, corpus_version, category, started=started)
        raise EvidenceReductionExecutionError(category) from None
    except TypeError:
        raise
    except Exception:
        category = EvidenceReductionFailureCategory.REDUCTION_FAILURE
        _log_failure(corpus_id, corpus_version, category, started=started)
        raise EvidenceReductionExecutionError(category) from None


def _source_ranges_for_candidate(
    snapshot: CorpusSnapshot,
    generation: LexicalGeneration,
    record_id: RecordId,
    content: str,
    sources: tuple[EvidenceSource, ...],
) -> tuple[EvidenceTextSourceRange, ...] | None:
    """Build exact per-span text mappings, returning None when proof fails."""
    try:
        chunk = generation.get_chunk(record_id)
        if chunk is None or not _candidate_matches_chunk(chunk, content, sources):
            return None
        metadata = resolve_lexical_candidate_metadata(snapshot, chunk)
        boundary_map = build_canonical_source_boundary_map(
            NormalisedText(source_text=metadata.parsed_representation.source_text)
        )
        chunk_canonical_start = boundary_map.canonical_offset_for_source(
            chunk.source_start
        )
        chunk_canonical_end = boundary_map.canonical_offset_for_source(chunk.source_end)
        if (
            chunk_canonical_end - chunk_canonical_start != len(content)
            or boundary_map.canonical_text[chunk_canonical_start:chunk_canonical_end]
            != content
        ):
            return None

        source_by_span = {source.source_span_id: source for source in sources}
        location_by_span = {
            location.source_span_id: location
            for location in metadata.parsed_representation.source_locations
        }
        ranges: list[EvidenceTextSourceRange] = []
        for link in chunk.source_links:
            source = source_by_span.get(link.source_span_id)
            location = location_by_span.get(link.source_span_id)
            if (
                source is None
                or location is None
                or link.source_start < location.start_offset
                or link.source_end > location.end_offset
            ):
                return None
            canonical_start = boundary_map.canonical_offset_for_source(
                link.source_start
            )
            canonical_end = boundary_map.canonical_offset_for_source(link.source_end)
            if not (
                chunk_canonical_start
                <= canonical_start
                < canonical_end
                <= chunk_canonical_end
            ):
                return None
            boundaries = tuple(
                TextSourceBoundary(
                    text_offset=boundary.canonical_offset - chunk_canonical_start,
                    source_offset=boundary.source_offset,
                )
                for boundary in boundary_map.boundaries
                if link.source_start <= boundary.source_offset <= link.source_end
                and canonical_start <= boundary.canonical_offset <= canonical_end
            )
            ranges.append(
                EvidenceTextSourceRange(
                    source=source,
                    text_start=canonical_start - chunk_canonical_start,
                    text_end=canonical_end - chunk_canonical_start,
                    source_start=link.source_start,
                    source_end=link.source_end,
                    boundaries=boundaries,
                )
            )
        return tuple(sorted(ranges, key=lambda item: item.text_start))
    except Exception:
        return None


def _candidate_matches_chunk(
    chunk: LexicalIndexedChunk,
    content: str,
    sources: tuple[EvidenceSource, ...],
) -> bool:
    """Ensure the result candidate is the same immutable indexed chunk."""
    expected_sources = tuple(
        EvidenceSource(chunk.document_id, link.document_version_id, link.source_span_id)
        for link in chunk.source_links
    )
    return chunk.content == content and set(expected_sources) == set(sources)


def _validate_result_source_versions(
    snapshot: CorpusSnapshot,
    result: EvidenceResult,
) -> None:
    """Reject any candidate source that is outside the selected snapshot."""
    for hit in result.hits:
        for source in hit.candidate.sources:
            version = snapshot.get_document_version(source.document_version_id)
            if (
                version is None
                or version.document_id != source.document_id
                or version.document_version_id != source.document_version_id
            ):
                raise EvidenceReductionExecutionError(
                    EvidenceReductionFailureCategory.CORPUS_VERSION_MISMATCH
                )


def _log_outcome(
    outcome: EvidenceReductionOutcome,
    *,
    started: float,
) -> None:
    """Emit bounded completion and explicit truncation records."""
    identity = outcome.identity
    for item in outcome.items:
        if item.action.value == "truncated":
            _LOGGER.warning(
                "evidence_truncated",
                module="inference_lab.retrieval",
                pipeline_stage="retrieval_context_reduction",
                corpus_id=outcome.corpus_id.as_str(),
                corpus_version=outcome.corpus_version.as_str(),
                record_id=item.hit.candidate.record_id.as_str(),
                retained_range_count=len(item.source_ranges),
                token_budget=identity.token_budget,
                policy_version=identity.policy_version,
                identity_digest=identity.digest,
            )
    degraded = bool(outcome.failures) or outcome.source_status.value != "completed"
    log = _LOGGER.warning if degraded else _LOGGER.info
    log(
        "evidence_reduction_completed",
        module="inference_lab.retrieval",
        pipeline_stage="retrieval_context_reduction",
        corpus_id=outcome.corpus_id.as_str(),
        corpus_version=outcome.corpus_version.as_str(),
        status="degraded" if degraded else "completed",
        degraded=degraded,
        source_status=outcome.source_status.value,
        source_failure_count=len(outcome.source_failures),
        source_failure_categories=tuple(
            sorted({failure.category.value for failure in outcome.source_failures})
        ),
        failure_categories=tuple(
            sorted({failure.category for failure in outcome.failures})
        ),
        failure_count=len(outcome.failures),
        input_count=outcome.input_count,
        retained_count=len(outcome.items),
        dropped_count=outcome.dropped_count,
        truncated_count=sum(item.action.value == "truncated" for item in outcome.items),
        token_count=outcome.token_count,
        token_budget=identity.token_budget,
        item_budget=identity.item_budget,
        diversity_policy=identity.diversity_policy,
        tokeniser_implementation=identity.tokeniser.implementation,
        tokeniser_revision=identity.tokeniser.revision,
        tokeniser_contract_version=identity.tokeniser.contract_version,
        policy_version=identity.policy_version,
        accounting_version=identity.accounting_version,
        identity_digest=identity.digest,
        duration_ms=max(0.0, (time.perf_counter() - started) * 1000),
    )


def _log_failure(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId | None,
    category: EvidenceReductionFailureCategory,
    *,
    started: float,
) -> None:
    """Emit one bounded error record without exception or payload details."""
    _LOGGER.error(
        "evidence_reduction_failed",
        module="inference_lab.retrieval",
        pipeline_stage="retrieval_context_reduction",
        corpus_id=corpus_id.as_str() if isinstance(corpus_id, CorpusId) else None,
        corpus_version=(
            corpus_version.as_str()
            if isinstance(corpus_version, CorpusVersionId)
            else None
        ),
        failure_category=category.value,
        duration_ms=max(0.0, (time.perf_counter() - started) * 1000),
    )
