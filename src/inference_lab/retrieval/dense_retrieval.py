"""Pinned-snapshot execution for the dense retrieval channel."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType
from typing import cast

from inference_lab.backends.runtime import (
    EmbeddingBackend,
    EmbeddingOptions,
    EmbeddingRequest,
)
from inference_lab.provenance import CorpusId
from inference_lab.retrieval.dense import (
    DENSE_COMPONENT_KIND,
    DENSE_SCORE_LABEL,
    DenseIndexMatch,
    DenseVector,
    InvalidDenseContractError,
    rank_dense_entries,
)
from inference_lab.retrieval.dense_store import (
    DenseError,
    DenseFailureCategory,
    DenseGeneration,
    open_dense_generation,
)
from inference_lab.retrieval.evidence import (
    ChannelScore,
    EvidenceHit,
    RetrievalCandidate,
)
from inference_lab.retrieval.metadata_filters import (
    InvalidMetadataFilterError,
    MetadataFilterValidation,
    MetadataValues,
    evaluate_metadata_filter,
    validate_metadata_filter,
)
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.values import RetrievalChannel
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    CorpusSnapshot,
    CorruptStorageError,
    MetadataRepository,
    StorageError,
)


def _failed_result(
    query: RetrievalQuery,
    category: FailureCategory,
    *,
    retryable: bool = False,
) -> EvidenceResult:
    """Return one payload-free failed dense result."""
    return EvidenceResult.from_query(
        query,
        observed_corpus_version=query.corpus_version,
        hits=(),
        status=ResultStatus.FAILED,
        failures=(
            RetrievalFailure(
                category=category,
                retryable=retryable,
                channel=RetrievalChannel.DENSE,
            ),
        ),
    )


def _map_dense_failure(error: DenseError) -> FailureCategory:
    """Map the dense store's bounded failures into RET-01's result vocabulary."""
    if error.category in {
        DenseFailureCategory.GENERATION_UNAVAILABLE,
        DenseFailureCategory.CORRUPT_GENERATION,
    }:
        return FailureCategory.CHANNEL_UNAVAILABLE
    if error.category in {
        DenseFailureCategory.INCOMPATIBLE_GENERATION,
        DenseFailureCategory.INVALID_REQUEST,
    }:
        return FailureCategory.INVALID_CHANNEL_RESPONSE
    return FailureCategory.CHANNEL_EXECUTION_FAILURE


def execute_dense_retrieval(
    repository: MetadataRepository,
    index_dir: Path,
    corpus_id: CorpusId,
    query: RetrievalQuery,
    embedder: EmbeddingBackend,
    *,
    snapshot: CorpusSnapshot | None = None,
    generation: DenseGeneration | None = None,
) -> EvidenceResult:
    """Execute one dense query against a single pinned corpus snapshot.

    Supplied snapshot and generation handles are borrowed and remain owned by
    their caller.  Filters are evaluated in exact dense rank order before the
    requested result limit is applied.
    """
    if not callable(getattr(repository, "open_corpus_snapshot", None)):
        raise TypeError("repository must provide corpus snapshots.")
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(query, RetrievalQuery):
        raise TypeError("query must be a RetrievalQuery.")
    if not callable(getattr(embedder, "embed", None)):
        raise TypeError("embedder must implement EmbeddingBackend.")
    if snapshot is not None and not _is_snapshot_reader(snapshot):
        raise TypeError("snapshot must provide corpus snapshot reads or be None.")
    if generation is not None and not isinstance(generation, DenseGeneration):
        raise TypeError("generation must be a DenseGeneration or None.")

    validation = (
        None if query.filter is None else validate_metadata_filter(query.filter)
    )
    try:
        snapshot_context = (
            nullcontext(snapshot)
            if snapshot is not None
            else repository.open_corpus_snapshot(corpus_id, query.corpus_version)
        )
        with snapshot_context as selected_snapshot:
            if selected_snapshot is None:
                return _failed_result(query, FailureCategory.CHANNEL_UNAVAILABLE)
            if selected_snapshot.corpus_id != corpus_id:
                return _failed_result(query, FailureCategory.INVALID_CHANNEL_RESPONSE)
            if selected_snapshot.version.corpus_version != query.corpus_version:
                return EvidenceResult.from_query(
                    query,
                    observed_corpus_version=selected_snapshot.version.corpus_version,
                    hits=(),
                    status=ResultStatus.FAILED,
                    failures=(),
                )

            references = tuple(
                reference
                for reference in selected_snapshot.manifest.component_generations
                if reference.component_kind == DENSE_COMPONENT_KIND
            )
            if len(references) != 1:
                return _failed_result(query, FailureCategory.INVALID_CHANNEL_RESPONSE)
            reference = references[0]
            if (
                reference.state is not ComponentGenerationState.READY
                or reference.corpus_id != corpus_id
                or reference.corpus_version != query.corpus_version
            ):
                return _failed_result(query, FailureCategory.INVALID_CHANNEL_RESPONSE)

            generation_context = (
                nullcontext(generation)
                if generation is not None
                else open_dense_generation(index_dir, reference.generation_id)
            )
            with generation_context as selected_generation:
                if selected_generation is None or not _generation_matches_reference(
                    selected_generation,
                    reference,
                    corpus_id,
                    query,
                ):
                    return _failed_result(
                        query,
                        FailureCategory.INVALID_CHANNEL_RESPONSE,
                    )

                if selected_generation.manifest.entry_count == 0:
                    return _completed_result(query, selected_snapshot, ())

                try:
                    response = embedder.embed(
                        EmbeddingRequest(
                            input_texts=(query.text,),
                            options=EmbeddingOptions(
                                dimensions=(
                                    selected_generation.manifest.compatibility.dimensions
                                ),
                                normalise=(
                                    selected_generation.manifest.compatibility.embedding_profile.l2_normalised
                                ),
                            ),
                            purpose="query",
                        )
                    )
                except Exception:
                    return _failed_result(
                        query,
                        FailureCategory.CHANNEL_EXECUTION_FAILURE,
                        retryable=True,
                    )

                compatibility = selected_generation.manifest.compatibility
                if (
                    response.model_metadata != compatibility.model_metadata
                    or response.embedding_profile != compatibility.embedding_profile
                    or len(response.vectors) != 1
                ):
                    return _failed_result(
                        query,
                        FailureCategory.INVALID_CHANNEL_RESPONSE,
                    )
                try:
                    query_vector = DenseVector(tuple(response.vectors[0]))
                except (TypeError, ValueError, InvalidDenseContractError):
                    return _failed_result(
                        query,
                        FailureCategory.INVALID_CHANNEL_RESPONSE,
                    )
                if len(query_vector.values) != compatibility.dimensions:
                    return _failed_result(
                        query,
                        FailureCategory.INVALID_CHANNEL_RESPONSE,
                    )

                entries = selected_generation.iter_entries()
                if not entries:
                    return _completed_result(query, selected_snapshot, ())
                ranking_limit = (
                    len(entries)
                    if validation is not None
                    else min(query.limit.value, len(entries))
                )
                matches = rank_dense_entries(
                    entries,
                    query_vector,
                    limit=ranking_limit,
                )
                hits = _build_hits(
                    query,
                    selected_snapshot,
                    matches,
                    validation,
                )
                return _completed_result(query, selected_snapshot, hits)
    except InvalidMetadataFilterError:
        return _failed_result(query, FailureCategory.INVALID_CHANNEL_RESPONSE)
    except DenseError as error:
        return _failed_result(query, _map_dense_failure(error))
    except StorageError as error:
        category = (
            FailureCategory.CHANNEL_EXECUTION_FAILURE
            if validation is not None and isinstance(error, CorruptStorageError)
            else FailureCategory.CHANNEL_UNAVAILABLE
        )
        return _failed_result(query, category)
    except (AttributeError, OSError, TypeError, ValueError, KeyError, IndexError):
        return _failed_result(query, FailureCategory.CHANNEL_EXECUTION_FAILURE)


def _is_snapshot_reader(snapshot: CorpusSnapshot) -> bool:
    """Check the narrow snapshot surface used by dense result construction."""
    return all(
        callable(getattr(snapshot, method_name, None))
        for method_name in (
            "get_document_version",
            "get_source_artefact",
            "get_parsed_representation",
            "resolve_source_span",
        )
    ) and all(
        hasattr(snapshot, property_name)
        for property_name in ("corpus_id", "version", "manifest")
    )


def _generation_matches_reference(
    generation: DenseGeneration,
    reference: ComponentGenerationReference,
    corpus_id: CorpusId,
    query: RetrievalQuery,
) -> bool:
    """Require the opened immutable generation to match its ready reference."""
    return (
        generation.manifest.corpus_id == corpus_id
        and generation.manifest.corpus_version == query.corpus_version
        and generation.manifest.generation_id == reference.generation_id
        and generation.manifest.compatibility.configuration.implementation
        == reference.implementation
        and generation.manifest.compatibility.configuration.format_version
        == reference.format_version
        and generation.manifest.compatibility.configuration.contract_version
        == reference.contract_version
        and generation.manifest.configuration_digest == reference.configuration_digest
        and generation.manifest.source_manifest_digest
        == reference.source_manifest_digest
    )


def _build_hits(
    query: RetrievalQuery,
    snapshot: CorpusSnapshot,
    matches: tuple[DenseIndexMatch, ...],
    validation: MetadataFilterValidation | None,
) -> tuple[EvidenceHit, ...]:
    """Filter ranked candidates through the pinned snapshot and build RET-01 hits."""
    hits: list[EvidenceHit] = []
    for match in matches:
        _validate_candidate_snapshot_links(snapshot, match.candidate)
        if validation is not None:
            values = _candidate_metadata_values(snapshot, match.candidate)
            if not evaluate_metadata_filter(validation, values):
                continue
        hits.append(
            EvidenceHit(
                candidate=match.candidate,
                contributions=(
                    ChannelScore(
                        channel=RetrievalChannel.DENSE,
                        rank=len(hits) + 1,
                        raw_score=match.cosine_similarity,
                        score_label=DENSE_SCORE_LABEL,
                    ),
                ),
                rank=len(hits) + 1,
            )
        )
        if len(hits) == query.limit.value:
            break
    return tuple(hits)


def _validate_candidate_snapshot_links(
    snapshot: CorpusSnapshot,
    candidate: RetrievalCandidate,
) -> None:
    """Reject any candidate whose complete source lineage is outside the snapshot."""
    for source in candidate.sources:
        document_version = snapshot.get_document_version(source.document_version_id)
        if (
            document_version is None
            or document_version.document_id != source.document_id
        ):
            raise CorruptStorageError()
        resolved = snapshot.resolve_source_span(source.source_span_id)
        if (
            resolved is None
            or resolved.document_id != source.document_id
            or resolved.location.document_version_id != source.document_version_id
            or resolved.location.source_span_id != source.source_span_id
        ):
            raise CorruptStorageError()


def _candidate_metadata_values(
    snapshot: CorpusSnapshot,
    candidate: RetrievalCandidate,
) -> MetadataValues:
    """Resolve RET-03 metadata from canonical records visible in the snapshot."""
    source = candidate.sources[0]
    source_artefact = snapshot.get_source_artefact(source.document_version_id)
    if source_artefact is None or (
        source_artefact.document_id != source.document_id
        or source_artefact.document_version_id != source.document_version_id
    ):
        raise CorruptStorageError()
    source_spans = tuple(
        snapshot.resolve_source_span(item.source_span_id) for item in candidate.sources
    )
    if any(span is None for span in source_spans):
        raise CorruptStorageError()
    parsed_record_ids = {
        span.location.parsed_record_id for span in source_spans if span is not None
    }
    if len(parsed_record_ids) != 1:
        raise CorruptStorageError()
    parsed_record_id = next(iter(parsed_record_ids))
    parsed = snapshot.get_parsed_representation(parsed_record_id)
    if parsed is None or (
        parsed.document_id != source.document_id
        or parsed.document_version_id != source.document_version_id
        or parsed.record_id != parsed_record_id
    ):
        raise CorruptStorageError()
    return cast(
        MetadataValues,
        MappingProxyType(
            {
                "document-type": source_artefact.media_type,
                "source-byte-count": source_artefact.byte_count,
                "language": parsed.language,
                "page-count": parsed.page_count,
                "character-count": parsed.character_count,
                "token-count": parsed.token_count,
            }
        ),
    )


def _completed_result(
    query: RetrievalQuery,
    snapshot: CorpusSnapshot,
    hits: tuple[EvidenceHit, ...],
) -> EvidenceResult:
    """Return a valid completed result, including an empty result."""
    return EvidenceResult.from_query(
        query,
        observed_corpus_version=snapshot.version.corpus_version,
        hits=hits,
        status=ResultStatus.COMPLETED,
        failures=(),
    )
