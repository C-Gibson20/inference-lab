"""Pure construction and outcome orchestration for ingest chunks."""

from typing import cast

from inference_lab.ingest._chunk_identity import (
    _CHUNK_RECORD_KIND,
    _chunk_operation,
    _chunk_output_digest_for_values,
    _configuration_identity,
    _record_id,
    _source_inputs,
)
from inference_lab.ingest._chunk_records import (
    Chunk,
    ChunkingContractError,
    ChunkingFailure,
    ChunkingFailureCategory,
    ChunkingOutcome,
    ChunkMinimumException,
    ChunkSet,
    ChunkSourceReference,
    ChunkSplitReason,
)
from inference_lab.ingest._chunk_windows import (
    ChunkDraft,
    ChunkWindowError,
    build_chunk_drafts,
)
from inference_lab.ingest.chunking import CURRENT_CHUNKER_VERSION, ChunkingRequest
from inference_lab.ingest.parser import ParsedDocument
from inference_lab.ingest.sources import StructuralBlockKind
from inference_lab.ingest.tokenisation import (
    ChunkTokeniser,
    ChunkTokeniserError,
    validate_chunk_tokeniser,
)
from inference_lab.provenance import (
    CURRENT_LINEAGE_SCHEMA_VERSION,
    ConfigurationIdentity,
    DerivedRecordMetadata,
    InvalidLineageError,
    KnownSourceSpan,
    LineageBundle,
    build_derivation_fingerprint,
)


def build_chunk_set(request: ChunkingRequest) -> ChunkSet:
    """Build immutable provenance-backed chunks from one validated request."""
    if not isinstance(request, ChunkingRequest):
        raise TypeError("request must be a ChunkingRequest.")
    tokeniser = validate_chunk_tokeniser(cast(ChunkTokeniser, request.tokeniser))
    drafts = build_chunk_drafts(request)
    configuration_identity = _configuration_identity(request, tokeniser.identity)
    chunks = tuple(
        _chunk_from_draft(
            request=request,
            tokeniser=tokeniser,
            configuration_identity=configuration_identity,
            draft=draft,
        )
        for draft in drafts
    )
    source_spans = tuple(
        sorted(
            {
                KnownSourceSpan(
                    document_version_id=reference.document_version_id,
                    source_span_id=reference.source_span_id,
                )
                for chunk in chunks
                for reference in chunk.source_references
            },
            key=lambda source: source.key(),
        )
    )
    lineage = LineageBundle.from_records(
        source_spans=source_spans,
        records=tuple(chunk.metadata for chunk in chunks),
    )
    return ChunkSet(
        corpus_id=request.corpus_id,
        document_id=request.parsed_document.document_id,
        document_version_id=request.parsed_document.document_version_id,
        parser_version=request.parsed_document.parser_version,
        chunker_version=CURRENT_CHUNKER_VERSION,
        tokeniser_identity=tokeniser.identity,
        policy=request.policy,
        configuration_identity=configuration_identity,
        chunks=chunks,
        lineage=lineage,
    )


def build_chunking_outcome(request: ChunkingRequest) -> ChunkingOutcome:
    """Return a complete success or a bounded failure for one request."""
    if not isinstance(request, ChunkingRequest):
        raise TypeError("request must be a ChunkingRequest.")
    try:
        return ChunkingOutcome(
            corpus_id=request.corpus_id,
            document_id=request.parsed_document.document_id,
            document_version_id=request.parsed_document.document_version_id,
            chunk_set=build_chunk_set(request),
        )
    except ChunkWindowError as error:
        return _failure_outcome(request, _window_failure_category(error.category))
    except ChunkTokeniserError as error:
        return _failure_outcome(request, _tokeniser_failure_category(error.category))
    except ChunkingContractError as error:
        return _failure_outcome(request, error.category)
    except InvalidLineageError:
        return _failure_outcome(request, ChunkingFailureCategory.INVALID_LINEAGE)


def _failure_outcome(
    request: ChunkingRequest,
    category: ChunkingFailureCategory,
) -> ChunkingOutcome:
    """Build a payload-free failed outcome from a valid request."""
    return ChunkingOutcome(
        corpus_id=request.corpus_id,
        document_id=request.parsed_document.document_id,
        document_version_id=request.parsed_document.document_version_id,
        failure=ChunkingFailure(category),
    )


def _chunk_from_draft(
    *,
    request: ChunkingRequest,
    tokeniser: ChunkTokeniser,
    configuration_identity: ConfigurationIdentity,
    draft: object,
) -> Chunk:
    """Translate one private draft into a fully lineaged public chunk."""
    if not isinstance(draft, ChunkDraft):
        raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT)
    document = request.parsed_document
    references = _references_for_draft(document, draft)
    if not references:
        raise ChunkingContractError(ChunkingFailureCategory.INVALID_LINEAGE)
    chunker_version = CURRENT_CHUNKER_VERSION
    operation = _chunk_operation(
        chunker_version=chunker_version,
        tokeniser_identity=tokeniser.identity,
        configuration_identity=configuration_identity,
    )
    inputs = _source_inputs(document.document_version_id, references)
    payload_values = {
        "document_id": document.document_id,
        "document_version_id": document.document_version_id,
        "ordinal": draft.ordinal,
        "text": draft.text,
        "token_count": draft.token_count,
        "source_start": draft.source_start,
        "source_end": draft.source_end,
        "target_overlap_tokens": draft.target_overlap_tokens,
        "overlap_source_start": draft.overlap_source_start,
        "overlap_source_end": draft.overlap_source_end,
        "overlap_token_count": draft.overlap_token_count,
        "source_references": references,
        "structural_kinds": draft.structural_kinds,
        "structural_block_ordinals": draft.structural_block_ordinals,
        "minimum_exception": _minimum_exception(draft.minimum_exception),
        "split_reason": _split_reason(draft.split_reason),
    }
    output_digest = _chunk_output_digest_for_values(**payload_values)
    chunk_values = {
        **payload_values,
        "chunker_version": chunker_version,
        "tokeniser_identity": tokeniser.identity,
        "configuration_identity": configuration_identity,
    }
    fingerprint = build_derivation_fingerprint(
        record_kind=_CHUNK_RECORD_KIND,
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs,
        operation=operation,
        output_digest=output_digest,
    )
    metadata = DerivedRecordMetadata(
        record_id=_record_id(fingerprint),
        record_kind=_CHUNK_RECORD_KIND,
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs,
        operation=operation,
        output_digest=output_digest,
        derivation_fingerprint=fingerprint,
    )
    return Chunk(metadata=metadata, **chunk_values)


def _references_for_draft(
    document: ParsedDocument,
    draft: object,
) -> tuple[ChunkSourceReference, ...]:
    """Clip every intersecting parser span to one draft's source range."""
    if not isinstance(draft, ChunkDraft):
        raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT)
    if len(draft.structural_kinds) != len(draft.structural_block_ordinals):
        raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT)
    references: list[ChunkSourceReference] = []
    for block_ordinal in draft.structural_block_ordinals:
        if not 0 <= block_ordinal < len(document.blocks):
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT)
        block = document.blocks[block_ordinal]
        span = block.source_span
        source_start = max(draft.source_start, span.start_offset)
        source_end = min(draft.source_end, span.end_offset)
        if source_start >= source_end:
            raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT)
        references.append(
            ChunkSourceReference(
                source_span_id=span.source_span_id,
                document_version_id=span.document_version_id,
                block_ordinal=block_ordinal,
                block_kind=cast(StructuralBlockKind, block.kind),
                source_start=source_start,
                source_end=source_end,
            )
        )
    return tuple(sorted(references, key=ChunkSourceReference.key))


def _minimum_exception(value: str | None) -> ChunkMinimumException | None:
    """Validate and convert a private minimum-exception annotation."""
    if value is None:
        return None
    try:
        return ChunkMinimumException(value)
    except ValueError:
        raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT) from None


def _split_reason(value: str | None) -> ChunkSplitReason | None:
    """Validate and convert a private split-reason annotation."""
    if value is None:
        return None
    try:
        return ChunkSplitReason(value)
    except ValueError:
        raise ChunkingContractError(ChunkingFailureCategory.INVALID_OUTPUT) from None


def _window_failure_category(category: str) -> ChunkingFailureCategory:
    """Map private window failures to the bounded public categories."""
    try:
        return ChunkingFailureCategory(category)
    except ValueError:
        return ChunkingFailureCategory.INVALID_OUTPUT


def _tokeniser_failure_category(category: str) -> ChunkingFailureCategory:
    """Map private tokeniser failures without exposing their details."""
    if category in {
        "unavailable",
        "unsafe_artefact",
        "artefact_changed_during_load",
        "slow_tokeniser",
    }:
        return ChunkingFailureCategory.TOKENISER_UNAVAILABLE
    return ChunkingFailureCategory.INVALID_TOKENISER
