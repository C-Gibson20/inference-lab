"""Pinned lexical-generation and parser-source location reconstruction."""

from __future__ import annotations

from pathlib import Path

from inference_lab.provenance import RecordId
from inference_lab.retrieval import (
    EvidenceSource,
    LexicalError,
    open_lexical_generation,
)
from inference_lab.retrieval.lexical_store_models import (
    LexicalIndexedChunk,
    LexicalSourceLink,
)
from inference_lab.serving.search_contracts import (
    ServingSearchError,
    ServingSearchFailureCategory,
    ServingSearchSourceLocation,
)
from inference_lab.storage import (
    ComponentGenerationState,
    CorpusSnapshot,
    StructuralBlockKind,
)


def resolve_source_locations(
    snapshot: CorpusSnapshot,
    index_dir: Path,
    record_id: RecordId,
    candidate_content: str,
    sources: tuple[EvidenceSource, ...],
) -> tuple[ServingSearchSourceLocation, ...]:
    """Resolve one hit's exact clipped ranges from its pinned generation."""
    try:
        reference = _lexical_reference(snapshot)
        with open_lexical_generation(index_dir, reference.generation_id) as generation:
            _validate_generation_manifest(generation.manifest, reference)
            chunk = generation.get_chunk(record_id)
            if chunk is None:
                raise _invalid_location()
            _validate_chunk(chunk, record_id, candidate_content, sources)
            links_by_span = {link.source_span_id: link for link in chunk.source_links}
            if len(links_by_span) != len(chunk.source_links):
                raise _invalid_location()
            return tuple(
                _resolve_one(
                    snapshot,
                    links_by_span[source.source_span_id],
                    source,
                )
                for source in sources
            )
    except ServingSearchError:
        raise
    except LexicalError as error:
        del error
        raise _invalid_location() from None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise _invalid_location() from None


def _lexical_reference(snapshot: CorpusSnapshot):
    """Select exactly one ready lexical reference from the pinned manifest."""
    references = tuple(
        reference
        for reference in snapshot.manifest.component_generations
        if reference.component_kind.as_str() == "lexical-index"
    )
    if len(references) != 1:
        raise _invalid_location()
    reference = references[0]
    if (
        reference.state is not ComponentGenerationState.READY
        or reference.corpus_id != snapshot.corpus_id
        or reference.corpus_version != snapshot.version.corpus_version
    ):
        raise _invalid_location()
    return reference


def _validate_generation_manifest(manifest: object, reference: object) -> None:
    """Require the opened immutable generation to match its manifest reference."""
    for field_name in (
        "corpus_id",
        "corpus_version",
        "generation_id",
        "implementation",
        "format_version",
        "contract_version",
        "configuration_digest",
        "source_manifest_digest",
    ):
        if getattr(manifest, field_name) != getattr(reference, field_name):
            raise _invalid_location()


def _validate_chunk(
    chunk: LexicalIndexedChunk,
    record_id: RecordId,
    candidate_content: str,
    sources: tuple[EvidenceSource, ...],
) -> None:
    """Require the generation row to agree with the returned evidence."""
    if (
        chunk.record_id != record_id
        or chunk.content != candidate_content
        or not sources
        or chunk.document_id != sources[0].document_id
        or chunk.document_version_id != sources[0].document_version_id
        or {link.source_span_id for link in chunk.source_links}
        != {source.source_span_id for source in sources}
    ):
        raise _invalid_location()
    if any(
        source.document_id != chunk.document_id
        or source.document_version_id != chunk.document_version_id
        for source in sources
    ):
        raise _invalid_location()


def _resolve_one(
    snapshot: CorpusSnapshot,
    link: LexicalSourceLink,
    source: EvidenceSource,
) -> ServingSearchSourceLocation:
    """Resolve and validate one clipped lexical source reference."""
    if link.source_span_id != source.source_span_id:
        raise _invalid_location()
    if link.document_version_id != source.document_version_id:
        raise _invalid_location()
    resolved = snapshot.resolve_source_span(source.source_span_id)
    if resolved is None:
        raise _invalid_location()
    location = resolved.location
    if (
        resolved.document_id != source.document_id
        or location.source_span_id != source.source_span_id
        or location.document_version_id != source.document_version_id
    ):
        raise _invalid_location()
    parsed = snapshot.get_parsed_representation(location.parsed_record_id)
    if (
        parsed is None
        or parsed.record_id != location.parsed_record_id
        or parsed.document_id != source.document_id
        or parsed.document_version_id != source.document_version_id
        or link.source_start < location.start_offset
        or link.source_end > location.end_offset
        or link.source_end > len(parsed.source_text)
    ):
        raise _invalid_location()
    try:
        block = parsed.blocks[link.block_ordinal]
        block_kind = StructuralBlockKind(link.block_kind)
    except (IndexError, TypeError, ValueError):
        raise _invalid_location() from None
    if (
        block.ordinal != link.block_ordinal
        or block.kind is not block_kind
        or block.source_span_id != source.source_span_id
    ):
        raise _invalid_location()
    return ServingSearchSourceLocation(
        source_span_id=source.source_span_id,
        document_id=source.document_id,
        document_version_id=source.document_version_id,
        parsed_record_id=location.parsed_record_id,
        start_offset=link.source_start,
        end_offset=link.source_end,
        span_start_offset=location.start_offset,
        span_end_offset=location.end_offset,
        content_origin=location.content_origin,
        block_ordinal=link.block_ordinal,
        block_kind=block_kind,
        page_number=location.page_number,
        bounding_box=location.bounding_box,
        page_layout=resolved.page_layout,
    )


def _invalid_location() -> ServingSearchError:
    """Return one bounded fail-closed source-location error."""
    return ServingSearchError(ServingSearchFailureCategory.CORRUPT_STATE)
