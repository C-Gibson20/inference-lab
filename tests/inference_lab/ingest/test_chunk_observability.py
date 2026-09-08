"""Tests for observed chunking lifecycle logging and failure boundaries."""

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

import inference_lab.ingest.chunk_observability as chunk_observability
from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ChunkingFailureCategory,
    ChunkingPolicy,
    ChunkingRequest,
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    generate_source_span_id,
    observe_chunking,
)
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.ingest.tokenisation import FakeChunkTokeniser
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)

_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000301")
_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000302")
_DOCUMENT_VERSION_ID = DocumentVersionId.from_str(
    "00000000-0000-4000-8000-000000000303"
)


def _document(source_text: str, *, include_block: bool = True) -> ParsedDocument:
    """Build a valid document whose source can act as a privacy canary."""
    blocks: tuple[StructuralBlock, ...] = ()
    if include_block:
        span = SourceSpan(
            source_span_id=generate_source_span_id(
                _DOCUMENT_VERSION_ID,
                parser_version=CURRENT_PARSER_VERSION,
                ordinal=0,
                kind="paragraph",
                start_offset=0,
                end_offset=len(source_text),
            ),
            document_version_id=_DOCUMENT_VERSION_ID,
            start_offset=0,
            end_offset=len(source_text),
        )
        blocks = (
            StructuralBlock(
                kind="paragraph",
                text=normalise_source_text(source_text),
                source_span=span,
            ),
        )
    return ParsedDocument(
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        text=NormalisedText(source_text),
        blocks=blocks,
        pages=None,
        sections=None,
        paragraphs=blocks or None,
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(normalise_source_text(source_text)),
        token_count=None,
    )


def _request(document: ParsedDocument) -> ChunkingRequest:
    """Build one corpus-scoped request for lifecycle tests."""
    return ChunkingRequest(
        corpus_id=_CORPUS_ID,
        parsed_document=document,
        tokeniser=FakeChunkTokeniser(),
        policy=ChunkingPolicy(),
    )


def test_observed_chunking_binds_scope_and_logs_safe_completion() -> None:
    private_source = "private-chunk-content-marker"
    request = _request(_document(private_source))
    corpus_version = CorpusVersionId.generate()
    clock_values = iter((1_000_000_000, 1_025_000_000))

    with capture_logs() as logs:
        with observe_chunking(
            request,
            corpus_version=corpus_version,
            clock=clock_values.__next__,
        ) as lifecycle:
            assert get_contextvars() == {
                "corpus_id": _CORPUS_ID.as_str(),
                "corpus_version": corpus_version.as_str(),
                "document_id": _DOCUMENT_ID.as_str(),
            }
            outcome = lifecycle.complete()

    assert get_contextvars() == {}
    assert outcome.succeeded
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("chunk_started", "info"),
        ("chunk_completed", "info"),
    ]
    assert logs[0]["pipeline_stage"] == "chunk"
    assert logs[0]["chunker_contract_version"] == "1.0.1"
    assert logs[0]["tokeniser_contract_version"] == "1.0.0"
    assert logs[1]["chunk_count"] == 1
    assert logs[1]["input_token_count"] == 1
    assert logs[1]["output_token_count"] == 1
    assert logs[1]["overlap_token_count_total"] == 0
    assert logs[1]["overlap_token_count_max"] == 0
    assert logs[1]["undersized_chunk_count"] == 1
    assert logs[1]["oversize_split_count"] == 0
    assert logs[1]["duration_ms"] == 25.0
    assert all(marker not in repr(logs) for marker in (private_source,))


def test_observed_chunking_logs_bounded_expected_failure_without_partial_data() -> None:
    private_source = "private-uncovered-content-marker"
    request = _request(_document(private_source, include_block=False))
    clock_values = iter((2_000_000_000, 2_010_000_000))

    with capture_logs() as logs:
        with observe_chunking(request, clock=clock_values.__next__) as lifecycle:
            outcome = lifecycle.complete()

    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.category is ChunkingFailureCategory.UNCOVERED_CONTENT
    assert [record["event"] for record in logs] == [
        "chunk_started",
        "chunk_failed",
    ]
    assert logs[-1]["log_level"] == "error"
    assert logs[-1]["error_category"] == "uncovered_content"
    assert "chunk_count" not in logs[-1]
    assert private_source not in repr(outcome)
    assert private_source not in repr(logs)


def test_observed_chunking_reraises_unexpected_errors_and_logs_bounded_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(_document("private-unexpected-content-marker"))
    primary = RuntimeError("private-primary-error")

    def raise_primary(_: ChunkingRequest) -> object:
        raise primary

    monkeypatch.setattr(
        chunk_observability,
        "build_chunking_outcome",
        raise_primary,
    )
    clock_values = iter((3_000_000_000, 3_010_000_000))

    with capture_logs() as logs:
        with pytest.raises(RuntimeError, match="private-primary-error") as caught:
            with observe_chunking(request, clock=clock_values.__next__) as lifecycle:
                lifecycle.complete()

    assert caught.value is primary
    assert [record["event"] for record in logs] == [
        "chunk_started",
        "chunk_failed",
    ]
    assert logs[-1]["error_category"] == "unexpected_error"
    assert "private-primary-error" not in repr(logs)


def test_observed_chunking_emits_one_terminal_event_when_context_body_fails() -> None:
    request = _request(_document("private-body-error-marker"))
    primary = RuntimeError("private-body-primary")

    with capture_logs() as logs:
        with pytest.raises(RuntimeError, match="private-body-primary"):
            with observe_chunking(request, clock=lambda: 0):
                raise primary

    assert [record["event"] for record in logs] == [
        "chunk_started",
        "chunk_failed",
    ]
    assert logs[-1]["error_category"] == "unexpected_error"
    assert "private-body-primary" not in repr(logs)


def test_observed_chunking_logs_missing_terminal_result_as_unexpected_failure() -> None:
    request = _request(_document("private-missing-terminal-marker"))

    with capture_logs() as logs:
        with pytest.raises(RuntimeError, match="without a terminal result"):
            with observe_chunking(request, clock=lambda: 0):
                pass

    assert [record["event"] for record in logs] == [
        "chunk_started",
        "chunk_failed",
    ]
    assert logs[-1]["error_category"] == "unexpected_error"
    assert "private-missing-terminal-marker" not in repr(logs)


def test_observed_chunking_rejects_second_completion_without_duplicate_logging() -> (
    None
):
    request = _request(_document("private-double-completion-marker"))

    with capture_logs() as logs:
        with observe_chunking(request, clock=lambda: 0) as lifecycle:
            lifecycle.complete()
            with pytest.raises(RuntimeError, match="already been completed"):
                lifecycle.complete()

    assert [record["event"] for record in logs] == [
        "chunk_started",
        "chunk_completed",
    ]
