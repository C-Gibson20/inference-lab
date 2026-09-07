"""Bounded large-document measurement for the ING-04 chunking baseline."""

from __future__ import annotations

import json
import time
import tracemalloc

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ChunkingRequest,
    ChunkTokeniser,
    FakeChunkTokeniser,
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    TokenisedText,
    TokeniserIdentity,
    build_chunk_set,
    generate_source_span_id,
)
from inference_lab.ingest._chunk_windows import _MAX_COUNT_CACHE_ENTRIES
from inference_lab.provenance import CorpusId, DocumentId, DocumentVersionId

_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000405")
_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000406")
_DOCUMENT_VERSION_ID = DocumentVersionId.from_str(
    "00000000-0000-4000-8000-000000000407"
)


class MeasuredFakeTokeniser:
    """Count exact tokeniser calls and distinct slice inputs for one run."""

    def __init__(self) -> None:
        self._delegate = FakeChunkTokeniser()
        self.count_calls = 0
        self.unique_count_inputs: set[str] = set()

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the deterministic fake identity."""
        return self._delegate.identity

    def tokenise(self, text: str) -> TokenisedText:
        """Delegate exact token offsets to the fake tokeniser."""
        return self._delegate.tokenise(text)

    def count_tokens(self, text: str) -> int:
        """Record one exact standalone count request."""
        self.count_calls += 1
        self.unique_count_inputs.add(text)
        return self._delegate.count_tokens(text)


def _words(count: int) -> str:
    """Build one deterministic large synthetic document."""
    return " ".join(f"word{index}" for index in range(count))


def _request(token_count: int, tokeniser: ChunkTokeniser) -> ChunkingRequest:
    """Build a request whose one parser span covers all synthetic content."""
    source = _words(token_count)
    span = SourceSpan(
        source_span_id=generate_source_span_id(
            _DOCUMENT_VERSION_ID,
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=0,
            end_offset=len(source),
        ),
        document_version_id=_DOCUMENT_VERSION_ID,
        start_offset=0,
        end_offset=len(source),
    )
    block = StructuralBlock("paragraph", source, span)
    document = ParsedDocument(
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        text=NormalisedText(source),
        blocks=(block,),
        pages=None,
        sections=None,
        paragraphs=(block,),
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(source),
        token_count=None,
    )
    return ChunkingRequest(
        corpus_id=_CORPUS_ID,
        parsed_document=document,
        tokeniser=tokeniser,
    )


def _measure(token_count: int) -> tuple[MeasuredFakeTokeniser, float, int, int]:
    """Measure one deterministic run and return bounded operational metrics."""
    tokeniser = MeasuredFakeTokeniser()
    tracemalloc.start()
    started = time.perf_counter()
    chunk_set = build_chunk_set(_request(token_count, tokeniser))
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert chunk_set.chunks
    return tokeniser, elapsed, peak_bytes, len(chunk_set.chunks)


def test_large_document_measurement_reports_metrics_and_bounded_rescanning() -> None:
    """Report baseline metrics without imposing hardware-specific throughput."""
    token_count = 2_000
    tokeniser, elapsed, peak_bytes, chunk_count = _measure(token_count)
    metrics = {
        "cache_size_bound": _MAX_COUNT_CACHE_ENTRIES,
        "chunk_count": chunk_count,
        "count_calls": tokeniser.count_calls,
        "peak_memory_bytes": peak_bytes,
        "tokens_per_second": token_count / elapsed if elapsed else None,
        "unique_count_inputs": len(tokeniser.unique_count_inputs),
    }
    print("chunk_measurement=" + json.dumps(metrics, sort_keys=True))

    assert metrics["peak_memory_bytes"] > 0
    assert metrics["tokens_per_second"] is not None
    assert metrics["unique_count_inputs"] <= metrics["count_calls"]
    assert metrics["unique_count_inputs"] <= _MAX_COUNT_CACHE_ENTRIES
    assert metrics["count_calls"] < token_count * 100


def test_count_requests_scale_without_quadratic_rescanning() -> None:
    """Compare instrumented count work at two deterministic document sizes."""
    small, _, _, _ = _measure(1_000)
    large, _, _, _ = _measure(2_000)

    assert large.count_calls <= (small.count_calls * 3) + 1_000
    assert len(large.unique_count_inputs) <= _MAX_COUNT_CACHE_ENTRIES
