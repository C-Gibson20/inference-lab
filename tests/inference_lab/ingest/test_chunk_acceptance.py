"""Golden, invariant, and privacy acceptance tests for ING-04 chunking."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    BoundingBox,
    ChunkingPolicy,
    ChunkingRequest,
    ChunkMinimumException,
    ChunkSet,
    ChunkSplitReason,
    ChunkTokeniser,
    FakeChunkTokeniser,
    NormalisedText,
    PageLayoutDescriptor,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    StructuralBlockKind,
    TokenisedText,
    TokeniserIdentity,
    TokenOffset,
    build_chunk_set,
    build_chunking_outcome,
    build_normalisation_boundary_map,
    generate_source_span_id,
    observe_chunking,
    serialise_chunk,
    serialise_chunk_set,
)
from inference_lab.ingest.normalisation import normalise_source_text
from inference_lab.provenance import CorpusId, DocumentId, DocumentVersionId

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "ingest"
    / "chunking"
    / "golden_cases.json"
)
_CORPUS_ID = "00000000-0000-4000-8000-000000000401"
_OTHER_CORPUS_ID = "00000000-0000-4000-8000-000000000402"
_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000403")
_DOCUMENT_VERSION_ID = DocumentVersionId.from_str(
    "00000000-0000-4000-8000-000000000404"
)
_PRIVATE_TOKENISER_MARKER = "private tokeniser payload marker"


@dataclass(frozen=True, slots=True)
class SpanSpec:
    """Describe one deterministic parser span for an acceptance document."""

    kind: str
    start: int
    end: int
    page_number: int | None = None
    bounding_box: BoundingBox | None = None


class ContextualTestTokeniser:
    """Fake tokeniser whose first-word split exposes substring count mistakes."""

    _IDENTITY = TokeniserIdentity(
        implementation="contextual-test",
        contract_version="1.0.0",
        artefact_identity="acceptance-fixture",
        artefact_revision="v1",
        options=(("contextual-first-word", "true"),),
    )

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the stable acceptance tokeniser identity."""
        return self._IDENTITY

    def tokenise(self, text: str) -> TokenisedText:
        """Split a leading ``prefix`` word into two exact token offsets."""
        offsets = [
            TokenOffset(match.start(), match.end())
            for match in re.finditer(r"\S+", text)
        ]
        if text.startswith("prefix") and offsets:
            first = offsets[0]
            offsets[:1] = (
                TokenOffset(first.start, first.start + 3),
                TokenOffset(first.start + 3, first.end),
            )
        return TokenisedText(
            token_ids=tuple(range(1, len(offsets) + 1)),
            offsets=tuple(offsets),
        )

    def count_tokens(self, text: str) -> int:
        """Count the exact standalone token sequence for one text slice."""
        return len(self.tokenise(text).token_ids)


class NonMonotonicTestTokeniser:
    """Expose valid standalone counts that repeatedly cross a token limit."""

    _IDENTITY = TokeniserIdentity(
        implementation="non-monotonic-test",
        contract_version="1.0.0",
        artefact_identity="acceptance-fixture",
        artefact_revision="v1",
        options=(("non-monotonic-counts", "true"),),
    )
    _COUNTS = {1: 1, 2: 3, 3: 2, 4: 5, 5: 1, 6: 4}

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the stable acceptance tokeniser identity."""
        return self._IDENTITY

    def tokenise(self, text: str) -> TokenisedText:
        """Return offsets consistent with the adversarial standalone count."""
        words = tuple(re.finditer(r"\S+", text))
        word_count = len(words)
        token_count = self._COUNTS.get(word_count, word_count)
        offsets = [TokenOffset(match.start(), match.end()) for match in words]
        if token_count < word_count:
            offsets = offsets[:token_count]
        elif token_count > word_count:
            whitespace = (
                TokenOffset(index, index + 1)
                for index, character in enumerate(text)
                if character.isspace()
            )
            offsets.extend(tuple(whitespace)[: token_count - word_count])
            offsets.sort(key=lambda offset: offset.start)
        return TokenisedText(
            token_ids=tuple(range(1, len(offsets) + 1)),
            offsets=tuple(offsets),
        )

    def count_tokens(self, text: str) -> int:
        """Return deterministic counts with no monotonicity guarantee."""
        word_count = len(tuple(re.finditer(r"\S+", text)))
        return self._COUNTS.get(word_count, word_count)


class MissingIdentityTestTokeniser:
    """Implement token operations without the required stable identity."""

    def __init__(self) -> None:
        self._delegate = FakeChunkTokeniser()

    def tokenise(self, text: str) -> TokenisedText:
        """Delegate tokenisation for the missing-identity case."""
        return self._delegate.tokenise(text)

    def count_tokens(self, text: str) -> int:
        """Delegate counting for the missing-identity case."""
        return self._delegate.count_tokens(text)


class InvalidBoundaryTestTokeniser:
    """Inject one selected tokeniser contract violation on document text."""

    def __init__(self, failure: str) -> None:
        self._failure = failure
        self._delegate = FakeChunkTokeniser()

    @property
    def identity(self) -> object:
        """Return either the valid fixture identity or an invalid value."""
        if self._failure == "identity_exception":
            raise RuntimeError(_PRIVATE_TOKENISER_MARKER)
        if self._failure == "invalid_identity":
            return _PRIVATE_TOKENISER_MARKER
        return self._delegate.identity

    def tokenise(self, text: str) -> TokenisedText:
        """Inject malformed offsets or a private backend exception."""
        if self._failure == "probe_exception" and text == "one café":
            raise RuntimeError(_PRIVATE_TOKENISER_MARKER)
        if _PRIVATE_TOKENISER_MARKER in text:
            if self._failure == "malformed_offsets":
                return TokenisedText(
                    (1,),
                    (TokenOffset(0, len(text) + 1),),
                )
            if self._failure == "backend_exception":
                raise RuntimeError(_PRIVATE_TOKENISER_MARKER)
        return self._delegate.tokenise(text)

    def count_tokens(self, text: str) -> int:
        """Inject an actual-document count inconsistent with tokenisation."""
        if self._failure == "count_exception" and _PRIVATE_TOKENISER_MARKER in text:
            raise RuntimeError(_PRIVATE_TOKENISER_MARKER)
        count = self._delegate.count_tokens(text)
        if self._failure == "inconsistent_count" and _PRIVATE_TOKENISER_MARKER in text:
            return count + 1
        return count


def _words(count: int) -> str:
    """Build deterministic whitespace-tokenised source text."""
    return " ".join(f"word{index}" for index in range(count))


def _document(
    source_text: str,
    spans: tuple[SpanSpec, ...],
    *,
    page_layouts: tuple[PageLayoutDescriptor, ...] = (),
) -> ParsedDocument:
    """Build one validated parser document with stable provenance IDs."""
    blocks = tuple(
        StructuralBlock(
            kind=cast(StructuralBlockKind, span.kind),
            text=normalise_source_text(source_text[span.start : span.end]),
            source_span=SourceSpan(
                source_span_id=generate_source_span_id(
                    _DOCUMENT_VERSION_ID,
                    parser_version=CURRENT_PARSER_VERSION,
                    ordinal=ordinal,
                    kind=span.kind,
                    start_offset=span.start,
                    end_offset=span.end,
                ),
                document_version_id=_DOCUMENT_VERSION_ID,
                start_offset=span.start,
                end_offset=span.end,
                page_number=span.page_number,
                bounding_box=span.bounding_box,
            ),
        )
        for ordinal, span in enumerate(spans)
    )
    return ParsedDocument(
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        text=NormalisedText(source_text),
        blocks=blocks,
        pages=tuple(block for block in blocks if block.kind == "page") or None,
        sections=tuple(block for block in blocks if block.kind == "section") or None,
        paragraphs=tuple(block for block in blocks if block.kind == "paragraph")
        or None,
        page_layouts=page_layouts,
        language=None,
        page_count=len(page_layouts) or None,
        character_count=len(normalise_source_text(source_text)),
        token_count=None,
    )


def _request(
    document: ParsedDocument,
    *,
    tokeniser: object | None = None,
    corpus_id: str = _CORPUS_ID,
    policy: ChunkingPolicy | None = None,
) -> ChunkingRequest:
    """Build one acceptance request from a document and explicit components."""
    return ChunkingRequest(
        corpus_id=CorpusId.from_str(corpus_id),
        parsed_document=document,
        tokeniser=tokeniser or FakeChunkTokeniser(),
        policy=policy or ChunkingPolicy(),
    )


def _build_set(
    source_text: str,
    spans: tuple[SpanSpec, ...],
    *,
    page_layouts: tuple[PageLayoutDescriptor, ...] = (),
    tokeniser: object | None = None,
    corpus_id: str = _CORPUS_ID,
    policy: ChunkingPolicy | None = None,
) -> tuple[ParsedDocument, ChunkSet]:
    """Build a document and complete chunk set for one acceptance case."""
    document = _document(source_text, spans, page_layouts=page_layouts)
    chunk_set = build_chunk_set(
        _request(
            document,
            tokeniser=tokeniser,
            corpus_id=corpus_id,
            policy=policy,
        )
    )
    return document, chunk_set


def _golden_cases() -> tuple[tuple[str, dict[str, object]], ...]:
    """Load the checked-in deterministic size matrix."""
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return tuple(
        (name, cast(dict[str, object], case)) for name, case in payload.items()
    )


def _case_source(case: dict[str, object]) -> str:
    """Materialise one fixture case without introducing runtime randomness."""
    source_kind = cast(str, case["source_kind"])
    if source_kind == "empty":
        return ""
    if source_kind == "whitespace":
        return " \r\n\t "
    if source_kind == "words":
        return _words(cast(int, case["token_count"]))
    raise AssertionError(f"unsupported fixture source kind: {source_kind}")


def _full_span(source_text: str) -> tuple[SpanSpec, ...]:
    """Return one paragraph span covering a non-empty fixture source."""
    return (SpanSpec("paragraph", 0, len(source_text)),)


def _reconstruct(document: ParsedDocument, chunk_set: ChunkSet) -> str:
    """Remove adjacent overlap from chunks and reconstruct their source slice."""
    boundary_map = build_normalisation_boundary_map(document.text)
    chunks = chunk_set.chunks
    assert chunks
    reconstructed = chunks[0].text
    for chunk in chunks[1:]:
        overlap_text = boundary_map.canonical_text_for_source_range(
            chunk.overlap_source_start,
            chunk.overlap_source_end,
        )
        assert chunk.text.startswith(overlap_text)
        reconstructed += chunk.text[len(overlap_text) :]
    expected = boundary_map.canonical_text_for_source_range(
        chunks[0].source_start,
        chunks[-1].source_end,
    )
    assert reconstructed == expected
    return reconstructed


def _assert_invariants(
    document: ParsedDocument,
    chunk_set: ChunkSet,
    tokeniser: ChunkTokeniser,
) -> None:
    """Check hard limits, exact counts, progress, overlap, and navigation."""
    policy = chunk_set.policy
    boundary_map = build_normalisation_boundary_map(document.text)
    chunks = chunk_set.chunks
    assert tuple(chunk.ordinal for chunk in chunks) == tuple(range(len(chunks)))
    for previous, current in zip(chunks, chunks[1:]):
        assert current.source_start <= previous.source_end
        assert current.source_end > previous.source_end
        assert current.overlap_source_start == current.source_start
        assert current.overlap_source_end == previous.source_end
        assert current.overlap_token_count <= current.target_overlap_tokens
    for chunk in chunks:
        assert 0 < chunk.token_count <= policy.max_tokens
        assert chunk.token_count == tokeniser.count_tokens(chunk.text)
        assert chunk.text == boundary_map.canonical_text_for_source_range(
            chunk.source_start,
            chunk.source_end,
        )
        if chunk.token_count < policy.min_tokens:
            assert chunk.minimum_exception is not None
        else:
            assert chunk.minimum_exception is None
        assert chunk.source_references
        for reference in chunk.source_references:
            assert chunk.source_start <= reference.source_start
            assert reference.source_end <= chunk.source_end


@pytest.mark.parametrize(
    ("case_name", "case"),
    _golden_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_checked_in_golden_size_matrix(case_name: str, case: dict[str, object]) -> None:
    """Cover empty, undersized, and 256/512/513-token boundaries."""
    source_text = _case_source(case)
    spans = () if not source_text else _full_span(source_text)
    document, chunk_set = _build_set(source_text, spans)

    assert len(chunk_set.chunks) == case["expected_chunk_count"]
    if chunk_set.chunks:
        _assert_invariants(document, chunk_set, FakeChunkTokeniser())
        expected_exception = case.get("expected_minimum_exception")
        assert chunk_set.chunks[0].minimum_exception == expected_exception
        _reconstruct(document, chunk_set)
    else:
        assert chunk_set.lineage.source_spans == ()


def test_long_flat_and_many_short_paragraphs_preserve_order_without_duplication() -> (
    None
):
    long_source = _words(1_200)
    document, chunk_set = _build_set(long_source, _full_span(long_source))
    _assert_invariants(document, chunk_set, FakeChunkTokeniser())
    assert _reconstruct(document, chunk_set) == document.text.text
    assert all(
        chunk.overlap_token_count == chunk.target_overlap_tokens
        for chunk in chunk_set.chunks[1:]
    )

    paragraphs = tuple(f"paragraph {index} " + _words(18) for index in range(32))
    paragraph_source = "\n\n".join(paragraphs)
    offset = 0
    spans: list[SpanSpec] = []
    for paragraph in paragraphs:
        spans.append(SpanSpec("paragraph", offset, offset + len(paragraph)))
        offset += len(paragraph) + 2
    paragraph_document, paragraph_set = _build_set(
        paragraph_source,
        tuple(spans),
    )
    _assert_invariants(paragraph_document, paragraph_set, FakeChunkTokeniser())
    assert (
        _reconstruct(paragraph_document, paragraph_set) == paragraph_document.text.text
    )
    assert (
        len(
            {
                reference.block_ordinal
                for chunk in paragraph_set.chunks
                for reference in chunk.source_references
            }
        )
        > 1
    )


def test_tail_rebalancing_and_audited_short_tail_exception() -> None:
    rebalanced_source = _words(600)
    document, chunk_set = _build_set(rebalanced_source, _full_span(rebalanced_source))
    assert all(chunk.minimum_exception is None for chunk in chunk_set.chunks)
    _assert_invariants(document, chunk_set, FakeChunkTokeniser())

    before = _words(200)
    protected = _words(100)
    after = _words(250)
    source = f"{before}\n{protected}\n{after}"
    protected_start = len(before) + 1
    protected_end = protected_start + len(protected)
    tail_document, tail_set = _build_set(
        source,
        (
            SpanSpec("paragraph", 0, len(before)),
            SpanSpec("table", protected_start, protected_end),
            SpanSpec("paragraph", protected_end + 1, len(source)),
        ),
    )
    _assert_invariants(tail_document, tail_set, FakeChunkTokeniser())
    assert any(
        chunk.minimum_exception
        in {
            ChunkMinimumException.FINAL_REMAINDER,
            ChunkMinimumException.PROTECTED_STRUCTURE_NEIGHBOUR,
        }
        for chunk in tail_set.chunks
    )


def test_nested_overlapping_structures_and_page_layout_retain_source_anchors() -> None:
    source = _words(700)
    words = tuple(re.finditer(r"\S+", source))
    table_start = words[100].start()
    table_end = words[150].end()
    caption_start = words[550].start()
    caption_end = words[600].end()
    layout = PageLayoutDescriptor(
        page_number=1,
        coordinate_unit="pdf_points",
        coordinate_origin="top_left",
        page_width=612,
        page_height=792,
        rotation=0,
    )
    bbox = BoundingBox(10, 20, 300, 400)
    document, chunk_set = _build_set(
        source,
        (
            SpanSpec("page", 0, len(source), 1, bbox),
            SpanSpec("section", 0, len(source)),
            SpanSpec("paragraph", 0, len(source)),
            SpanSpec("other", 0, len(source)),
            SpanSpec("table", table_start, table_end),
            SpanSpec("figure_caption", caption_start, caption_end),
        ),
        page_layouts=(layout,),
    )
    _assert_invariants(document, chunk_set, FakeChunkTokeniser())
    _reconstruct(document, chunk_set)
    kinds = {
        reference.block_kind
        for chunk in chunk_set.chunks
        for reference in chunk.source_references
    }
    assert {"page", "section", "paragraph", "table", "figure_caption", "other"} <= kinds
    for chunk in chunk_set.chunks:
        for reference in chunk.source_references:
            if reference.block_kind == "page":
                assert (
                    document.blocks[reference.block_ordinal].source_span.bounding_box
                    == bbox
                )


@pytest.mark.parametrize("kind", ("table", "figure_caption"))
def test_protected_structures_are_atomic_when_fitting_and_split_when_oversize(
    kind: str,
) -> None:
    fitting_before = _words(400)
    fitting_protected = _words(100)
    fitting_after = _words(400)
    fitting_source = f"{fitting_before}\n{fitting_protected}\n{fitting_after}"
    fitting_start = len(fitting_before) + 1
    fitting_end = fitting_start + len(fitting_protected)
    fitting_document, fitting_set = _build_set(
        fitting_source,
        (
            SpanSpec("paragraph", 0, len(fitting_before)),
            SpanSpec(kind, fitting_start, fitting_end),
            SpanSpec("paragraph", fitting_end + 1, len(fitting_source)),
        ),
    )
    _assert_invariants(fitting_document, fitting_set, FakeChunkTokeniser())
    assert all(chunk.split_reason is None for chunk in fitting_set.chunks)
    assert any(
        chunk.overlap_token_count < chunk.target_overlap_tokens
        for chunk in fitting_set.chunks[1:]
    )
    assert any(chunk.overlap_token_count == 0 for chunk in fitting_set.chunks[1:])

    oversize_source = _words(600)
    oversize_document, oversize_set = _build_set(
        oversize_source,
        (SpanSpec(kind, 0, len(oversize_source)),),
    )
    _assert_invariants(oversize_document, oversize_set, FakeChunkTokeniser())
    assert len(oversize_set.chunks) > 1
    assert all(
        chunk.split_reason is ChunkSplitReason.OVERSIZE_STRUCTURE
        for chunk in oversize_set.chunks
    )


def test_unicode_newlines_composition_astral_and_repeated_passages_navigate_exactly() -> (
    None
):
    source = "\t" + ("e\u0301 café 😀 repeated\r\n" * 300) + "\r  "
    document, chunk_set = _build_set(source, _full_span(source))

    _assert_invariants(document, chunk_set, FakeChunkTokeniser())
    assert document.text.text.startswith("\té café 😀 repeated\n")
    reconstructed = _reconstruct(document, chunk_set)
    assert reconstructed.startswith("\té café 😀 repeated\n")
    assert reconstructed.endswith("repeated")
    assert all(
        document.text.source_text[chunk.source_start : chunk.source_end].strip()
        for chunk in chunk_set.chunks
    )


def test_contextual_tokeniser_uses_exact_standalone_chunk_text_counts() -> None:
    tokeniser = ContextualTestTokeniser()
    source = " ".join(("prefix", *(f"word{index}" for index in range(800))))
    document, chunk_set = _build_set(
        source,
        _full_span(source),
        tokeniser=tokeniser,
    )

    whole_document_count = tokeniser.count_tokens(document.text.text)
    assert any(chunk.token_count != whole_document_count for chunk in chunk_set.chunks)
    _assert_invariants(document, chunk_set, tokeniser)
    assert _reconstruct(document, chunk_set) == document.text.text


def test_non_monotonic_counts_choose_furthest_end_and_best_overlap() -> None:
    tokeniser = NonMonotonicTestTokeniser()
    source = "a b c d e f g h"
    document, chunk_set = _build_set(
        source,
        _full_span(source),
        tokeniser=tokeniser,
        policy=ChunkingPolicy(
            min_tokens=1,
            max_tokens=4,
            overlap_numerator=1,
            overlap_denominator=2,
        ),
    )

    assert len(chunk_set.chunks) == 2
    first, second = chunk_set.chunks
    assert first.source_end == source.index("g")
    assert first.token_count == 4
    assert second.source_start == source.index("d")
    assert second.target_overlap_tokens == 2
    assert second.overlap_token_count == 2
    _assert_invariants(document, chunk_set, tokeniser)
    assert _reconstruct(document, chunk_set) == document.text.text


def test_chunk_identity_is_corpus_independent_and_cold_process_stable() -> None:
    source = _words(600)
    document, first = _build_set(source, _full_span(source), corpus_id=_CORPUS_ID)
    _, second = _build_set(source, _full_span(source), corpus_id=_OTHER_CORPUS_ID)

    assert first.corpus_id != second.corpus_id
    assert tuple(serialise_chunk(chunk) for chunk in first.chunks) == tuple(
        serialise_chunk(chunk) for chunk in second.chunks
    )
    assert tuple(chunk.record_id for chunk in first.chunks) == tuple(
        chunk.record_id for chunk in second.chunks
    )
    assert serialise_chunk_set(first) == serialise_chunk_set(
        build_chunk_set(_request(document))
    )

    child = textwrap.dedent(
        """
        from inference_lab.ingest import (
            CURRENT_PARSER_VERSION,
            ChunkingRequest,
            FakeChunkTokeniser,
            NormalisedText,
            ParsedDocument,
            SourceSpan,
            StructuralBlock,
            build_chunk_set,
            generate_source_span_id,
            serialise_chunk_set,
        )
        from inference_lab.provenance import CorpusId, DocumentId, DocumentVersionId
        import base64

        source = " ".join(f"word{index}" for index in range(600))
        document_version_id = DocumentVersionId.from_str(
            "00000000-0000-4000-8000-000000000404"
        )
        span = SourceSpan(
            source_span_id=generate_source_span_id(
                document_version_id,
                parser_version=CURRENT_PARSER_VERSION,
                ordinal=0,
                kind="paragraph",
                start_offset=0,
                end_offset=len(source),
            ),
            document_version_id=document_version_id,
            start_offset=0,
            end_offset=len(source),
        )
        block = StructuralBlock("paragraph", source, span)
        document = ParsedDocument(
            document_id=DocumentId.from_str("00000000-0000-4000-8000-000000000403"),
            document_version_id=document_version_id,
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
        request = ChunkingRequest(
            corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000401"),
            parsed_document=document,
            tokeniser=FakeChunkTokeniser(),
        )
        print(base64.b64encode(serialise_chunk_set(build_chunk_set(request))).decode())
        """
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "1"
    first_process = subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    environment["PYTHONHASHSEED"] = "2"
    second_process = subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert first_process.stdout == second_process.stdout


def test_acceptance_failure_and_logs_exclude_private_canaries() -> None:
    private_source = "private acceptance source marker"
    document = _document(private_source, ())
    outcome = build_chunking_outcome(_request(document))
    assert outcome.failure is not None

    assert private_source not in str(outcome)
    assert private_source not in repr(outcome)

    with capture_logs() as logs:
        with observe_chunking(
            _request(document), clock=iter((0, 1)).__next__
        ) as lifecycle:
            observed = lifecycle.complete()

    assert observed.failure is not None
    assert private_source not in repr(logs)


@pytest.mark.parametrize(
    ("tokeniser", "expected_category"),
    (
        (MissingIdentityTestTokeniser(), "invalid_tokeniser"),
        (InvalidBoundaryTestTokeniser("invalid_identity"), "invalid_tokeniser"),
        (
            InvalidBoundaryTestTokeniser("identity_exception"),
            "tokeniser_unavailable",
        ),
        (InvalidBoundaryTestTokeniser("probe_exception"), "tokeniser_unavailable"),
        (InvalidBoundaryTestTokeniser("malformed_offsets"), "invalid_tokeniser"),
        (InvalidBoundaryTestTokeniser("inconsistent_count"), "invalid_tokeniser"),
        (InvalidBoundaryTestTokeniser("backend_exception"), "tokeniser_unavailable"),
        (InvalidBoundaryTestTokeniser("count_exception"), "tokeniser_unavailable"),
    ),
)
def test_tokeniser_contract_failures_produce_bounded_outcomes(
    tokeniser: object,
    expected_category: str,
) -> None:
    document = _document(
        _PRIVATE_TOKENISER_MARKER,
        _full_span(_PRIVATE_TOKENISER_MARKER),
    )

    with capture_logs() as logs:
        outcome = build_chunking_outcome(_request(document, tokeniser=tokeniser))

    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.category.value == expected_category
    assert _PRIVATE_TOKENISER_MARKER not in repr(outcome)
    assert logs == []
