"""Observed corpus-scoped operation for deterministic chunking."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final, cast

from structlog.contextvars import bound_contextvars

from inference_lab._validation import is_strict_integer
from inference_lab.ingest._chunk_records import (
    ChunkingOutcome,
    ChunkingRequest,
    build_chunking_outcome,
)
from inference_lab.ingest.chunk_logging import emit_chunk_event
from inference_lab.ingest.chunking import CURRENT_CHUNKER_VERSION, TokeniserIdentity
from inference_lab.provenance import CorpusVersionId

Clock = Callable[[], int]

_UNEXPECTED_ERROR_CATEGORY: Final[str] = "unexpected_error"


def _read_clock(clock: Clock) -> int:
    """Read one strict monotonic nanosecond value."""
    value = clock()
    if not is_strict_integer(value):
        raise TypeError("clock must return an integer nanosecond value.")
    return value


def _duration_ms(started_ns: int, finished_ns: int) -> float:
    """Convert a non-negative interval to milliseconds."""
    if finished_ns < started_ns:
        raise ValueError("clock must not move backwards during chunking.")
    return (finished_ns - started_ns) / 1_000_000


class _ChunkingLifecycle:
    """Run and log one terminal result for an observed chunking request."""

    def __init__(
        self,
        request: ChunkingRequest,
        *,
        started_ns: int,
        clock: Clock,
    ) -> None:
        self._request = request
        self._started_ns = started_ns
        self._clock = clock
        self._completed = False

    @property
    def completed(self) -> bool:
        """Return whether this lifecycle has emitted a terminal event."""
        return self._completed

    def complete(self) -> ChunkingOutcome:
        """Run the pure engine and emit exactly one terminal event."""
        if self._completed:
            raise RuntimeError("chunking lifecycle has already been completed.")

        try:
            outcome = build_chunking_outcome(self._request)
            duration_ms = _duration_ms(self._started_ns, _read_clock(self._clock))
        except BaseException:
            self._emit_unexpected_failure()
            self._completed = True
            raise

        self._completed = True
        if outcome.succeeded:
            assert outcome.chunk_set is not None
            self._emit_completion(outcome, duration_ms)
        else:
            assert outcome.failure is not None
            emit_chunk_event(
                __name__,
                "error",
                "chunk_failed",
                error_category=outcome.failure.category.value,
                duration_ms=duration_ms,
            )
        return outcome

    def _emit_completion(self, outcome: ChunkingOutcome, duration_ms: float) -> None:
        """Emit bounded counts and versions for a successful chunk set."""
        assert outcome.chunk_set is not None
        chunk_set = outcome.chunk_set
        chunks = chunk_set.chunks
        undersized_count = sum(
            chunk.token_count < chunk_set.policy.min_tokens for chunk in chunks
        )
        oversize_split_count = sum(chunk.split_reason is not None for chunk in chunks)
        overlap_counts = tuple(chunk.overlap_token_count for chunk in chunks)
        emit_chunk_event(
            __name__,
            "info",
            "chunk_completed",
            chunker_implementation=chunk_set.chunker_version.implementation,
            chunker_contract_version=chunk_set.chunker_version.contract_version,
            tokeniser_implementation=chunk_set.tokeniser_identity.implementation,
            tokeniser_contract_version=chunk_set.tokeniser_identity.contract_version,
            chunk_count=len(chunks),
            input_token_count=self._input_token_count(),
            output_token_count=sum(chunk.token_count for chunk in chunks),
            overlap_token_count_total=sum(overlap_counts),
            overlap_token_count_max=max(overlap_counts, default=0),
            undersized_chunk_count=undersized_count,
            oversize_split_count=oversize_split_count,
            duration_ms=duration_ms,
        )

    def _input_token_count(self) -> int | None:
        """Return the exact whole-document count when the tokeniser exposes it."""
        fallback = self._request.parsed_document.token_count
        count_tokens = getattr(self._request.tokeniser, "count_tokens", None)
        if not callable(count_tokens):
            return fallback
        try:
            count = count_tokens(self._request.parsed_document.text.text)
        except Exception:
            return fallback
        if not is_strict_integer(count):
            return fallback
        count_value = cast(int, count)
        return count_value if count_value >= 0 else fallback

    def _emit_unexpected_failure(self) -> None:
        """Emit a bounded terminal event while preserving the original error."""
        try:
            finished_ns = _read_clock(self._clock)
            duration_ms = _duration_ms(self._started_ns, finished_ns)
        except Exception:
            duration_ms = None
        emit_chunk_event(
            __name__,
            "error",
            "chunk_failed",
            error_category=_UNEXPECTED_ERROR_CATEGORY,
            duration_ms=duration_ms,
        )


@contextmanager
def observe_chunking(
    request: ChunkingRequest,
    *,
    corpus_version: CorpusVersionId | None = None,
    clock: Clock = time.monotonic_ns,
) -> Iterator[_ChunkingLifecycle]:
    """Bind context and log one observed chunking operation."""
    if not isinstance(request, ChunkingRequest):
        raise TypeError("request must be a ChunkingRequest.")
    if corpus_version is not None and not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId or None.")
    if not callable(clock):
        raise TypeError("clock must be callable.")

    started_ns = _read_clock(clock)
    lifecycle = _ChunkingLifecycle(
        request,
        started_ns=started_ns,
        clock=clock,
    )
    context = {
        "corpus_id": request.corpus_id.as_str(),
        "document_id": request.parsed_document.document_id.as_str(),
        "corpus_version": (
            corpus_version.as_str() if corpus_version is not None else None
        ),
    }
    with bound_contextvars(**context):
        tokeniser_identity = getattr(request.tokeniser, "identity", None)
        tokeniser_fields = (
            {
                "tokeniser_implementation": tokeniser_identity.implementation,
                "tokeniser_contract_version": tokeniser_identity.contract_version,
            }
            if isinstance(tokeniser_identity, TokeniserIdentity)
            else {
                "tokeniser_implementation": None,
                "tokeniser_contract_version": None,
            }
        )
        emit_chunk_event(
            __name__,
            "info",
            "chunk_started",
            chunker_implementation=CURRENT_CHUNKER_VERSION.implementation,
            chunker_contract_version=CURRENT_CHUNKER_VERSION.contract_version,
            min_tokens=request.policy.min_tokens,
            max_tokens=request.policy.max_tokens,
            overlap_numerator=request.policy.overlap_numerator,
            overlap_denominator=request.policy.overlap_denominator,
            **tokeniser_fields,
        )
        try:
            yield lifecycle
        except BaseException:
            if not lifecycle.completed:
                lifecycle._emit_unexpected_failure()
                lifecycle._completed = True
            raise
        else:
            if not lifecycle.completed:
                lifecycle._emit_unexpected_failure()
                lifecycle._completed = True
                raise RuntimeError(
                    "chunking lifecycle exited without a terminal result."
                )
