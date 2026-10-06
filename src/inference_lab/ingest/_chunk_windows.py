"""Private deterministic structural chunk-window construction."""

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, cast

from inference_lab._validation import is_strict_integer
from inference_lab.ingest.chunking import (
    CanonicalSourceBoundaryMap,
    ChunkingRequest,
    SourceBoundary,
    UnsafeSourceBoundaryError,
)
from inference_lab.ingest.parser import ParsedDocument
from inference_lab.ingest.sources import StructuralBlock
from inference_lab.ingest.tokenisation import (
    ChunkTokeniser,
    TokenisedText,
    validate_chunk_tokeniser,
    validate_tokenised_text,
)

StructuralBoundaryRole = Literal["start", "end"]
CountFn = Callable[[int, int], int]

_STRUCTURAL_PRIORITY: Final[dict[str, int]] = {
    "section": 0,
    "table": 1,
    "figure_caption": 1,
    "paragraph": 2,
    "page": 3,
    "other": 4,
}
_PROTECTED_KINDS: Final[frozenset[str]] = frozenset({"table", "figure_caption"})
_MAX_COUNT_CACHE_ENTRIES: Final[int] = 4096
_STANDALONE_COUNT_CORRECTION_TOKENS: Final[int] = 32


class ChunkWindowError(ValueError):
    """Report a bounded failure while constructing private chunk windows."""

    def __init__(self, category: str) -> None:
        """Create a payload-free chunk-window failure."""
        super().__init__(category)
        self.category = category


@dataclass(frozen=True, slots=True)
class StructuralBoundary:
    """Represent one parser-provided candidate boundary."""

    source_offset: int
    canonical_offset: int
    kind: str
    block_ordinal: int
    role: StructuralBoundaryRole

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Enforce non-negative offsets, non-empty kind, and supported role."""
        if not is_strict_integer(self.source_offset) or self.source_offset < 0:
            raise ValueError("structural boundary source offset must be non-negative")
        if not is_strict_integer(self.canonical_offset) or self.canonical_offset < 0:
            raise ValueError(
                "structural boundary canonical offset must be non-negative"
            )
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("structural boundary kind must be non-empty")
        if not is_strict_integer(self.block_ordinal) or self.block_ordinal < 0:
            raise ValueError("structural boundary ordinal must be non-negative")
        if self.role not in ("start", "end"):
            raise ValueError("unsupported structural boundary role")

    @property
    def priority(self) -> int:
        """Return the closed boundary-kind priority."""
        try:
            return _STRUCTURAL_PRIORITY[self.kind]
        except KeyError:
            raise ChunkWindowError("unsupported_structure") from None

    def key(self) -> tuple[int, int, int, str]:
        """Return the deterministic equal-offset tie-break key."""
        return (
            self.priority,
            self.block_ordinal,
            0 if self.role == "end" else 1,
            self.kind,
        )


@dataclass(frozen=True, slots=True)
class ProtectedStructure:
    """Represent a fitting table or caption that must not be split."""

    source_start: int
    source_end: int
    kind: str
    block_ordinal: int
    token_count: int


@dataclass(frozen=True, slots=True)
class StructuralCatalogue:
    """One non-duplicating structural overlay over canonical source text."""

    boundaries: tuple[StructuralBoundary, ...]
    blocks: tuple[StructuralBlock, ...]
    protected: tuple[ProtectedStructure, ...]
    oversize: tuple[ProtectedStructure, ...]

    def ends_at(self, source_offset: int) -> tuple[StructuralBoundary, ...]:
        """Return candidate boundaries at one source offset in tie order."""
        return tuple(
            sorted(
                (
                    boundary
                    for boundary in self.boundaries
                    if boundary.source_offset == source_offset
                ),
                key=StructuralBoundary.key,
            )
        )


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """Private source-linked draft consumed by the provenance PR."""

    ordinal: int
    source_start: int
    source_end: int
    canonical_start: int
    canonical_end: int
    text: str
    token_count: int
    target_overlap_tokens: int
    overlap_source_start: int
    overlap_source_end: int
    overlap_token_count: int
    minimum_exception: str | None
    split_reason: str | None
    structural_kinds: tuple[str, ...]
    structural_block_ordinals: tuple[int, ...]


def build_structural_catalogue(
    document: ParsedDocument,
    boundary_map: CanonicalSourceBoundaryMap,
    tokeniser: ChunkTokeniser,
    *,
    max_tokens: int,
) -> StructuralCatalogue:
    """Build parser boundaries and protected structures without duplicating text."""
    if not isinstance(document, ParsedDocument):
        raise TypeError("document must be a ParsedDocument")
    if not isinstance(boundary_map, CanonicalSourceBoundaryMap):
        raise TypeError("boundary_map must be a CanonicalSourceBoundaryMap")
    if boundary_map.source_text != document.text.source_text:
        raise ChunkWindowError("invalid_input")
    if not is_strict_integer(max_tokens) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    validate_chunk_tokeniser(tokeniser)

    safe_sources = {boundary.source_offset for boundary in boundary_map.boundaries}
    boundaries: list[StructuralBoundary] = []
    protected: list[ProtectedStructure] = []
    oversize: list[ProtectedStructure] = []
    for ordinal, block in enumerate(document.blocks):
        span = block.source_span
        block_token_count = _count_exact_block(tokeniser, block)
        if block.kind in _PROTECTED_KINDS:
            structure = ProtectedStructure(
                source_start=span.start_offset,
                source_end=span.end_offset,
                kind=block.kind,
                block_ordinal=ordinal,
                token_count=block_token_count,
            )
            if block_token_count <= max_tokens:
                protected.append(structure)
            else:
                oversize.append(structure)
        for role, source_offset in (
            ("start", span.start_offset),
            ("end", span.end_offset),
        ):
            if source_offset not in safe_sources:
                continue
            canonical_offset = boundary_map.canonical_offset_for_source(source_offset)
            boundaries.append(
                StructuralBoundary(
                    source_offset=source_offset,
                    canonical_offset=canonical_offset,
                    kind=block.kind,
                    block_ordinal=ordinal,
                    role=cast(StructuralBoundaryRole, role),
                )
            )

    return StructuralCatalogue(
        boundaries=tuple(
            sorted(
                boundaries,
                key=lambda boundary: (
                    boundary.source_offset,
                    boundary.key(),
                ),
            )
        ),
        blocks=document.blocks,
        protected=tuple(protected),
        oversize=tuple(oversize),
    )


def build_chunk_drafts(request: ChunkingRequest) -> tuple[ChunkDraft, ...]:
    """Construct deterministic private windows for one parsed document."""
    if not isinstance(request, ChunkingRequest):
        raise TypeError("request must be a ChunkingRequest")
    tokeniser = validate_chunk_tokeniser(cast(ChunkTokeniser, request.tokeniser))
    boundary_map = _build_boundary_map(request.parsed_document)
    catalogue = build_structural_catalogue(
        request.parsed_document,
        boundary_map,
        tokeniser,
        max_tokens=request.policy.max_tokens,
    )
    useful_range = _find_useful_range(request.parsed_document, boundary_map, catalogue)
    if useful_range is None:
        return ()

    count_cache: dict[tuple[int, int], int] = {}

    def count(source_start: int, source_end: int) -> int:
        key = (source_start, source_end)
        cached = count_cache.get(key)
        if cached is not None:
            return cached
        try:
            text = boundary_map.canonical_text_for_source_range(
                source_start,
                source_end,
            )
        except UnsafeSourceBoundaryError:
            raise ChunkWindowError("unsafe_normalisation_boundary") from None
        result = tokeniser.count_tokens(text)
        if not is_strict_integer(result) or result < 0:
            raise ChunkWindowError("invalid_tokeniser")
        if len(count_cache) < _MAX_COUNT_CACHE_ENTRIES:
            count_cache[key] = result
        return result

    useful_range, token_boundaries = _token_boundaries(
        boundary_map,
        tokeniser,
        useful_range,
    )
    useful_start, useful_end = useful_range
    drafts: list[ChunkDraft] = []
    current_start = useful_start
    while current_start < useful_end:
        previous = drafts[-1] if drafts else None
        overlap = _overlap_for(
            previous,
            boundary_map=boundary_map,
            tokeniser=tokeniser,
            safe_boundaries=token_boundaries,
            catalogue=catalogue,
            count=count,
            numerator=request.policy.overlap_numerator,
            denominator=request.policy.overlap_denominator,
        )
        end = _choose_end(
            progress_start=current_start,
            window_start=overlap[0],
            useful_end=useful_end,
            safe_boundaries=token_boundaries,
            catalogue=catalogue,
            count=count,
            min_tokens=request.policy.min_tokens,
            max_tokens=request.policy.max_tokens,
            candidate_limit=_prefix_candidate_limit(
                window_start=overlap[0],
                useful_end=useful_end,
                boundary_map=boundary_map,
                tokeniser=tokeniser,
                max_tokens=request.policy.max_tokens,
            ),
        )
        if end <= current_start:
            raise ChunkWindowError("no_progress")
        draft = _make_draft(
            ordinal=len(drafts),
            source_start=overlap[0],
            source_end=end,
            previous=previous,
            boundary_map=boundary_map,
            catalogue=catalogue,
            count=count,
            overlap=overlap,
            useful_end=useful_end,
            min_tokens=request.policy.min_tokens,
        )

        if (
            draft.source_end == useful_end
            and draft.token_count < request.policy.min_tokens
        ):
            rebalanced = _rebalance_tail(
                drafts=drafts,
                useful_end=useful_end,
                safe_boundaries=token_boundaries,
                catalogue=catalogue,
                boundary_map=boundary_map,
                tokeniser=tokeniser,
                count=count,
                numerator=request.policy.overlap_numerator,
                denominator=request.policy.overlap_denominator,
                min_tokens=request.policy.min_tokens,
                max_tokens=request.policy.max_tokens,
            )
            if rebalanced is not None:
                previous_draft, draft = rebalanced
                drafts[-1] = previous_draft

        drafts.append(draft)
        current_start = draft.source_end

    _validate_drafts(
        drafts,
        boundary_map=boundary_map,
        catalogue=catalogue,
        min_tokens=request.policy.min_tokens,
        max_tokens=request.policy.max_tokens,
    )
    return tuple(drafts)


def _build_boundary_map(document: ParsedDocument) -> CanonicalSourceBoundaryMap:
    """Build the boundary map without exposing it as a public chunk API."""
    from inference_lab.ingest.chunking import build_canonical_source_boundary_map

    return build_canonical_source_boundary_map(document.text)


def _token_boundaries(
    boundary_map: CanonicalSourceBoundaryMap,
    tokeniser: ChunkTokeniser,
    useful_range: tuple[int, int],
) -> tuple[tuple[int, int], tuple[SourceBoundary, ...]]:
    """Return the covered range expanded to safe source-mapped token edges."""
    useful_start, useful_end = useful_range
    tokenised = validate_tokenised_text(
        boundary_map.canonical_text,
        tokeniser.tokenise(boundary_map.canonical_text),
    )
    canonical_offsets = {0, len(boundary_map.canonical_text)}
    for start, end, _, _ in _token_offset_groups(tokenised):
        canonical_offsets.update((start, end))
    candidates = tuple(
        boundary
        for boundary in boundary_map.boundaries
        if boundary.canonical_offset in canonical_offsets
    )
    starts = tuple(
        boundary for boundary in candidates if boundary.source_offset <= useful_start
    )
    ends = tuple(
        boundary for boundary in candidates if boundary.source_offset >= useful_end
    )
    if not starts or not ends:
        raise ChunkWindowError("unsafe_normalisation_boundary")
    expanded_start = starts[-1].source_offset
    expanded_end = ends[0].source_offset
    if expanded_start >= expanded_end:
        raise ChunkWindowError("unsafe_normalisation_boundary")
    selected = tuple(
        boundary
        for boundary in candidates
        if expanded_start <= boundary.source_offset <= expanded_end
    )
    return (expanded_start, expanded_end), selected


def _token_offset_groups(
    tokenised: TokenisedText,
) -> tuple[tuple[int, int, int, int], ...]:
    """Group token spans that overlap so chunk boundaries cannot split them."""
    if not tokenised.offsets:
        return ()
    groups: list[tuple[int, int, int, int]] = []
    first = 0
    group_start = tokenised.offsets[0].start
    group_end = tokenised.offsets[0].end
    for index, offset in enumerate(tokenised.offsets[1:], start=1):
        if offset.start < group_end:
            group_end = max(group_end, offset.end)
            continue
        groups.append((group_start, group_end, first, index - 1))
        first = index
        group_start = offset.start
        group_end = offset.end
    groups.append((group_start, group_end, first, len(tokenised.offsets) - 1))
    return tuple(groups)


def _count_exact_block(tokeniser: ChunkTokeniser, block: StructuralBlock) -> int:
    """Count one parser block's exact canonical text."""
    result = tokeniser.count_tokens(block.text)
    if not is_strict_integer(result) or result < 0:
        raise ChunkWindowError("invalid_tokeniser")
    return result


def _find_useful_range(
    document: ParsedDocument,
    boundary_map: CanonicalSourceBoundaryMap,
    catalogue: StructuralCatalogue,
) -> tuple[int, int] | None:
    """Find the smallest safe range containing all covered non-whitespace text."""
    source_text = document.text.source_text
    intervals = sorted(
        (block.source_span.start_offset, block.source_span.end_offset)
        for block in catalogue.blocks
        if block.source_span.start_offset < block.source_span.end_offset
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    covered_positions: list[int] = []
    interval_index = 0
    for position, character in enumerate(source_text):
        while interval_index < len(merged) and position >= merged[interval_index][1]:
            interval_index += 1
        covered = (
            interval_index < len(merged)
            and merged[interval_index][0] <= position < merged[interval_index][1]
        )
        if not character.isspace() and not covered:
            raise ChunkWindowError("uncovered_content")
        if covered and not character.isspace():
            covered_positions.append(position)
    if not covered_positions:
        return None

    safe_sources = tuple(boundary.source_offset for boundary in boundary_map.boundaries)
    first_content = covered_positions[0]
    last_content_end = covered_positions[-1] + 1
    start_index = max(0, bisect_right(safe_sources, first_content) - 1)
    end_index = bisect_left(safe_sources, last_content_end)
    if end_index == len(safe_sources):
        raise ChunkWindowError("unsafe_normalisation_boundary")
    useful_start = safe_sources[start_index]
    useful_end = safe_sources[end_index]
    if useful_start >= useful_end:
        raise ChunkWindowError("no_progress")
    return useful_start, useful_end


def _prefix_candidate_limit(
    *,
    window_start: int,
    useful_end: int,
    boundary_map: CanonicalSourceBoundaryMap,
    tokeniser: ChunkTokeniser,
    max_tokens: int,
) -> int:
    """Bound exact prefix probes using native offsets plus correction headroom."""
    text = boundary_map.canonical_text_for_source_range(window_start, useful_end)
    tokenised = validate_tokenised_text(text, tokeniser.tokenise(text))
    probe_tokens = max_tokens + _STANDALONE_COUNT_CORRECTION_TOKENS
    if len(tokenised.offsets) <= probe_tokens:
        return useful_end
    canonical_start = boundary_map.canonical_offset_for_source(window_start)
    probe_group = next(
        group
        for group in _token_offset_groups(tokenised)
        if group[2] <= probe_tokens - 1 <= group[3]
    )
    canonical_limit = canonical_start + probe_group[1]
    return max(
        boundary.source_offset
        for boundary in boundary_map.boundaries
        if boundary.canonical_offset <= canonical_limit
    )


def _suffix_candidate_limit(
    previous: ChunkDraft,
    *,
    boundary_map: CanonicalSourceBoundaryMap,
    tokeniser: ChunkTokeniser,
    target_tokens: int,
) -> int:
    """Bound exact suffix probes using native offsets plus correction headroom."""
    tokenised = validate_tokenised_text(
        previous.text,
        tokeniser.tokenise(previous.text),
    )
    probe_tokens = target_tokens + _STANDALONE_COUNT_CORRECTION_TOKENS
    if len(tokenised.offsets) <= probe_tokens:
        return previous.source_start
    probe_index = len(tokenised.offsets) - probe_tokens
    probe_group = next(
        group
        for group in _token_offset_groups(tokenised)
        if group[2] <= probe_index <= group[3]
    )
    canonical_limit = previous.canonical_start + probe_group[0]
    return min(
        boundary.source_offset
        for boundary in boundary_map.boundaries
        if boundary.canonical_offset >= canonical_limit
    )


def _choose_end(
    *,
    progress_start: int,
    window_start: int,
    useful_end: int,
    safe_boundaries: tuple[SourceBoundary, ...],
    catalogue: StructuralCatalogue,
    count: CountFn,
    min_tokens: int,
    max_tokens: int,
    candidate_limit: int,
) -> int:
    """Choose the furthest legal structural or token boundary."""
    count_range = count
    safe_source_offsets = {boundary.source_offset for boundary in safe_boundaries}

    structural_candidates: list[tuple[int, StructuralBoundary]] = []
    for boundary in catalogue.boundaries:
        if (
            not progress_start < boundary.source_offset <= candidate_limit
            or boundary.source_offset not in safe_source_offsets
        ):
            continue
        if _splits_protected(
            window_start,
            boundary.source_offset,
            catalogue.protected,
        ):
            continue
        token_count = count_range(window_start, boundary.source_offset)
        if min_tokens <= token_count <= max_tokens:
            structural_candidates.append((boundary.source_offset, boundary))
    if structural_candidates:
        furthest = max(source_offset for source_offset, _ in structural_candidates)
        tied = tuple(
            boundary
            for source_offset, boundary in structural_candidates
            if source_offset == furthest
        )
        return min(tied, key=StructuralBoundary.key).source_offset

    fallback = tuple(
        boundary.source_offset
        for boundary in safe_boundaries
        if progress_start < boundary.source_offset <= candidate_limit
        and not _splits_protected(
            window_start,
            boundary.source_offset,
            catalogue.protected,
        )
    )
    if not fallback:
        raise ChunkWindowError("token_budget_failure")
    for source_offset in reversed(fallback):
        token_count = count_range(window_start, source_offset)
        if token_count <= max_tokens:
            return source_offset
    raise ChunkWindowError("token_budget_failure")


def _overlap_for(
    previous: ChunkDraft | None,
    *,
    boundary_map: CanonicalSourceBoundaryMap,
    tokeniser: ChunkTokeniser,
    safe_boundaries: tuple[SourceBoundary, ...],
    catalogue: StructuralCatalogue,
    count: CountFn,
    numerator: int,
    denominator: int,
) -> tuple[int, int, int, int]:
    """Choose the greatest legal previous suffix at or below the target."""
    if previous is None:
        return (0, 0, 0, 0)
    count_range = count
    target = previous.token_count * numerator // denominator
    candidate_limit = _suffix_candidate_limit(
        previous,
        boundary_map=boundary_map,
        tokeniser=tokeniser,
        target_tokens=target,
    )
    candidates = tuple(
        boundary.source_offset
        for boundary in safe_boundaries
        if candidate_limit <= boundary.source_offset <= previous.source_end
        and not _starts_inside_protected(boundary.source_offset, catalogue.protected)
    )
    if not candidates:
        return (previous.source_end, previous.source_end, target, 0)
    best: tuple[int, int] | None = None
    for source_start in candidates:
        overlap_count = count_range(source_start, previous.source_end)
        if overlap_count > target:
            continue
        candidate = (overlap_count, source_start)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return (previous.source_end, previous.source_end, target, 0)
    overlap_count, start = best
    return (start, previous.source_end, target, overlap_count)


def _make_draft(
    *,
    ordinal: int,
    source_start: int,
    source_end: int,
    previous: ChunkDraft | None,
    boundary_map: CanonicalSourceBoundaryMap,
    catalogue: StructuralCatalogue,
    count: CountFn,
    overlap: tuple[int, int, int, int],
    useful_end: int,
    min_tokens: int,
) -> ChunkDraft:
    """Build one validated private draft from a contiguous source slice."""
    count_range = count
    try:
        canonical_start = boundary_map.canonical_offset_for_source(source_start)
        canonical_end = boundary_map.canonical_offset_for_source(source_end)
        text = boundary_map.canonical_text_for_source_range(source_start, source_end)
    except UnsafeSourceBoundaryError:
        raise ChunkWindowError("unsafe_normalisation_boundary") from None
    token_count = count_range(source_start, source_end)
    minimum_exception = None
    if token_count < min_tokens:
        if previous is None and source_end == useful_end:
            minimum_exception = "short_document"
        elif _has_protected_barrier(source_end, catalogue.protected):
            minimum_exception = "protected_structure_neighbour"
        elif source_end == useful_end:
            minimum_exception = "final_remainder_minimum_exception"
    intersecting = tuple(
        (ordinal_index, block.kind)
        for ordinal_index, block in enumerate(catalogue.blocks)
        if block.source_span.start_offset < source_end
        and block.source_span.end_offset > source_start
    )
    split_reason = (
        "oversize_structure"
        if any(
            structure.source_start < source_end and structure.source_end > source_start
            for structure in catalogue.oversize
        )
        else None
    )
    return ChunkDraft(
        ordinal=ordinal,
        source_start=source_start,
        source_end=source_end,
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        text=text,
        token_count=token_count,
        target_overlap_tokens=overlap[2],
        overlap_source_start=overlap[0],
        overlap_source_end=overlap[1],
        overlap_token_count=overlap[3],
        minimum_exception=minimum_exception,
        split_reason=split_reason,
        structural_kinds=tuple(kind for _, kind in intersecting),
        structural_block_ordinals=tuple(index for index, _ in intersecting),
    )


def _rebalance_tail(
    *,
    drafts: list[ChunkDraft],
    useful_end: int,
    safe_boundaries: tuple[SourceBoundary, ...],
    catalogue: StructuralCatalogue,
    boundary_map: CanonicalSourceBoundaryMap,
    tokeniser: ChunkTokeniser,
    count: CountFn,
    numerator: int,
    denominator: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[ChunkDraft, ChunkDraft] | None:
    """Move the preceding boundary when both resulting chunks fit legally."""
    if not drafts:
        return None
    previous = drafts[-1]
    count_range = count
    candidates: list[tuple[int, int]] = []
    for boundary in safe_boundaries:
        candidate_end = boundary.source_offset
        if not previous.source_start < candidate_end < useful_end:
            continue
        if _splits_protected(
            previous.source_start,
            candidate_end,
            catalogue.protected,
        ):
            continue
        previous_count = count_range(previous.source_start, candidate_end)
        if not min_tokens <= previous_count <= max_tokens:
            continue
        candidates.append((abs(previous_count - previous.token_count), candidate_end))
    for _, candidate_end in sorted(candidates):
        candidate_previous = _replace_draft_range(
            previous,
            source_end=candidate_end,
            boundary_map=boundary_map,
            catalogue=catalogue,
            count=count_range,
            minimum_exception=None,
        )
        overlap = _overlap_for(
            candidate_previous,
            boundary_map=boundary_map,
            tokeniser=tokeniser,
            safe_boundaries=safe_boundaries,
            catalogue=catalogue,
            count=count_range,
            numerator=numerator,
            denominator=denominator,
        )
        current_count = count_range(overlap[0], useful_end)
        if not min_tokens <= current_count <= max_tokens:
            continue
        current = _make_draft(
            ordinal=candidate_previous.ordinal + 1,
            source_start=overlap[0],
            source_end=useful_end,
            previous=candidate_previous,
            boundary_map=boundary_map,
            catalogue=catalogue,
            count=count_range,
            overlap=overlap,
            useful_end=useful_end,
            min_tokens=min_tokens,
        )
        return candidate_previous, current
    return None


def _replace_draft_range(
    draft: ChunkDraft,
    *,
    source_end: int,
    boundary_map: CanonicalSourceBoundaryMap,
    catalogue: StructuralCatalogue,
    count: CountFn,
    minimum_exception: str | None,
) -> ChunkDraft:
    """Replace one draft's end while preserving its incoming overlap metadata."""
    count_range = count
    canonical_end = boundary_map.canonical_offset_for_source(source_end)
    text = boundary_map.canonical_text_for_source_range(draft.source_start, source_end)
    token_count = count_range(draft.source_start, source_end)
    intersecting = tuple(
        (index, block.kind)
        for index, block in enumerate(catalogue.blocks)
        if block.source_span.start_offset < source_end
        and block.source_span.end_offset > draft.source_start
    )
    split_reason = (
        "oversize_structure"
        if any(
            structure.source_start < source_end
            and structure.source_end > draft.source_start
            for structure in catalogue.oversize
        )
        else None
    )
    return ChunkDraft(
        ordinal=draft.ordinal,
        source_start=draft.source_start,
        source_end=source_end,
        canonical_start=draft.canonical_start,
        canonical_end=canonical_end,
        text=text,
        token_count=token_count,
        target_overlap_tokens=draft.target_overlap_tokens,
        overlap_source_start=draft.overlap_source_start,
        overlap_source_end=draft.overlap_source_end,
        overlap_token_count=draft.overlap_token_count,
        minimum_exception=minimum_exception,
        split_reason=split_reason,
        structural_kinds=tuple(kind for _, kind in intersecting),
        structural_block_ordinals=tuple(index for index, _ in intersecting),
    )


def _starts_inside_protected(
    source_start: int,
    protected: tuple[ProtectedStructure, ...],
) -> bool:
    """Return whether a new window starts strictly inside a protected unit."""
    return any(
        structure.source_start < source_start < structure.source_end
        for structure in protected
    )


def _splits_protected(
    source_start: int,
    source_end: int,
    protected: tuple[ProtectedStructure, ...],
) -> bool:
    """Return whether an end would cut through a protected unit."""
    return any(
        source_start < structure.source_start < source_end < structure.source_end
        or structure.source_start <= source_start < structure.source_end < source_end
        for structure in protected
    )


def _has_protected_barrier(
    source_end: int,
    protected: tuple[ProtectedStructure, ...],
) -> bool:
    """Return whether a protected unit begins at the selected boundary."""
    return any(structure.source_start == source_end for structure in protected)


def _validate_drafts(
    drafts: list[ChunkDraft],
    *,
    boundary_map: CanonicalSourceBoundaryMap,
    catalogue: StructuralCatalogue,
    min_tokens: int,
    max_tokens: int,
) -> None:
    """Enforce hard maxima, progress, overlap, and exact source reconstruction."""
    previous: ChunkDraft | None = None
    for expected_ordinal, draft in enumerate(drafts):
        if draft.ordinal != expected_ordinal:
            raise ChunkWindowError("invalid_output")
        if draft.token_count > max_tokens or draft.token_count < 0:
            raise ChunkWindowError("token_budget_failure")
        if draft.token_count < min_tokens and draft.minimum_exception is None:
            raise ChunkWindowError("invalid_minimum_exception")
        if draft.token_count >= min_tokens and draft.minimum_exception is not None:
            raise ChunkWindowError("invalid_minimum_exception")
        if not draft.text.strip() or not draft.source_start < draft.source_end:
            raise ChunkWindowError("invalid_output")
        if previous is None:
            if draft.overlap_token_count != 0:
                raise ChunkWindowError("invalid_overlap")
        else:
            if draft.source_start > previous.source_end:
                raise ChunkWindowError("content_gap")
            if draft.source_end <= previous.source_end:
                raise ChunkWindowError("no_progress")
            if draft.overlap_source_start != draft.source_start:
                raise ChunkWindowError("invalid_overlap")
            if draft.overlap_source_end != previous.source_end:
                raise ChunkWindowError("invalid_overlap")
            if draft.overlap_token_count > draft.target_overlap_tokens:
                raise ChunkWindowError("invalid_overlap")
            if _starts_inside_protected(draft.source_start, catalogue.protected):
                raise ChunkWindowError("protected_structure_split")
        if (
            boundary_map.canonical_text_for_source_range(
                draft.source_start,
                draft.source_end,
            )
            != draft.text
        ):
            raise ChunkWindowError("invalid_output")
        previous = draft
