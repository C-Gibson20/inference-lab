"""Deterministic, source-preserving reduction of retrieved evidence."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from inference_lab._validation import is_canonical_semantic_version, is_strict_integer
from inference_lab.provenance import CorpusId, CorpusVersionId, RecordId
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.evidence import (
    EvidenceHit,
    EvidenceSource,
    RetrievalCandidate,
)
from inference_lab.retrieval.results import (
    EvidenceResult,
    ResultStatus,
    RetrievalFailure,
)

EVIDENCE_SEPARATOR = "\n\n"
REDUCTION_POLICY_VERSION = "1.0.0"
REDUCTION_ACCOUNTING_VERSION = "1.0.0"


class EvidenceTokeniserError(ValueError):
    """Report a bounded failure at the evidence tokeniser boundary."""

    def __init__(self, category: str) -> None:
        """Create a payload-free tokeniser error."""
        super().__init__(category)
        self.category = category


@dataclass(frozen=True, slots=True)
class EvidenceTokeniserIdentity:
    """Identify a local tokeniser without retaining its deployment path."""

    implementation: str
    revision: str
    contract_version: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the tokeniser identity is path-free and well-formed."""
        if not isinstance(self.implementation, str) or not self.implementation:
            raise TypeError("implementation must be a non-empty string.")
        if not isinstance(self.revision, str) or not self.revision:
            raise TypeError("revision must be a non-empty string.")
        if any(
            len(value) > 128
            or any(char in value for char in "\\/\x00")
            or any(ord(char) < 32 for char in value)
            for value in (self.implementation, self.revision)
        ):
            raise ValueError("tokeniser identity must be path-free.")
        if not isinstance(self.contract_version, str):
            raise TypeError("contract_version must be a string.")
        if not is_canonical_semantic_version(self.contract_version):
            raise ValueError("contract_version must be semantic versioning.")
        if not isinstance(self.options, tuple):
            raise TypeError("options must be a tuple.")
        if any(
            not isinstance(option, tuple)
            or len(option) != 2
            or not all(isinstance(value, str) and value for value in option)
            or any(
                len(value) > 128
                or any(char in value for char in "\\/\x00")
                or any(ord(char) < 32 for char in value)
                for value in option
            )
            for option in self.options
        ):
            raise TypeError("options must contain non-empty string pairs.")
        keys = tuple(key for key, _ in self.options)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("options must be sorted with unique keys.")


@dataclass(frozen=True, slots=True)
class EvidenceTokenOffset:
    """Locate a token in Python Unicode code-point offsets."""

    start: int
    end: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the token offset is valid."""
        if not is_strict_integer(self.start) or not is_strict_integer(self.end):
            raise TypeError("token offsets must be integers.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("token offsets must be non-empty ranges.")


@dataclass(frozen=True, slots=True)
class EvidenceTokenisation:
    """Token offsets for one exact evidence string."""

    offsets: tuple[EvidenceTokenOffset, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the tokenisation offsets are valid."""
        if not isinstance(self.offsets, tuple):
            raise TypeError("offsets must be a tuple.")
        if any(not isinstance(item, EvidenceTokenOffset) for item in self.offsets):
            raise TypeError("offsets must contain EvidenceTokenOffset values.")


@runtime_checkable
class EvidenceTokeniser(Protocol):
    """Local deterministic tokeniser used for exact evidence accounting."""

    @property
    def identity(self) -> EvidenceTokeniserIdentity:
        """Return the complete path-free tokeniser identity."""
        ...

    def tokenise(self, text: str) -> EvidenceTokenisation:
        """Return token code-point offsets without special tokens."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the exact token count for the complete supplied string."""
        ...


@dataclass(frozen=True, slots=True)
class TextSourceBoundary:
    """Pair a candidate-text boundary with its exact decoded-source offset."""

    text_offset: int
    source_offset: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not is_strict_integer(self.text_offset) or not is_strict_integer(
            self.source_offset
        ):
            raise TypeError("source boundaries must be integer offsets.")
        if self.text_offset < 0 or self.source_offset < 0:
            raise ValueError("source boundaries must not be negative.")


@dataclass(frozen=True, slots=True)
class EvidenceTextSourceRange:
    """Map exact candidate-text boundaries to one original source span.

    The boundary table may omit unsafe normalisation boundaries. A reduction
    may end only at a listed boundary, so a clipped range remains reversible.
    """

    source: EvidenceSource
    text_start: int
    text_end: int
    source_start: int
    source_end: int
    boundaries: tuple[TextSourceBoundary, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the text source range is well-formed."""
        if not isinstance(self.source, EvidenceSource):
            raise TypeError("source must be an EvidenceSource.")
        values = (self.text_start, self.text_end, self.source_start, self.source_end)
        if any(not is_strict_integer(value) for value in values):
            raise TypeError("source range offsets must be integers.")
        if self.text_start < 0 or self.text_end <= self.text_start:
            raise ValueError("candidate text range must be non-empty.")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("decoded source range must be non-empty.")
        if not isinstance(self.boundaries, tuple) or len(self.boundaries) < 2:
            raise TypeError("boundaries must contain at least two values.")
        if any(not isinstance(value, TextSourceBoundary) for value in self.boundaries):
            raise TypeError("boundaries must contain TextSourceBoundary values.")
        if self.boundaries[0] != TextSourceBoundary(self.text_start, self.source_start):
            raise ValueError("source map must begin at its declared range start.")
        if self.boundaries[-1] != TextSourceBoundary(self.text_end, self.source_end):
            raise ValueError("source map must end at its declared range end.")
        if any(
            right.text_offset <= left.text_offset
            or right.source_offset <= left.source_offset
            for left, right in zip(self.boundaries, self.boundaries[1:])
        ):
            raise ValueError("source map boundaries must increase strictly.")

    def source_offset_at(self, text_offset: int) -> int | None:
        """Return an exact source offset, or None for an unsafe boundary."""
        for boundary in self.boundaries:
            if boundary.text_offset == text_offset:
                return boundary.source_offset
        return None


class EvidenceReductionAction(StrEnum):
    """Decision applied to one ranked evidence item."""

    KEPT = "kept"
    TRUNCATED = "truncated"
    DROPPED_ITEM_LIMIT = "dropped_item_limit"
    DROPPED_DIVERSITY = "dropped_diversity"
    DROPPED_BUDGET = "dropped_budget"
    FAILED_SOURCE_MAP = "failed_source_map"


@dataclass(frozen=True, slots=True)
class ReducedEvidenceItem:
    """One retained hit and its unchanged or clipped source mapping."""

    hit: EvidenceHit
    text: str
    source_ranges: tuple[EvidenceTextSourceRange, ...]
    action: EvidenceReductionAction

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the reduced evidence item is well-formed."""
        if not isinstance(self.hit, EvidenceHit):
            raise TypeError("hit must be an EvidenceHit.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("retained evidence text must not be blank.")
        if not isinstance(self.source_ranges, tuple) or any(
            not isinstance(item, EvidenceTextSourceRange) for item in self.source_ranges
        ):
            raise TypeError("source_ranges must contain source range values.")
        if not isinstance(self.action, EvidenceReductionAction):
            raise TypeError("action must be an EvidenceReductionAction.")
        if self.action is EvidenceReductionAction.TRUNCATED and not self.source_ranges:
            raise ValueError("truncated evidence must retain source ranges.")


@dataclass(frozen=True, slots=True)
class EvidenceReductionFailure:
    """Bounded reason an evidence item could not be retained."""

    record_id: RecordId
    category: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the reduction failure is well-formed."""
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string.")
        if self.category not in {"source_map_unavailable", "source_map_invalid"}:
            raise ValueError("unsupported reduction failure category.")


@dataclass(frozen=True, slots=True)
class EvidenceReductionDecision:
    """Record the keep, drop, truncate, or failure decision for one hit."""

    record_id: RecordId
    action: EvidenceReductionAction

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the reduction decision is well-formed."""
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        if not isinstance(self.action, EvidenceReductionAction):
            raise TypeError("action must be an EvidenceReductionAction.")


@dataclass(frozen=True, slots=True)
class EvidenceReductionIdentity:
    """Stable path-free description of one evidence reduction policy."""

    tokeniser: EvidenceTokeniserIdentity
    token_budget: int
    item_budget: int
    diversity_policy: Literal["none", "per_document"] = "none"
    separator: str = EVIDENCE_SEPARATOR
    policy_version: str = REDUCTION_POLICY_VERSION
    accounting_version: str = REDUCTION_ACCOUNTING_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the reduction identity is well-formed."""
        if not isinstance(self.tokeniser, EvidenceTokeniserIdentity):
            raise TypeError("tokeniser must be an EvidenceTokeniserIdentity.")
        if not is_strict_integer(self.token_budget) or self.token_budget <= 0:
            raise ValueError("token_budget must be a positive integer.")
        if not is_strict_integer(self.item_budget) or self.item_budget <= 0:
            raise ValueError("item_budget must be a positive integer.")
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string.")
        if not isinstance(self.policy_version, str) or not isinstance(
            self.accounting_version, str
        ):
            raise TypeError("reduction versions must be strings.")
        if not is_canonical_semantic_version(
            self.policy_version
        ) or not is_canonical_semantic_version(self.accounting_version):
            raise ValueError("reduction versions must be semantic versions.")
        if not isinstance(self.diversity_policy, str):
            raise TypeError("diversity_policy must be a string.")
        if self.diversity_policy not in {"none", "per_document"}:
            raise ValueError("unsupported diversity policy.")

    @property
    def digest(self) -> str:
        """Return a stable digest of every reduction-affecting identity field."""
        payload = {
            "accounting_version": self.accounting_version,
            "diversity_policy": self.diversity_policy,
            "item_budget": self.item_budget,
            "policy_version": self.policy_version,
            "separator": self.separator,
            "token_budget": self.token_budget,
            "tokeniser": {
                "contract_version": self.tokeniser.contract_version,
                "implementation": self.tokeniser.implementation,
                "options": self.tokeniser.options,
                "revision": self.tokeniser.revision,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceReductionRequest:
    """Reduce one reranked result for exactly one corpus and version."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    result: EvidenceResult
    tokeniser: EvidenceTokeniser
    token_budget: int
    item_budget: int
    source_ranges: Mapping[RecordId, tuple[EvidenceTextSourceRange, ...]]
    diversity_policy: Literal["none", "per_document"] = "none"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the reduction request is well-formed."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.result, EvidenceResult):
            raise TypeError("result must be an EvidenceResult.")
        if self.result.corpus_version != self.corpus_version:
            raise InvalidRetrievalContractError(
                "evidence result must match the selected corpus version."
            )
        if not isinstance(self.tokeniser, EvidenceTokeniser):
            raise TypeError("tokeniser must implement EvidenceTokeniser.")
        if not isinstance(self.tokeniser.identity, EvidenceTokeniserIdentity):
            raise TypeError("tokeniser identity is invalid.")
        if not is_strict_integer(self.token_budget) or self.token_budget <= 0:
            raise ValueError("token_budget must be a positive integer.")
        if not is_strict_integer(self.item_budget) or self.item_budget <= 0:
            raise ValueError("item_budget must be a positive integer.")
        if self.item_budget > self.result.requested_limit.value:
            raise ValueError("item_budget must not exceed the retrieval result limit.")
        if not isinstance(self.diversity_policy, str):
            raise TypeError("diversity_policy must be a string.")
        if self.diversity_policy not in {"none", "per_document"}:
            raise ValueError("unsupported diversity policy.")
        if not isinstance(self.source_ranges, Mapping):
            raise TypeError("source_ranges must be a mapping.")
        for record_id, ranges in self.source_ranges.items():
            if not isinstance(record_id, RecordId):
                raise TypeError("source range keys must be RecordId values.")
            if not isinstance(ranges, tuple) or any(
                not isinstance(item, EvidenceTextSourceRange) for item in ranges
            ):
                raise TypeError("source range values must be tuples of source ranges.")
        object.__setattr__(
            self, "source_ranges", MappingProxyType(dict(self.source_ranges))
        )


@dataclass(frozen=True, slots=True)
class EvidenceReductionOutcome:
    """Bounded evidence, diagnostics, and identity for downstream context."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    items: tuple[ReducedEvidenceItem, ...]
    decisions: tuple[EvidenceReductionDecision, ...]
    input_count: int
    dropped_count: int
    token_count: int
    source_status: ResultStatus
    source_failures: tuple[RetrievalFailure, ...]
    identity: EvidenceReductionIdentity
    failures: tuple[EvidenceReductionFailure, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that the reduction outcome is well-formed."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.items, tuple) or any(
            not isinstance(item, ReducedEvidenceItem) for item in self.items
        ):
            raise TypeError("items must contain ReducedEvidenceItem values.")
        if not isinstance(self.decisions, tuple) or any(
            not isinstance(item, EvidenceReductionDecision) for item in self.decisions
        ):
            raise TypeError("decisions must contain EvidenceReductionDecision values.")
        if not isinstance(self.failures, tuple) or any(
            not isinstance(item, EvidenceReductionFailure) for item in self.failures
        ):
            raise TypeError("failures must contain EvidenceReductionFailure values.")
        if not isinstance(self.source_status, ResultStatus):
            raise TypeError("source_status must be a ResultStatus.")
        if not isinstance(self.source_failures, tuple) or any(
            not isinstance(item, RetrievalFailure) for item in self.source_failures
        ):
            raise TypeError("source_failures must contain RetrievalFailure values.")
        for name in ("input_count", "dropped_count", "token_count"):
            value = getattr(self, name)
            if not is_strict_integer(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.identity, EvidenceReductionIdentity):
            raise TypeError("identity must be an EvidenceReductionIdentity.")
        if self.token_count > self.identity.token_budget:
            raise ValueError("token_count exceeds the reduction budget.")
        if len(self.items) > self.identity.item_budget:
            raise ValueError("items exceed the reduction item budget.")
        if len(self.decisions) != self.input_count:
            raise ValueError("every input hit must have one reduction decision.")
        dropped_actions = {
            EvidenceReductionAction.DROPPED_ITEM_LIMIT,
            EvidenceReductionAction.DROPPED_DIVERSITY,
            EvidenceReductionAction.DROPPED_BUDGET,
            EvidenceReductionAction.FAILED_SOURCE_MAP,
        }
        if self.dropped_count != sum(
            decision.action in dropped_actions for decision in self.decisions
        ):
            raise ValueError("dropped_count does not match reduction decisions.")


def reduce_evidence(request: EvidenceReductionRequest) -> EvidenceReductionOutcome:
    """Fit ranked hits into exact joined-text budgets without inventing spans."""
    if not isinstance(request, EvidenceReductionRequest):
        raise TypeError("request must be an EvidenceReductionRequest.")
    source_hits = request.result.hits
    considered = source_hits[: request.item_budget]
    dropped = len(source_hits) - len(considered)
    failures: list[EvidenceReductionFailure] = []
    items: list[ReducedEvidenceItem] = []
    decisions: list[EvidenceReductionDecision] = []
    seen_documents: set[str] = set()

    for hit in considered:
        candidate = hit.candidate
        document_ids = {source.document_id.as_str() for source in candidate.sources}
        if request.diversity_policy == "per_document" and (
            not document_ids or seen_documents.intersection(document_ids)
        ):
            dropped += 1
            decisions.append(
                EvidenceReductionDecision(
                    candidate.record_id, EvidenceReductionAction.DROPPED_DIVERSITY
                )
            )
            continue
        if request.diversity_policy == "per_document":
            seen_documents.update(document_ids)
        full_text = candidate.content
        if (
            _count_joined(
                request.tokeniser, [*(item.text for item in items), full_text]
            )
            <= request.token_budget
        ):
            items.append(
                ReducedEvidenceItem(
                    hit=hit,
                    text=full_text,
                    source_ranges=(),
                    action=EvidenceReductionAction.KEPT,
                )
            )
            decisions.append(
                EvidenceReductionDecision(
                    candidate.record_id, EvidenceReductionAction.KEPT
                )
            )
            continue

        ranges = request.source_ranges.get(candidate.record_id)
        if not ranges:
            dropped += 1
            failures.append(
                EvidenceReductionFailure(candidate.record_id, "source_map_unavailable")
            )
            decisions.append(
                EvidenceReductionDecision(
                    candidate.record_id, EvidenceReductionAction.FAILED_SOURCE_MAP
                )
            )
            continue
        if not _valid_source_ranges(candidate, ranges):
            dropped += 1
            failures.append(
                EvidenceReductionFailure(candidate.record_id, "source_map_invalid")
            )
            decisions.append(
                EvidenceReductionDecision(
                    candidate.record_id, EvidenceReductionAction.FAILED_SOURCE_MAP
                )
            )
            continue
        prefix, has_safe_prefix = _longest_fitting_prefix(request, items, hit, ranges)
        if prefix is None:
            dropped += 1
            if has_safe_prefix:
                action = EvidenceReductionAction.DROPPED_BUDGET
            else:
                failures.append(
                    EvidenceReductionFailure(candidate.record_id, "source_map_invalid")
                )
                action = EvidenceReductionAction.FAILED_SOURCE_MAP
            decisions.append(EvidenceReductionDecision(candidate.record_id, action))
            continue
        text, clipped_ranges, token_count = prefix
        if not text.strip():
            dropped += 1
            decisions.append(
                EvidenceReductionDecision(
                    candidate.record_id, EvidenceReductionAction.DROPPED_BUDGET
                )
            )
            continue
        items.append(
            ReducedEvidenceItem(
                hit=hit,
                text=text,
                source_ranges=clipped_ranges,
                action=EvidenceReductionAction.TRUNCATED,
            )
        )
        decisions.append(
            EvidenceReductionDecision(
                candidate.record_id, EvidenceReductionAction.TRUNCATED
            )
        )
        # The next iteration recounts the complete joined representation.
        if token_count > request.token_budget:
            raise InvalidRetrievalContractError("reducer exceeded the token budget.")

    for hit in source_hits[len(considered) :]:
        decisions.append(
            EvidenceReductionDecision(
                hit.candidate.record_id, EvidenceReductionAction.DROPPED_ITEM_LIMIT
            )
        )

    final_count = _count_joined(request.tokeniser, [item.text for item in items])
    while items and final_count > request.token_budget:
        removed = items.pop()
        dropped += 1
        final_count = _count_joined(request.tokeniser, [item.text for item in items])
        if removed.action is EvidenceReductionAction.TRUNCATED:
            failures.append(
                EvidenceReductionFailure(
                    removed.hit.candidate.record_id, "source_map_invalid"
                )
            )
            action = EvidenceReductionAction.FAILED_SOURCE_MAP
        else:
            action = EvidenceReductionAction.DROPPED_BUDGET
        decisions = [
            EvidenceReductionDecision(
                decision.record_id,
                action
                if decision.record_id == removed.hit.candidate.record_id
                else decision.action,
            )
            for decision in decisions
        ]

    identity = EvidenceReductionIdentity(
        tokeniser=request.tokeniser.identity,
        token_budget=request.token_budget,
        item_budget=request.item_budget,
        diversity_policy=request.diversity_policy,
    )
    return EvidenceReductionOutcome(
        corpus_id=request.corpus_id,
        corpus_version=request.corpus_version,
        items=tuple(items),
        decisions=tuple(decisions),
        input_count=len(source_hits),
        dropped_count=dropped,
        token_count=final_count,
        source_status=request.result.status,
        source_failures=request.result.failures,
        identity=identity,
        failures=tuple(failures),
    )


def _longest_fitting_prefix(
    request: EvidenceReductionRequest,
    retained: list[ReducedEvidenceItem],
    hit: EvidenceHit,
    ranges: tuple[EvidenceTextSourceRange, ...],
) -> tuple[tuple[str, tuple[EvidenceTextSourceRange, ...], int] | None, bool]:
    """Try safe token boundaries longest-first against the complete joined text."""
    text = hit.candidate.content
    tokenisation = _tokenise(request.tokeniser, text)
    boundaries = sorted({offset.end for offset in tokenisation.offsets}, reverse=True)
    has_safe_prefix = False
    for end in boundaries:
        if end <= 0 or end >= len(text):
            continue
        clipped = _clip_source_ranges(ranges, end)
        if clipped is None:
            continue
        has_safe_prefix = True
        prefix = text[:end]
        token_count = _count_joined(
            request.tokeniser, [*(item.text for item in retained), prefix]
        )
        if token_count <= request.token_budget:
            return (prefix, clipped, token_count), True
    return None, has_safe_prefix


def _valid_source_ranges(
    candidate: RetrievalCandidate,
    ranges: tuple[EvidenceTextSourceRange, ...],
) -> bool:
    """Check that source mappings exactly cover candidate text and links."""
    cursor = 0
    for item in ranges:
        if item.source not in candidate.sources or item.text_start != cursor:
            return False
        cursor = item.text_end
    mapped_sources = {item.source for item in ranges}
    return cursor == len(candidate.content) and mapped_sources == set(candidate.sources)


def _clip_source_ranges(
    ranges: tuple[EvidenceTextSourceRange, ...], end: int
) -> tuple[EvidenceTextSourceRange, ...] | None:
    """Return exact source maps clipped to a candidate prefix boundary."""
    clipped: list[EvidenceTextSourceRange] = []
    cursor = 0
    for item in ranges:
        if item.text_start != cursor:
            return None
        if item.text_start >= end:
            break
        selected_end = min(end, item.text_end)
        source_end = item.source_offset_at(selected_end)
        if source_end is None:
            return None
        selected_boundaries = tuple(
            boundary
            for boundary in item.boundaries
            if item.text_start <= boundary.text_offset <= selected_end
        )
        clipped.append(
            EvidenceTextSourceRange(
                source=item.source,
                text_start=item.text_start,
                text_end=selected_end,
                source_start=item.source_start,
                source_end=source_end,
                boundaries=selected_boundaries,
            )
        )
        cursor = selected_end
        if cursor == end:
            return tuple(clipped)
    return None


def _count_joined(tokeniser: EvidenceTokeniser, values: list[str]) -> int:
    """Count the complete canonical evidence representation exactly."""
    if not values:
        return 0
    try:
        count = tokeniser.count_tokens(EVIDENCE_SEPARATOR.join(values))
    except Exception:
        raise EvidenceTokeniserError("tokeniser_unavailable") from None
    if not is_strict_integer(count) or count < 0:
        raise EvidenceTokeniserError("unmeasurable_text")
    return count


def _tokenise(tokeniser: EvidenceTokeniser, text: str) -> EvidenceTokenisation:
    """Validate exact, ordered token offsets returned by the local tokeniser."""
    try:
        result = tokeniser.tokenise(text)
    except Exception:
        raise EvidenceTokeniserError("tokeniser_unavailable") from None
    if not isinstance(result, EvidenceTokenisation):
        raise EvidenceTokeniserError("invalid_tokeniser_output")
    previous_end = 0
    for offset in result.offsets:
        if offset.end > len(text) or offset.start < previous_end:
            raise EvidenceTokeniserError("invalid_token_offsets")
        previous_end = offset.end
    try:
        count = tokeniser.count_tokens(text)
    except Exception:
        raise EvidenceTokeniserError("tokeniser_unavailable") from None
    if not is_strict_integer(count) or count != len(result.offsets):
        raise EvidenceTokeniserError("inconsistent_token_count")
    return result
