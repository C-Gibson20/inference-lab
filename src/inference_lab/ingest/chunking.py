"""Contracts for deterministic, source-navigable ingest chunking."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._normalisation import normalise_source_text
from inference_lab._validation import is_canonical_semantic_version, is_strict_integer
from inference_lab.ingest.parser import NormalisedText, ParsedDocument
from inference_lab.provenance import CorpusId

_SLUG_PATTERN: Final = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


class StructuralBoundaryPolicy(StrEnum):
    """Closed policy names for structure-aware chunk packing."""

    GREEDY = "structure-aware-greedy"


class OversizeUnitPolicy(StrEnum):
    """Closed policy names for structures larger than the token ceiling."""

    TOKEN_WINDOW = "token-window"


@dataclass(frozen=True, slots=True)
class ChunkerVersion:
    """Identify the implementation and semantic contract of a chunker."""

    implementation: str
    contract_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, str):
            raise TypeError("chunker implementation must be a string")
        if _SLUG_PATTERN.fullmatch(self.implementation) is None:
            raise ValueError("chunker implementation must be a lowercase slug")
        if not is_canonical_semantic_version(self.contract_version):
            raise ValueError("chunker contract version must be semantic version")

    @property
    def value(self) -> str:
        """Return the canonical chunker-version value."""
        return f"{self.implementation}/contract-{self.contract_version}"


CURRENT_CHUNKER_VERSION: Final = ChunkerVersion(
    implementation="structure-aware-greedy",
    contract_version="1.0.0",
)


@dataclass(frozen=True, slots=True)
class TokeniserIdentity:
    """Identify every versioned input that can affect tokenisation."""

    implementation: str
    contract_version: str
    artefact_identity: str
    artefact_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the tokeniser identity fields."""
        if not isinstance(self.implementation, str):
            raise TypeError("tokeniser implementation must be a string")
        if _SLUG_PATTERN.fullmatch(self.implementation) is None:
            raise ValueError("tokeniser implementation must be a lowercase slug")
        if not is_canonical_semantic_version(self.contract_version):
            raise ValueError("tokeniser contract version must be semantic version")
        for field_name, value in (
            ("artefact_identity", self.artefact_identity),
            ("artefact_revision", self.artefact_revision),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value or "/" in value or "\\" in value or "\x00" in value:
                raise ValueError(f"{field_name} must be a non-path identifier")
        if not isinstance(self.options, tuple):
            raise TypeError("tokeniser options must be a tuple")
        option_keys: list[str] = []
        for option in self.options:
            if (
                not isinstance(option, tuple)
                or len(option) != 2
                or not isinstance(option[0], str)
                or not isinstance(option[1], str)
            ):
                raise TypeError("tokeniser options must contain string pairs")
            key, value = option
            if not key or key.lower() != key or "\x00" in key or "=" in key:
                raise ValueError("tokeniser option keys must be lowercase names")
            if not value or "\x00" in value or "/" in value or "\\" in value:
                raise ValueError("tokeniser option values must be path-free")
            option_keys.append(key)
        if tuple(option_keys) != tuple(sorted(set(option_keys))):
            raise ValueError("tokeniser options must be sorted and unique")

    def as_str(self) -> str:
        """Return a stable, path-free identity string."""
        options = ",".join(f"{key}={value}" for key, value in self.options)
        return (
            f"{self.implementation}/contract-{self.contract_version}/"
            f"artefact-{self.artefact_identity}@{self.artefact_revision}/options-{options}"
        )


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    """Validated size and boundary policy for deterministic chunking."""

    min_tokens: int = 256
    max_tokens: int = 512
    overlap_numerator: int = 15
    overlap_denominator: int = 100
    structural_boundary_policy: StructuralBoundaryPolicy = (
        StructuralBoundaryPolicy.GREEDY
    )
    oversize_unit_policy: OversizeUnitPolicy = OversizeUnitPolicy.TOKEN_WINDOW

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the chunking policy is internally consistent and well-typed."""
        for field_name in (
            "min_tokens",
            "max_tokens",
            "overlap_numerator",
            "overlap_denominator",
        ):
            value = getattr(self, field_name)
            if not is_strict_integer(value):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.min_tokens == 0 or self.max_tokens == 0:
            raise ValueError("chunk token limits must be positive")
        if self.min_tokens > self.max_tokens:
            raise ValueError("minimum token limit must not exceed maximum")
        if self.overlap_denominator == 0:
            raise ValueError("overlap denominator must be positive")
        if self.overlap_numerator > self.overlap_denominator:
            raise ValueError("overlap numerator must not exceed denominator")
        if not isinstance(
            self.structural_boundary_policy,
            StructuralBoundaryPolicy,
        ):
            raise TypeError("invalid structural boundary policy")
        if not isinstance(self.oversize_unit_policy, OversizeUnitPolicy):
            raise TypeError("invalid oversize unit policy")


@dataclass(frozen=True, slots=True)
class ChunkingRequest:
    """Describe one corpus-scoped, storage-neutral chunking operation."""

    corpus_id: CorpusId
    parsed_document: ParsedDocument
    tokeniser: object
    policy: ChunkingPolicy = ChunkingPolicy()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the chunking request is internally consistent and well-typed."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId")
        if not isinstance(self.parsed_document, ParsedDocument):
            raise TypeError("parsed_document must be a ParsedDocument")
        if not hasattr(self.tokeniser, "tokenise") or not hasattr(
            self.tokeniser, "count_tokens"
        ):
            raise TypeError("tokeniser must implement the chunk tokeniser boundary")
        if not isinstance(self.policy, ChunkingPolicy):
            raise TypeError("policy must be a ChunkingPolicy")


@dataclass(frozen=True, slots=True)
class SourceBoundary:
    """Pair one safe decoded-source boundary with its canonical offset."""

    source_offset: int
    canonical_offset: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the source boundary is well-typed and non-negative."""
        if not is_strict_integer(self.source_offset):
            raise TypeError("source offset must be an integer")
        if not is_strict_integer(self.canonical_offset):
            raise TypeError("canonical offset must be an integer")
        if self.source_offset < 0 or self.canonical_offset < 0:
            raise ValueError("boundary offsets must not be negative")


class UnsafeSourceBoundaryError(ValueError):
    """Report that a canonical/decoded boundary cannot be mapped exactly."""

    category = "unsafe_normalisation_boundary"

    def __init__(self) -> None:
        """Create a payload-free boundary error."""
        super().__init__(self.category)


@dataclass(frozen=True, slots=True)
class CanonicalSourceBoundaryMap:
    """Map only exact, bidirectional boundaries between source and text."""

    source_text: str
    canonical_text: str
    boundaries: tuple[SourceBoundary, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the boundary map is internally consistent and well-typed."""
        if not isinstance(self.source_text, str) or not isinstance(
            self.canonical_text, str
        ):
            raise TypeError("boundary-map text must be strings")
        if self.canonical_text != normalise_source_text(self.source_text):
            raise ValueError("canonical text does not match source normalisation")
        if not isinstance(self.boundaries, tuple) or not self.boundaries:
            raise ValueError("boundary map must contain boundaries")
        if any(
            not isinstance(boundary, SourceBoundary) for boundary in self.boundaries
        ):
            raise TypeError("boundaries must contain SourceBoundary values")
        expected_first = SourceBoundary(0, 0)
        expected_last = SourceBoundary(len(self.source_text), len(self.canonical_text))
        if self.boundaries[0] != expected_first:
            raise ValueError("boundary map must start at zero")
        if self.boundaries[-1] != expected_last:
            raise ValueError("boundary map must end at both text lengths")
        for previous, current in zip(self.boundaries, self.boundaries[1:]):
            if current.source_offset <= previous.source_offset:
                raise ValueError("source boundaries must be strictly increasing")
            if current.canonical_offset <= previous.canonical_offset:
                raise ValueError("canonical boundaries must be strictly increasing")
        for start, end in zip(self.boundaries, self.boundaries[1:]):
            if (
                normalise_source_text(
                    self.source_text[start.source_offset : end.source_offset]
                )
                != self.canonical_text[start.canonical_offset : end.canonical_offset]
            ):
                raise ValueError("boundary map contains a non-exact segment")

    def source_offset_for_canonical(self, canonical_offset: int) -> int:
        """Return the source offset for one safe canonical boundary."""
        if not is_strict_integer(canonical_offset):
            raise TypeError("canonical offset must be an integer")
        for boundary in self.boundaries:
            if boundary.canonical_offset == canonical_offset:
                return boundary.source_offset
        raise UnsafeSourceBoundaryError()

    def canonical_offset_for_source(self, source_offset: int) -> int:
        """Return the canonical offset for one safe source boundary."""
        if not is_strict_integer(source_offset):
            raise TypeError("source offset must be an integer")
        for boundary in self.boundaries:
            if boundary.source_offset == source_offset:
                return boundary.canonical_offset
        raise UnsafeSourceBoundaryError()

    def source_range_for_canonical_range(
        self,
        canonical_start: int,
        canonical_end: int,
    ) -> tuple[int, int]:
        """Map an exact canonical range to its decoded source range."""
        if canonical_end < canonical_start:
            raise ValueError("canonical range end must not precede start")
        source_start = self.source_offset_for_canonical(canonical_start)
        source_end = self.source_offset_for_canonical(canonical_end)
        return source_start, source_end

    def canonical_text_for_source_range(
        self, source_start: int, source_end: int
    ) -> str:
        """Normalise one safe source slice and verify its canonical text."""
        if source_end < source_start:
            raise ValueError("source range end must not precede start")
        canonical_start = self.canonical_offset_for_source(source_start)
        canonical_end = self.canonical_offset_for_source(source_end)
        selected = normalise_source_text(self.source_text[source_start:source_end])
        expected = self.canonical_text[canonical_start:canonical_end]
        if selected != expected:
            raise UnsafeSourceBoundaryError()
        return selected


def _normalisation_units(source_text: str) -> tuple[tuple[str, int, int], ...]:
    """Represent newline-normalised characters with original source spans."""
    units: list[tuple[str, int, int]] = []
    offset = 0
    while offset < len(source_text):
        character = source_text[offset]
        if character == "\r":
            if offset + 1 < len(source_text) and source_text[offset + 1] == "\n":
                units.append(("\n", offset, offset + 2))
                offset += 2
            else:
                units.append(("\n", offset, offset + 1))
                offset += 1
        else:
            units.append((character, offset, offset + 1))
            offset += 1
    return tuple(units)


def build_canonical_source_boundary_map(
    text: NormalisedText,
) -> CanonicalSourceBoundaryMap:
    """Build a deterministic map for exact decoded-source navigation."""
    if not isinstance(text, NormalisedText):
        raise TypeError("text must be NormalisedText")
    units = _normalisation_units(text.source_text)
    if not units:
        return CanonicalSourceBoundaryMap(
            source_text=text.source_text,
            canonical_text=text.text,
            boundaries=(SourceBoundary(0, 0),),
        )

    transformed = "".join(unit[0] for unit in units)
    segments: list[tuple[int, int]] = []
    segment_start = 0
    segment_text = units[0][0]
    for unit_index in range(1, len(units)):
        next_character = units[unit_index][0]
        combined = segment_text + next_character
        if normalise_source_text(combined) == (
            normalise_source_text(segment_text) + normalise_source_text(next_character)
        ):
            segments.append((segment_start, unit_index))
            segment_start = unit_index
            segment_text = next_character
        else:
            segment_text = combined
    segments.append((segment_start, len(units)))

    boundaries: list[SourceBoundary] = [SourceBoundary(0, 0)]
    canonical_offset = 0
    for segment_start, segment_end in segments:
        raw_segment = transformed[segment_start:segment_end]
        normalised_segment = normalise_source_text(raw_segment)
        for relative_offset in range(1, len(raw_segment)):
            left = normalise_source_text(raw_segment[:relative_offset])
            right = normalise_source_text(raw_segment[relative_offset:])
            if left + right != normalised_segment:
                continue
            source_offset = units[segment_start + relative_offset - 1][2]
            candidate = SourceBoundary(
                source_offset=source_offset,
                canonical_offset=canonical_offset + len(left),
            )
            if candidate != boundaries[-1]:
                boundaries.append(candidate)
        canonical_offset += len(normalised_segment)
        source_offset = units[segment_end - 1][2]
        terminal = SourceBoundary(source_offset, canonical_offset)
        if terminal != boundaries[-1]:
            boundaries.append(terminal)

    return CanonicalSourceBoundaryMap(
        source_text=text.source_text,
        canonical_text=text.text,
        boundaries=tuple(boundaries),
    )


NormalisationBoundaryMap = CanonicalSourceBoundaryMap
build_normalisation_boundary_map = build_canonical_source_boundary_map
