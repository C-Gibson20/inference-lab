"""Typed request and result values for model runtime backends."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class BackendOperation(StrEnum):
    """Operation implemented by a backend."""

    EMBED = "embed"
    RERANK = "rerank"
    GENERATE = "generate"


class BackendCapability(StrEnum):
    """Optional behavior supported by a backend."""

    BATCHING = "batching"
    STREAMING = "streaming"
    CANCELLATION = "cancellation"
    TOKEN_USAGE = "token_usage"


class GenerationEventType(StrEnum):
    """Kind of event emitted by a generator backend."""

    DELTA = "delta"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    """Stable identity and capabilities reported by a backend."""

    backend_name: str
    model_name: str
    model_revision: str
    supported_operations: tuple[BackendOperation, ...]
    capabilities: tuple[BackendCapability, ...]
    embedding_dimensions: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate backend identity and capability metadata."""
        if not self.backend_name:
            raise ValueError("backend_name must not be empty")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.model_revision:
            raise ValueError("model_revision must not be empty")
        if self.embedding_dimensions is not None and self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")


@dataclass(frozen=True, slots=True)
class EmbeddingOptions:
    """Optional controls for embedding generation."""

    dimensions: int | None = None
    normalize: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate embedding options."""
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """A batch of texts to embed in input order."""

    input_texts: tuple[str, ...]
    options: EmbeddingOptions = field(default_factory=EmbeddingOptions)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate embedding inputs."""
        if not self.input_texts:
            raise ValueError("input_texts must not be empty")
        if any(not text for text in self.input_texts):
            raise ValueError("input_texts must not contain empty text")


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    """Token usage reported for an embedding request."""

    input_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate embedding usage counts."""
        if self.input_tokens < 0 or self.total_tokens < 0:
            raise ValueError("token counts must not be negative")
        if self.total_tokens < self.input_tokens:
            raise ValueError("total_tokens must be at least input_tokens")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors returned for an embedding request, in input order."""

    vectors: tuple[tuple[float, ...], ...]
    model_metadata: BackendMetadata
    usage: EmbeddingUsage | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate vector count and dimensions."""
        if not self.vectors:
            raise ValueError("vectors must not be empty")
        dimensions = len(self.vectors[0])
        if dimensions == 0:
            raise ValueError("vectors must not be empty vectors")
        if any(len(vector) != dimensions for vector in self.vectors):
            raise ValueError("all vectors must have the same dimension")
        declared_dimensions = self.model_metadata.embedding_dimensions
        if declared_dimensions is not None and declared_dimensions != dimensions:
            raise ValueError("vector dimension does not match backend metadata")


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """A candidate document or passage presented to a reranker."""

    candidate_id: str
    candidate_text: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate candidate identity and text."""
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.candidate_text:
            raise ValueError("candidate_text must not be empty")


@dataclass(frozen=True, slots=True)
class RerankRequest:
    """A query and ordered candidates to score."""

    query: str
    candidates: tuple[RerankCandidate, ...]
    top_k: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate reranking inputs."""
        if not self.query:
            raise ValueError("query must not be empty")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate identifier with its reranker score and rank."""

    candidate_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate reranking output values."""
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.rank <= 0:
            raise ValueError("rank must be positive")


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Candidates ranked by score, with stable candidate identities."""

    scored_candidates: tuple[ScoredCandidate, ...]
    model_metadata: BackendMetadata

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate rank uniqueness and ordering."""
        ranks = tuple(candidate.rank for candidate in self.scored_candidates)
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranks must be unique")
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("ranks must be contiguous and start at one")


@dataclass(frozen=True, slots=True)
class GenerationMessage:
    """One message in a structured generation prompt."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate message content."""
        if not self.content:
            raise ValueError("content must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Sampling and output controls for generation."""

    maximum_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate generation options."""
        if self.maximum_output_tokens is not None and self.maximum_output_tokens <= 0:
            raise ValueError("maximum_output_tokens must be positive")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        if any(not sequence for sequence in self.stop_sequences):
            raise ValueError("stop_sequences must not contain empty strings")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A structured prompt and options for text generation."""

    messages: tuple[GenerationMessage, ...]
    options: GenerationOptions = field(default_factory=GenerationOptions)
    request_id: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate generation inputs."""
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.request_id is not None and not self.request_id:
            raise ValueError("request_id must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """Token usage reported for a generation request."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate generation usage counts."""
        if min(self.input_tokens, self.output_tokens, self.total_tokens) < 0:
            raise ValueError("token counts must not be negative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input plus output tokens")


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """One ordered event emitted by a generator backend."""

    event_type: GenerationEventType
    sequence_number: int
    text_delta: str | None = None
    finish_reason: str | None = None
    usage: GenerationUsage | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate event-specific fields."""
        if self.sequence_number < 0:
            raise ValueError("sequence_number must not be negative")
        if self.event_type is GenerationEventType.DELTA:
            if not self.text_delta:
                raise ValueError("delta events require text_delta")
            if self.finish_reason is not None or self.usage is not None:
                raise ValueError("delta events cannot contain terminal fields")
        elif self.event_type is GenerationEventType.COMPLETED:
            if self.text_delta is not None:
                raise ValueError("completed events cannot contain text_delta")
            if not self.finish_reason:
                raise ValueError("completed events require finish_reason")
        elif self.event_type is GenerationEventType.FAILED:
            if self.text_delta is not None:
                raise ValueError("failed events cannot contain text_delta")
            if self.finish_reason != "error":
                raise ValueError("failed events require finish_reason='error'")
