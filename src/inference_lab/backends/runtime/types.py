"""Typed request and result values for model runtime backends."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal, get_args

from inference_lab.backends.runtime._validation import (
    validate_finite_number,
    validate_integer,
    validate_non_empty_string,
    validate_semantic_version,
)


class BackendOperation(StrEnum):
    """Operation implemented by a backend."""

    EMBED = "embed"
    RERANK = "rerank"
    GENERATE = "generate"


class BackendCapability(StrEnum):
    """Optional behaviour supported by a backend."""

    BATCHING = "batching"
    STREAMING = "streaming"
    CANCELLATION = "cancellation"
    TOKEN_USAGE = "token_usage"


class GenerationEventType(StrEnum):
    """Kind of event emitted by a generator backend.

    Backend failures are raised as typed exceptions from the stream. They are
    not represented as events, which keeps successful completion distinct from
    failure and lets callers preserve partial output naturally.
    """

    DELTA = "delta"
    COMPLETED = "completed"


GenerationRole = Literal["system", "user", "assistant", "tool"]
EmbeddingPurpose = Literal["query", "document"]
EmbeddingOverLimitPolicy = Literal["truncate_tail", "reject"]

_GENERATION_ROLES: Final[frozenset[GenerationRole]] = frozenset(
    get_args(GenerationRole)
)
_EMBEDDING_OVER_LIMIT_POLICIES: Final[frozenset[EmbeddingOverLimitPolicy]] = frozenset(
    get_args(EmbeddingOverLimitPolicy)
)
_EMBEDDING_PURPOSES: Final[frozenset[EmbeddingPurpose]] = frozenset(
    get_args(EmbeddingPurpose)
)


def _validate_path_free_identifier(value: str, field_name: str) -> None:
    """Reject local paths where an immutable artefact identity is required."""
    if (
        value.startswith(("/", "\\", "~"))
        or "://" in value
        or (len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\"))
    ):
        raise ValueError(f"{field_name} must not be a local path")


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Describe vector-affecting embedding behaviour without local paths."""

    profile_version: str
    tokenizer_id: str
    tokenizer_revision: str
    query_prefix: str
    document_prefix: str
    pooling_strategy: str
    input_normalisation: str
    max_input_tokens: int | None
    over_limit_policy: EmbeddingOverLimitPolicy | None
    l2_normalised: bool
    inference_dtype: str
    device_type: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the versioned vector-preparation identity."""
        validate_semantic_version(self.profile_version, "profile_version")
        for name, value in (
            ("tokenizer_id", self.tokenizer_id),
            ("tokenizer_revision", self.tokenizer_revision),
            ("pooling_strategy", self.pooling_strategy),
            ("input_normalisation", self.input_normalisation),
            ("inference_dtype", self.inference_dtype),
            ("device_type", self.device_type),
        ):
            validate_non_empty_string(value, name)
        _validate_path_free_identifier(self.tokenizer_id, "tokenizer_id")
        _validate_path_free_identifier(self.tokenizer_revision, "tokenizer_revision")
        if not isinstance(self.query_prefix, str):
            raise TypeError("query_prefix must be a string")
        if not isinstance(self.document_prefix, str):
            raise TypeError("document_prefix must be a string")
        if self.max_input_tokens is not None:
            validate_integer(self.max_input_tokens, "max_input_tokens")
            if self.max_input_tokens <= 0:
                raise ValueError("max_input_tokens must be positive")
        if self.over_limit_policy is not None:
            if not isinstance(self.over_limit_policy, str):
                raise TypeError("over_limit_policy must be a string or None")
            if self.over_limit_policy not in _EMBEDDING_OVER_LIMIT_POLICIES:
                raise ValueError("unsupported embedding over-limit policy")
        if (self.max_input_tokens is None) != (self.over_limit_policy is None):
            raise ValueError(
                "max_input_tokens and over_limit_policy must be set together"
            )
        if not isinstance(self.l2_normalised, bool):
            raise TypeError("l2_normalised must be a boolean")


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
        validate_non_empty_string(self.backend_name, "backend_name")
        validate_non_empty_string(self.model_name, "model_name")
        validate_non_empty_string(self.model_revision, "model_revision")
        if not isinstance(self.supported_operations, tuple):
            raise TypeError("supported_operations must be a tuple")
        if not self.supported_operations:
            raise ValueError("supported_operations must not be empty")
        if any(
            not isinstance(operation, BackendOperation)
            for operation in self.supported_operations
        ):
            raise TypeError("supported_operations must contain BackendOperation values")
        if len(set(self.supported_operations)) != len(self.supported_operations):
            raise ValueError("supported_operations must not contain duplicates")
        if not isinstance(self.capabilities, tuple):
            raise TypeError("capabilities must be a tuple")
        if any(
            not isinstance(capability, BackendCapability)
            for capability in self.capabilities
        ):
            raise TypeError("capabilities must contain BackendCapability values")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must not contain duplicates")

        supports_generation = BackendOperation.GENERATE in self.supported_operations
        supports_streaming = BackendCapability.STREAMING in self.capabilities
        if supports_generation != supports_streaming:
            raise ValueError(
                "generate operation and streaming capability must be advertised together"
            )

        if self.embedding_dimensions is not None:
            validate_integer(self.embedding_dimensions, "embedding_dimensions")
            if self.embedding_dimensions <= 0:
                raise ValueError("embedding_dimensions must be positive")
            if BackendOperation.EMBED not in self.supported_operations:
                raise ValueError("embedding_dimensions requires the embed operation")


@dataclass(frozen=True, slots=True)
class EmbeddingOptions:
    """Optional controls for embedding generation."""

    dimensions: int | None = None
    normalise: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate embedding options."""
        if self.dimensions is not None:
            validate_integer(self.dimensions, "dimensions")
            if self.dimensions <= 0:
                raise ValueError("dimensions must be positive")
        if not isinstance(self.normalise, bool):
            raise TypeError("normalise must be a boolean")


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """A batch of query or document texts to embed in input order."""

    input_texts: tuple[str, ...]
    options: EmbeddingOptions = field(default_factory=EmbeddingOptions)
    purpose: EmbeddingPurpose = "document"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate embedding inputs."""
        if not isinstance(self.input_texts, tuple):
            raise TypeError("input_texts must be a tuple")
        if not self.input_texts:
            raise ValueError("input_texts must not be empty")
        if any(not isinstance(text, str) for text in self.input_texts):
            raise TypeError("input_texts must contain strings")
        if any(not text for text in self.input_texts):
            raise ValueError("input_texts must not contain empty text")
        if not isinstance(self.options, EmbeddingOptions):
            raise TypeError("options must be EmbeddingOptions")
        if not isinstance(self.purpose, str):
            raise TypeError("purpose must be a string")
        if self.purpose not in _EMBEDDING_PURPOSES:
            raise ValueError("purpose must be 'query' or 'document'")


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    """Token usage reported for an embedding request."""

    input_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate embedding usage counts."""
        validate_integer(self.input_tokens, "input_tokens")
        validate_integer(self.total_tokens, "total_tokens")
        if self.input_tokens < 0 or self.total_tokens < 0:
            raise ValueError("token counts must not be negative")
        if self.total_tokens < self.input_tokens:
            raise ValueError("total_tokens must be at least input_tokens")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors returned for an embedding request, in input order."""

    vectors: tuple[tuple[float, ...], ...]
    model_metadata: BackendMetadata
    embedding_profile: EmbeddingProfile
    usage: EmbeddingUsage | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate vector count and dimensions."""
        if not isinstance(self.vectors, tuple):
            raise TypeError("vectors must be a tuple")
        if not self.vectors:
            raise ValueError("vectors must not be empty")
        if any(not isinstance(vector, tuple) for vector in self.vectors):
            raise TypeError("vectors must contain tuples")
        for vector in self.vectors:
            for value in vector:
                validate_finite_number(value, "vector values")
        if not isinstance(self.model_metadata, BackendMetadata):
            raise TypeError("model_metadata must be BackendMetadata")
        if BackendOperation.EMBED not in self.model_metadata.supported_operations:
            raise ValueError("embedding result metadata must advertise embed")
        if not isinstance(self.embedding_profile, EmbeddingProfile):
            raise TypeError("embedding_profile must be EmbeddingProfile")
        if self.usage is not None and not isinstance(self.usage, EmbeddingUsage):
            raise TypeError("usage must be EmbeddingUsage or None")
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
        validate_non_empty_string(self.candidate_id, "candidate_id")
        validate_non_empty_string(self.candidate_text, "candidate_text")


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
        validate_non_empty_string(self.query, "query")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if any(
            not isinstance(candidate, RerankCandidate) for candidate in self.candidates
        ):
            raise TypeError("candidates must contain RerankCandidate values")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if self.top_k is not None:
            validate_integer(self.top_k, "top_k")
            if self.top_k <= 0:
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
        validate_non_empty_string(self.candidate_id, "candidate_id")
        validate_finite_number(self.score, "score")
        validate_integer(self.rank, "rank")
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
        """Validate candidate identity, rank, score ordering, and metadata."""
        if not isinstance(self.scored_candidates, tuple):
            raise TypeError("scored_candidates must be a tuple")
        if not self.scored_candidates:
            raise ValueError("scored_candidates must not be empty")
        if any(
            not isinstance(candidate, ScoredCandidate)
            for candidate in self.scored_candidates
        ):
            raise TypeError("scored_candidates must contain ScoredCandidate values")
        if not isinstance(self.model_metadata, BackendMetadata):
            raise TypeError("model_metadata must be BackendMetadata")
        if BackendOperation.RERANK not in self.model_metadata.supported_operations:
            raise ValueError("rerank result metadata must advertise rerank")

        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.scored_candidates
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("scored candidate IDs must be unique")
        ranks = tuple(candidate.rank for candidate in self.scored_candidates)
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranks must be unique")
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("ranks must be contiguous and start at one")
        scores = tuple(candidate.score for candidate in self.scored_candidates)
        if scores != tuple(sorted(scores, reverse=True)):
            raise ValueError("scores must be ordered from highest to lowest")


@dataclass(frozen=True, slots=True)
class GenerationMessage:
    """One message in a structured generation prompt."""

    role: GenerationRole
    content: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate message role and content."""
        if not isinstance(self.role, str):
            raise TypeError("role must be a string")
        if self.role not in _GENERATION_ROLES:
            raise ValueError("role must be a supported generation role")
        validate_non_empty_string(self.content, "content")


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
        if self.maximum_output_tokens is not None:
            validate_integer(self.maximum_output_tokens, "maximum_output_tokens")
            if self.maximum_output_tokens <= 0:
                raise ValueError("maximum_output_tokens must be positive")
        if self.temperature is not None:
            validate_finite_number(self.temperature, "temperature")
            if self.temperature < 0:
                raise ValueError("temperature must not be negative")
        if self.top_p is not None:
            validate_finite_number(self.top_p, "top_p")
            if not 0 < self.top_p <= 1:
                raise ValueError("top_p must be greater than zero and at most one")
        if not isinstance(self.stop_sequences, tuple):
            raise TypeError("stop_sequences must be a tuple")
        if any(not isinstance(sequence, str) for sequence in self.stop_sequences):
            raise TypeError("stop_sequences must contain strings")
        if any(not sequence for sequence in self.stop_sequences):
            raise ValueError("stop_sequences must not contain empty strings")
        if len(set(self.stop_sequences)) != len(self.stop_sequences):
            raise ValueError("stop_sequences must not contain duplicates")
        if self.seed is not None:
            validate_integer(self.seed, "seed")


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
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple")
        if not self.messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, GenerationMessage) for message in self.messages):
            raise TypeError("messages must contain GenerationMessage values")
        if not isinstance(self.options, GenerationOptions):
            raise TypeError("options must be GenerationOptions")
        if self.request_id is not None:
            validate_non_empty_string(self.request_id, "request_id")


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
        validate_integer(self.input_tokens, "input_tokens")
        validate_integer(self.output_tokens, "output_tokens")
        validate_integer(self.total_tokens, "total_tokens")
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
        if not isinstance(self.event_type, GenerationEventType):
            raise TypeError("event_type must be a GenerationEventType")
        validate_integer(self.sequence_number, "sequence_number")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must not be negative")
        if self.event_type is GenerationEventType.DELTA:
            validate_non_empty_string(self.text_delta, "text_delta")
            if self.finish_reason is not None or self.usage is not None:
                raise ValueError("delta events cannot contain terminal fields")
        elif self.event_type is GenerationEventType.COMPLETED:
            if self.text_delta is not None:
                raise ValueError("completed events cannot contain text_delta")
            validate_non_empty_string(self.finish_reason, "finish_reason")
            if self.usage is not None and not isinstance(self.usage, GenerationUsage):
                raise TypeError("usage must be GenerationUsage or None")
