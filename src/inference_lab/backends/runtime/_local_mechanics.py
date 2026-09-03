"""Implementation-only observations of local prefix-cache mechanics."""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import structlog
import torch
from transformers.cache_utils import DynamicCache

from inference_lab.backends.runtime._local_generation import _CausalModel
from inference_lab.backends.runtime._validation import validate_integer
from inference_lab.backends.runtime.errors import (
    BackendError,
    BackendExecutionError,
    BackendNotReadyError,
    BackendResourceError,
)
from inference_lab.backends.runtime.local import (
    _TOKENISER_REVISION,
    LocalCausalBackend,
)
from inference_lab.backends.runtime.types import BackendOperation, GenerationRequest

_LOGGER = structlog.get_logger(__name__)


class CacheObservationOutcome(StrEnum):
    """Lifecycle outcome observed by the implementation mechanics harness."""

    CREATED = "created"
    REUSED = "reused"
    REJECTED = "rejected"
    CLEARED = "cleared"


@dataclass(frozen=True, slots=True)
class LocalMechanicsObservation:
    """Payload-free observation of one native prefix-cache comparison."""

    rendered_input_tokens: int
    common_prefix_tokens: int
    tokens_evaluated_without_cache: int
    tokens_evaluated_to_create_cache: int
    tokens_evaluated_after_cache: int
    cache_outcomes: tuple[CacheObservationOutcome, ...]
    logits_equivalent: bool | None
    selection_equivalent: bool | None
    backend_name: str
    model_name: str
    model_revision: str
    tokeniser_revision: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate immutable mechanics observations."""
        for field_name in (
            "rendered_input_tokens",
            "common_prefix_tokens",
            "tokens_evaluated_without_cache",
            "tokens_evaluated_to_create_cache",
            "tokens_evaluated_after_cache",
        ):
            value = getattr(self, field_name)
            validate_integer(value, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isinstance(self.cache_outcomes, tuple):
            raise TypeError("cache_outcomes must be a tuple")
        if not self.cache_outcomes or any(
            not isinstance(outcome, CacheObservationOutcome)
            for outcome in self.cache_outcomes
        ):
            raise ValueError("cache_outcomes must contain lifecycle outcomes")
        for field_name in ("logits_equivalent", "selection_equivalent"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean or None")
        for field_name in (
            "backend_name",
            "model_name",
            "model_revision",
            "tokeniser_revision",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value:
                raise ValueError(f"{field_name} must not be empty")


class LocalMechanicsHarness:
    """Implementation-only seam for observing native prefix-cache behaviour."""

    def __init__(self, backend: LocalCausalBackend) -> None:
        """Create a mechanics harness bound to one local backend."""
        if not isinstance(backend, LocalCausalBackend):
            raise TypeError("backend must be LocalCausalBackend")
        self._backend = backend
        self._active_cache_state: DynamicCache | None = None

    def observe_prefixes(
        self,
        first_request: GenerationRequest,
        second_request: GenerationRequest,
        *,
        cache_revision: str | None = None,
    ) -> LocalMechanicsObservation:
        """Compare a full prompt with a compatible cached-prefix continuation."""
        first = self._backend._prepare_generation(first_request)
        second = self._backend._prepare_generation(second_request)
        return self.observe_token_prefixes(
            first.input_ids,
            second.input_ids,
            rendered_input_tokens=second.input_tokens,
            cache_revision=cache_revision,
        )

    def observe_token_prefixes(
        self,
        first_token_ids: tuple[int, ...],
        second_token_ids: tuple[int, ...],
        *,
        rendered_input_tokens: int | None = None,
        cache_revision: str | None = None,
    ) -> LocalMechanicsObservation:
        """Observe prefix reuse for token IDs produced by the local tokeniser."""
        rendered_input_tokens = self._validate_observation(
            first_token_ids,
            second_token_ids,
            rendered_input_tokens=rendered_input_tokens,
            cache_revision=cache_revision,
        )
        model = self._backend._model
        if model is None:
            raise BackendNotReadyError()
        common_prefix_tokens = _common_prefix_length(
            first_token_ids,
            second_token_ids,
        )
        can_reuse = common_prefix_tokens > 0 and self._cache_is_compatible(
            cache_revision
        )

        try:
            comparison = self._execute_comparison(
                cast(_CausalModel, model),
                first_token_ids,
                second_token_ids,
                common_prefix_tokens=common_prefix_tokens,
                can_reuse=can_reuse,
            )
        except BackendError:
            raise
        except MemoryError:
            raise BackendResourceError() from None
        except Exception:
            raise BackendExecutionError() from None
        finally:
            self._active_cache_state = None

        outcomes = comparison.cache_outcomes + (CacheObservationOutcome.CLEARED,)
        return self._complete_observation(
            rendered_input_tokens=rendered_input_tokens,
            common_prefix_tokens=common_prefix_tokens,
            second_token_ids=second_token_ids,
            comparison=comparison,
            outcomes=outcomes,
        )

    def _validate_observation(
        self,
        first_token_ids: tuple[int, ...],
        second_token_ids: tuple[int, ...],
        *,
        rendered_input_tokens: int | None,
        cache_revision: str | None,
    ) -> int:
        """Validate observation inputs and return the rendered token count."""
        self._backend._ensure_available()
        _validate_token_ids(first_token_ids, "first_token_ids")
        _validate_token_ids(second_token_ids, "second_token_ids")
        if rendered_input_tokens is None:
            rendered_input_tokens = len(second_token_ids)
        validate_integer(rendered_input_tokens, "rendered_input_tokens")
        if rendered_input_tokens != len(second_token_ids):
            raise ValueError("rendered_input_tokens must match second token IDs")
        if cache_revision is not None and not isinstance(cache_revision, str):
            raise TypeError("cache_revision must be a string or None")
        return rendered_input_tokens

    def _cache_is_compatible(self, cache_revision: str | None) -> bool:
        """Return whether a requested cache revision matches this backend."""
        return (
            cache_revision is None
            or cache_revision == self._backend.metadata.model_revision
        )

    def _execute_comparison(
        self,
        model: _CausalModel,
        first_token_ids: tuple[int, ...],
        second_token_ids: tuple[int, ...],
        *,
        common_prefix_tokens: int,
        can_reuse: bool,
    ) -> "_CacheComparison":
        """Execute uncached and optional cached-prefix model evaluations."""
        with torch.inference_mode():
            cache = None
            cached_logits = None
            if can_reuse:
                cache, cached_logits = _create_prefix_cache(
                    model,
                    first_token_ids,
                    common_prefix_tokens,
                )
                self._active_cache_state = cache

            full_logits = _execute_full(model, second_token_ids)
            if cache is None or cached_logits is None:
                return _CacheComparison.rejected()

            cached_logits, tokens_after_cache = _execute_reuse(
                model,
                cache,
                cached_logits,
                second_token_ids,
                common_prefix_tokens,
            )
            logits_equivalent, selection_equivalent = _compare_logits(
                full_logits,
                cached_logits,
            )
            return _CacheComparison.reused(
                tokens_to_create_cache=len(first_token_ids),
                tokens_after_cache=tokens_after_cache,
                logits_equivalent=logits_equivalent,
                selection_equivalent=selection_equivalent,
            )

    def _complete_observation(
        self,
        *,
        rendered_input_tokens: int,
        common_prefix_tokens: int,
        second_token_ids: tuple[int, ...],
        comparison: "_CacheComparison",
        outcomes: tuple[CacheObservationOutcome, ...],
    ) -> LocalMechanicsObservation:
        """Log and construct one completed mechanics observation."""
        metadata = self._backend.metadata
        _LOGGER.debug(
            "backend_cache_observation_completed",
            pipeline_stage="runtime",
            backend_name=metadata.backend_name,
            operation=BackendOperation.GENERATE.value,
            rendered_input_tokens=rendered_input_tokens,
            common_prefix_tokens=common_prefix_tokens,
            tokens_evaluated_without_cache=len(second_token_ids),
            tokens_evaluated_to_create_cache=comparison.tokens_to_create_cache,
            tokens_evaluated_after_cache=comparison.tokens_after_cache,
            cache_outcome=outcomes[-2].value,
        )
        return LocalMechanicsObservation(
            rendered_input_tokens=rendered_input_tokens,
            common_prefix_tokens=common_prefix_tokens,
            tokens_evaluated_without_cache=len(second_token_ids),
            tokens_evaluated_to_create_cache=comparison.tokens_to_create_cache,
            tokens_evaluated_after_cache=comparison.tokens_after_cache,
            cache_outcomes=outcomes,
            logits_equivalent=comparison.logits_equivalent,
            selection_equivalent=comparison.selection_equivalent,
            backend_name=metadata.backend_name,
            model_name=metadata.model_name,
            model_revision=metadata.model_revision,
            tokeniser_revision=_TOKENISER_REVISION,
        )


@dataclass(frozen=True, slots=True)
class _CacheComparison:
    """Scalar result of uncached and optional cached-prefix execution."""

    cache_outcomes: tuple[CacheObservationOutcome, ...]
    tokens_to_create_cache: int
    tokens_after_cache: int
    logits_equivalent: bool | None
    selection_equivalent: bool | None

    @classmethod
    def rejected(cls) -> "_CacheComparison":
        """Create the result for an unavailable or incompatible prefix."""
        return cls(
            cache_outcomes=(CacheObservationOutcome.REJECTED,),
            tokens_to_create_cache=0,
            tokens_after_cache=0,
            logits_equivalent=None,
            selection_equivalent=None,
        )

    @classmethod
    def reused(
        cls,
        *,
        tokens_to_create_cache: int,
        tokens_after_cache: int,
        logits_equivalent: bool,
        selection_equivalent: bool,
    ) -> "_CacheComparison":
        """Create the result for a completed cached-prefix comparison."""
        return cls(
            cache_outcomes=(
                CacheObservationOutcome.CREATED,
                CacheObservationOutcome.REUSED,
            ),
            tokens_to_create_cache=tokens_to_create_cache,
            tokens_after_cache=tokens_after_cache,
            logits_equivalent=logits_equivalent,
            selection_equivalent=selection_equivalent,
        )


def _create_prefix_cache(
    model: _CausalModel,
    token_ids: tuple[int, ...],
    common_prefix_tokens: int,
) -> tuple[DynamicCache, torch.Tensor]:
    """Execute the first request and retain logits at the shared prefix."""
    cache = DynamicCache()
    outputs = model(
        input_ids=torch.tensor([token_ids], dtype=torch.long),
        attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    return cache, outputs.logits[:, common_prefix_tokens - 1, :]


def _execute_full(
    model: _CausalModel,
    token_ids: tuple[int, ...],
) -> torch.Tensor:
    """Execute a complete request without cache reuse and return final logits."""
    outputs = model(
        input_ids=torch.tensor([token_ids], dtype=torch.long),
        attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
        use_cache=False,
    )
    return outputs.logits[:, -1, :]


def _execute_reuse(
    model: _CausalModel,
    cache: DynamicCache,
    cached_logits: torch.Tensor,
    token_ids: tuple[int, ...],
    common_prefix_tokens: int,
) -> tuple[torch.Tensor, int]:
    """Crop a first-request cache and evaluate the second-request suffix."""
    cache.crop(common_prefix_tokens)
    if cache.get_seq_length() != common_prefix_tokens:
        raise BackendExecutionError()
    suffix = token_ids[common_prefix_tokens:]
    if not suffix:
        return cached_logits, 0
    outputs = model(
        input_ids=torch.tensor([suffix], dtype=torch.long),
        attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    return outputs.logits[:, -1, :], len(suffix)


def _compare_logits(
    full_logits: torch.Tensor,
    cached_logits: torch.Tensor,
) -> tuple[bool, bool]:
    """Compare full and cached logits and their greedy token selections."""
    return (
        bool(torch.allclose(full_logits, cached_logits, rtol=0, atol=1e-6)),
        bool(
            torch.equal(
                torch.argmax(full_logits, dim=-1),
                torch.argmax(cached_logits, dim=-1),
            )
        ),
    )


def _validate_token_ids(token_ids: tuple[int, ...], field_name: str) -> None:
    """Validate a token-ID sequence without retaining its values."""
    if not isinstance(token_ids, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if not token_ids:
        raise ValueError(f"{field_name} must not be empty")
    for token_id in token_ids:
        validate_integer(token_id, f"{field_name} values")
        if token_id < 0:
            raise ValueError(f"{field_name} values must not be negative")


def _common_prefix_length(
    first_token_ids: tuple[int, ...],
    second_token_ids: tuple[int, ...],
) -> int:
    """Count equal token IDs from the beginning without exposing them."""
    for index, (first, second) in enumerate(
        zip(first_token_ids, second_token_ids, strict=False)
    ):
        if first != second:
            return index
    return min(len(first_token_ids), len(second_token_ids))
