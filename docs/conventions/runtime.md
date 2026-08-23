# Runtime backend contracts

The `inference_lab.backends.runtime` package defines the model-backend boundary
used by retrieval, serving, ingest, evaluation, and later runtime scheduling.
It provides protocols, immutable request and result values, typed errors,
stream validation, and a deterministic fake. It does not select or integrate
a real model implementation.

## Public API and protocol detection

Consumers import runtime contracts from `inference_lab.backends.runtime`, not
from implementation submodules. The lifecycle, embedding, reranking, and
generation protocols are structural and runtime-checkable. A backend may
satisfy them without inheriting from a project base class.

Runtime protocol checks establish that the required attributes are present.
They do not replace typed request validation or backend integration tests.

## Lifecycle

A backend begins unloaded. `load()` transitions it to ready and is idempotent
while ready. `close()` releases resources, is idempotent, and permanently
closes that backend instance.

Inference before loading raises `BackendNotReadyError`. Inference after closing
raises `BackendClosedError`; a closed backend cannot be loaded again. Backend
errors carry a boolean `retryable` hint, but retry and recovery policy belongs
to the caller.

Embedding and reranking calls are synchronous. Generation returns an
asynchronous iterator so callers can consume partial output and cancel without
waiting for complete generation.

## Metadata and value contracts

`BackendMetadata.supported_operations` is authoritative. A backend must reject
an operation it does not advertise. Operation and capability values are unique
enum members, generation and streaming are advertised together, and declared
embedding dimensions are present only for a backend that supports embedding.

Request, result, usage, and event records are immutable. Tuple fields require
tuples rather than accepting mutable containers implicitly. Booleans are not
accepted as integers or measurements, numeric values must be finite, candidate
identities are unique within one request or result, and rerank results use
contiguous ranks in descending score order.

Wrong runtime types raise `TypeError`. Values of the correct type that violate
the contract raise `ValueError`. Unsupported backend operations or request
options raise `BackendRequestError`.

## Generation stream

A valid generation stream:

1. contains only `GenerationEvent` values;
2. starts at sequence number zero and remains contiguous;
3. emits zero or more delta events;
4. ends with exactly one completion event; and
5. emits nothing after completion.

The validation adapter withholds the completion event until the upstream
iterator is exhausted. This prevents a consumer from observing successful
completion before the complete stream contract is known to be valid.

Typed `BackendError` instances and consumer cancellation propagate unchanged.
An unexpected implementation exception becomes a generic
`BackendExecutionError`; its original message and payload are suppressed from
the rendered exception chain. Invalid event objects, ordering, or termination
raise `BackendProtocolError`. Partial deltas already consumed remain available
when a later failure occurs.

## Deterministic fake backend

`FakeBackend` implements all three operation protocols by default. Custom
metadata may advertise a subset, and unadvertised operations are rejected.
The fake must be loaded before use and follows the normal permanent-close
lifecycle.

`FakeBackendConfig` fails fast when controls are malformed or contradictory:

- delays are finite, non-negative seconds;
- generation chunks are ordered, non-empty strings;
- configured embedding vectors have one finite dimension matching metadata;
- configured rerank score overrides have unique IDs and finite values;
- failure mode and operation are configured together; and
- a post-chunk generation failure names a reachable emitted-chunk boundary.

Injected synchronous and asynchronous sleepers make delay assertions
deterministic without requiring wall-clock waits.

### Embedding behaviour

Without configured vectors, the fake derives deterministic vectors from input
order. Configured vectors must match the request batch size. A requested
dimension may reduce the available vector dimension but may not exceed it.
Normalisation uses the L2 norm; an all-zero vector remains all zero. Result
metadata reports the dimension actually returned.

### Reranking behaviour

Configured candidate scores override deterministic position-based defaults.
Results are sorted by descending score with candidate ID as the stable tie
breaker, truncated by `top_k`, and assigned contiguous ranks from one.

### Generation behaviour

Configured chunks define the deterministic response. The fake uses whitespace
token counting for both usage and `maximum_output_tokens`. Token limits and the
earliest matching stop sequence truncate the response while preserving the
configured chunk boundaries. Completion reports `length` only when the token
limit truncated output; otherwise it reports `stop`.

`temperature`, `top_p`, and `seed` are rejected explicitly. A fixed-output fake
cannot represent their sampling semantics honestly.

## Logging and privacy

Runtime code uses the shared `structlog` pipeline. Lifecycle events are normal
operational records, while an operation that raises a backend failure is logged
at `ERROR`. A retryable hint does not mean the failed operation recovered;
retry attempts and successful fallback belong to the consuming layer and may
produce their own `WARNING` records.

Logs may include backend/model identity, operation, counts, error category, and
retryability. They must not include input text, candidate text, prompts,
generated chunks, raw exception messages, model output, credentials, or
sensitive paths.

## Scope boundary

The runtime contract layer does not include a tokenizer or local model, memory
accounting, real embedding, reranking, or generation integrations, cache
lifecycle operations, scheduling, residency, or microbatching. Those
capabilities build on these contracts without changing the fake into a
production model backend.
