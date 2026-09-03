# Runtime backend contracts

The `inference_lab.backends.runtime` package defines the model-backend boundary
used by retrieval, serving, ingest, evaluation, and later runtime scheduling.
It provides protocols, immutable request and result values, typed errors,
stream validation, and a deterministic fake. RUN-02 additionally supplies a
small local mechanics backend for the focused acceptance suite; it does not
introduce production model selection or deployment configuration.

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

## RUN-02 local mechanics backend

RUN-02 uses the pinned PyTorch, Transformers, Tokenizers, and Safetensors stack
recorded in [`docs/runtime/local-mechanics-stack.md`](../runtime/local-mechanics-stack.md).
The checked-in fixture is a one-layer GPT-2-compatible causal model and a
project-owned BPE tokeniser. It is CPU-only, loaded from an explicit local
directory, and never acquires model artefacts from a network or model registry.
The fixture is for runtime mechanics and contract evidence, not answer quality,
production sizing, or 27B-model feasibility.

The backend pins the exact inventory and SHA-256 digest of every model,
generation, tokeniser, special-token, and chat-template file used by the
fixture. It verifies that manifest before loading and again before publishing a
ready backend. Missing, changed, or additional files fail closed under the
privacy-safe load error; a directory cannot load while advertising the pinned
fixture revision unless its complete byte identity matches. Model and tokeniser
revision strings are derived from the expected manifest rather than maintained
as independent labels.

`LocalCausalBackend` is generation-only. Its metadata advertises streaming,
cancellation, and token usage, and does not advertise embedding, reranking, or
batching. Loading is explicit and idempotent while ready; closing is idempotent
and permanently prevents reuse. The backend uses the minimum of its configured
context limit and the model's declared context limit.

Prompt accounting is based on the complete string rendered by the versioned
chat template, followed by the fast BPE tokeniser. The rendered prompt is not
reconstructed from message lengths. Generation is greedy and repeatable for
the pinned fixture; sampling options (`temperature`, `top_p`, and `seed`) are
rejected because this backend does not implement their semantics. Usage reports
the rendered input token count, generated token count, and their total. Stop
sequences are buffered across token and delta boundaries and matched text is
never emitted.

Generation uses the model's native past/KV state for autoregressive
continuation and returns an asynchronous stream validated by the RUN-01
stream adapter. The implementation-only `LocalMechanicsHarness` compares full
uncached execution with cache state produced by executing the complete first
request. For an extending, diverging, or shorter second request, it crops that
native cache to the shared prefix and evaluates only the uncached suffix. It
checks next-token selection and logits equivalence and separately records the
tokens evaluated to create the first cache and after reuse. It records token
counts, cache outcomes, and stable identity only; it does not expose cache
handles, ownership, persistence, eviction, memory bytes, or a public cache
protocol. Those lifecycle and accounting concerns remain downstream work for
RUN-07 and RUN-03 respectively.

The focused offline CPU acceptance suite is:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run pytest tests/inference_lab/backends/runtime/test_local.py
```

The normal repository gate remains authoritative and must also pass Ruff,
Ruff format checking, Pyright, and the complete Pytest suite. RUN-02 logging
uses the runtime policy: lifecycle records at `INFO`, mechanics at `DEBUG`,
failures at `ERROR`, and no prompts, generated chunks, raw exception messages,
credentials, or sensitive paths.

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

The runtime contract layer does not include a tokenizer or local model, real
embedding, reranking, or generation integrations, cache lifecycle operations,
scheduling, residency, or microbatching. Those capabilities build on these
contracts without changing the fake into a production model backend.

## Component memory accounting

The optional `MemoryAccountingAdapter` capability reports one-shot,
component-attributed observations through the runtime memory contract. It is
not part of the lifecycle, embedding, reranking, or generation protocols, so
backends that do not implement memory reporting remain valid.

Memory measurements use the profiling memory categories and
`measured`/`unavailable`/`unreliable` statuses. Byte values are canonical
non-negative integers. Component role and memory category are independent: for
example, generator weights are `model_weights`, while an explicitly coarse
embedder total may use `embedding_reranker_memory`.

Memory snapshot schema version 2 separates the identity of the collector that
assembles the snapshot from the identity of each adapter that supplies a
measurement. Every snapshot declares a non-empty expected-contribution
inventory. Each entry binds one component, role, category, memory space and
device slot to its responsible adapter, so one snapshot may compose multiple
adapters without weakening attribution. Schema version 1 snapshots are not
accepted as version 2 data.

Exclusive contributions are counted once. The `disallow` overlap policy
rejects duplicate expected slots and duplicate or unplanned exclusive
measurements. Informational roll-ups declare the measurement IDs they cover
and are excluded from totals. Snapshot validation rejects missing roll-up
coverage, cross-space or cross-device coverage, and cycles. Adapters and the
collector remain responsible for the physical non-overlap invariant: distinct
exclusive slots must refer to disjoint allocations.

Memory totals are derived from the expected inventory, not merely from the
measurements that happened to arrive. An absent expected contribution is
reported separately as `missing`; unreliable, unavailable, and missing counts
all make the affected total incomplete. Memory totals are never combined
across accelerator, ordinary host, pinned-host, and storage-cache spaces.
Accelerator contributions without a resolved device make every concrete
device budget in that snapshot incomplete, and a measured accelerator value
must identify its device. A budget comparison is `within_budget` only when all
expected contributions affecting that device are measured and reliable; an
incomplete result is `indeterminate` unless the reliable measured subtotal
already breaches the budget.

Budget comparison state, provisional headroom, and completeness are derived
facts, not independently selectable fields. New comparisons must be created
from the budget and accounting counts through the canonical factory. Direct
construction and deserialisation validate the same relationships: headroom is
the budget minus measured bytes, completeness agrees with all uncertainty
counts, positive bytes and complete results have measured coverage, and state
agrees with breach and completeness.

The PyTorch calibration adapter is deliberately limited to inspectable tensor
backing storage. It counts each storage allocation once, including tied
parameters and multiple tensor views. It does not claim exact allocator
overhead, process RSS deltas, activation lifetime, or device hardware capacity.
Unsupported or non-inspectable categories are reported as `unavailable`.
