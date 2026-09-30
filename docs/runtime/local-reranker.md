# RUN-05 local reranker backend

`LocalRerankerBackend` implements the runtime `RerankerBackend` contract using
the pinned BGE-reranker-v2-m3 sequence classifier. This is the provisional
integration candidate selected for PR2 because its CPU throughput and measured
weight storage were substantially lower than the two decoder candidates in the
synthetic pilot. The pilot is not representative retrieval evidence and does
not establish the production model choice. See the
[candidate comparison](../../benchmarks/run-05/README.md).

## Local model setup

Acquire `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` from the official model repository
through the approved local acquisition process. Store it outside the checkout,
then configure `models.reranker_path` in the deployment's uncommitted
`config.toml` to point at that directory. The backend does not download files,
resolve a repository ID, run remote model code, or fall back to a registry.

Before and after loading, the backend verifies the exact file inventory and
SHA-256 digests for the model, tokenizer, configuration, repository card, and
included repository assets recorded in
[`benchmarks/run-05/artifact-manifest.json`](../../benchmarks/run-05/artifact-manifest.json).
Missing, changed, symlinked, or unexpected files fail closed. Model and
tokenizer loading both set `local_files_only=True` and
`trust_remote_code=False`. Hugging Face's `.cache` metadata directory is
excluded from the artefact inventory.

## Request and ranking behaviour

- The query and each candidate passage are encoded as one BGE-M3 tokenizer pair
  with its native special tokens. Higher raw classifier logits indicate a
  stronger match; scores are not normalised.
- A query may contain at most 256 tokenizer tokens, excluding special tokens.
  Longer queries raise `BackendRequestError` so the search intent is not
  silently changed.
- A query/candidate pair may contain at most 512 tokens including special
  tokens. Candidate text is truncated at the tail to fit while preserving the
  complete query. The `reranker_candidates_truncated` warning includes only
  affected and original/retained token counts and the pair limit.
- A request may contain at most 100 candidates. Larger requests raise
  `BackendRequestError`; no subset is silently scored. Internal inference
  batches contain four pairs. All batch results map back to their original
  candidate IDs before sorting.
- Scores must be finite. Results are sorted by descending score, then ascending
  `candidate_id` for ties, assigned contiguous ranks, and truncated to `top_k`
  only after the complete request has been scored.
- The preprocessing compatibility profile is version `1.0.0`. This backend
  runs CPU float32 inference without weight quantization. It advertises only
  reranking and batching.

The shared request value rejects empty requests, duplicate candidate IDs, and
invalid `top_k` values before model inference. Lifecycle follows the runtime
contract: `load()` is idempotent while ready; inference before loading or after
closing raises the corresponding typed error; and a closed instance cannot be
reloaded.

## Errors, logging, and memory

Unexpected load or inference failures become generic typed backend errors;
out-of-memory failures become retryable `BackendResourceError` values. Logs use
the shared structured `structlog` pipeline. Lifecycle and operation events
contain bounded model identity, counts, duration, and failure category. They do
not include request text, candidate IDs or text, scores, paths, raw dependency
messages, or token IDs.

`measure_memory()` reports unique PyTorch tensor storage for model weights with
role `reranker` and category `model_weights`. It reports unavailable before
loading and when the allocation cannot be inspected. It does not claim peak
process memory, allocator reservation, activation peaks, or an accelerator
cache measurement.

The pilot measured M3 at about 2.27 GB of inspectable CPU weight storage and
19.45 candidate pairs per second on an arm64 Mac CPU with Python 3.13.12,
PyTorch 2.8.0, Transformers 4.55.4, a 40-pair pass, and batch size four. These
are local measurements, not deployment guarantees. The fixture does not
exercise long-candidate truncation or the planned approximately 100-candidate
workload. Re-run evaluation on representative labelled data and the intended
deployment hardware before fixing a production revision.
