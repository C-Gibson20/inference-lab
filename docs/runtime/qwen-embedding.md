# RUN-04 local Qwen embedding backend

`QwenEmbeddingBackend` is the provisional RUN-04 embedding implementation. It
implements the public runtime contract with a pinned Qwen3-Embedding-0.6B
snapshot and runs inference locally on CPU in float32. Its candidate selection
is based on the development pilot recorded in
[`benchmarks/run-04/candidate-pilot.md`](../../benchmarks/run-04/candidate-pilot.md);
that small, self-curated pilot is not final quality evidence.

## Local model files

Use the repository `Qwen/Qwen3-Embedding-0.6B` at revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` from a trusted model-acquisition
process, then set `models.embedder_path` in local `config.toml` to that
directory. The application configuration already defines this path; the
backend adds no environment variables or model-download behaviour.

The backend verifies SHA-256 identity for these required files:

| File | SHA-256 |
| --- | --- |
| `config.json` | `b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9` |
| `merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `model.safetensors` | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| `tokenizer.json` | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |
| `tokenizer_config.json` | `253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

Additional snapshot files are allowed. Missing files, symlinks in place of a
required file, changed hashes, or incompatible model/tokeniser configuration
fail loading with a generic backend error. Loading and inference use local
files only; no model registry fallback is enabled.

## Request behaviour

- `EmbeddingRequest.purpose="document"` (the default) embeds passage text
  without a prefix.
- `EmbeddingRequest.purpose="query"` prepends `Instruct: Given a scientific
  literature search query, retrieve passages that answer the query`, a newline,
  and `Query: ` before the text.
- Input text is otherwise passed to the pinned tokeniser unchanged.
- Prefix and special tokens count toward the 4096-token limit. Oversized input
  is truncated at the text tail while preserving the prefix and special tokens.
  A structured `embedding_input_truncated` warning reports only bounded counts
  and the token limit.
- The fixed internal batch size is two. Results keep the request order.
- Dimensions from 32 through 1024 are supported by retaining the leading
  dimensions. L2 normalisation follows `EmbeddingOptions.normalise` (default
  `false`).
- Empty batches and empty strings are rejected by the shared request value;
  whitespace-only text raises `BackendRequestError`.
- Lifecycle calls are idempotent for `load()` while ready and `close()`. A
  closed instance cannot be loaded or used again.

Logs include model identity, operation, counts, and failure category. They do
not include input text, token IDs, local paths, or raw library exceptions.
Errors loading a model are surfaced as generic typed backend errors so that
dependency messages cannot leak paths or payloads.

The development pilot measured 457 input tokens per second, a 1.6-second cold
load, and 5.11 GB peak process RSS on an Apple Silicon CPU with batch size two
and float32. These are local observations, not performance guarantees. The
pilot does not establish GPU throughput or full allocator/activation memory.

For a repeatable local load and throughput observation, use the pinned model
directory and committed input fixture:

```sh
uv run python scripts/inference_lab/backends/benchmark_local_embedding.py \
  --model-path /path/to/pinned/local/model \
  --output benchmarks/run-04/results/qwen-local-measurement.json
```

The report records the verified model and tokenizer identity, vector profile,
host/software metadata, request and internal batch sizes, cold load time, warm
inference throughput, and input fixture digest. It contains aggregate counts,
not input text or the local model path. The benchmark uses a non-corpus fixed
input fixture to measure backend throughput; the separate candidate pilot
contains the provisional corpus-sample comparison.

`QwenEmbeddingBackend.measure_memory()` implements RUN-03's optional memory
accounting capability. It reports unique model tensor storage as an embedder
measurement. The current calibration does not measure allocator reservation,
process RSS, activation peaks, framework overhead, power, or thermal behaviour.
The included report labels those limits rather than treating them as zero.

The verified local measurement on 2026-09-29 recorded a 1,776.869 ms model
load, 12.05 embeddings/s, and 120.46 input tokens/s using the eight-input fixed
fixture, two warmups, five measured batches, and backend batch size two. The
real backend returned eight finite 1,024-dimensional vectors, repeated within
`rtol=1e-5, atol=1e-6`, and preserved order for reversed input. RUN-03 measured
2,383,106,304 bytes of unique model tensor storage. These are local
observations; the short fixed-input throughput is not comparable to the
full-page candidate comparison. See the [machine-readable report](../../benchmarks/run-04/results/qwen-local-measurement.json)
for the full environment, profile, and memory status.
