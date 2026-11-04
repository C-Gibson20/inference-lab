# Instructions for AI coding agents

Read [docs/spec.md](docs/spec.md) before changing architecture. This project is
a local, persistent multi-corpus tool for papers and patents—not a small
uploaded-file session app. Every first-build operation targets exactly one
corpus; federated cross-corpus querying is out of scope. Evidence and
source-span provenance come before answer fluency; unsupported answers should
abstain.

## Agent role in this repo

Code implementation is done by the maintainer, not by agents, unless
explicitly asked otherwise in a session. Default agent tasks are:
- Planning: breaking spec items into issues, proposing build order
- Review: assessing skeleton/structure/PRs against the spec
- Docs/tooling: hooks, CI config, non-core-logic scripts (fair game to write)

Do not write or scaffold implementation code in `src/` (or your actual
module dirs) unless the user explicitly requests it in that session.

## Architecture

- `src/inference_lab/ingest/`: identify, deduplicate, parse/OCR, preserve spans,
  chunk, summarise, and extract.
- `src/inference_lab/provenance/`: stable document/version identity and lineage.
- `src/inference_lab/storage/`: persistent metadata and indexes.
- `src/inference_lab/retrieval/`: lexical, dense, metadata, summary, and
  structured retrieval, fusion, reranking, and deduplication.
- `src/inference_lab/serving/`: query planning, sessions, answering, citations,
  verification, and abstention.
- `src/inference_lab/backends/`: model and hardware integrations; runtime owns
  scheduling, residency, and memory accounting.
- `src/inference_lab/evaluation/`: question sets and quality/system metrics.
- `src/inference_lab/profiling/`: reusable profiling code; experiment outputs go
  in top-level `benchmarks/`.

Keep batch ingestion separate from interactive serving. Preserve links from
every derived record to source documents and spans. Prefer measurable baselines
and deterministic fixtures over speculative abstractions. Keep the 24GB
single-GPU constraint visible in runtime decisions.

Treat document and document-version IDs as deployment-global immutable
identities, with visibility controlled by explicit corpus membership. Require a
target corpus for ingest, retrieval, evaluation, sessions, and version
operations, and test that shared storage never leaks records between corpora.

## Conventions and boundaries

Use the project’s Conventional Commit format, Ruff formatting/linting, public
API docstrings, concise comments for reasoning, and privacy-safe structured
logging. Use `structlog` inside `inference_lab`; use standard-library logging
only for dependencies and process-level handlers. Do not add third-party
model/data APIs to the deployed path.

Follow the [Python style conventions](docs/conventions/python-style.md),
including the absolute-import policy for production modules.
Do not introduce entity graphs until simpler retrieval and structured queries
are shown insufficient. Do not move domain-specific ADMET experiments into the
core product boundary.

Avoid changing `docs/spec.md` to justify an implementation. Update it only when
the product decision itself changes. Avoid committing generated caches, model
artefacts, confidential documents, or benchmark outputs containing sensitive
data.

## Validation

From the repository root, run:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov
```

Storage, index, and retrieval behaviour is gated by acceptance journeys, not by
a line-coverage threshold. The journeys that must pass for a change touching
persistence or retrieval are the corpus-foundation, recovery, persistent-index,
and hybrid-pipeline suites under `tests/inference_lab/`. Read coverage as a
diagnostic that points at untested behaviour, never as a target to be met by
adding tests:

```sh
uv run coverage report --include='src/inference_lab/storage/*'
```

For changes affecting retrieval or answering, also update or run the relevant
evaluation cases and check citation correctness, faithfulness, latency, and
memory—not only Recall@10. For changes affecting ingest or runtime, report
incremental behaviour, throughput, memory, and failure modes when measurable.
