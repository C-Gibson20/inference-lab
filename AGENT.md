# Instructions for AI coding agents

Read [docs/spec.md](docs/spec.md) before changing architecture. This project is
a local, persistent corpus tool for papers and patents—not a small uploaded-file
session app. Evidence and source-span provenance come before answer fluency;
unsupported answers should abstain.

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
  chunk, summarize, and extract.
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

## Conventions and boundaries

Use the project’s Conventional Commit format, Ruff formatting/linting, public
API docstrings, concise comments for reasoning, and privacy-safe standard
library logging. Do not add third-party model/data APIs to the deployed path.
Do not introduce entity graphs until simpler retrieval and structured queries
are shown insufficient. Do not move domain-specific ADMET experiments into the
core product boundary.

Avoid changing `docs/spec.md` to justify an implementation. Update it only when
the product decision itself changes. Avoid committing generated caches, model
artifacts, confidential documents, or benchmark outputs containing sensitive
data.

## Validation

From the repository root, run:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

For changes affecting retrieval or answering, also update or run the relevant
evaluation cases and check citation correctness, faithfulness, latency, and
memory—not only Recall@10. For changes affecting ingest or runtime, report
incremental behavior, throughput, memory, and failure modes when measurable.
