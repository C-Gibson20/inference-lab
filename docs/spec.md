# Specification

## Summary

A locally hosted tool for asking questions across a corpus of papers and patents. Documents are processed once at ingest; questions are answered in seconds with verifiable citations. In deployment, no document, query, or telemetry leaves the machine.

The engineering focus is the boundary between batch throughput and interactive serving: efficient ingestion, evidence-grounded retrieval, context management, and hardware-aware scheduling on one GPU.

## Product decision

This is a persistent corpus tool for thousands of documents. Ingest may run overnight; query-time context should remain small, typically around 8K tokens.

A session tool for a handful of uploaded files is a separate product. It may be added later using the same indexing infrastructure, but it is not part of the first build.

## Design principles

1. **Evidence before fluency.** Answers must be grounded in source passages, not merely plausible retrieved text.
2. **Hybrid retrieval.** Exact terms, semantic similarity, metadata, summaries, and structured data each solve different classes of question.
3. **Derived knowledge remains traceable.** Every summary, extraction, and aggregate links back to source documents and spans.
4. **Measure before optimizing.** The project’s question set is the primary test suite.
5. **Keep the deployed system local.** Development and benchmarking may use public models and public corpora; confidential deployment data may not leave the machine.

## Supported questions

- **Lookup:** “What assay did this paper use?”
- **Comparison:** “How do these two approaches differ?”
- **Thematic:** “What methods have been tried for X?”
- **Aggregation:** “How many patents mention Y after 2018?”
- **Follow-up:** “What about the second compound?”

The system should support more than one retrieval mode for a question. Routing selects and budgets retrieval channels; it does not make them mutually exclusive.

## Risk posture

This is an experimental first system, not a claim that the initial ingest and retrieval design is universally optimal. The project should prefer measurement and observation over speculative complexity:

- establish a baseline;
- instrument quality, latency, throughput, memory, and failure modes;
- test against real questions and documents;
- change the design when evidence shows that the current approach is insufficient.

The document types, ingest strategy, summaries, extraction schema, and routing policy are therefore hypotheses to be tested. They should not be treated as permanent architecture until the evaluation set and real use expose their limits.

## System architecture

```
Documents
  → identify, deduplicate, parse, OCR when needed
  → preserve pages, sections, layout, and source spans
  → chunk and index
  → summarize and extract structured facts

Question
  → rewrite using conversation state
  → classify intent and entities
  → query lexical, dense, metadata, summary, and structured indexes
  → fuse, rerank, and deduplicate evidence
  → generate an answer with source-span citations
  → verify citations and abstain when evidence is insufficient
```

## Ingestion

### Source preservation

For every document, retain:

- stable document ID and content hash;
- original file and publication identifiers;
- document version and duplicate relationships;
- page, section, paragraph, and bounding-box information where available;
- parser and OCR versions;
- links from every derived record to its source span.

The system must distinguish document identity from document version. Re-ingesting an unchanged document should be a no-op.

### Parsing and OCR

The corpus audit must record:

- document types and languages;
- page counts and token counts;
- scanned versus text-native documents;
- OCR coverage and quality;
- duplicate and family-member rates;
- tables, figures, claims, examples, references, and other important structure.

OCR is enabled only for documents that need it. OCR quality is evaluated by its effect on retrieval and citations, not only by character-level accuracy.

### Chunking and indexing

The baseline uses 256–512-token chunks with approximately 15% overlap, but chunk boundaries should respect sections, paragraphs, claims, tables, and figure captions where possible.

Build these indexes:

- lexical search, such as BM25, for exact terms and identifiers;
- dense vector search for semantic similarity;
- metadata filters for dates, assignees, document type, and other fields;
- structured database indexes for extracted entities and facts;
- summary indexes for corpus navigation.

The system must support incremental updates and persistent indexes without rebuilding unchanged documents.

### Summaries

Generate summaries at multiple levels:

1. document;
2. topic or cluster;
3. corpus.

Summaries are retrieval and navigation aids, not authoritative evidence. Each material claim in a summary should retain links to the supporting document and source span. Summaries are versioned and invalidated when their inputs change.

### Structured extraction

Initial fields include:

- compounds;
- targets;
- assays;
- dates;
- assignees and inventors;
- publication and priority identifiers;
- methods, outcomes, and relevant measurements.

Each extraction stores the normalized value, original mention, source span, model version, and confidence. Aggregations must be reproducible from stored records and must expose uncertainty where normalization is ambiguous.

Entity graphs are out of scope until retrieval, structured extraction, and simpler relationship queries have been shown to be insufficient.

## Query and serving

### Retrieval

For each question, the planner may combine:

- lexical retrieval;
- dense retrieval;
- metadata filtering;
- structured database queries;
- summary retrieval;
- query rewriting and synonym expansion.

Results are fused, reranked, deduplicated, and reduced to a small evidence set. The final answer should cite original passages wherever possible, rather than citing only a generated summary.

### Answering

Answers must:

- cite the document and precise page or source span;
- distinguish reported facts from model synthesis;
- state when the corpus contains insufficient evidence;
- avoid implying that an unobserved result is a negative result;
- provide a route to open the cited source location.

Citation verification should check that each citation actually supports the associated claim.

### Sessions and caching

Support multi-turn conversations with:

- conversation-aware query rewriting;
- stable system prompt and history placement;
- reusable prompt-prefix and KV-cache state where supported by the runtime;
- an evidence ledger containing source IDs, spans, and prior claims;
- eviction of full retrieved text when it is no longer needed;
- cache invalidation when the corpus version changes.

The target is sub-second time to first token and stable latency across a ten-turn session. Measure first-token latency, total answer latency, tokens per second, cache hit rate, and context size separately.

## Models and runtime

The initial model plan is:

| Role | Initial choice | Notes |
| --- | --- | --- |
| Embedder | Qwen3-Embedding-0.6B or BGE-M3 | Validate license, quality, and memory use before committing |
| Reranker | BGE-reranker-v2 | Rerank approximately 100 candidates to approximately 5–10 |
| Generator | Qwen3.8-27B-class model at Q4 | Deliberately optimized for a 24GB single-GPU constraint |
| OCR | Ingest-only OCR model or system | Run only for scanned pages |

The 27B generator is a deliberate project constraint, not merely a default model choice. Its quality and broad capability reduce the risk of ruling out useful corpus-level behaviour too early. The 24GB limit creates the central systems challenge: model residency, KV-cache use, batching, scheduling, quantization, memory bandwidth, and thermal behavior must be engineered carefully.

The same generator may be used for ingest initially. This creates a known shared-resource risk: summarization, extraction, and answering may bottleneck one another or expose different quality requirements. The first implementation should measure that risk before introducing additional model complexity. Smaller or specialized models remain fallback options if observation shows that they are necessary for throughput, latency, or reliability.

The runtime must support a scheduler for:

- latency-sensitive generation;
- batch summarization and extraction;
- embedding;
- reranking;
- CPU parsing, OCR, storage, and index updates.

Benchmark model residency, model swapping, microbatching, concurrent workloads, quantization, power limits, and thermal behavior. Report quality together with throughput and memory use.

## Memory and storage accounting

Track separately:

- model weights;
- KV cache;
- activations;
- embedding and reranker memory;
- index memory;
- framework overhead;
- pinned host memory;
- storage cache.

The initial 27B Q4 design is expected to require roughly 24GB of VRAM after runtime overhead, but this must be measured rather than assumed. The embedder and reranker do not necessarily need to remain resident during interactive generation. Working within this limit is a primary engineering objective, not a reason to remove the 27B model from scope.

Also measure cold and warm retrieval, index build time, restart-to-serve time, incremental update time, and concurrent reads during ingest.

## Evaluation

### Question set

Before implementation, create 30–50 real questions with known supporting evidence. Include lookups, comparisons, thematic questions, aggregations, follow-ups, and deliberately difficult terminology.

Each question should record:

- expected answer characteristics;
- supporting documents;
- supporting passages or structured records;
- acceptable alternative answers;
- whether abstention is the correct behavior.

Keep a held-out subset for final evaluation.

### Metrics

Retrieval:

- document Recall@10;
- passage Recall@k;
- reranker precision at 5;
- duplicate and source-family handling.

Answer quality:

- citation correctness;
- citation completeness;
- faithfulness to cited evidence;
- aggregation accuracy;
- abstention accuracy;
- structured extraction precision and recall.

Systems:

- time to first token;
- end-to-end latency;
- tokens per second;
- ingest pages/minute and tokens/second;
- GPU utilization and VRAM;
- CPU utilization;
- storage throughput;
- power and thermal behavior where measurable.

The system is not considered improved if Recall@10 rises while citation correctness or answer faithfulness falls.

## Development phases

### Phase 1 — Corpus and questions

Audit the corpus and write the evaluation set. Establish scanned-page rates, duplicates, document families, and expected source evidence.

**Gate:** corpus characteristics and evaluation questions are documented.

### Phase 2 — Reproducible baseline retrieval

Implement parsing, provenance-preserving chunking, lexical search, dense search, metadata filtering, reranking, and Recall@10 evaluation.

**Gate:** the correct passages are reliably retrievable before adding summaries or extraction.

### Phase 3 — Evidence-grounded answers

Add answer generation, citations, citation verification, abstention, source navigation, and held-out answer evaluation.

**Gate:** answers are useful and verifiably supported.

### Phase 4 — Ingest knowledge layer

Add structured extraction, normalization, document summaries, topic summaries, aggregation queries, versioning, and invalidation.

**Gate:** derived records remain traceable and incremental ingest does not recompute unchanged work.

### Phase 5 — Hardware-aware serving

Add the scheduler, persistent caches, session handling, batching, model residency experiments, and full profiling.

**Gate:** the best configuration is selected using quality, latency, throughput, memory, and power together.

### Phase 6 — Interface and hardening

Add chat, inline citations, jump-to-source, dataset upload, failure recovery, corpus updates, restart behavior, and operational privacy checks.

## Development model

Most development should run locally. Build these components early:

- a fake backend with configurable delay and streaming;
- a memory accountant calibrated against real measurements;
- a small local model for tokenization and cache mechanics;
- deterministic pipeline fixtures and evaluation runs;
- profiling output that records each ingest and query stage.

Use rented GPUs only for focused measurements: generator quality, absolute latency, memory ceilings, throughput, and model comparisons. Batch those experiments into planned sessions. The deployed system must self-host its models and data; documents, queries, project data, and operational data must not be sent through third-party model or data APIs.

If a failure can be reproduced without a GPU, it should be reproduced without a GPU.

## Success criteria

The project succeeds when:

1. Answers cite sufficient, correct source passages.
2. Unsupported questions produce uncertainty or abstention rather than invention.
3. Hybrid retrieval outperforms each individual retrieval method on the held-out set.
4. Structured aggregations match manually verified results.
5. Incremental ingest avoids recomputing unchanged derived data.
6. Query latency remains stable across ten-turn sessions.
7. The corpus can be served after restart without reconstruction.
8. Ingest speedups are documented with stage-level profiles.
9. Memory, throughput, quality, power, and thermal results are recorded together.
10. You use the tool every week without thinking about its internals.