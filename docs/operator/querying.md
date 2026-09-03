# Running queries

## Current status

The `inference-lab` command is currently a skeleton; no operator query command
or API is available yet. The first executable workflow is planned to select a
named corpus, ingest sources, and return lexical passages with precise source
locations.

Do not infer that an application startup message means that a corpus has been
ingested or that queries are available.

## Baseline query workflow

When that workflow is implemented, this page will document the concrete command
or API request, required arguments, query text, corpus/index selection, output
format, and source-location fields. The baseline workflow is:

1. create, ingest, or open a named persistent corpus;
2. select that corpus and issue a query through the documented local CLI/API;
3. inspect the resolved corpus version plus the returned document IDs,
   passages, pages, and source spans; and
4. open the cited source location before relying on a result.

Every query targets one corpus. Results from different corpora are not fused,
and shared document storage must not make a document visible outside explicit
membership. Cross-corpus comparison requires separate queries until a future
federated-query contract is added.

The baseline is lexical search. It is not yet a generated answer and should
not be described as supporting citation verification, abstention, sessions, or
follow-up questions until the relevant serving issues land.

## Supported question shapes

The product is intended to support these question types as serving capability
is added:

- lookup: identify a method, assay, or other reported detail;
- comparison: contrast two approaches;
- thematic: find methods tried for a topic;
- aggregation: count or filter structured records; and
- follow-up: resolve a question using conversation state.

Each example added here must identify the serving capability that makes it
executable. Results should prefer original passages over generated summaries.

## Evidence and privacy rules

Answers must distinguish reported facts from model synthesis, cite precise
pages or source spans, and state when the corpus is insufficient. An
unobserved result is not evidence of a negative result. Citation verification,
abstention, source opening, sessions, and chat will be documented when those
capabilities are implemented.

Queries and results remain local. Do not place query text, prompts, outputs, or
source passages in operational logs; follow the
[logging conventions](../conventions/logging.md).
