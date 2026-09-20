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

Serving integrations must hold a context-managed corpus snapshot for the
request so every metadata and index read uses one immutable manifest. See the
[corpus-version runbook](corpus-versions.md) for snapshot lifetime and
historical selection.

## Supported metadata filters

The query command/API is not exposed yet, but integrations implementing the
baseline query contract must use the following closed metadata registry
(`1.0.0`). Filters are evaluated against canonical metadata reconstructed from
the same pinned corpus snapshot as lexical retrieval.

| Field | Values | Operators | Missing values |
| --- | --- | --- | --- |
| `character-count` | integer | exact, one-of, range, exists, missing | invalid record state |
| `document-type` | string | exact, one-of, exists, missing | invalid record state |
| `language` | string | exact, one-of, exists, missing | allowed; `missing` matches |
| `page-count` | integer | exact, one-of, range, exists, missing | allowed; `missing` matches |
| `source-byte-count` | integer | exact, one-of, range, exists, missing | invalid record state |
| `token-count` | integer | exact, one-of, range, exists, missing | allowed; `missing` matches |

`range` is available only for integer fields and uses the bounds explicitly
specified by the caller. `exists` and `missing` refer to the canonical value,
not to text inferred from a passage. An unsupported field is rejected; it is
not converted into a `missing` match. Dates and assignees are intentionally
deferred and are not supported filter fields in this release.

Invalid fields, operators, scalar types, or oversized filter trees fail before
ranked execution. A valid filter that retains no candidates is a successful
empty result. Storage, index, or provenance corruption fails closed without
returning hits. Query text, filter values, metadata values, passages, paths,
SQL, and raw exception details must remain out of operational logs and
profiling records; see the [logging conventions](../conventions/logging.md).

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
