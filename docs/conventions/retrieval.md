# Retrieval contract and boundaries

The retrieval package owns immutable, channel-neutral values shared by lexical,
dense, metadata, structured, and summary retrieval. It carries private query
and candidate content, but it does not log or persist those payloads.

## Query and filters

`RetrievalQuery` preserves non-blank query text exactly and selects one
`CorpusVersionId`, whose owning `CorpusId` is validated by the execution and
storage boundary. It optionally carries a typed filter tree and requires a
strictly positive `RetrievalLimit`. Filters support exact values, non-empty
typed sets, bounded ranges, presence/missing checks, and non-empty all/any/not
composition. Field names are lowercase project-owned slugs. Filter values are
strict strings, integers, finite floats, booleans, or calendar dates; no value
is normalised, rewritten, or inferred. `OneOfFilter` values are set-like and
are sorted canonically within their single scalar type. `AllFilter` and
`AnyFilter` child order remains the producer's expression/planning order and is
preserved in canonical bytes; consumers must not infer relevance or authority
from that order.

### Executable metadata filters

RET-03's executable metadata registry is versioned independently from the
retrieval wire schema. The current registry is `1.0.0` and is deliberately
closed: a field is executable only when it appears in this matrix.

| Field | Type | Allowed operators | Missing policy | Durable source |
| --- | --- | --- | --- | --- |
| `character-count` | integer | exact, one-of, range, exists, missing | required | parsed representation |
| `document-type` | string | exact, one-of, exists, missing | required | source artefact |
| `language` | string | exact, one-of, exists, missing | nullable | parsed representation |
| `page-count` | integer | exact, one-of, range, exists, missing | nullable | parsed representation |
| `source-byte-count` | integer | exact, one-of, range, exists, missing | required | source artefact |
| `token-count` | integer | exact, one-of, range, exists, missing | nullable | parsed representation |

`range` is restricted to integer fields and honours the filter's explicit
inclusive or exclusive bounds. `exists` and `missing` test the reconstructed
value in the pinned snapshot. A missing value is not the same as an unsupported
field: `MissingFilter("language")` is valid, while an unregistered field is
invalid input. Required fields are expected to be present in a valid reachable
record; corruption or ambiguity is a failure, not a normal missing value.

The execution boundary validates the complete filter before ranked index
execution, resolves metadata and lexical candidates from one owned immutable
corpus snapshot, applies the predicate before the final lexical limit, and
fails closed if the required source or parsed provenance cannot be reconstructed.
Dates and assignees are not in the current registry and are assigned to
RET-11 after provenance-backed extraction and normalisation. Callers must
not approximate them with string fields or interpret their absence as a
negative result.

## Candidates, scores, and results

Every `RetrievalCandidate` has a `RecordId`, a candidate-kind authority label,
non-empty retrieval content, and one or more unique source links. Each source
link retains `DocumentId`, `DocumentVersionId`, and `SourceSpanId`. Original
passages, structured records, and generated summaries remain distinguishable;
kind, rank, and score never prove claim support.

For baseline passage retrieval, ING-04 supplies the exact immutable chunk text,
record identity, token count, and source-linked ranges. Retrieval indexes and
filters those chunks within the selected corpus version; it does not rewrite
chunk content or infer new source locations. See the [chunking conventions](chunking.md)
for the producer contract.

Every channel contribution has a positive channel-local rank. A raw score is
optional, and when present it must have a non-empty channel-owned semantics
label and remains uninterpreted. Scores from different channels are not
comparable here. Evidence hits have contiguous result-local ranks, unique
candidate identities, and at most one contribution per channel. Producers use
canonical `RecordId` ordering to break otherwise equal exposed ordering inputs.

Results are scoped to one corpus version:

- `completed` may contain hits or zero hits and has no failures;
- `partial` has at least one hit and at least one typed failure; and
- `failed` has no hits and at least one typed failure.

Every result retains the query's `requested_limit` and must contain no more
hits than that limit. A corpus-version mismatch fails closed: an unscoped
mismatch requires a failed, hit-free result, while a channel-scoped mismatch
may appear in a partial aggregate only when none of its hits contains a score
contribution from that channel. Failure values expose only a bounded category,
optional channel, and retryability; they never carry raw exception messages or
payloads.

Every retrieval execution therefore targets exactly one corpus. Each channel
must validate that its index generation belongs to the query's corpus version
and must restrict candidates to document and record membership in that
generation. A shared physical catalogue or index does not relax this filter.
Fusion and reranking cannot introduce a candidate that was not visible in the
selected corpus version.

Cross-corpus or federated retrieval is outside this contract and the first
build. Comparing corpora requires separate queries and results; callers must
not combine their hits into a result labelled as one corpus version.

## RET-06 limits

Reranking uses the typed `[retrieval]` settings owned by
`inference_lab.settings.RetrievalConfig`:

| Setting | Environment override | Default | Valid range |
| --- | --- | --- | --- |
| `retrieval.evidence_count` | `RETRIEVAL__EVIDENCE_COUNT` | `8` | `5` through `10` |
| `retrieval.candidate_limit` | `RETRIEVAL__CANDIDATE_LIMIT` | `100` | `1` through the RUN-05 hard cap of `100` |

The candidate limit bounds the pool presented to RUN-05. The selected evidence
count is a separate downstream maximum; fewer passages may be returned when
fewer eligible candidates exist. Neither setting changes the RET-01 wire
schema or makes reranker scores evidence of claim support.

RET-06 exact deduplication policy `1.0.0` keys evidence by candidate kind and
the complete set of `(document_version_id, source_span_id)` links. It retains
the highest-ranked candidate for a repeated key and then makes result ranks
contiguous. It does not collapse candidates because they share a document,
family, or overlapping source ranges.

Retryable RUN-05 availability and resource failures may return the bounded
fused order as an `EvidenceResult` with `partial` status and a typed failure.
The RET-06 execution outcome exposes `degraded=true` separately, leaving the
RET-01 serialised result contract unchanged. Non-retryable backend failures,
malformed responses, and corpus-version mismatches fail closed without hits.

## Serialisation and privacy

Queries and evidence results use retrieval schema version `2.0.0` and
deterministic UTF-8 JSON: sorted keys, compact separators, full field names,
and one trailing newline. Filter scalars are tagged so dates, booleans,
integers, and floats cannot be confused. Deserialisation rejects malformed
UTF-8/JSON, duplicate or unknown fields, unsupported versions, non-finite
values, implicit coercion, non-canonical identifiers, and invalid value
invariants. Version `1.0.0` results did not persist their requested limit and
are not accepted as version `2.0.0` data; the shared schema version is bumped
for queries as well so one version continues to identify the complete
retrieval wire contract. Malformed filter discriminators at any nesting depth
raise `InvalidRetrievalSerialisationError`; mapping implementation exceptions
are not part of the public failure surface.

Pure retrieval values and serialisation emit no logs. Operational retrieval
stages may log bounded lifecycle information through the shared `structlog`
pipeline, but must not log queries, candidate content, source passages,
prompts, model output, credentials, or sensitive paths.

## Scope boundary

RET-01 defines values and in-memory serialisation only. It does not implement
indexes, retrieval execution, metadata-field registries, score normalisation,
fusion, reranking, planning, budgeting, persistence, model calls, networking,
or source existence checks. Later issues own those responsibilities while
retaining these identities and provenance links.
