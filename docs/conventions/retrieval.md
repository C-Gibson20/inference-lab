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

The lexical generation also retains the complete decoded-source interval for
each indexed chunk, alongside each source link's clipped interval. Its SQLite
format and lexical component contract versions identify this range-bearing
layout. Consumers that need candidate-text offsets must derive them through the
chunk's canonical source-boundary map and the parsed representation; they must
not search for chunk text in source text to guess offsets. This interval is the
source-range contract used by RET-07 to build exact mappings before shortening
evidence.

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

## RET-04 dense index

The dense channel is a durable, corpus-version-scoped vector index built from
the same canonical chunks the lexical channel indexes. It is an **exact flat
scan**, not an approximate-nearest-neighbour index: the declared
implementation is `sqlite-flat` and every query scores every entry in the
selected generation. Approximate search, quantised vectors, and partitioned
indexes are outside this contract.

### Component identity and compatibility

| Field | Value |
| --- | --- |
| `ComponentKind` | `dense-index` |
| Implementation | `sqlite-flat` |
| Format version | `1.0.0` |
| Contract version | `1.0.0` |
| Distance metric | `cosine` |
| Tie-break policy | `record-id-ascending` |
| Score semantics label | `cosine_similarity` |

`DenseIndexCompatibility` is the complete answer to "may these vectors be
reused?". It pairs the backend's `BackendMetadata`, the returned output
dimension, and the runtime `EmbeddingProfile` with the versioned
`DenseIndexConfiguration`. Its `configuration_digest` is the SHA-256 of the
canonical JSON of all four, so any change to tokeniser identity or revision,
query or document prefix, pooling, input normalisation, token limit or
over-limit policy, L2 normalisation, dtype, device type, dimension, distance
metric, tie-break policy, or the format and contract versions produces a
different, incompatible generation. Compatibility fields are path-free; model
and backend names are validated as non-path identifiers.

A generation ID is `"dense-"` followed by the SHA-256 of the canonical JSON of
the compatibility digest, `corpus_id`, `corpus_version`, and the chunk
`source_manifest_digest` — 70 characters, with no path, corpus name, or
payload content. Two builds with identical inputs therefore derive the same
immutable identity, which is what makes an interrupted build safe to repeat.

### Durable layout

A generation is a directory under `storage.index_dir`:

```text
<index_dir>/dense/<generation_id>/
  vectors.sqlite3    # PRAGMA user_version = 1
  manifest.json      # canonical, path-free DenseIndexManifest payload
```

The database is written with `foreign_keys = ON`, `journal_mode = DELETE`, and
`synchronous = FULL` so a published generation is a single self-contained file
with no sidecar journal. It holds a `members` table of selected document
versions with their chunk-set selection digests and chunk counts, and an
`entries` table of one row per chunk carrying its record identity, owning
document and version, ordinal, exact chunk text, canonical source links, and
vector blob. Vectors are packed little-endian with one IEEE-754 **double** per
dimension; `manifest.json` records the SHA-256 of the complete database file.

Manifest members are unique and ordered by `document_id`. The mutable
`active` link of any component must never be a reader selection mechanism:
readers open a generation only by the ID recorded in a committed corpus-version
manifest.

### Opening and validation

`open_dense_generation` rejects a symlinked directory or artefact, a manifest
whose generation ID does not match the requested one, a manifest declaring a
different implementation, format, or contract version, a database whose
SHA-256 does not match the manifest, a failed `integrity_check`, an unexpected
`user_version`, a failed `foreign_key_check`, member rows that disagree with
the manifest, and per-document entry counts that disagree with declared chunk
counts. The connection is opened read-only and immutable.

Query execution additionally requires exactly one ready `dense-index`
reference in the pinned snapshot manifest, owned by the selected corpus and
version, whose implementation, format, contract, configuration, and
source-manifest digests all match the opened manifest.

### Query execution

The query text is embedded through the runtime embedding boundary with
`purpose="query"`, the dimension recorded in the generation's compatibility,
and the normalisation recorded in its embedding profile. The response is
rejected unless its `BackendMetadata` and `EmbeddingProfile` are equal to the
generation's, which is how an incompatible or silently re-configured embedder
is caught before it can contaminate results.

Candidates are scored by cosine similarity, clamped to `[-1.0, 1.0]`, and
ordered by descending similarity with ascending canonical `RecordId` as the
only tie-break. Zero vectors are invalid for cosine and are rejected at
construction. When the query carries a metadata filter, the predicate is
evaluated in exact dense rank order before the requested limit is applied, so
filtering never reorders surviving candidates.

### Incremental reuse

A build may name a parent generation reference. Vectors are reused from the
parent only when the parent's `configuration_digest` equals the proposed
compatibility digest **and** the candidate — record identity, kind, exact
text, and complete source links — is unchanged. Everything else is
re-embedded. A parent owned by another corpus, in a non-ready state, or
declaring a different implementation or configuration digest fails closed
rather than being partially trusted.

Re-publishing an identical generation ID is a no-op: the existing durable
generation is revalidated against the proposed inputs and returned with
`generation_reused=true`. Build statistics report members processed, chunks
processed, and vectors reused, embedded, replaced, and removed.

### Failures

`DenseError` carries one bounded category and a retryability hint:
`invalid_request`, `generation_unavailable`, `corrupt_generation`,
`incompatible_generation`, `embedding_failure`, `execution_failure`. It never
carries SQL, paths, vectors, chunk text, query text, or raw exception
messages. At the retrieval boundary, unavailable and corrupt generations
become `channel_unavailable`; incompatible generations and invalid requests
become `invalid_channel_response`; everything else becomes
`channel_execution_failure`.

### Storage participation

The dense owner participates in corpus-version publication through
`DenseGenerationParticipant` and `DenseGenerationValidator`. The validator
returns a strict boolean and confirms that the registered reference is exactly
the one derived from the manifest, that staged membership matches the chunk
source manifest entry for entry, that each document version's persisted
chunk-set selection still matches, and that the durable generation reopens by
ID and matches its inputs. A database commit failure may leave an
unreferenced immutable generation; it is unreachable through any committed
manifest, and the dense owner retains cleanup responsibility under a future
retention policy.

### Outstanding against this contract

These are known gaps, recorded here so the contract is not read as a
description of finished work:

- **Query cost grows with corpus size.** Opening a generation materialises
  every entry to verify declared counts, and scoring materialises every
  vector again. With a metadata filter, candidate provenance is also resolved
  for every entry rather than for a bounded window. The specification requires
  candidate provenance checks to stay bounded as result limits and corpus size
  grow; this implementation does not yet meet that requirement.
- **No operational logging or stage instrumentation.** The lexical build emits
  an observed stage; the dense build emits neither logs nor profiling records.
  The retrieval logging policy applies to this channel and is not yet
  implemented for it.
- **`library_version` defaults to the host SQLite version** and participates
  in the configuration digest, so an unrelated SQLite upgrade invalidates
  existing generations and blocks cross-machine builds. The library version
  belongs in recorded compatibility metadata, not in the reuse identity.
- **Vectors are stored as doubles** although the backend produces single
  precision, doubling index size for no additional precision.

## RET-05 fusion and document deduplication

Fusion combines the ranked lexical and dense channels into one corpus-version
scoped result without treating their incomparable raw scores as a single
scale. Both channels are executed against **one** pinned snapshot, and every
candidate's source links are revalidated against that snapshot before fusion;
a channel whose candidates do not resolve becomes a channel-scoped failure
rather than contributing unverified evidence.

### Ranked channels and the metadata predicate

Only `lexical` and `dense` are ranked, fusible channels. **Metadata is a
predicate, not a channel**: `RET-03` filters are applied inside each channel's
execution before its result limit, so there is no metadata candidate list to
fuse and no metadata weight. A metadata-only comparison is therefore
unavailable by construction, and an evaluation that reports one must report it
as unavailable rather than as zero recall. Presenting a metadata result
alongside ranked channels would imply a ranking the system does not compute.

### Algorithm

The baseline policy is weighted reciprocal-rank fusion, version `1.0.0`:

```text
fused_score(record) = sum over each contributing channel of
                          weight(channel)
                      ───────────────────────────────
                      rank_constant + rank(record, channel)
```

| Parameter | Default | Constraint |
| --- | --- | --- |
| `rank_constant` | `60` | positive integer |
| `lexical_weight` | `1.0` | finite, strictly positive |
| `dense_weight` | `1.0` | finite, strictly positive |

Ranks are channel-local and strictly positive; raw scores are never read.
`FusionPolicy.identity` renders every ranking parameter as
`reciprocal_rank_fusion/<version>/k=<constant>/dense=<weight>/lexical=<weight>`
with weights formatted at full double precision so an identity cannot hide a
weight change.

Records are merged by `RecordId`. A merge requires identical candidate kind
and byte-identical content; a source span identity appearing twice must carry
identical ownership. Merged source links are the union, in canonical
`(document_id, document_version_id, source_span_id)` order. A record
contributes at most once per channel, and every contributing channel's
contribution is retained in the fused hit so downstream stages can see which
channels found the evidence.

Fused order is descending `fused_score`, then ascending canonical `RecordId`.
Channel results are themselves processed in canonical channel order, so fusion
is deterministic for identical inputs.

### Deduplication, version `1.0.0`

Deduplication runs **after** ranking and **before** the requested limit, so
the highest-ranked member of a duplicate group survives and the caller still
receives up to its requested number of distinct hits.

Byte-identical duplicate relationships visible in the pinned snapshot are
resolved into connected components by union–find; a component's root is its
lexicographically smallest `DocumentVersionId`. Only
`DuplicateClass.BYTE_IDENTICAL` participates, because only that class
guarantees identical content, and persistence already enforces matching hash
algorithm and digest for it. A relationship whose endpoint is not visible in
the snapshot is a corrupt read, not a silently dropped edge.

The key for one candidate is its kind plus the sorted set of per-source keys:

| Condition | Per-source key |
| --- | --- |
| Owning document version is in a duplicate component of size > 1 | `("byte_identical_source_range", component_root, start_offset, end_offset, page_number, content_origin)` |
| Otherwise | `("source_span", source_span_id)` |

An absent page number is normalised to `-1` so a page-less span cannot collide
with page one. The effect is that the same passage reached through two
byte-identical copies of a document collapses to one hit, while two genuinely
different passages never collapse.

### Document families are deliberately not collapsed

Family membership is resolved and validated against the snapshot, but it does
**not** affect the deduplication key. Two members of one publication or patent
family remain separate hits.

This follows from the provenance contract: `DocumentFamilyId` "identifies a
family grouping, not a document", family membership "does not merge member
documents or their immutable source versions", and "each member remains
independently citable". Collapsing family members would discard distinct,
separately citable evidence — precisely the outcome the issue boundary warns
against — and would do so before the stage that is actually responsible for
diversity.

Resolved family identities are therefore exposed as **diagnostics**: the fused
completion event reports how many distinct families the evidence set touches.
Later channel planning and answer construction may use that signal to
diversify or to annotate an answer; fusion does not act on it.

> **Open decision for the maintainer.** `RET-05`'s scope line in
> `build-details.md` still reads "document-version and family deduplication",
> which this section does not implement and argues against. Either amend that
> scope line to "document-version deduplication and family-aware diagnostics",
> or keep it and record family-level collapse as a successor issue sequenced
> with multi-channel planning. This section documents shipped behaviour and
> its rationale; it does not resolve which of those two routes the plan takes.

### Degraded channels and corpus-version mismatch

Channel failures are preserved in the fused result and determine its status
under the shared result rules: hits with failures is `partial`, no hits with
failures is `failed`. Fusion requires at least one ranked channel result, and
a channel result whose requested limit differs from the query's is invalid
input.

Any channel observing a corpus version other than the query's makes the whole
fused result fail closed with no hits and an unscoped corpus-version mismatch
failure. Fusion does not emit the channel-scoped partial form; a mismatch at
this level means the snapshot and the query disagree, which is not a
recoverable per-channel degradation.

### Logging

`retrieval_fusion_completed` is emitted once per execution at `INFO` when
completed, `WARNING` when partial, and `ERROR` when failed. It carries the
caller-supplied `corpus_id`, the result's `corpus_version`, the fusion policy
identity, the deduplication policy version, result status, bounded per-channel
candidate counts, the final result count, the family-diagnostic count, and the
bounded failure categories. It excludes query text, candidate content, source
passages, raw scores, paths, and raw exception messages.

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

Each RET-06 execution outcome carries a stable `RerankingIdentity` for
evaluation and replay. It includes backend/model name and revision, the
backend's semantic preprocessing profile version, both limits, and the RET-06
policy and exact-deduplication versions. Its SHA-256 digest covers all of
these fields. Identity values must not include deployment paths; raw scores,
queries, passages, and prompts are not part of the identity.

The `reranking_completed` structured event includes the caller-supplied
`corpus_id` and query-pinned `corpus_version`, bounded candidate and evidence
counts, configured limits, result status, degraded state, a typed failure
category when present, the stable identity fields and digest, and elapsed
milliseconds. It excludes query and candidate text, source passages, raw
scores, prompts, raw exception messages, credentials, and deployment paths.
Completed runs log at `INFO`, degraded runs at `WARNING`, and failed runs at
`ERROR`.

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

## RET-07 evidence reduction

`reduce_retrieval_evidence` accepts one RET-06 `EvidenceResult`, a pinned
`CorpusSnapshot`, the matching immutable lexical generation, an explicit local
offset-bearing tokeniser, and validated `RetrievalConfig`. It rejects a corpus
or version mismatch before reading candidate source data. The default token
ceiling is `retrieval.evidence_token_budget = 4096`; a request may supply a
positive token-budget override. The default item ceiling is RET-06's configured
`evidence_count` (at most eight by default), further capped by the request's
result limit and an optional lower per-request item limit. The configured
diversity policy defaults to `none`; `per_document` keeps the first
(highest-ranked) candidate for a document and drops later candidates for that
document. A request may override this policy for one call. Selection follows
RET-06 order and never re-ranks evidence.

This is an evidence-only ceiling, not a generator context size. SERV-02 must
reserve prompt, user, history and template overhead from
`runtime.max_context_tokens` and pass the smaller remaining
headroom as the effective request budget. RET-07 counts the complete joined
evidence text using the fixed `\n\n` separator after every selection or
truncation.

Before shortening a candidate, the adapter reopens its exact lexical chunk and
parsed representation from the same snapshot, checks candidate identity and
text, and derives text-to-source boundaries through ING-04's
`CanonicalSourceBoundaryMap`. It does not search for matching substrings.
Incomplete or unsafe maps do not prevent keeping a complete passage; if that
passage needs truncation, the reducer drops it with a bounded source-map
failure. Truncated items retain the original hit and exact clipped source
ranges. The transient outcome carries its tokeniser, budget, separator,
diversity and policy identity with a stable digest; it has no durable wire
schema.

The `evidence_reduction_completed` event records corpus context, status,
bounded counts, tokeniser and policy identity, and elapsed time. Each
`evidence_truncated` event is emitted at `WARNING` with the record identity and
bounded mapping counts. Failed calls emit a typed `ERROR` event. None of these
events includes candidate text, queries, prompts, token IDs, paths or raw
exceptions.

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
