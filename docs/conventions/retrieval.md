# Retrieval contract and boundaries

The retrieval package owns immutable, channel-neutral values shared by lexical,
dense, metadata, structured, and summary retrieval. It carries private query
and candidate content, but it does not log or persist those payloads.

## Query and filters

`RetrievalQuery` preserves non-blank query text exactly, selects one
`CorpusVersionId`, optionally carries a typed filter tree, and requires a
strictly positive `RetrievalLimit`. Filters support exact values, non-empty
typed sets, bounded ranges, presence/missing checks, and non-empty all/any/not
composition. Field names are lowercase project-owned slugs. Filter values are
strict strings, integers, finite floats, booleans, or calendar dates; no value
is normalised, rewritten, or inferred.

## Candidates, scores, and results

Every `RetrievalCandidate` has a `RecordId`, a candidate-kind authority label,
non-empty retrieval content, and one or more unique source links. Each source
link retains `DocumentId`, `DocumentVersionId`, and `SourceSpanId`. Original
passages, structured records, and generated summaries remain distinguishable;
kind, rank, and score never prove claim support.

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

A corpus-version mismatch fails closed and emits no hits from the incompatible
snapshot. Failure values expose only a bounded category, optional channel, and
retryability; they never carry raw exception messages or payloads.

## Serialisation and privacy

Queries and evidence results use retrieval schema version `1.0.0` and
deterministic UTF-8 JSON: sorted keys, compact separators, full field names,
and one trailing newline. Filter scalars are tagged so dates, booleans,
integers, and floats cannot be confused. Deserialisation rejects malformed
UTF-8/JSON, duplicate or unknown fields, unsupported versions, non-finite
values, implicit coercion, non-canonical identifiers, and invalid value
invariants.

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
