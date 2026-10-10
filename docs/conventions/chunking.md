# Ingest chunking conventions

ING-04 turns one validated `ParsedDocument` into an immutable, ordered,
storage-neutral `ChunkSet`. The operation targets exactly one `CorpusId`, but
corpus identity is an operation and visibility concern: it is deliberately
excluded from chunk payload identity so one immutable document version can be
reused by explicit membership in more than one corpus.

## Request and tokenisation

`ChunkingRequest` contains:

- the target `CorpusId`;
- one validated `ParsedDocument`;
- one explicitly constructed object implementing the ingest `ChunkTokeniser`
  protocol; and
- one validated `ChunkingPolicy`.

The selected tokeniser is the source of truth for counts and boundaries. It
must expose deterministic code-point offsets, exact standalone token counts,
and a complete path-free `TokeniserIdentity`. Tokenisation is performed
without special tokens. A chunk's stored count is the count of its exact
canonical text, not a slice of whole-document token IDs; this matters for
contextual BPE tokenisers.

Native token offsets establish a bounded candidate neighbourhood for end and
overlap selection. Exact standalone counts choose between candidates within
that neighbourhood and independently enforce the hard maximum. Selection must
not binary-search scalar counts as though they were globally monotonic, nor
rescan the entire source-boundary map or retokenise the remaining suffix for
each chunk. Build token and source-boundary indexes once per document, slice
the indexed neighbourhood for each window, and retain exact standalone counts
for candidate validation. This avoids work growing with the product of
document length and chunk count. For normal-sized chunks, test the exact
token-offset overlap candidate first and stop when its standalone count is
within the target; retain exhaustive comparison for short windows, where the
bounded search is cheap and useful for adversarial count contracts. The
baseline guarantees deterministic, contract-safe packing rather than globally
optimal packing for an arbitrary pathological count oracle.

The baseline local adapter loads a fast tokeniser from an explicit local
artefact directory with `local_files_only=True`, rejects slow or changing
artefacts, and records a manifest revision. It never falls back to a network
load. The tokeniser identity and every count-affecting option participate in
chunk configuration identity.

## Baseline policy

The closed baseline policy is:

| Field | Value |
| --- | --- |
| Minimum target | `256` tokens |
| Hard maximum | `512` tokens |
| Overlap target | `floor(previous standalone count * 15 / 100)` |
| Boundary policy | `structure-aware-greedy` |
| Oversize policy | deterministic `token-window` splitting |

The minimum is a packing goal, not a hard validity requirement. A chunk below
the minimum must carry exactly one audited exception:

- `short_document` for the whole useful document;
- `protected_structure_neighbour` when a fitting table or figure caption
  forces a short adjacent chunk; or
- `final_remainder_minimum_exception` when a legal tail cannot be rebalanced.

The maximum is hard. A fitting table or figure caption is not split. A table
or caption larger than the maximum is split into safe token/source windows and
each affected chunk carries `oversize_structure`.

Overlap is the decoded-source intersection with the preceding chunk. The
later chunk stores its overlap source range and actual standalone count. The
actual count may be lower than the target when a protected structure or safe
source boundary prevents a larger overlap; it must never be higher. The first
chunk has zero overlap. Every later chunk must make forward source progress.

## Structure and source coordinates

The canonical text stream is chunked once. Parser blocks are boundary and
provenance overlays and are never concatenated, because pages, sections,
paragraphs, tables, captions, and `other` spans may nest or overlap.

Candidate structural ends use this closed tie-break priority:

1. `section`
2. `table` and `figure_caption`
3. `paragraph`
4. `page`
5. `other`

Parser source spans remain half-open Unicode code-point ranges into decoded
`NormalisedText.source_text`. Chunk text is the normalised form of its exact
source slice. A segment-aware boundary map admits only exact bidirectional
boundaries through newline and NFC normalisation; positions are never
recovered with substring search, which would be ambiguous for repeated text.
Each chunk retains ordered clipped `ChunkSourceReference` values back to the
ING-01 source spans and complete PROV-03 lineage.

Empty and whitespace-only documents produce zero chunks. Non-whitespace source
content without covering parser provenance fails closed as `uncovered_content`.

## Identity, serialisation, and invalidation

Chunk payload schema version `1.0.0` is canonical UTF-8 JSON with sorted keys,
compact separators, and one trailing newline. Serialisation and
deserialisation are strict and validate payload fields, counts, ranges,
configuration identity, output digest, fingerprint, source catalogue, and
lineage.

Each chunk record ID is a project-owned UUIDv5 derived from its complete
PROV-03 derivation fingerprint. The fingerprint includes the document/version,
source anchors, parser/chunker/tokeniser components, policy/configuration
identity, canonical payload, output digest, and schema versions. It excludes
`CorpusId`, storage locations, and index identity.

Changing parser or normalisation version, chunker contract, policy,
tokeniser contract, local artefact manifest, configuration, source payload, or
lineage creates a new immutable chunk generation. Existing records are never
mutated or silently aliased.

## Failures and operational logging

The pure chunking engine emits no logs. It returns either a complete
`ChunkSet` or a payload-free `ChunkingFailure` with one bounded category:
invalid input, unsupported structure, tokeniser unavailable/invalid, unsafe
normalisation boundary, token budget failure, uncovered content, invalid
lineage, or invalid output.

The observed wrapper binds `corpus_id`, `document_id`, nullable
`corpus_version`, and `pipeline_stage="chunk"`. It emits stable
`chunk_started`, `chunk_completed`, and `chunk_failed` lifecycle events.
Completion fields may include component contract versions, chunk and token
counts, overlap totals, undersized/split counts, and elapsed time. Failure
events contain only a bounded category. Logs and failure representations must
not contain source/chunk text, paths, token IDs, raw artefact metadata,
hashes, or raw exceptions. Logging is best effort and must not replace the
primary operation outcome.

## Validation gates

Run the focused acceptance and measurement suites with:

```sh
uv run pytest -s \
  tests/inference_lab/ingest/test_chunk_acceptance.py \
  tests/inference_lab/ingest/test_chunk_measurement.py
```

The acceptance suite uses checked-in deterministic cases for size boundaries,
structure overlays, protected units, Unicode/source navigation, contextual
token counts, corpus isolation, privacy, and cold-process determinism. The
measurement suite reports tokens/second, peak traced memory when available,
count-cache bound, observed count calls, and distinct count inputs. It does
not assert a hardware-dependent throughput threshold; it does reject
unbounded cache growth and clearly quadratic count rescanning.

The complete repository gate remains the lint, format, type-check, coverage,
and test sequence in `AGENT.md`.

## Ownership boundaries

The parser owns decoded text, normalisation, structural blocks, and source
spans. Chunking owns deterministic windows, overlap, chunk identity, and
storage-neutral lineage. ING-05 owns persistence, checkpointing, component
generations, membership revalidation, and corpus-version publication.
Retrieval indexes exact chunk text and source links; storage and backup preserve
the canonical chunk and lineage records without changing their identity.
