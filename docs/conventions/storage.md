# Storage conventions

## Scope

The storage package owns restart-safe persistence for deployment-global
document identities, immutable source versions, managed original bytes, parsed
representations, parse outcomes, and resolvable source locations. Every public
operation that can expose source or parsed content requires a `CorpusId` and
checks explicit corpus membership before returning data.

Storage does not discover files, assign identity, run parsers or OCR, select an
active parse, publish a corpus version, build indexes, or present citations to
an operator. Those operations compose with this persistence boundary in later
ingest, storage, retrieval, and serving work.

## Configuration and lifecycle

Application composition opens storage through
`inference_lab.storage.open_metadata_repository()`, passing the validated
`storage.metadata_db_path` and `storage.source_dir` paths. Both paths belong to
one local deployment and may contain records shared by multiple named corpora.
They are not corpus selectors.

The metadata database, source directory, and index directory must be distinct.
The repository creates required parent directories, but the deployment must
provide sufficient permissions and capacity. It must not follow symbolic links
inside managed source storage.

Repository opening is non-mutating by default. Callers must opt in with
`apply_migrations=True` to create a database or apply a supported forward
migration. Missing, older, malformed, and unsupported-newer schemas fail with
typed storage errors and must not be guessed or rewritten. Callers close the
repository explicitly; operations after closure are rejected.

## Identity and corpus visibility

`DocumentId` and `DocumentVersionId` are deployment-global immutable
identities. Physical source bytes and parsed records may be reused across
corpora, but reuse never grants visibility. A public source or parsed-record
read returns data only when the requested version is explicitly present in the
selected corpus catalogue.

An absent record and a record outside the selected corpus both return `None`
where the reader contract is nullable. This non-disclosure rule prevents a
caller from probing another corpus through shared physical storage. A reachable
record whose persisted state or managed bytes are invalid instead fails closed
with `CorruptStorageError`.

STOR-02 enforces the current `CorpusId` catalogue boundary. STOR-03 will add
immutable `CorpusVersionId` publication and active-version selection without
changing the deployment-global identity or physical source layout.

## Durable source and parser records

A `SourceArtefact` records one immutable original's document and version IDs,
SHA-256 identity, byte count, canonical media type, managed storage key, and
record schema version. The storage key is an internal relative locator, not a
caller path or public identity.

A `ParsedRepresentation` retains:

- decoded source text and canonical normalised text;
- parser and optional OCR producer identity;
- normalisation and record schema versions;
- language and page, character, and token counts;
- explicit collection availability;
- ordered structural blocks;
- source locations with half-open decoded-text offsets; and
- optional page coordinate spaces, rotations, origins, and bounding boxes.

The aggregate validates its canonical output digest, source ownership, block
text, spans, coordinate bounds, producer attribution, and distinction between
unavailable (`None`) and available-empty (`()`) data before it becomes durable.
Multiple immutable parser outputs may exist for one source version, with
explicit supersession where applicable; storage does not choose an active one.

A `ParseOutcome` preserves succeeded, recoverable, and fatal attempts.
Recoverable outcomes keep their valid representation and bounded omitted-unit
records. Fatal outcomes have no partial representation. Failure records contain
only closed categories, severities, stages, and optional unit kind/index values;
they never persist source excerpts or exception payloads.

Storage-owned records have an explicit schema version independent of the
metadata database migration revision. Readers reconstruct and validate the
complete domain aggregate rather than treating internal SQLite rows as a public
format.

## Atomic source publication

`MetadataRepository.persist_source_artefact()` is the public publication
boundary for managed bytes, source metadata, and an optional parse bundle. It:

1. validates the complete bundle and corpus membership;
2. acquires bounded cross-process publication access;
3. streams into private staging under the managed source directory;
4. enforces the caller's source-size limit and verifies byte count and SHA-256;
5. atomically promotes the immutable blob; and
6. commits source metadata and parse records in one SQLite transaction.

Exact replay is a no-op with the same observable result. Conflicting identity,
content, metadata, or parse records are rejected. An interruption cannot expose
staging data or partial metadata. A failure after blob promotion may leave an
unreferenced final blob, which is safe because public visibility starts at the
metadata membership boundary.

Source reads verify the complete stored byte count and hash before returning
bytes. The streaming reader returns a detached, fully verified iterator so the
repository may be closed before the caller consumes it. Missing, truncated,
replaced, linked, or otherwise invalid reachable bytes fail closed.

## Reconciliation and ownership

Repository startup performs bounded, conservative source reconciliation. The
same operation is available explicitly through
`MetadataRepository.reconcile_source_storage()` for controlled maintenance.
Reconciliation removes abandoned staging files and inventories final blobs; it
does not delete metadata-referenced or ambiguous entries. It reports only
bounded counts and status fields.

Operators and other modules must treat `storage.source_dir` as repository-owned
state. Do not rename, edit, deduplicate, or delete its contents directly. A
manually modified blob no longer matches its immutable identity and is treated
as corrupt.

## Errors and observability

Failures crossing the public adapter use `StorageError` subclasses with fixed,
payload-free messages. Categories distinguish repository availability,
closure, immutable conflict, bounded concurrency, transaction failure, corrupt
state, migration failure, incompatible schema, and required migration.
`StorageConcurrencyError` is retryable; callers must not infer retryability from
raw SQLite or filesystem messages.

Storage events follow the shared structured-logging schema and bind opaque
`corpus_id` and `document_id` values at source-publication boundaries.
`corpus_version` remains nullable until STOR-03 supplies it. Logs may contain
stable event names, typed failure categories, algorithms, counts, sizes,
durations, and bounded reconciliation results. They must not contain source
bytes or text, filenames, original or managed paths, storage keys, content
digests, publication identifiers, parser failure payloads, SQL, or raw exception
messages.

## Downstream boundaries

- **ING-03** maps the ingest parser contract into the storage-owned aggregate
  and records target-corpus membership. Storage does not import ingest types.
- **STOR-03** publishes immutable corpus generations and adds version-scoped
  visibility over the records defined here.
- **SERV-04** resolves accepted citations to stored text ranges and optional
  page regions, then exposes an operator-facing source-opening workflow.
- **STOR-08** includes canonical metadata and source integrity inventory in
  corpus-state backups. It excludes managed original bytes, which require a
  separate source-data backup or availability policy.

See the [configuration conventions](configuration.md),
[provenance conventions](provenance.md),
[logging conventions](logging.md), and
[backup and restore conventions](backup-restore.md) for their respective
cross-cutting policies.
