# Storage conventions

These conventions define the durable corpus-version publication and reader
boundary. They apply to ingest orchestration, persistent index owners,
retrieval, serving, evaluation, sessions, and any future component that reads
or contributes to a published corpus.

## Consistency boundary

A `CorpusVersion` is an immutable, corpus-owned consistency boundary. Its
`CorpusVersionManifest` contains the complete materialised document-version
membership and the complete ordered set of immutable component-generation
references. Membership and components in one manifest always name the same
`corpus_id` and proposed `corpus_version`.

The metadata database's per-corpus active-version pointer is the only
publication point. Filesystem presence, a component's mutable `active` link,
deployment-global catalogue existence, or a staged database row never grants
published visibility. Product-facing reads use `CorpusSnapshot`; trusted
ingest and identity-resolution code may continue to use the global catalogue
APIs for their narrower purposes.

Each corpus advances independently. A version has a positive per-corpus
sequence and, after the first publication, the immediately preceding active
version as its parent. Version IDs are deployment-global opaque identities,
while their records and manifests remain owned by exactly one corpus.

## Repository and transaction ownership

Open the public `MetadataRepository` with `open_metadata_repository()` and
close it explicitly. Apply schema migrations only through the factory's
explicit `apply_migrations` policy. Application integrations depend on the
protocols exported by `inference_lab.storage`, not private SQLite modules or
connections.

All corpus-update writes occur inside `with repository.transaction() as
transaction:`. Successful exit commits the entire metadata unit; exceptional
exit rolls it back. Transactions are non-nestable and single-use. Do not hold
a metadata write transaction while parsing documents, calling a model, or
building an index. Build immutable component data outside the transaction and
use a short transaction to validate and register it.

Metadata transactions expose database-backed metadata reads but deliberately
do not expose `read_source_bytes()` or `stream_source_bytes()`. Read managed
source files through the repository outside a write transaction, or through a
version-pinned snapshot for product-facing work. Commit any source metadata
changes before reading their bytes through the repository.

An error makes the transaction rollback-only. Never catch an operation error
inside the context and continue issuing writes. Exit, let the transaction roll
back, inspect durable state through a new unit, and then decide whether a safe
retry is possible.

## Update lifecycle

`begin_corpus_update()` requires one corpus, a caller-generated proposed
`CorpusVersionId`, the exact expected active version (including explicit
`None`), and the complete required component-kind set. The proposed version is
also the durable update identity; do not introduce a parallel job or staging
identity as a substitute in storage contracts.

Begin verifies the active-version expectation and seeds proposed membership
from the expected parent. While the update is `preparing`, use the typed
membership operations:

- add a document version already present in that corpus's catalogue;
- replace exactly the expected visible version with another catalogue version
  of the same logical document; or
- remove exactly the expected membership.

The operations are compare-and-set changes, not best-effort edits. Exact
replay is idempotent; a different payload for the same identity fails closed.
Never mutate staged tables directly.

`seal_corpus_update()` requires complete valid membership and exactly one ready
generation for each component kind declared at begin. It canonicalises the
manifest, persists its SHA-256 digest, and makes the update `ready` and
immutable. If the complete semantic manifest is unchanged from the expected
active version, sealing returns `NoCorpusChangeOutcome`, marks the proposal
terminal, creates no sequence, and leaves the active pointer unchanged.
Semantic equality compares membership plus component implementation, format,
contract, configuration, and source-manifest digests; it deliberately ignores
the new generation IDs. The persisted manifest digest still covers those exact
generation IDs so committed snapshots reopen the precise immutable artefacts
that were published.

`commit_corpus_update()` accepts only a ready ordinary update. In one database
transaction it rechecks the expected active version, creates the immutable
version and materialised membership, advances the active pointer, and marks the
update committed. Failure exposes neither a partial new version nor staged
records. Exact commit replay returns the already committed result.

`abort_corpus_update()` may terminate a preparing or ready proposal but never
changes active state. Committed and aborted updates are terminal. Preserve
historical versions and durable update records; retention and destructive
garbage collection require a separately specified policy.

## Downstream component participation

A persistent component owner, such as a lexical or dense index, participates
without giving storage knowledge of its internal files. Its integration must:

1. Define a stable lowercase-hyphenated `ComponentKind` and documented
   implementation, format, and contract versions.
2. Accept the selected `corpus_id`, proposed `corpus_version`, and the complete
   staged membership as build inputs.
3. Write to a generation-specific immutable location. A mutable active path
   must never be the generation identity or the reader selection mechanism.
4. Produce a canonical non-path generation ID plus SHA-256 configuration and
   source-manifest digests. Do not put paths, corpus names, or payload data in
   these identity fields.
5. Construct a `ComponentGenerationReference` in `preparing` state for the
   exact corpus, proposed version, and declared kind.
6. Supply a `ComponentGenerationValidator` when registering the reference.
   The validator returns a strict boolean and verifies that the generation is
   durable, immutable, complete for exactly the supplied membership, owned by
   the named corpus/version, compatible with its declared format and contract,
   and reopenable by generation ID without consulting a mutable active link.
7. Open a published component only from the generation reference in the
   snapshot manifest. Reject missing, foreign, unready, incompatible, corrupt,
   or mixed-generation state before serving results.

Validation exceptions and non-boolean results cross the adapter boundary as a
privacy-safe repository-unavailable failure. A `False` result is a conflict.
Registration persists a validated reference as `ready`; it does not mutate the
component artefact or make it visible.

If a preparing generation is superseded before sealing, replacement requires
the exact registered reference and a newly validated reference of the same
component kind. Adding or replacing a component after seal requires a new
corpus version.

A database commit failure may leave an unreferenced immutable component
generation. Readers cannot discover it through a committed manifest. The
component owner retains cleanup responsibility, but must not remove any
generation reachable from a committed or potentially recoverable update until
a future retention policy makes that safe.

## Snapshot readers

Use `open_active_corpus_snapshot(corpus_id)` for the active version or
`open_corpus_snapshot(corpus_id, corpus_version)` for an owned historical
version. Both return unopened, context-managed `CorpusSnapshot` instances.
Enter the context before reading and treat the object as permanently closed
after exit.

Snapshot entry opens a dedicated query-only SQLite connection, begins a read
transaction, selects one committed version, and reconstructs its complete
manifest. Every read remains scoped to that membership. Relationships are
visible only when their required endpoints are visible; global catalogue
records and another corpus's records remain hidden.

Source artefacts, original bytes, parsed representations, parse outcomes, and
source spans are immutable records owned by a document version. Snapshot
readers expose exact-identity lookups for those records only when the owning
`DocumentVersionId` occurs in the pinned manifest. Catalogue enumeration of
parser outputs remains an ingest and audit operation on `MetadataRepository`;
it is not a product-facing snapshot selection mechanism. Persistent component
generations record the exact source-manifest digest used to build their derived
state.

The selected version does not change during the snapshot, even when another
writer publishes. Keep the snapshot open for one bounded request or job unit,
then close it promptly so SQLite can reclaim reader resources and checkpoint
the WAL. Do not nest a snapshot, use it after close, or treat the repository's
ordinary global reads as equivalent published visibility.

## Concurrency and retry policy

Publication uses a per-corpus expected-active compare-and-set. Two proposals
may prepare from the same parent, but exactly one can advance that corpus. The
loser receives `StorageConflictError`; it must not be rebased or retried
blindly because its membership and components were built for a stale parent.
Start a fresh proposal only after the owning workflow re-evaluates the new
active state. Conflict subtypes are not a public result contract; bounded
operational reasons such as an expected-active mismatch or reused version ID
may appear only in privacy-safe structured logs.

SQLite uses a bounded lock wait. Lock expiry becomes
`StorageConcurrencyError`, the only retryable storage error. Callers may retry
the complete failed transaction with bounded attempts and backoff appropriate
to their service boundary. They must not loop indefinitely or retry only a
suffix of a rolled-back transaction. Physical SQLite writer serialisation does
not turn different-corpus work into a semantic version conflict.

Use each operation's exact-replay behaviour to recover from an ambiguous caller
acknowledgement. Reuse precisely the same corpus, proposed version, expected
parent, required kinds, and payload. Never reuse an ID with changed inputs.

## Restart and recovery

Repository reopen selects committed state from the durable active pointer; it
does not infer state from component paths. On startup, orchestration may use
`list_non_terminal_corpus_updates()` to inventory all `preparing` and `ready`
work. Storage never resumes, commits, aborts, or deletes those updates
automatically.

The orchestration owner decides whether the original inputs and component
artefacts are still valid enough to resume or commit. Otherwise it explicitly
aborts the update. A staged artefact with no matching durable update, or a
durable update whose artefact cannot be validated, is never published merely
to complete recovery.

## Rollback as a new version

Operational rollback never rewinds or edits the active pointer directly.
Begin a new proposal against the current active version and name an owned,
non-parent historical version in `restores_corpus_version`. Storage seeds the
historical membership and derives its required component kinds. Component
owners must provide new references owned by the proposed version whose
implementation, format, contract, configuration digest, and source-manifest
digest reproduce the historical component semantics.

Seal verifies equivalence with the historical target.
`commit_corpus_rollback()` then publishes the next sequence, records the
current active version as parent, and retains the historical restore link for
audit. Cross-corpus targets, current-parent targets, ordinary commit of a
rollback proposal, and rollback commit of an ordinary proposal fail closed.

## Errors, logging, and configuration

Storage adapters translate implementation failures into fixed-message
`StorageError` subclasses. Only `StorageConcurrencyError` is retryable.
`StorageConflictError` means the caller must reconcile state;
`CorruptStorageError` requires stopping the affected publication or read path.
Do not expose SQL, database messages, paths, validator exceptions, component
payloads, or record values in public exceptions.

Corpus update, commit, rollback, and snapshot events use the shared structured
storage logger. Bind only opaque `corpus_id` and `corpus_version` values,
sequences, bounded counts, lifecycle states, stable error categories, and
retryability. Never log corpus display names, generation payloads, document
content, source passages, sensitive paths, raw validator output, database
messages, or SQL. Best-effort logging and cleanup must preserve the primary
storage outcome.

The public repository uses the configured local `storage.metadata_db_path`.
Component owners use `storage.index_dir` through their own typed integration.
Do not add ad hoc environment reads, hidden timeout switches, or per-corpus
database paths. Any new setting requires the project configuration contract,
validation, documentation, and tests. The SQLite adapter owns its bounded lock
policy; callers own only the bounded operation-level retry policy described
above.

## Managed source persistence scope

The storage package owns restart-safe persistence for deployment-global
document identities, immutable source versions, managed original bytes, parsed
representations, parse outcomes, and resolvable source locations. Every public
operation that can expose source or parsed content requires a `CorpusId` and
checks explicit corpus membership before returning data.

Managed source persistence does not discover files, assign identity, run
parsers or OCR, select an active parse, build indexes, or present citations to
an operator. Corpus-version publication composes with this persistence boundary
through the lifecycle and snapshot operations described above.

## Managed source configuration and lifecycle

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
corpora, but reuse never grants visibility. A catalogue source or parsed-record
read returns data only when the requested version is explicitly present in the
selected corpus catalogue. Product reads use the snapshot methods, which
additionally require that version in the selected immutable corpus-version
manifest.

An absent record and a record outside the selected corpus both return `None`
where the reader contract is nullable. This non-disclosure rule prevents a
caller from probing another corpus through shared physical storage. A reachable
record whose persisted state or managed bytes are invalid instead fails closed
with `CorruptStorageError`.

STOR-02 enforces the `CorpusId` catalogue boundary. STOR-03 adds immutable
`CorpusVersionId` publication and active-version selection without changing the
deployment-global identity or physical source layout.

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

This representation is a canonical source record, not a PROV-03 derived
record. Its producer identity, schema version, output digest, document-version
ownership, and source-span catalogue are its complete provenance contract.
Downstream derived records use `DerivedRecordMetadata` to cite the specific
source spans or parent records they consume; storage does not manufacture a
whole-document span solely to make parser output fit that envelope.

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

Ingest integrations construct the canonical managed-source locator with the
public `inference_lab.storage.build_source_blob_storage_key(ContentHash)`
helper. They must not import the private blob-store implementation, duplicate
its directory layout, or manufacture a path from implementation details. The
helper returns an opaque relative storage key; it is not a public identity and
must not appear in logs, errors, or request/result representations.

The bundled publication returns no new-versus-replay flag. Callers may rely on
stable IDs and parse status after success, but must not infer novelty from a
race-prone preflight read. Exact replay remains a no-op; changed immutable
payloads or reused record IDs remain typed storage conflicts.

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

## Managed source errors and observability

Failures crossing the public adapter use `StorageError` subclasses with fixed,
payload-free messages. Categories distinguish repository availability,
closure, immutable conflict, bounded concurrency, transaction failure, corrupt
state, migration failure, incompatible schema, and required migration.
`StorageConcurrencyError` is retryable; callers must not infer retryability from
raw SQLite or filesystem messages.

Storage events follow the shared structured-logging schema and bind opaque
`corpus_id` and `document_id` values at source-publication boundaries.
`corpus_version` may remain nullable for source-publication operations that are
not yet associated with a published version. Logs may contain stable event
names, typed failure categories, algorithms, counts, sizes, durations, and
bounded reconciliation results. They must not contain source bytes or text,
filenames, original or managed paths, storage keys, content digests,
publication identifiers, parser failure payloads, SQL, or raw exception
messages.

## Integration boundaries

- **ING-03** maps the ingest parser contract into the storage-owned aggregate
  and records target-corpus membership. Storage does not import ingest types.
- **ING-04** produces storage-neutral canonical chunks and lineage. ING-05
  owns their persistence, checkpointing, membership revalidation, and
  corpus-version publication. See the [chunking conventions](chunking.md).
- **STOR-03** publishes immutable corpus generations and applies version-scoped
  visibility to the records defined here.
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
