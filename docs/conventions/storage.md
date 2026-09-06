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
active state.

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
