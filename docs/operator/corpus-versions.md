# Corpus-version operations

> **Status:** Corpus-version staging, publication, historical selection, and
> rollback are available through the Python storage and serving boundaries. The
> `inference-lab` command does not yet expose these operations. Use the local
> Python API for ingest and the fresh-process query example in
> [Running queries](querying.md) for a restart-safe search check.

This runbook describes the operational guarantees around one named corpus.
The normative integration contract is defined in the
[storage conventions](../conventions/storage.md).

## Publication model

Every published corpus state is an immutable `CorpusVersion` containing a
complete document-version membership and the complete set of participating
component generations. One active-version pointer per corpus determines what a
new active snapshot selects.

An update progresses through these durable states:

```text
preparing -> ready -> committed
     |          |
     +----------+-> aborted
```

- `preparing` is private staging work and may be changed.
- `ready` has a sealed, immutable manifest and may only be committed or
  aborted.
- `committed` is published history and cannot be changed.
- `aborted` is terminal and never becomes visible to readers.

Staged rows, component files, and catalogue records are not publication. The
SQLite active-version pointer is the sole publication point, and it advances
in the same transaction that records the complete committed manifest.

## Before an update

Record these values in the orchestration job before doing work:

- the opaque `corpus_id`, never only its display name;
- a newly generated proposed `corpus_version`;
- the exact active corpus version expected by the job, including an explicit
  absence for the first publication; and
- the complete set of component kinds required by the resulting corpus.

Read the active version once. Do not silently substitute a newer active
version if another writer publishes while the job is running. That would
combine work based on different parents.

## Stage and publish an update

The integration responsible for ingest should perform the following logical
steps. Each metadata mutation takes place in a context-managed repository
transaction; expensive parsing and component building take place outside that
transaction.

1. Begin the proposal against the recorded expected active version. Existing
   committed membership is copied into private staging for an incremental
   update.
2. Add, replace, or remove exact document-version memberships. A version must
   already be in the selected corpus's catalogue before it can be staged.
3. Build every declared component generation as immutable data owned by the
   proposed corpus and version.
4. Validate and register each generation. Registration changes a valid
   `preparing` reference to `ready`; it does not publish the generation.
5. Seal the proposal. Sealing checks complete membership and exactly one valid
   generation for every declared component kind, then fixes the manifest
   digest.
6. Commit the ready proposal. Commit atomically materialises the version,
   advances only that corpus's active pointer, and marks the update committed.
7. Open a fresh active snapshot and verify its corpus, version, manifest
   digest, membership count, and required component kinds.

If sealing returns a no-change outcome, no version or sequence is created and
the proposal becomes terminal. Treat this as a successful idempotent operation,
not an error. Newly built component generations may have different generation
IDs and still be unchanged when their implementation, compatibility,
configuration, and source-manifest semantics match the active generations.

An exact retry of begin, seal, abort, or commit returns the durable prior
outcome. Reusing the same proposed version with different inputs is a conflict.

## Interrupted and concurrent work

After process restart, inspect `list_non_terminal_corpus_updates()`. It returns
all durable `preparing` and `ready` proposals; startup does not resume, commit,
abort, or delete them automatically.

For each proposal, the owning orchestration layer must decide whether to:

- resume from its recorded membership and component references;
- commit it after independently confirming that it is sealed, complete, and
  still based on the active version it expected; or
- explicitly abort it.

Do not delete staging rows or component directories as a recovery shortcut.
Retention and garbage collection of aborted work, historical versions, and
unreferenced immutable generations are not implemented by this feature.

Two writers based on the same corpus parent cannot both publish. The loser
receives a non-retryable storage conflict and must start a new proposal after
re-evaluating its work. SQLite lock contention is reported separately as a
retryable concurrency error after a bounded wait. Retry the complete failed
transaction with a bounded caller-owned policy; do not retry forever. Updates
to different corpora remain semantically independent even though SQLite may
briefly serialise their write transactions.

## Reader snapshot lifetime

Product reads must use `open_active_corpus_snapshot(corpus_id)` or
`open_corpus_snapshot(corpus_id, corpus_version)` as a context manager. Opening
an active snapshot resolves the pointer once. The snapshot then keeps one
dedicated read transaction and one complete manifest pinned until it closes.
A concurrent publication does not change what that open reader sees; a later
snapshot sees the newly active version.

Open source bytes, parsed records, parse outcomes, and source spans through the
snapshot's exact-identity methods. They return no result when the owning
document version is outside the pinned manifest, even if an ingest catalogue
record for it remains in storage. Do not fall back to repository catalogue
reads for product retrieval or citation navigation.

Keep a snapshot for one bounded request or job unit and close it promptly.
Never cache the snapshot object, use it after context exit, nest its context,
or share it as a mutable active-version handle. A long-lived SQLite reader can
retain resources and delay WAL checkpoint progress even though it does not
block publication.

The repository may be closed after a snapshot is open because the snapshot
owns its dedicated connection. The snapshot itself still must be closed.

## Canonical chunk persistence

Baseline ingest stores one complete immutable `ChunkSet` for each document
version and compatible parser, chunker, tokeniser, and policy identity. The
chunk set is deployment-global physical state; explicit corpus membership still
controls whether a catalogue or pinned snapshot can select it. Sharing a
document version or its canonical bytes between corpora does not grant
cross-corpus visibility.

The write boundary is atomic. An interrupted or rolled-back write leaves no
partially visible chunk set, and an exact retry returns the same result without
creating a duplicate. A changed immutable payload, configuration, ownership,
or source association is a storage conflict and must be reconciled by the
owning workflow rather than overwritten. A reachable row that fails digest or
metadata reconstruction is corrupt state; readers fail closed instead of
returning partial payloads.

Repository creation and supported forward migrations are explicit operations.
Open an older database through the configured metadata-repository factory with
`apply_migrations=True`; do not edit `chunk_sets`, migration history, or other
SQLite tables by hand. After a clean restart, reopen the repository and select
the same corpus and exact configuration to verify replay or catalogue reads.

If a chunk read or write reports `CorruptStorageError`, preserve the database
and the associated source state for investigation. Do not delete rows, replace
canonical payloads, change stored digests, or infer a repair from a mutable
filesystem path. Use a validated deployment backup or the owning ingest
rebuild workflow when that workflow is available. The command-line entry point
does not yet expose a chunk repair command.

## Baseline ingest restart and replay

The baseline ingest integration uses the proposed `CorpusVersionId` as its
durable update identity. A process restart must reopen the same metadata
database and resume or reconcile that proposal; it must not generate a second
proposal merely because the previous process ended before returning a result.

For a restarted or uncertain operation:

1. Reopen the repository with supported migrations enabled and inspect the
   proposal's durable state.
2. Reconstruct the same request, corpus version, configuration identity, and
   component-generation identity from the private job state.
3. Retry the exact begin and publication operations in a new transaction.
4. Treat an exact committed result with `replayed = true` as success. It must
   retain the original version, sequence, manifest, and timestamp.
5. If the proposal is still `preparing` or `ready`, validate its staged
   membership and component references before deciding whether to resume or
   abort it explicitly.

An acknowledgement failure is not evidence that publication failed. Re-read
the active version and proposal before starting any compensating work. The
retry may return the already committed result, a semantic no-change outcome,
or a named conflict if the retry payload no longer matches the durable
proposal. Never create a new proposal to guess what an ambiguous commit did.

Baseline publication is corpus-scoped. A second corpus may publish while the
first corpus is idle or advancing, but each operation must use its own
`corpus_id`, expected active version, membership, and snapshot. Verify each
result through a fresh active snapshot; a snapshot for one corpus must not
select the other corpus's document versions or chunk sets.

## Roll back an active corpus

Rollback is a new publication, not a pointer rewind:

1. Select an owned historical version from `list_corpus_versions(corpus_id)`.
2. Record the current active version as the new proposal's expected parent.
3. Begin a new update with a fresh proposed version and the historical version
   as `restores_corpus_version`.
4. Prepare and register new component-generation references owned by the new
   proposed version. Their implementation, formats, contracts, configuration,
   and source-manifest semantics must match the historical target.
5. Seal the proposal, which verifies that membership and component semantics
   reproduce the target.
6. Publish it with `commit_corpus_rollback`, then verify through a fresh active
   snapshot.

The result receives the next sequence, names the previously active version as
its parent, and records the selected historical version as its restore target.
This preserves audit history and causes version-keyed sessions and caches to
observe a genuine state change. A target from another corpus is rejected.

## Diagnose failures safely

| Symptom | Meaning and response |
| --- | --- |
| `StorageConflictError` | Inputs, ownership, lifecycle, or expected active state disagree. Do not blind-retry; inspect opaque IDs and restart planning from the current active version when appropriate. |
| `StorageConcurrencyError` | The bounded SQLite lock wait expired. Retry the complete transaction under a bounded caller policy. |
| `CorruptStorageError` | Durable state failed reconstruction or consistency checks. Stop publication and preserve the database for investigation. |
| `RepositoryUnavailableError` | The local repository or a component validator could not be used. Check availability and permissions without logging paths or payloads. |
| `StorageTransactionError` | Transaction lifecycle, commit, or rollback failed. Treat the attempted unit as unsuccessful and re-read durable state before deciding what to do. |

Storage events are written to stdout through the configured structured logging
pipeline. Correlate with `corpus_id`, `corpus_version`, sequence, bounded
counts, error category, and retryability. Corpus display names, paths,
document content, component payloads, raw validator output, database messages,
and SQL must not be added to logs or error messages.
