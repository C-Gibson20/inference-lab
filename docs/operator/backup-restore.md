# Backup and restore runbook

> **Status:** The workflow and initial backup contract are defined. Backup
> operations, exact commands, and storage-specific procedures remain deferred
> until the persistent component interfaces are implemented.

This runbook covers backup and restore of persistent multi-corpus state. The
portable baseline covers one selected corpus; compatible recovery snapshots
may cover the complete deployment. The normative policy is defined in the
[backup-restore conventions](../conventions/backup-restore.md).

## Recovery objectives

Recovery should restore a validated selected-corpus state that can serve
queries without reconstructing unchanged data and without changing unrelated
corpora. A complete recovery also requires the original source documents to be
available through the separate source-data backup or availability policy. See
[source availability](source-availability.md) for the current managed-original
integrity and operator boundary.

## Backup procedure

1. Confirm that the destination has sufficient local storage.
2. Identify the `CorpusId` and owned `CorpusVersionId` to capture.
3. Coordinate with ingest so the corpus version cannot advance during capture.
4. Keep read-only queries available where supported.
5. Capture the selected corpus's membership, reachable canonical metadata, and
   all persistent indexes for its selected version. Do not include another
   corpus's membership or private derived state.
6. Include the recovery manifest and integrity information.
7. Validate every component, version, generation, and checksum.
8. Publish the backup atomically only after validation succeeds.
9. Record the backup identity, corpus ID and version, component status, and
   validation result.

Failed or interrupted work must remain unpublished and must not replace the
previous published backup.

## Restore procedure

1. Select a published backup and confirm that it is the intended corpus and
   version.
2. Confirm that the required source documents are available.
3. Validate the manifest, checksums, schema version, index format versions, and
   software compatibility.
4. Confirm that the restore destination has sufficient storage.
5. Create a pre-restore backup when replacing an active corpus or installation.
6. Restore into a separate staging location.
7. Open and validate the metadata store and every included index.
8. Run smoke checks for metadata queries, index queries, provenance, and source
   spans.
9. Record whether indexes were restored directly or rebuilt from canonical
   records.
10. Reuse exact deployment-global immutable records already present, and reject
    conflicting content for an existing ID.
11. Atomically activate the validated staged restore without changing
    unrelated corpora.
12. Restart or reconnect serving components.
13. Verify restart-to-serve behaviour and run post-activation checks.

The active installation must remain unchanged if any validation, activation, or
post-activation check fails.

## Failure and rollback

- Treat incomplete, corrupt, mixed-version, and incompatible backups as
  unusable.
- Keep the previous active state available until activation succeeds.
- Roll back to the previous state if activation or post-activation checks
  fail.
- Keep the last known-good published backup available.
- Make retries idempotent and do not publish partial state.
- Report whether recovery remains available after every failed operation.

## Recovery verification

After restore, verify:

- The expected corpus version is active.
- Other corpora retain their membership and active versions.
- The metadata store opens and returns expected fixture records.
- Each persistent index opens and returns expected results.
- Results resolve to document IDs, source spans, and source locations.
- Required source documents can be opened for citation navigation.
- The service remains available after restart without reconstruction of
  unchanged data.
- Backup and restore events contain privacy-safe structured logs.

## Recovery drills

Perform restore drills in a clean location rather than over the active
installation. Record the backup identity, restore outcome, duration,
reconstruction performed, validation results, and any failure modes.

Portable backups should be used for cross-machine and deterministic-fixture
testing. Compatible recovery snapshots should be tested separately for fast
local recovery.

## Initial backup contract

The initial implementation uses immutable, generation-based full portable
backups of one selected corpus. A portable backup contains that corpus's
membership, a logical export of reachable canonical metadata, chunks and
provenance, index configuration and generation metadata, a manifest, schema
information, and checksums. Serialised indexes are included only when their
format is documented and portable; otherwise they are rebuilt from canonical
records during restore. Compatible recovery snapshots may cover the complete
deployment by listing the active version of every included corpus; they are an
optimisation and do not replace the per-corpus portable baseline.

Backup settings belong under `[storage]` and follow the normal precedence order
(explicit `Settings(...)` values, environment variables, then `config.toml`):

| Setting | TOML field | Environment override | Default or policy |
| --- | --- | --- | --- |
| Backup destination | `storage.backup_dir` | `STORAGE__BACKUP_DIR` | Required when backup operations are enabled. |
| Restore staging directory | `storage.restore_staging_dir` | `STORAGE__RESTORE_STAGING_DIR` | Required when restore operations are enabled. |
| Retained generations | `storage.backup_retention_count` | `STORAGE__BACKUP_RETENTION_COUNT` | Defaults to `5`. |
| Compression | `storage.backup_compression` | `STORAGE__BACKUP_COMPRESSION` | Defaults to `none`. |
| Automatic backups | `storage.automatic_backup_enabled` | `STORAGE__AUTOMATIC_BACKUP_ENABLED` | Defaults to `false`; explicit backups are the baseline. |

These settings are the selected contract but are not available until the
backup lifecycle is implemented. Paths must be local filesystem paths, and
backup and restore must report clear errors for inaccessible or undersized
destinations.

## To be completed during implementation

- Final backup and restore commands or API operations.
- Concrete backup layout and storage paths.
- Database- and index-specific snapshot procedures.
- Permissions and service-management steps.
- Retention and pruning commands.
- Expected output and troubleshooting guidance.
- Measured backup, restore, and index-rebuild times.
