# Backup and restore runbook

> **Status:** Workflow and initial backup contract defined; backup operations,
> exact commands, and storage-specific procedures remain deferred under TOOL-04.

This runbook covers backup and restore of the persistent corpus state. The
normative policy is defined in the [backup-restore conventions](../conventions/backup-restore.md).

## Recovery objectives

Recovery should restore a validated corpus state that can serve queries
without reconstructing unchanged data. A complete recovery also requires the
original source documents to be available through the separate source-data
backup or availability policy.

## Backup procedure

1. Confirm that the destination has sufficient local storage.
2. Identify the corpus version to capture.
3. Coordinate with ingest so the corpus version cannot advance during capture.
4. Keep read-only queries available where supported.
5. Capture the metadata store and all persistent indexes for the selected
   corpus version.
6. Include the recovery manifest and integrity information.
7. Validate every component, version, generation, and checksum.
8. Publish the backup atomically only after validation succeeds.
9. Record the backup identity, corpus version, component status, and validation
   result.

Failed or interrupted work must remain unpublished and must not replace the
previous published backup.

## Restore procedure

1. Select a published backup and confirm that it is the intended corpus
   version.
2. Confirm that the required source documents are available.
3. Validate the manifest, checksums, schema version, index format versions, and
   software compatibility.
4. Confirm that the restore destination has sufficient storage.
5. Create a pre-restore backup when replacing an active installation.
6. Restore into a separate staging location.
7. Open and validate the metadata store and every included index.
8. Run smoke checks for metadata queries, index queries, provenance, and source
   spans.
9. Record whether indexes were restored directly or rebuilt from canonical
   records.
10. Atomically activate the validated staged restore.
11. Restart or reconnect serving components.
12. Verify restart-to-serve behavior and run post-activation checks.

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
backups. A portable backup contains a logical metadata-store export, canonical
chunks and provenance, index configuration and generation metadata, a manifest,
schema information, and checksums. Serialized indexes are included only when
their format is documented and portable; otherwise they are rebuilt from the
canonical records during restore. Compatible recovery snapshots are an
optimization and do not replace this baseline contract.

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
