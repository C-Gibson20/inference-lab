# Backup and restore runbook

> **Status:** Provisional. This runbook defines the operator workflow from the
> backup policy. Exact commands, configuration names, paths, and
> storage-specific procedures will be added when implementation begins.

This runbook covers backup and restore of the persistent corpus state. The
normative policy is defined in the [backup conventions](../conventions/backup.md).

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

## To be completed during implementation

- Final backup and restore commands or API operations.
- Configuration names, defaults, and environment overrides.
- Concrete backup layout and storage paths.
- Database- and index-specific snapshot procedures.
- Permissions and service-management steps.
- Retention and pruning commands.
- Expected output and troubleshooting guidance.
- Measured backup, restore, and index-rebuild times.
