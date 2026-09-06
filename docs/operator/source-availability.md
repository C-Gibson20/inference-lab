# Source availability

## Current capability

Inference Lab now has a restart-safe storage boundary for managed original
bytes, parsed representations, parser outcomes, and source locations. The
current command-line entry point does not yet expose ingest, source-opening, or
repair commands. STOR-03 now publishes immutable corpus versions through this
boundary. ING-03 will connect parser output to it, and SERV-04 will provide
citation opening and navigation.

This page describes the operator responsibilities that already apply to the
persistent source store. The normative contract is in the
[storage conventions](../conventions/storage.md).

## Managed-source layout

Set `storage.source_dir` to a persistent local directory separate from
`storage.metadata_db_path` and `storage.index_dir`. The directory may contain
immutable originals reused by multiple corpora, but corpus membership still
controls every public read. Sharing bytes never makes a document visible in an
unselected corpus.

The source directory is repository-owned. Do not rename, edit, move,
deduplicate, or delete individual entries. Managed names are internal
content-addressed locators and are not original filenames. The service account
must be able to create directories, stage files, atomically rename them, read
them after restart, and acquire the publication lock. Ensure the filesystem has
enough capacity for an incoming source plus temporary staging data.

Configuration changes take effect after process restart. Moving only the
metadata database or only the source directory produces an incomplete
deployment even if startup succeeds.

## Integrity and startup

Publication verifies the configured size limit, expected byte count, and
SHA-256 identity before metadata becomes reachable. Reads verify the complete
managed file again before returning it. A missing or non-member source is not
disclosed and may return no result; a source that is reachable through metadata
but missing, truncated, replaced, or linked fails closed as corrupt storage.

Opening the repository performs bounded conservative reconciliation. It removes
abandoned staging files and inventories final blobs without deleting referenced
or ambiguous data. This maintenance is not a substitute for source backup or
capacity monitoring.

## Backup responsibility

Back up `storage.source_dir` separately from corpus-state backups. The planned
STOR-08 portable backup includes canonical metadata and source integrity
inventory but deliberately excludes managed original bytes. A metadata or index
restore is not operationally complete until every required original is also
available and passes its stored byte-count and SHA-256 checks.

The source-data backup workflow and commands have not yet been implemented.
Until they are, use deployment-owned local backup facilities and treat the
source directory and metadata database as related state. Do not claim a
successful recovery based only on restoring metadata or indexes. Preserve the
previous known-good copies until source validation succeeds.

See the [backup and restore runbook](backup-restore.md) for the corpus-state
contract and its current implementation status.

## Diagnosing unavailable sources

Use opaque `corpus_id` and `document_id` values plus the structured
`error_category` when correlating an incident. Logs intentionally omit source
text, filenames, paths, storage keys, content hashes, publication identifiers,
parser failure payloads, SQL, and raw filesystem or database messages.

Common categories are:

| Symptom | Meaning | Operator action |
| --- | --- | --- |
| No source result | The record is absent or not visible in the selected corpus. | Verify the selected corpus and its explicit membership through the future application workflow; do not probe managed files directly. |
| `corrupt_state` | Reachable metadata or managed bytes failed reconstruction or integrity validation. | Preserve the affected state, check source-data backup availability, and restore only bytes that match the recorded immutable identity. |
| `migration_required` | The database has a supported older schema and was opened without migration permission. | Run the future documented migration workflow; do not modify schema tables manually. |
| `incompatible_schema` | The schema is malformed or newer/unsupported. | Stop and use compatible software or a validated recovery copy. |
| `concurrency` | Bounded database or source-publication access was unavailable. | Retry with bounded backoff after checking for another active writer. |
| `repository_unavailable` | A configured path, permission, device, or database resource is unavailable. | Check the configured paths, filesystem permissions, capacity, and process ownership. |

There is not yet an operator repair command. Do not silence integrity failures
by deleting metadata, replacing bytes with a different file, changing a stored
hash, or assigning a new identity manually.

## Recovery verification

When the complete backup and citation workflows land, a recovery is complete
only after a clean process can:

1. open the metadata repository without an unsupported schema change;
2. select the intended corpus and recover its parsed representation;
3. resolve stored spans to decoded text and optional page regions;
4. open every required original with byte-count and hash verification; and
5. confirm another corpus cannot see sources without explicit membership.

Record only privacy-safe IDs, categories, counts, and validation status in the
recovery report.
