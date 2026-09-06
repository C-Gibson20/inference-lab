# Troubleshooting

## Startup fails with a configuration error

Check that the process is running from the directory containing `config.toml`
and that every required setting is present:

- `models.embedder_path`;
- `models.reranker_path`;
- `models.generator_path`;
- `runtime.vram_budget_gb`;
- `storage.index_dir`; and
- `storage.metadata_db_path`.

Check the TOML type and any environment override. Environment values take
precedence over the file, and settings changes require a process restart.
The loader does not guess missing paths or runtime values.

## A configured path cannot be used

Configuration loading accepts path values even when their targets do not yet
exist. The operation using the path must still be able to create or access it,
and the host must have sufficient storage. Check permissions, available space,
and whether the path is shared with another active installation.

## No query result is available

The current command is a skeleton and does not yet expose the query workflow.
After the first executable query workflow is available, check that the corpus
was ingested, the persistent index is available, and the documented query
command is being run from the correct deployment directory. An empty result is
not by itself evidence of a failed operation.

## A local source was skipped or failed

Check the bounded `error_category`, `failure_stage`, and `retryable` fields in
the structured logs. Common policy outcomes include unsupported or unknown
format, symbolic link, special file, nested mount, oversized source, unstable
read, ambiguous observation, and exhausted write contention. Paths are omitted
from logs; use the private in-process result returned to the integrating local
workflow to associate a source ordinal with its locator.

An empty directory is a successful result. PDF files are currently recognised
but unsupported, and only `.txt` files have a parser capability. Discovery does
not follow links. Move detection is intentionally conservative when a scan is
incomplete, the prior path is still live, or more than one candidate matches.
See [local source discovery](source-discovery.md) for the complete policy.

If `source_changed_during_read` occurs, stop the writer that is modifying the
file and retry discovery from a stable snapshot. If a later managed-source copy
cannot revalidate a discovery snapshot, rediscover the source rather than
reusing the earlier identity decision.

## Logs need inspection

Logs are written to stdout. Use `logging.format = "console"` for interactive
inspection or `"json"` for scripts and tooling. Operational logs must remain
privacy-safe: do not add queries, prompts, outputs, document text, source
passages, credentials, or sensitive paths to diagnostic output.

For backup or restore failures, preserve the last known-good published backup,
check the manifest and validation result, and follow the
[backup and restore runbook](backup-restore.md). Do not activate an incomplete,
mixed-version, or unvalidated backup.
