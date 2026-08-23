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

## Logs need inspection

Logs are written to stdout. Use `logging.format = "console"` for interactive
inspection or `"json"` for scripts and tooling. Operational logs must remain
privacy-safe: do not add queries, prompts, outputs, document text, source
passages, credentials, or sensitive paths to diagnostic output.

For backup or restore failures, preserve the last known-good published backup,
check the manifest and validation result, and follow the
[backup and restore runbook](backup-restore.md). Do not activate an incomplete,
mixed-version, or unvalidated backup.
