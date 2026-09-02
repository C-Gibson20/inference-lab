# Configuration reference

Inference Lab loads typed settings from `config.toml` in the current working
directory. The file is deployment-specific and must not be committed.

Create it from the tracked example:

```sh
cp config.example.toml config.toml
```

## Precedence and lifecycle

Values are resolved from highest to lowest precedence:

1. explicit `Settings(...)` values;
2. environment variables; and
3. values in `config.toml`.

Environment overrides are field-level and use uppercase names with `__`
between nested sections and fields. For example:

```sh
RUNTIME__VRAM_BUDGET_GB=24 \
STORAGE__INDEX_DIR=/mnt/data/indexes \
uv run inference-lab
```

Settings are validated before startup continues. Paths are parsed as `Path`
values, but their targets do not need to exist during loading. Operations that
cannot create or access a required path must report a clear error.

`get_settings()` caches settings for the process lifetime. Changes to the TOML
file or environment take effect after a process restart.

## Settings

| TOML field | Environment override | Policy |
| --- | --- | --- |
| `models.embedder_path` | `MODELS__EMBEDDER_PATH` | Required local embedder path. |
| `models.reranker_path` | `MODELS__RERANKER_PATH` | Required local reranker path. |
| `models.generator_path` | `MODELS__GENERATOR_PATH` | Required local generator path. |
| `runtime.vram_budget_gb` | `RUNTIME__VRAM_BUDGET_GB` | Required VRAM budget in decimal GB; accounting converts `1 GB` to `1,000,000,000` bytes. |
| `runtime.max_context_tokens` | `RUNTIME__MAX_CONTEXT_TOKENS` | Defaults to `8192`. |
| `storage.index_dir` | `STORAGE__INDEX_DIR` | Required persistent index path. |
| `storage.metadata_db_path` | `STORAGE__METADATA_DB_PATH` | Required metadata database path. |
| `logging.level` | `LOGGING__LEVEL` | Logging threshold; example default is `INFO`. |
| `logging.format` | `LOGGING__FORMAT` | `console`, `json`, or `auto`; example default is `auto`. |

The configuration loader must reject missing required values and invalid types.
It must not substitute guessed model paths, storage paths, or runtime values.
Backup-specific settings are defined but not yet available; see the
[backup and restore runbook](backup-restore.md) for the selected contract and
current implementation status.

### Memory-budget conversion

`runtime.vram_budget_gb` uses decimal gigabytes because the field is named
`gb`. The runtime converts the validated value exactly once at the accounting
boundary: `budget_bytes = vram_budget_gb * 1,000,000,000`. It must not be
interpreted as GiB or multiplied again by downstream components. A future
binary unit would require a separately named setting such as `gib`.

Feature flags are not currently part of the configuration contract. They will
be added as typed, feature-owned settings when the corresponding feature is
introduced; a generic top-level `feature_flags` section is unsupported.

## Logging

Logs are written to stdout. `console` is intended for interactive use, `json`
for scripts and tooling, and `auto` selects console output on a TTY and JSON
otherwise. JSON records retain structured fields for operational tooling.

Logs must not contain document text, source passages, user queries, prompts,
model outputs, credentials, or sensitive paths. See the
[logging conventions](../conventions/logging.md) for the record schema and
level policy.

## Adding settings

Every new setting requires a typed field, TOML location, environment override,
explicit default or required policy, validation, and tests for loading,
precedence, conversion, missing values, and invalid values. Update this table
when the setting becomes part of the deployed contract.
