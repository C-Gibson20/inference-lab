# Configuration conventions

Inference Lab uses typed settings for local model, runtime, and storage
configuration. Settings are loaded and validated by
`inference_lab.settings` before application startup continues.

The configuration is deployment-specific. Keep model and storage paths local,
and do not commit `config.toml`.

## Configuration file

The default file is `config.toml` in the current working directory. Create it
from [config.example.toml](../../config.example.toml), then update the paths
for the local machine:

```sh
cp config.example.toml config.toml
```

The current sections are:

| Section | Purpose |
| --- | --- |
| `[models]` | Local embedder, reranker, and generator paths |
| `[runtime]` | VRAM budget and maximum context size |
| `[storage]` | Persistent index, metadata database, and managed source paths |
| `[logging]` | Log level and output format |
| `[ingest]` | Controls safety limits for batch ingestion |

Run the application from the directory containing `config.toml`.

Storage paths configure one local deployment, not one corpus. The metadata
database and index root may contain multiple named corpora. Corpus creation and
selection are runtime domain operations using `CorpusId`; they are not achieved
by swapping configuration files or assigning one database path per corpus.

There is no generic `[feature_flags]` section. Feature flags are introduced
only alongside the feature they control, in the relevant typed settings model.
The owning feature must document each flag's TOML field, environment override,
default, and operational effect. Until then, a proposed flag is unsupported
and must be rejected by configuration validation.

Chunking policy and tokeniser identity are explicit validated inputs to an
ingest request, not deployment settings or feature flags in the baseline. A
tokeniser artefact is selected by composition code and must remain local and
path-free in its recorded identity. See the [chunking conventions](chunking.md)
for the closed policy and invalidation rules.

## Source precedence

Values are resolved from highest to lowest precedence:

1. Explicit values passed to `Settings(...)`.
2. Environment variables.
3. Values in `config.toml`.

The loader retains extension points for dotenv and secret-file sources, but
neither is part of the current deployment contract.

Overrides are field-level and nested settings remain mergeable. For example,
overriding `runtime.vram_budget_gb` does not discard
`runtime.max_context_tokens` loaded from TOML.

## Environment overrides

Environment variables use upper-case field names and `__` between nested
sections and fields:

```sh
MODELS__GENERATOR_PATH=/mnt/models/generator \
STORAGE__INDEX_DIR=/mnt/data/indexes \
STORAGE__SOURCE_DIR=/mnt/source-data \
uv run inference-lab
```

Environment values take precedence over the corresponding TOML values. This
is intended for one-off paths, CI jobs, and deployment-specific overrides.

## Validation and lifecycle

The loader rejects missing required values and invalid types. Paths are
validated as `Path` values; loading configuration does not require the target
paths to exist.

All path trust boundaries use the shared local-path syntax policy. Empty and
NUL-containing paths, URI schemes before or after `Path` coercion, and
UNC-shaped network locations are rejected. POSIX and relative paths are local;
Windows drive paths are local only on a Windows host. The shared check does not
resolve paths or touch the filesystem. Each consuming domain owns those later
actions and translates rejection into its own public error contract.

Required values are:

- `models.embedder_path`
- `models.reranker_path`
- `models.generator_path`
- `runtime.vram_budget_gb`
- `storage.index_dir`
- `storage.metadata_db_path`
- `storage.source_dir`

`runtime.max_context_tokens` defaults to `8192`.
`ingest.max_source_bytes` defaults to `52428800` bytes and must be greater than
zero. Source publication enforces this limit while streaming into managed
staging; it must not buffer or publish an oversized source first.

`get_settings()` caches the validated settings for the process lifetime.
Changes to `config.toml` or the environment take effect after a process
restart. Tests and other controlled callers may clear the cache explicitly.

Application composition opens metadata through
`inference_lab.storage.open_metadata_repository()`, passing the validated
`settings.storage.metadata_db_path` and `settings.storage.source_dir`. Schema
creation and supported forward migrations occur only when the caller explicitly
sets `apply_migrations=True`; the default open policy is non-mutating and rejects
a missing or older schema.

`storage.source_dir` is the deployment-owned root for managed copies of
original source files. It must be separate from `storage.index_dir` and the
metadata database. Original sources remain subject to the separate source-data
backup policy; they are not included in corpus-state backups. The repository
owns the directory layout, validates managed bytes on read, and must be the only
component that modifies its contents. See the
[storage conventions](storage.md) for lifecycle, integrity, and corpus
visibility rules.

“Separate” means that `storage.index_dir`, `storage.metadata_db_path`, and
`storage.source_dir` are neither equal nor ancestors or descendants of one
another. Settings enforce this after converting each path to a host-normalised
absolute lexical path; this does not require the targets to exist or resolve
symbolic links. The storage and index consumers must repeat the separation
check after safely resolving existing filesystem aliases before creating or
opening managed state.

Configuration errors must fail startup clearly. They must not be replaced with
guessed model paths, storage paths, or runtime values.

## Adding settings

Every new setting should have:

- a typed field on the relevant settings model;
- a documented TOML location and environment override;
- an explicit default or required-value policy;
- validation appropriate to its use; and
- tests covering TOML loading, precedence, valid conversion, missing values,
  and invalid values.

Domain modules should receive validated settings rather than reading
`os.environ` directly.

Feature flags follow the same rules. A flag must have a clear owning feature,
an explicit boolean type and default, and a field-specific environment name;
flags must not be represented by an untyped dictionary.

Logging settings follow this loading and override strategy.
