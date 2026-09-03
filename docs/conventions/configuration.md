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
| `[storage]` | Persistent index and metadata database paths |
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
uv run inference-lab
```

Environment values take precedence over the corresponding TOML values. This
is intended for one-off paths, CI jobs, and deployment-specific overrides.

## Validation and lifecycle

The loader rejects missing required values and invalid types. Paths are
validated as `Path` values; loading configuration does not require the target
paths to exist.

Required values are:

- `models.embedder_path`
- `models.reranker_path`
- `models.generator_path`
- `runtime.vram_budget_gb`
- `storage.index_dir`
- `storage.metadata_db_path`

`runtime.max_context_tokens` defaults to `8192`.

`get_settings()` caches the validated settings for the process lifetime.
Changes to `config.toml` or the environment take effect after a process
restart. Tests and other controlled callers may clear the cache explicitly.

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
