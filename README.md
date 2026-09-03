# inference-lab

`inference-lab` is a locally hosted tool for interrogating multiple named
corpora of papers and patents. It is designed around provenance-preserving
ingestion, corpus-isolated hybrid retrieval, evidence-grounded answers, and
hardware-aware serving on one GPU. Documents, queries, and telemetry remain
local in deployment. Each first-build operation targets one selected corpus;
federated cross-corpus querying is out of scope.

## Setup

Install Python 3.13 and [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync
./scripts/setup-hooks.sh
```

The setup script configures this checkout to use the versioned hooks in
`.githooks/`. Run it again after cloning a fresh checkout; Git does not version
`.git/hooks/`.

The command-line entry point is currently a skeleton:

```sh
uv run inference-lab
```

## Configuration

Configuration is loaded from `config.toml` in the project working directory.
The file is intentionally ignored by Git because model and storage paths are
specific to each machine or deployment.

Create a local configuration from the tracked example, then update its paths:

```sh
cp config.example.toml config.toml
```

The settings loader also supports nested environment-variable overrides. Use
`__` between the section and field names:

```sh
MODELS__GENERATOR_PATH=/mnt/models/generator \
STORAGE__INDEX_DIR=/mnt/data/indexes \
uv run inference-lab
```

Run the command from the project root unless the application is given a
different configuration-loading mechanism in the future.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov
```

Release versioning is derived from committed headers. Preview or apply the next
SemVer bump with `uv run python scripts/release.py --dry-run` or without the flag.
The release script also adds a section to `CHANGELOG.md`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit and review conventions, and
[docs/spec.md](docs/spec.md) for the product specification.
