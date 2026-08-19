# inference-lab

`inference-lab` is a locally hosted tool for interrogating papers and
patents. It is designed around provenance-preserving ingestion, hybrid retrieval,
evidence-grounded answers, and hardware-aware serving on one GPU. Documents,
queries, and telemetry remain local in deployment.

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

## Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The test directories are currently placeholders, so pytest reports that no
tests are collected until the first test cases are added.

Release versioning is derived from committed headers. Preview or apply the next
SemVer bump with `uv run python scripts/release.py --dry-run` or without the flag.
The release script also adds a section to `CHANGELOG.md`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit and review conventions, and
[docs/spec.md](docs/spec.md) for the product specification.