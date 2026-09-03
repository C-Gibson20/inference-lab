# Contributing

## Branches and commits

Use short branches named `feature/<topic>`, `fix/<topic>`, or
`experiment/<topic>`. Commits must use the enforced form:

```text
[<module>] <type>: <description>
```

Valid modules are `ingest`, `retrieval`, `serving`, `runtime`, `storage`,
`provenance`, `evaluation`, `profiling`, `docs`, and `tooling`. Valid types are
`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, and `chore`.
Add `!` after the type for a breaking change. `feat` produces a minor release;
`fix` and `perf` produce patch releases; `!` or a `BREAKING CHANGE:` footer
produces a major release.

## Local checks and style

Run `./scripts/setup-hooks.sh` once per checkout. The pre-commit hook runs Ruff
linting and formatting checks. The complete validation commands are:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov
```

Use Ruff’s 88-character formatting and keep imports/lint rules clean. Ruff
enforces docstrings on public source modules, classes, and functions. Comments
should explain non-obvious reasoning, not restate code. The test suite must
maintain at least 80% combined statement-and-branch coverage; the canonical
policy is configured in `pyproject.toml`. Focused test runs may omit `--cov`.
Use `structlog` inside `inference_lab`; use standard-library `logging` only for
dependencies and process-level handlers. Do not log documents, queries, or
other sensitive content.

## Pull requests

Keep changes focused, describe the evidence for behavior or performance claims,
and include tests or evaluation updates when applicable. Before requesting
review, run the complete validation commands and confirm that local deployment
data remains local.
