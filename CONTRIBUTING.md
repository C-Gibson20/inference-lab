# Contributing

## Branches and commits

Use short branches named `feature/<topic>`, `fix/<topic>`,
`experiment/<topic>`, or `review/<topic>` for work that remediates a recorded
review. Commits must use the enforced form:

```text
[<module>] <type>: <description>
```

Valid modules are `ingest`, `retrieval`, `serving`, `runtime`, `storage`,
`provenance`, `evaluation`, `profiling`, `docs`, `tooling`, and `src`. Use a
domain module whenever the change has one; `src` is only for a change that
genuinely spans several domains, such as a repository-wide rename. Valid types
are `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, and
`chore`.
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
should explain non-obvious reasoning, not restate code. The coverage floor
configured in `pyproject.toml` is a regression tripwire, not a target: do not
add tests solely to raise it. Behaviour is gated by the acceptance journeys
described in `docs/plans/test-suite.md`. Focused test runs may omit `--cov`.
Use `structlog` inside `inference_lab`; use standard-library `logging` only for
dependencies and process-level handlers. Do not log documents, queries, or
other sensitive content.

## Pull requests

Keep changes focused, describe the evidence for behaviour or performance
claims, and include tests or evaluation updates when applicable. When a
change necessarily fixes something owned by another backlog item, say so in
the description instead of letting it land silently: an unrecorded
cross-item fix is invisible to the item that owns it, and to anyone later
asking when the behaviour changed. Before requesting review, run the
complete validation commands and confirm that local deployment data remains
local.
