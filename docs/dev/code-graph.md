# Code graph

## Scope

The code graph is a local developer tool for visualizing application structure
and dependencies. It is not part of the deployed product and does not inspect
the corpus, models, queries, prompts, or runtime data.

Graphify scans the repository root in code-only mode. The tracked
`.graphifyignore` limits the graph to:

- `src/`;
- `tests/`.

This scope keeps the graph focused on production code, tests, and their
relationships. Developer scripts, documentation, configuration, caches, and
other repository artifacts are excluded.

## Installation

Install the pinned local Graphify tool:

```sh
uv tool install graphifyy==0.9.47
```

The package installs the `graphify` executable. Verify that it is available:

```sh
graphify --version
```

The graph generator does not require `config.toml`, model paths, or storage
paths.

## Generate or refresh

Run from any directory in this checkout:

```sh
uv run python scripts/graph/generate_graph.py
```

The command:

1. validates the Graphify installation and required scope directories;
2. extracts the scoped code graph without LLM or network-based extraction;
3. clusters the extracted graph without community-label generation;
4. generates the local HTML tree visualization;
5. validates the generated outputs; and
6. publishes the new graph after the complete refresh succeeds.

Logging can be selected explicitly:

```sh
uv run python scripts/graph/generate_graph.py \
  --log-level DEBUG \
  --log-format console
```

## Outputs

Generated files are written under the ignored `.dev/graphify-out/` directory:

- `graph.json`: dependency and structure graph data;
- `graph.html`: clustered community visualization;
- `GRAPH_TREE.html`: browser-openable hierarchy visualization;
- Graphify manifest and local cache files.

Generated artifacts are reproducible developer output. They must not be
committed, backed up as corpus state, or treated as application data.

## Refresh behavior

Refreshes are staged under a temporary directory in `.dev/`. The existing
graph remains available while extraction, clustering, and visualization run.
The current output is replaced only after `graph.json`, `graph.html`, and
`GRAPH_TREE.html` have been validated.

If extraction, visualization, validation, or publication fails, the previous
usable graph is restored or retained and the temporary output is removed.
Stale files from an earlier graph are therefore not carried into a successful
refresh.

## Privacy

Graph generation is local and uses Graphify's `--code-only` and `--no-cluster`
options. It must not use third-party model or data APIs.

The graph can contain source paths, module names, symbols, and code-derived
labels from `src/` and `tests/`. Do not place credentials or confidential
material in those directories. Before sharing generated output, inspect it for
sensitive content.

Logs are structured and local. They must not contain source text, credentials,
queries, prompts, model outputs, or other sensitive payloads.

## Failure handling

Common failures include:

- `graphify_not_found`: install the pinned Graphify tool and ensure its bin
  directory is on `PATH`;
- `scan_scope_missing`: restore the required `src/` or `tests/` directory;
- timeout: inspect Graphify output and retry after reducing repository changes;
- missing graph or tree output: treat the refresh as failed and retain the
  previous graph.

The generator bounds each Graphify subprocess to five minutes. It uses the
repository root as the working directory and does not load application
configuration.
