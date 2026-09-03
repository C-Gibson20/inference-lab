# Getting started

## Prerequisites

Install Python 3.13 and `uv`, then from the repository root run:

```sh
uv sync
```

The deployed system must use local models and local storage. Do not place
confidential documents, queries, telemetry, or model traffic behind a
third-party model or data API.

## Configure a local deployment

Create the ignored, machine-specific configuration file:

```sh
cp config.example.toml config.toml
```

Update the model and storage paths before starting the application. Required
settings and their environment overrides are listed in the
[configuration reference](configuration.md).

These storage paths belong to the local deployment and may eventually contain
multiple named corpora. Corpus selection is an application operation, not a
different configuration file or metadata database per corpus.

Run the application from the directory containing `config.toml`:

```sh
uv run inference-lab
```

Configuration is validated before application work begins. Missing required
values and invalid types fail startup; the application must not guess paths or
runtime values.

## Current status

The entry point is currently a skeleton. Ingest, search, and answer commands
will be added through the dependency-ordered serving work. Once the first
executable query workflow is available, this page will include a complete
clean-checkout walkthrough:

1. create or select a named corpus;
2. prepare deterministic or local sources and ingest them incrementally into
   that corpus;
3. restart the process;
4. run a query against the selected corpus; and
5. confirm source-linked results.

Do not treat successful process startup as evidence that a corpus is indexed
or queryable.
