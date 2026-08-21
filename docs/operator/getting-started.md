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

Run the application from the directory containing `config.toml`:

```sh
uv run inference-lab
```

Configuration is validated before application work begins. Missing required
values and invalid types fail startup; the application must not guess paths or
runtime values.

## Current status

The entry point is currently a skeleton. Ingest, search, and answer commands
will be added through the dependency-ordered serving work. Once `SERV-01` is
available, this page will include a complete clean-checkout walkthrough:

1. prepare a deterministic or local corpus;
2. ingest it incrementally;
3. restart the process;
4. run a query; and
5. confirm source-linked results.

Do not treat successful process startup as evidence that a corpus is indexed
or queryable.
