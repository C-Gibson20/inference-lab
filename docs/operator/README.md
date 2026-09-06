# Operator documentation

Inference Lab is a locally hosted tool for querying multiple named corpora of
papers and patents. Each operation selects one corpus; cross-corpus querying is
not part of the first build. Model paths, corpus data, queries, telemetry, and
operational logs remain local in deployment.

## Start here

- [Getting started](getting-started.md): install, configure, and start the
  application.
- [Configuration reference](configuration.md): typed settings, defaults, and
  environment overrides.
- [Running queries](querying.md): current query status and the operator
  workflow as serving capabilities land.
- [Corpus-version operations](corpus-versions.md): staging, atomic publication,
  reader snapshots, restart recovery, and rollback.
- [Evaluation dataset operations](evaluation.md): local validation,
  deterministic writing, held-out governance, and privacy boundaries.
- [Backup and restore](backup-restore.md): recover persistent corpus state.

Normative configuration, logging, and backup policy is defined in
[`docs/conventions/`](../conventions/). These operator documents describe how
to apply that policy.

## Documentation status

The command-line entry point currently starts a skeleton application. Durable
corpus-version operations are available to Python integrations but do not yet
have operator-facing commands. The first executable ingest-and-search workflow
will be documented when it is implemented.
Answer generation, citation verification, sessions, chat, named-corpus
creation/selection and updates, and privacy checks will extend these pages as
those capabilities become available.

When an operator-facing command or setting changes, update the relevant page in
the same change and verify the examples from a clean local configuration.
