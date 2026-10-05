# Preparing a local evaluation corpus

This procedure creates one pinned local corpus version from an explicit
directory of approved source documents. It uses the serving ingest API to
parse and chunk the sources, then publishes the lexical index and, when enabled,
the dense index under one immutable corpus version. It does not download
sources or models, or run retrieval evaluation.

## Requirements

- Run from the deployment directory containing its private `config.toml`.
- Keep the source directory outside the configured managed source directory.
- Point `EVALUATION_CORPUS_TOKENISER_DIRECTORY` at an existing local
  Transformers fast tokenizer directory. Loading is local only.
- Set `EVALUATION_CORPUS_INCLUDE_DENSE=true` to include the configured local
  Qwen embedding model in the same atomic publication. Its files must already
  be present at `models.embedder_path` in `config.toml` and pass the backend's
  artifact verification. The default is `false` for lexical-only preparation.
- Choose one stable `EVALUATION_CORPUS_ID` and a new
  `EVALUATION_CORPUS_PROPOSED_VERSION`. Retain both identities for the
  evaluation manifest. An update must also set
  `EVALUATION_CORPUS_EXPECTED_VERSION` to the currently active parent version.

The Python process and all corpus storage remain local. The command prints
only corpus/version identities and bounded outcome counts. It does not print
source paths, filenames, document text, or raw exceptions.

## First ingest

Set these values in the local shell, then run the script from the deployment
directory:

```sh
export EVALUATION_CORPUS_SOURCE_DIRECTORY=/absolute/path/to/approved/documents
export EVALUATION_CORPUS_TOKENISER_DIRECTORY=/absolute/path/to/local/fast-tokeniser
export EVALUATION_CORPUS_ID='paste-a-generated-corpus-id'
export EVALUATION_CORPUS_PROPOSED_VERSION='paste-a-new-corpus-version-id'
export EVALUATION_CORPUS_INCLUDE_DENSE=true

uv run --project /absolute/path/to/inference-lab python \
  /absolute/path/to/inference-lab/scripts/inference_lab/evaluation/prepare_local_evaluation_corpus.py
```

Generate opaque IDs with the project value constructors:

```sh
uv run --project /absolute/path/to/inference-lab python -c \
  'from inference_lab.provenance import CorpusId; print(CorpusId.generate().as_str())'
uv run --project /absolute/path/to/inference-lab python -c \
  'from inference_lab.provenance import CorpusVersionId; print(CorpusVersionId.generate().as_str())'
```

The command creates the named corpus if that ID is new. Replaying the same
corpus ID, version, source bytes, tokenizer, and policy is safe. A failed or
partially completed ingest is not an evaluation corpus version; inspect the
bounded status and fix the source/configuration before retrying.

## Updating an existing corpus

Keep the corpus ID, set a fresh proposed version, and pin the active parent:

```sh
export EVALUATION_CORPUS_EXPECTED_VERSION='paste-the-current-active-version-id'
export EVALUATION_CORPUS_PROPOSED_VERSION='paste-a-new-corpus-version-id'
```

Then rerun the same command. The ingest API publishes a complete immutable
version or leaves the previous version active. Do not change the source set,
tokenizer, parser, chunker, or policy after authoring evidence references for a
dataset version.

## Current scope

The output includes corpus/version identities and the component generation IDs
published in that version. With dense preparation enabled, the dense generation
is built before publication and committed together with chunks and the lexical
generation. The channel request factory, frozen benchmark, and held-out report
are separate steps. The report gate must remain inconclusive until every
planned channel has an honest result on the same corpus version.
