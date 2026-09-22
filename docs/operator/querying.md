# Running queries

## Current status

The local Python API is the supported initial surface for corpus creation,
incremental ingest/resume, filtered lexical search, and typed failures. The
`inference-lab` entry point remains a startup skeleton. The search-only script
below is a convenience example, not an ingest or general filter CLI.

Do not infer that an application startup message means that a corpus has been
ingested or that queries are available.

## Baseline query workflow

Use the same local settings, metadata database, index directory, and source
directory for corpus creation, ingest, and search. Keep the opaque corpus ID
and each committed corpus version returned by the API.

1. create, ingest, or open a named persistent corpus;
2. select that corpus and issue a query through the documented local API or
   fresh-process example;
3. inspect the resolved corpus version plus the returned document IDs,
   passages, pages, and source spans; and
4. open the cited source location before relying on a result.

Every query targets one corpus. Results from different corpora are not fused,
and shared document storage must not make a document visible outside explicit
membership. Cross-corpus comparison requires separate queries until a future
federated-query contract is added.

### Copyable create → ingest → resume → filter workflow

Run these files from the directory containing the deployment's `config.toml`.
The source file must be outside repository-managed `storage.source_dir`. Point
`TOKENISER_DIR` at an existing local Transformers **fast** tokeniser directory;
loading disables network access. `FakeChunkTokeniser` is reserved for tests.

Save this as `create-corpus.py`:

<!-- operator-create-example:start -->
```python
import json
import os
import sys

from inference_lab.logging.logging_config import configure_logging
from inference_lab.serving import LocalServingFacade
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository

settings = get_settings()
configure_logging(settings.logging, stream=sys.stderr)
repository = open_metadata_repository(
    settings.storage.metadata_db_path,
    source_dir=settings.storage.source_dir,
    apply_migrations=True,
)
try:
    corpus = LocalServingFacade(repository, settings).create_corpus(
        os.environ.get("CORPUS_NAME", "My local corpus")
    )
    print(json.dumps({"corpus_id": corpus.corpus_id.as_str()}))
finally:
    repository.close()
```
<!-- operator-create-example:end -->

Save this as `ingest-and-search.py`:

<!-- operator-ingest-example:start -->
```python
import json
import os
import sys
from pathlib import Path

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    ChunkingPolicy,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    LocalFastChunkTokeniser,
    derive_chunk_configuration_identity,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import ExactFilter
from inference_lab.serving import (
    LocalServingFacade,
    ServingIngestRequest,
    ServingSearchRequest,
)
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository

settings = get_settings()
configure_logging(settings.logging, stream=sys.stderr)
corpus_id = CorpusId.from_str(os.environ["CORPUS_ID"])
proposed_version = CorpusVersionId.from_str(os.environ["PROPOSED_VERSION"])
expected_text = os.environ.get("EXPECTED_VERSION")
expected_version = (
    None if expected_text is None else CorpusVersionId.from_str(expected_text)
)
tokeniser = LocalFastChunkTokeniser.from_directory(os.environ["TOKENISER_DIR"])
policy = ChunkingPolicy()
request = ServingIngestRequest(
    corpus_id=corpus_id,
    discovery_request=DiscoveryRequest(
        corpus_id=corpus_id,
        inputs=(
            DiscoveryInput(Path(os.environ["SOURCE_FILE"]), DiscoveryInputKind.FILE),
        ),
    ),
    proposed_corpus_version=proposed_version,
    expected_corpus_version=expected_version,
    configuration_identity=derive_chunk_configuration_identity(
        parser_version=CURRENT_PARSER_VERSION,
        chunker_version=CURRENT_CHUNKER_VERSION,
        tokeniser_identity=tokeniser.identity,
        policy=policy,
    ),
    chunking_policy=policy,
    tokeniser=tokeniser,
)
repository = open_metadata_repository(
    settings.storage.metadata_db_path,
    source_dir=settings.storage.source_dir,
    apply_migrations=True,
)
try:
    facade = LocalServingFacade(repository, settings)
    summary = facade.ingest(request)
    print(
        json.dumps(
            {
                "corpus_id": corpus_id.as_str(),
                "proposed_version": proposed_version.as_str(),
                "execution_status": summary.execution_status.value,
                "publication_status": summary.publication_status.value,
                "document_outcomes": len(summary.document_outcomes),
                "failures": [failure.category.value for failure in summary.failures],
            },
            sort_keys=True,
        )
    )
    if summary.publication_status.value != "not_published":
        result = facade.search(
            ServingSearchRequest(
                corpus_id=corpus_id,
                text=os.environ["QUERY"],
                limit=10,
                filter=ExactFilter(
                    "document-type", os.environ.get("DOCUMENT_TYPE", "text/plain")
                ),
            )
        )
        print(
            json.dumps(
                {
                    "resolved_version": result.corpus_version.as_str(),
                    "hits": [
                        {
                            "document_id": hit.document_id.as_str(),
                            "text": hit.text,
                            "source_locations": [
                                {
                                    "source_span_id": location.source_span_id.as_str(),
                                    "parsed_record_id": location.parsed_record_id.as_str(),
                                    "start_offset": location.start_offset,
                                    "end_offset": location.end_offset,
                                }
                                for location in hit.source_locations
                            ],
                        }
                        for hit in result.hits
                    ],
                },
                sort_keys=True,
            )
        )
finally:
    repository.close()
```
<!-- operator-ingest-example:end -->

Set the values explicitly. Generate a new proposed version with:

```sh
uv run python -c 'from inference_lab.provenance import CorpusVersionId; print(CorpusVersionId.generate())'
```

```sh
uv run python create-corpus.py
export CORPUS_ID="paste-corpus-id-from-output"
export SOURCE_FILE=/absolute/path/to/paper.txt
export TOKENISER_DIR=/absolute/path/to/local-fast-tokeniser
export PROPOSED_VERSION="paste-new-corpus-version-uuid"
export QUERY="reported method"
uv run python ingest-and-search.py
```

After a process restart, run `uv run python ingest-and-search.py` again with
the **same** corpus ID, proposed version, source file, tokeniser artefacts,
policy, and unset `EXPECTED_VERSION`. Exact committed replay reports zero
document outcomes and does not rebuild the index. To start a *new* update,
set `EXPECTED_VERSION` to the active `resolved_version`, generate a different
`PROPOSED_VERSION`, and run the script again. An unchanged new proposal may
return `no_change`, leaving the active version at the expected one. The first
JSON line reports ingest; the second reports filtered hits, passage text, and
source-linked locations. Operational logs go to stderr. Set
`DOCUMENT_TYPE=application/pdf` to see a valid empty filter match for a text
source. Unsupported fields such as `assignee` are rejected, not treated as
missing values.

`ServingSearchRequest` can also set `corpus_version` to an owned historical
`CorpusVersionId`; otherwise it pins the active version at snapshot open. The
API result includes source span IDs, parsed record IDs, offsets, content
origin, block coordinates, and optional page layout. Serving integrations
must keep the context-managed snapshot for the request; see the
[corpus-version runbook](corpus-versions.md). Typed ingest/search errors expose
bounded categories. A failed ingest summary has `not_published` status and
request-level failure categories; do not search it as if it were active.

### Fresh-process query example

After a corpus has been ingested, run this from the directory containing the
deployment's `config.toml`:

```sh
uv run python scripts/inference_lab/serving-search.py \
  --corpus-id 00000000-0000-4000-8000-000000000001 \
  --query "the reported method" \
  --limit 10
```

To query an exact historical version, add:

```sh
--corpus-version 00000000-0000-4000-8000-000000000002
```

The script writes a JSON result containing `corpus_id`, the resolved
`corpus_version`, and `hits`. Each hit contains `record_id`, document IDs,
rank, score, passage text, and `source_locations`, including offsets and any
available page or layout metadata. It writes only the result JSON to stdout
and sends structured operational logs to stderr. It remains a search-only
convenience script; use the Python API for filters and ingest. Invalid requests
and bounded serving failures return
exit code `2` with a category-only JSON error on stderr. Query text, paths,
passages, and raw exception details are not written to operational logs.

## Supported metadata filters

The API uses the following closed metadata registry
(`1.0.0`). Filters are evaluated against canonical metadata reconstructed from
the same pinned corpus snapshot as lexical retrieval.

| Field | Values | Operators | Missing values |
| --- | --- | --- | --- |
| `character-count` | integer | exact, one-of, range, exists, missing | invalid record state |
| `document-type` | string | exact, one-of, exists, missing | invalid record state |
| `language` | string | exact, one-of, exists, missing | allowed; `missing` matches |
| `page-count` | integer | exact, one-of, range, exists, missing | allowed; `missing` matches |
| `source-byte-count` | integer | exact, one-of, range, exists, missing | invalid record state |
| `token-count` | integer | exact, one-of, range, exists, missing | allowed; `missing` matches |

`range` is available only for integer fields and uses the bounds explicitly
specified by the caller. `exists` and `missing` refer to the canonical value,
not to text inferred from a passage. An unsupported field is rejected; it is
not converted into a `missing` match. Dates and assignees are intentionally
deferred to RET-11 and are not supported filter fields in this release.

Invalid fields, operators, scalar types, or oversized filter trees fail before
ranked execution. A valid filter that retains no candidates is a successful
empty result. Storage, index, or provenance corruption fails closed without
returning hits. Query text, filter values, metadata values, passages, paths,
SQL, and raw exception details must remain out of operational logs and
profiling records; see the [logging conventions](../conventions/logging.md).

The baseline is lexical search. It is not yet a generated answer and should
not be described as supporting citation verification, abstention, sessions, or
follow-up questions until the relevant serving issues land.

## Supported question shapes

The product is intended to support these question types as serving capability
is added:

- lookup: identify a method, assay, or other reported detail;
- comparison: contrast two approaches;
- thematic: find methods tried for a topic;
- aggregation: count or filter structured records; and
- follow-up: resolve a question using conversation state.

Each example added here must identify the serving capability that makes it
executable. Results should prefer original passages over generated summaries.

## Evidence and privacy rules

Answers must distinguish reported facts from model synthesis, cite precise
pages or source spans, and state when the corpus is insufficient. An
unobserved result is not evidence of a negative result. Citation verification,
abstention, source opening, sessions, and chat will be documented when those
capabilities are implemented.

Queries and results remain local. Do not place query text, prompts, outputs, or
source passages in operational logs; follow the
[logging conventions](../conventions/logging.md).
