# Running queries

## Current status

The local Python serving API supports corpus creation, incremental ingest, and
lexical search. The `inference-lab` entry point remains a startup skeleton; the
supported fresh-process query example is the script documented below.

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

### Create and ingest through the local API

The minimal local setup is:

```python
from inference_lab.serving import (
    LocalServingFacade,
    ServingIngestRequest,
)
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository

settings = get_settings()
repository = open_metadata_repository(
    settings.storage.metadata_db_path,
    source_dir=settings.storage.source_dir,
    apply_migrations=True,
)
try:
    facade = LocalServingFacade(repository, settings)
    corpus = facade.create_corpus("My local corpus")
    # Build a ServingIngestRequest with this corpus.corpus_id, a new
    # proposed CorpusVersionId, the source discovery inputs, and the selected
    # chunking/tokeniser configuration.
    result = facade.ingest(request)
finally:
    repository.close()
```

The complete request must include the owning `corpus_id`, a new proposed
`CorpusVersionId`, the expected active version (`None` for the first update),
the source discovery request, and the immutable parser, chunker, tokeniser,
and policy identity. Reuse the same proposed version and request after a
restart when the publication outcome is uncertain; an exact replay is a
successful idempotent operation.

### Search through the local API

Construct a `ServingSearchRequest` with the corpus ID, query text, and a
positive result limit. An optional owned `CorpusVersionId` selects an exact
historical snapshot; omitting it selects the active version at snapshot open.
The result includes the resolved corpus version, ranked passage text, and
source locations with source span IDs, parsed record IDs, offsets, content
origin, block coordinates, and optional page bounding boxes.

After a restart, reopen the same repository and create a new facade before
searching:

```python
from inference_lab.serving import LocalServingFacade
from inference_lab.provenance import CorpusVersionId
from inference_lab.serving import ServingSearchRequest
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository

settings = get_settings()
repository = open_metadata_repository(
    settings.storage.metadata_db_path,
    source_dir=settings.storage.source_dir,
    apply_migrations=True,
)
try:
    facade = LocalServingFacade(repository, settings)
    selected_corpus = facade.list_corpora()[0]

    active_result = facade.search(
        ServingSearchRequest(
            corpus_id=selected_corpus.corpus_id,
            text="the reported method",
            limit=10,
        )
    )

    historical_result = facade.search(
        ServingSearchRequest(
            corpus_id=selected_corpus.corpus_id,
            corpus_version=CorpusVersionId.from_str("<owned-version-id>"),
            text="the reported method",
            limit=10,
        )
    )
finally:
    repository.close()
```

Serving integrations must hold a context-managed corpus snapshot for the
request so every metadata and index read uses one immutable manifest. See the
[corpus-version runbook](corpus-versions.md) for snapshot lifetime and
historical selection.

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
available page or layout metadata. The configured structured operational logs
also use stdout, so this example is not a stable machine-readable CLI
serialization contract. Invalid requests and bounded serving failures return
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
deferred and are not supported filter fields in this release.

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
