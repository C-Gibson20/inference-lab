"""Query both persistent index channels in a fresh interpreter.

Run as a subprocess by the persistent-index acceptance suite with the metadata
database, managed source directory, and index directory as arguments. Exits
non-zero if either channel fails to serve a committed corpus version from
durable state alone.
"""

import sys
from pathlib import Path

from inference_lab.retrieval import (
    RetrievalLimit,
    RetrievalQuery,
    execute_dense_retrieval,
    execute_lexical_retrieval,
)
from inference_lab.retrieval.lexical import LexicalQuery
from inference_lab.storage import open_metadata_repository
from tests.inference_lab.storage.persistent_index_fixtures import (
    CORPUS_A,
    CORPUS_B,
    DOC_VERSION_A,
    DOC_VERSION_B,
    DOC_VERSION_SHARED,
    VERSION_A,
    VERSION_B,
    EmbeddingFixture,
)

# One expectation per corpus-scoped term: the exact chunk text and the member
# version that must own it after a cold start.
EXPECTATIONS = (
    (CORPUS_A, VERSION_A, "orchid", "orchid receptor evidence", DOC_VERSION_A),
    (CORPUS_A, VERSION_A, "common", "common assay terminology", DOC_VERSION_SHARED),
    (CORPUS_B, VERSION_B, "tulip", "tulip catalyst evidence", DOC_VERSION_B),
    (CORPUS_B, VERSION_B, "common", "common assay terminology", DOC_VERSION_SHARED),
)


def main(argv: list[str]) -> int:
    """Serve every expectation from both channels and report the outcome."""
    database_path, source_dir, index_dir = (Path(value) for value in argv)
    repository = open_metadata_repository(database_path, source_dir=source_dir)
    try:
        for corpus_id, version, term, text, document_version_id in EXPECTATIONS:
            query = RetrievalQuery(term, version, RetrievalLimit(1))
            lexical = execute_lexical_retrieval(
                repository, index_dir, corpus_id, LexicalQuery(query)
            )
            dense = execute_dense_retrieval(
                repository, index_dir, corpus_id, query, EmbeddingFixture()
            )
            for result in (lexical, dense):
                assert result.status.value == "completed"
                assert tuple(hit.candidate.content for hit in result.hits) == (text,)
                assert (
                    result.hits[0].candidate.sources[0].document_version_id
                    == document_version_id
                )
    finally:
        repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
