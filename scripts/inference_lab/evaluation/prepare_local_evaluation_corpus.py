#!/usr/bin/env python3
"""Prepare a version-pinned local corpus from an explicit source directory."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from inference_lab._validation import paths_overlap
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
from inference_lab.serving import LocalServingFacade, ServingIngestRequest
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository


def _required_path(name: str) -> Path:
    """Read one required local path without exposing it in validation errors."""
    value = os.environ.get(name)
    if not value:
        raise ValueError
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError
    return path


def _run() -> dict[str, object]:
    """Ingest the explicit source directory as one immutable corpus version."""
    settings = get_settings()
    configure_logging(settings.logging, stream=sys.stderr)

    source_directory = _required_path("EVALUATION_CORPUS_SOURCE_DIRECTORY")
    tokeniser_directory = _required_path("EVALUATION_CORPUS_TOKENISER_DIRECTORY")
    corpus_id_text = os.environ.get("EVALUATION_CORPUS_ID")
    proposed_version_text = os.environ.get("EVALUATION_CORPUS_PROPOSED_VERSION")
    if not corpus_id_text or not proposed_version_text:
        raise ValueError
    corpus_id = CorpusId.from_str(corpus_id_text)
    proposed_version = CorpusVersionId.from_str(proposed_version_text)
    expected_version_text = os.environ.get("EVALUATION_CORPUS_EXPECTED_VERSION")
    expected_version = (
        None
        if not expected_version_text
        else CorpusVersionId.from_str(expected_version_text)
    )

    if not source_directory.is_dir() or not tokeniser_directory.is_dir():
        raise ValueError
    input_root = source_directory.absolute()
    storage_roots = (
        settings.storage.index_dir.expanduser().absolute(),
        settings.storage.metadata_db_path.expanduser().absolute(),
        settings.storage.source_dir.expanduser().absolute(),
    )
    if any(paths_overlap(input_root, root) for root in storage_roots):
        raise ValueError

    tokeniser = LocalFastChunkTokeniser.from_directory(tokeniser_directory)
    policy = ChunkingPolicy()
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        if repository.get_corpus(corpus_id) is None:
            facade.create_corpus("SARM1 evaluation corpus", corpus_id=corpus_id)

        request = ServingIngestRequest(
            corpus_id=corpus_id,
            discovery_request=DiscoveryRequest(
                corpus_id=corpus_id,
                inputs=(
                    DiscoveryInput(
                        source_directory,
                        DiscoveryInputKind.DIRECTORY,
                    ),
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
        summary = facade.ingest(request)
        return {
            "corpus_id": corpus_id.as_str(),
            "proposed_version": proposed_version.as_str(),
            "execution_status": summary.execution_status.value,
            "publication_status": summary.publication_status.value,
            "document_outcome_count": len(summary.document_outcomes),
            "failure_categories": sorted(
                {failure.category.value for failure in summary.failures}
                | {
                    failure.category.value
                    for outcome in summary.document_outcomes
                    for failure in outcome.failures
                }
            ),
        }
    finally:
        repository.close()


def main(_argv: Sequence[str] | None = None) -> int:
    """Run the deployment-local preparation and print bounded JSON only."""
    try:
        result = _run()
    except Exception:
        print("local evaluation corpus preparation failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
