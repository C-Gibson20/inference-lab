#!/usr/bin/env python3
"""Prepare a version-pinned local corpus from an explicit source directory."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from inference_lab._validation import paths_overlap
from inference_lab.backends.runtime import (
    EmbeddingBackend,
    EmbeddingProfile,
    QwenEmbeddingBackend,
)
from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    BaselineIngestSummary,
    ChunkingPolicy,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    LocalFastChunkTokeniser,
    derive_chunk_configuration_identity,
)
from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import DenseIndexCompatibility
from inference_lab.serving import LocalServingFacade, ServingIngestRequest
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository


class DenseEmbeddingBackend(EmbeddingBackend, Protocol):
    """An embedding backend with a profile suitable for indexed vectors."""

    @property
    def output_profile(self) -> EmbeddingProfile:
        """Return the fixed profile attached to emitted vectors."""
        ...


def _required_path(name: str) -> Path:
    """Read one required local path without exposing it in validation errors."""
    value = os.environ.get(name)
    if not value:
        raise ValueError
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError
    return path


def _include_dense_index() -> bool:
    """Read the explicit opt-in for local dense index publication."""
    value = os.environ.get("EVALUATION_CORPUS_INCLUDE_DENSE", "false").lower()
    if value not in {"true", "false"}:
        raise ValueError
    return value == "true"


def _readiness_failure_categories(
    summary: BaselineIngestSummary,
    *,
    published_document_versions: set[tuple[str, str]],
    published_component_kinds: set[str],
    include_dense: bool,
) -> tuple[str, ...]:
    """Require complete document outcomes and matching published state."""
    failures: list[str] = []
    if summary.execution_status.value != "completed":
        failures.append("ingest_incomplete")
    if summary.publication_status.value not in {"committed", "no_change"}:
        failures.append("publication_incomplete")
    outcomes = summary.document_outcomes
    if not outcomes:
        failures.append("no_documents_discovered")
    if summary.failures or any(
        outcome.kind.value in {"ambiguous", "conflicting", "failed", "unsupported"}
        or outcome.failures
        or outcome.document_id is None
        or outcome.document_version_id is None
        for outcome in outcomes
    ):
        failures.append("document_outcomes_incomplete")
    expected_document_versions = {
        (outcome.document_id.as_str(), outcome.document_version_id.as_str())
        for outcome in outcomes
        if outcome.document_id is not None and outcome.document_version_id is not None
    }
    if (
        not expected_document_versions
        or expected_document_versions != published_document_versions
    ):
        failures.append("snapshot_membership_incomplete")
    required_components = {"chunks", "lexical-index"}
    if include_dense:
        required_components.add("dense-index")
    if not required_components.issubset(published_component_kinds):
        failures.append("required_components_missing")
    return tuple(failures)


def _run(
    *,
    embedding_backend_factory: Callable[
        [Path], DenseEmbeddingBackend
    ] = QwenEmbeddingBackend,
) -> dict[str, object]:
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
    dense_embedder = None
    dense_compatibility = None
    if _include_dense_index():
        dense_embedder = embedding_backend_factory(settings.models.embedder_path)
        try:
            dense_embedder.load()
            dimensions = dense_embedder.metadata.embedding_dimensions
            if dimensions is None:
                raise ValueError
            dense_compatibility = DenseIndexCompatibility(
                model_metadata=dense_embedder.metadata,
                dimensions=dimensions,
                embedding_profile=dense_embedder.output_profile,
            )
        except Exception:
            dense_embedder.close()
            raise
    policy = ChunkingPolicy()
    try:
        repository = open_metadata_repository(
            settings.storage.metadata_db_path,
            source_dir=settings.storage.source_dir,
            apply_migrations=True,
        )
    except Exception:
        if dense_embedder is not None:
            dense_embedder.close()
        raise
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
            dense_embedder=dense_embedder,
            dense_compatibility=dense_compatibility,
        )
        summary = facade.ingest(request)
        component_generation_ids: list[dict[str, str]] = []
        published_document_versions: set[tuple[str, str]] = set()
        published_component_kinds: set[str] = set()
        published_document_count = 0
        if summary.publication_status.value in {"committed", "no_change"}:
            with repository.open_corpus_snapshot(
                corpus_id, proposed_version
            ) as snapshot:
                memberships = snapshot.manifest.document_memberships
                published_document_count = len(memberships)
                published_document_versions = {
                    (
                        membership.document_id.as_str(),
                        membership.document_version_id.as_str(),
                    )
                    for membership in memberships
                }
                published_component_kinds = {
                    reference.component_kind.as_str()
                    for reference in snapshot.manifest.component_generations
                }
                component_generation_ids = [
                    {
                        "component_kind": reference.component_kind.as_str(),
                        "generation_id": reference.generation_id,
                    }
                    for reference in snapshot.manifest.component_generations
                ]
        readiness_failure_categories = _readiness_failure_categories(
            summary,
            published_document_versions=published_document_versions,
            published_component_kinds=published_component_kinds,
            include_dense=dense_embedder is not None,
        )
        return {
            "corpus_id": corpus_id.as_str(),
            "proposed_version": proposed_version.as_str(),
            "execution_status": summary.execution_status.value,
            "publication_status": summary.publication_status.value,
            "document_outcome_count": len(summary.document_outcomes),
            "published_document_count": published_document_count,
            "readiness_status": (
                "ready" if not readiness_failure_categories else "incomplete"
            ),
            "readiness_failure_categories": list(readiness_failure_categories),
            "failure_categories": sorted(
                {failure.category.value for failure in summary.failures}
                | {
                    failure.category.value
                    for outcome in summary.document_outcomes
                    for failure in outcome.failures
                }
            ),
            "component_generation_ids": component_generation_ids,
        }
    finally:
        if dense_embedder is not None:
            dense_embedder.close()
        repository.close()


def main(_argv: Sequence[str] | None = None) -> int:
    """Run the deployment-local preparation and print bounded JSON only."""
    try:
        result = _run()
    except Exception:
        print("local evaluation corpus preparation failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["readiness_status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
