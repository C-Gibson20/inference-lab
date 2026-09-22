"""Query behavior across published lexical-generation lifecycle failures."""

import shutil
from dataclasses import replace
from pathlib import Path

from inference_lab.ingest import build_and_publish_baseline_lexical_generation
from inference_lab.retrieval import (
    LexicalQuery,
    RetrievalLimit,
    RetrievalQuery,
    execute_lexical_retrieval,
    open_lexical_generation,
    serialise_lexical_manifest,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.retrieval import lexical_fixtures  # noqa: TID252
from tests.inference_lab.retrieval.lexical_fixtures import (  # noqa: TID252
    CORPUS_ID,
    publish_query_fixture,
    published_lexical_generation_id,
    seed_metadata_repository,
)


def test_pr4_query_returns_empty_completed_result_for_no_matches(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    try:
        seed_metadata_repository(repository, source_manifest)
        build_and_publish_baseline_lexical_generation(
            repository,
            tmp_path / "indexes",
            source_manifest,
            expected_corpus_version=None,
        )
        result = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "unmatchedterm",
                    source_manifest.corpus_version,
                    RetrievalLimit(3),
                )
            ),
        )

        assert result.status.value == "completed"
        assert result.hits == ()
        assert result.failures == ()
    finally:
        repository.close()


def test_pr4_query_reopens_published_generation_after_repository_restart(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    database_path = tmp_path / "metadata.sqlite3"
    index_dir = tmp_path / "indexes"
    repository = SQLiteMetadataRepository.open(database_path, apply_migrations=True)
    publish_query_fixture(repository, index_dir, source_manifest)
    repository.close()

    reopened = SQLiteMetadataRepository.open(database_path, apply_migrations=True)
    try:
        result = execute_lexical_retrieval(
            reopened,
            index_dir,
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "alpha",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )

        assert result.status.value == "completed"
        assert len(result.hits) == 1
        assert result.hits[0].candidate.content.startswith("alpha cites")
    finally:
        reopened.close()


def test_pr4_query_fails_closed_when_referenced_generation_is_missing(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    try:
        outcome = publish_query_fixture(
            repository, tmp_path / "indexes", source_manifest
        )
        generation_dir = (
            tmp_path / "indexes" / "lexical" / published_lexical_generation_id(outcome)
        )
        shutil.rmtree(generation_dir)

        result = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "alpha",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )

        assert result.status.value == "failed"
        assert result.hits == ()
        assert result.failures[0].category.value == "channel_unavailable"
    finally:
        repository.close()


def test_pr4_query_fails_closed_when_referenced_generation_is_corrupt(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    try:
        outcome = publish_query_fixture(
            repository, tmp_path / "indexes", source_manifest
        )
        database = (
            tmp_path
            / "indexes"
            / "lexical"
            / published_lexical_generation_id(outcome)
            / "index.sqlite3"
        )
        with database.open("ab") as stream:
            stream.write(b"corruption")

        result = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "alpha",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )

        assert result.status.value == "failed"
        assert result.hits == ()
        assert result.failures[0].category.value == "channel_unavailable"
        assert "corruption" not in repr(result)
        assert str(tmp_path) not in repr(result)
    finally:
        repository.close()


def test_pr4_query_rejects_generation_with_incompatible_reference_metadata(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = lexical_fixtures.source_manifest()
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
    )
    try:
        outcome = publish_query_fixture(
            repository, tmp_path / "indexes", source_manifest
        )
        manifest_path = (
            tmp_path
            / "indexes"
            / "lexical"
            / published_lexical_generation_id(outcome)
            / "manifest.json"
        )
        with open_lexical_generation(
            tmp_path / "indexes",
            published_lexical_generation_id(outcome),
        ) as generation:
            manifest = replace(generation.manifest, format_version="2.0.0")
        manifest_path.write_bytes(serialise_lexical_manifest(manifest))

        result = execute_lexical_retrieval(
            repository,
            tmp_path / "indexes",
            CORPUS_ID,
            LexicalQuery(
                RetrievalQuery(
                    "alpha",
                    source_manifest.corpus_version,
                    RetrievalLimit(5),
                )
            ),
        )

        assert result.status.value == "failed"
        assert result.hits == ()
        assert result.failures[0].category.value == "invalid_channel_response"
    finally:
        repository.close()
