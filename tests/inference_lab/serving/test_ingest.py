"""Acceptance tests for the local serving ingest facade."""

from pathlib import Path

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    ChunkingPolicy,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FakeChunkTokeniser,
)
from inference_lab.ingest._chunk_identity import _configuration_identity_from_values
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
)
from inference_lab.serving import LocalServingFacade, ServingIngestRequest
from inference_lab.settings import (
    ModelPaths,
    RuntimeConfig,
    Settings,
    StorageConfig,
)
from inference_lab.storage import MetadataRepository, open_metadata_repository


def _settings(tmp_path: Path) -> Settings:
    """Build isolated local settings for the facade acceptance test."""
    return Settings(
        models=ModelPaths(
            embedder_path=tmp_path / "models" / "embedder",
            reranker_path=tmp_path / "models" / "reranker",
            generator_path=tmp_path / "models" / "generator",
        ),
        runtime=RuntimeConfig(vram_budget_gb=24),
        storage=StorageConfig(
            index_dir=tmp_path / "index",
            metadata_db_path=tmp_path / "metadata.sqlite3",
            source_dir=tmp_path / "source-data",
        ),
    )


def _request(
    corpus_id: CorpusId,
    paths: tuple[Path, ...],
    *,
    proposed: CorpusVersionId,
    expected: CorpusVersionId | None,
) -> ServingIngestRequest:
    """Build one explicit corpus-scoped request over local files."""
    return ServingIngestRequest(
        corpus_id=corpus_id,
        discovery_request=DiscoveryRequest(
            corpus_id=corpus_id,
            inputs=tuple(
                DiscoveryInput(path, DiscoveryInputKind.FILE) for path in paths
            ),
        ),
        proposed_corpus_version=proposed,
        expected_corpus_version=expected,
        configuration_identity=_configuration_identity_from_values(
            parser_version=CURRENT_PARSER_VERSION,
            chunker_version=CURRENT_CHUNKER_VERSION,
            tokeniser_identity=FakeChunkTokeniser().identity,
            policy=ChunkingPolicy(min_tokens=1, max_tokens=32),
        ),
        chunking_policy=ChunkingPolicy(min_tokens=1, max_tokens=32),
        tokeniser=FakeChunkTokeniser(),
    )


def _membership_hashes(
    repository: MetadataRepository,
    request: ServingIngestRequest,
) -> set[str]:
    """Read content hashes through the selected corpus snapshot."""
    version = repository.get_active_corpus_version(request.corpus_id)
    assert version is not None
    hashes: set[str] = set()
    with repository.open_corpus_snapshot(
        request.corpus_id,
        version.corpus_version,
    ) as snapshot:
        for membership in snapshot.manifest.document_memberships:
            document_version = snapshot.get_document_version(
                membership.document_version_id
            )
            assert document_version is not None
            hashes.add(document_version.content_hash.as_str())
    return hashes


def test_ingest_is_corpus_scoped_and_replays_after_restart(tmp_path: Path) -> None:
    """Publish two isolated corpora and replay one proposal after reopening storage."""
    settings = _settings(tmp_path)
    shared = tmp_path / "shared.txt"
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    shared.write_text("shared source content", encoding="utf-8")
    alpha.write_text("alpha only content", encoding="utf-8")
    beta.write_text("beta only content", encoding="utf-8")

    corpus_a = CorpusId.generate()
    corpus_b = CorpusId.generate()
    request_a = _request(
        corpus_a,
        (shared, alpha),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )
    request_b = _request(
        corpus_b,
        (shared, beta),
        proposed=CorpusVersionId.generate(),
        expected=None,
    )

    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        facade = LocalServingFacade(repository, settings)
        facade.create_corpus("Corpus A", corpus_id=corpus_a)
        facade.create_corpus("Corpus B", corpus_id=corpus_b)

        first_a = facade.ingest(request_a)
        first_b = facade.ingest(request_b)

        assert first_a.publication_status.value == "committed"
        assert first_b.publication_status.value == "committed"
        assert len(repository.list_corpus_versions(corpus_a)) == 1
        assert len(repository.list_corpus_versions(corpus_b)) == 1
        hashes_a = _membership_hashes(repository, request_a)
        hashes_b = _membership_hashes(repository, request_b)
        assert len(hashes_a) == 2
        assert len(hashes_b) == 2
        assert len(hashes_a & hashes_b) == 1
        assert len(hashes_a - hashes_b) == 1
        assert len(hashes_b - hashes_a) == 1
    finally:
        repository.close()

    reopened = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=True,
    )
    try:
        replayed = LocalServingFacade(reopened, settings).ingest(request_a)
        assert replayed.publication_status.value == "committed"
        assert replayed.document_outcomes == ()
        assert len(reopened.list_corpus_versions(corpus_a)) == 1
    finally:
        reopened.close()
