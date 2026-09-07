"""Cross-boundary privacy acceptance for local source discovery logs."""

from hashlib import sha256
from pathlib import Path

from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    IdentityApplicationResult,
    IdentityPersistenceUnit,
    UuidDiscoveryIdentityIdAllocator,
    VerifiedSourceSnapshot,
    classify_source_inventory,
    hash_source_candidate,
    inventory_local_sources,
    observe_discovery,
    persist_discovered_source_identities,
)
from inference_lab.provenance import CorpusId, CorpusMetadata, ExternalIdentifier
from inference_lab.storage import open_metadata_repository

from .fixtures import fixture_settings  # noqa: TID252


def test_discovery_logs_exclude_private_data_across_every_boundary(
    tmp_path: Path,
) -> None:
    private_directory = tmp_path / "private-directory-marker"
    private_directory.mkdir()
    private_source = private_directory / "private-filename-marker.txt"
    private_bytes = b"private-content-marker\n"
    private_source.write_bytes(private_bytes)
    private_identifier = ExternalIdentifier(
        "doi",
        "10.1000/private-identifier-marker",
    )
    private_corpus_name = "private-corpus-name-marker"
    private_database = tmp_path / "private-database-name-marker.sqlite3"
    request = DiscoveryRequest(
        corpus_id=CorpusId("11111111-1111-4111-8111-111111111111"),
        inputs=(
            DiscoveryInput(
                private_source,
                DiscoveryInputKind.FILE,
                external_identifiers=(private_identifier,),
            ),
        ),
    )
    repository = open_metadata_repository(
        private_database,
        source_dir=tmp_path / "source-data",
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(request.corpus_id, private_corpus_name))

    lifecycle_clock = iter((1_000_000_000, 1_050_000_000))
    hashing_clock = iter((2_000_000_000, 2_025_000_000))
    try:
        with capture_logs() as logs:
            with observe_discovery(
                request,
                clock=lifecycle_clock.__next__,
            ) as lifecycle:
                traversal = inventory_local_sources(request, excluded_paths=())
                classification = classify_source_inventory(traversal)
                source = classification.hashable_candidates[0]
                snapshot = hash_source_candidate(
                    source,
                    max_source_bytes=fixture_settings().ingest.max_source_bytes,
                    clock=hashing_clock.__next__,
                )
                assert isinstance(snapshot, VerifiedSourceSnapshot)
                identity_results = persist_discovered_source_identities(
                    (IdentityPersistenceUnit(source, snapshot),),
                    request=request,
                    inventory=traversal,
                    repository=repository,
                    allocator=UuidDiscoveryIdentityIdAllocator(),
                )
                result = lifecycle.complete(
                    classification=classification,
                    identity_results=identity_results,
                )
    finally:
        repository.close()

    assert len(result.sources) == 1
    assert isinstance(identity_results[0], IdentityApplicationResult)
    events = [record["event"] for record in logs]
    assert events == [
        "discovery_started",
        "source_discovered",
        "source_hashed",
        "identity_resolution_completed",
        "source_identity_assigned",
        "transaction_completed",
        "identity_persistence_completed",
        "discovery_completed",
    ]

    payload = repr(logs)
    private_markers = (
        private_directory.name,
        private_source.name,
        str(private_source),
        private_bytes.decode().strip(),
        sha256(private_bytes).hexdigest(),
        private_identifier.value,
        private_corpus_name,
        private_database.name,
        str(private_database),
    )
    assert all(marker not in payload for marker in private_markers)
