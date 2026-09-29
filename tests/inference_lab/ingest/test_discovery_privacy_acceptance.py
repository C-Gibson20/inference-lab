"""Cross-boundary privacy acceptance for local source discovery logs."""

from hashlib import sha256
from pathlib import Path

from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    DiscoveryStatus,
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
    unsupported_source = private_directory / "private-unsupported-marker.docx"
    unsupported_source.write_bytes(b"private-docx-marker")
    unknown_source = private_directory / "private-unknown-marker.bin"
    unknown_source.write_bytes(b"private-binary-marker")
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
            DiscoveryInput(unsupported_source, DiscoveryInputKind.FILE),
            DiscoveryInput(unknown_source, DiscoveryInputKind.FILE),
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

    assert result.status is DiscoveryStatus.PARTIAL
    assert len(result.sources) == 1
    assert tuple(failure.ordinal for failure in result.failures) == (1, 2)
    assert result.counts.files_seen == 3
    assert result.counts.supported == 1
    assert result.counts.unsupported == 2
    assert result.counts.hashed == 1
    assert result.counts.new == 1
    assert result.counts.failed == 0
    assert isinstance(identity_results[0], IdentityApplicationResult)
    events = [
        record["event"]
        for record in logs
        if record["event"] not in {"stage_started", "stage_completed"}
    ]
    assert "source_discovered" in events
    assert "source_hashed" in events
    assert "identity_resolution_completed" in events
    assert "source_identity_assigned" in events
    assert "transaction_completed" in events
    assert "identity_persistence_completed" in events

    payload = repr(logs)
    private_markers = (
        private_directory.name,
        private_source.name,
        str(private_source),
        private_bytes.decode().strip(),
        sha256(private_bytes).hexdigest(),
        unsupported_source.name,
        str(unsupported_source),
        b"private-docx-marker".decode(),
        unknown_source.name,
        str(unknown_source),
        b"private-binary-marker".decode(),
        private_identifier.value,
        private_corpus_name,
        private_database.name,
        str(private_database),
    )
    assert all(marker not in payload for marker in private_markers)
