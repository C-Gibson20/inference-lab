"""SQLite acceptance tests for baseline generation publication."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_lab.ingest import (
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    begin_baseline_corpus_update,
    publish_baseline_chunk_generation,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    OutputDigest,
)
from inference_lab.storage import (
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
    CommitCorpusUpdateOutcome,
    ComponentGenerationState,
    NoCorpusChangeOutcome,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

_CORPUS_ID = CorpusId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_ID = DocumentId.from_str("22222222-2222-4222-8222-222222222222")
_DOCUMENT_VERSION_ID = DocumentVersionId.from_str(
    "33333333-3333-4333-8333-333333333333"
)
_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "a" * 64)
_CREATED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def _open(path: Path) -> SQLiteMetadataRepository:
    """Open a deterministic test repository."""
    return SQLiteMetadataRepository.open(
        path,
        apply_migrations=True,
        created_at_factory=lambda: _CREATED_AT,
    )


def _selection(
    *,
    document_id: DocumentId = _DOCUMENT_ID,
    document_version_id: DocumentVersionId = _DOCUMENT_VERSION_ID,
    payload: bytes = b"canonical-chunk-set",
) -> ChunkSetSelectionRecord:
    """Build a storage-valid exact chunk selection."""
    return ChunkSetSelectionRecord(
        document_id=document_id,
        document_version_id=document_version_id,
        parser_version=ChunkParserVersionRecord(
            implementation="text-native",
            contract_version="3.0.0",
            normalisation_version="1.0.0",
        ),
        chunker_version=ChunkerVersionRecord(
            implementation="structure-aware-greedy",
            contract_version="1.0.1",
        ),
        tokeniser_identity=ChunkTokeniserIdentityRecord(
            implementation="deterministic-tokeniser",
            contract_version="1.0.0",
            artefact_identity="fixture-tokeniser",
            artefact_revision="2026-01",
        ),
        policy=ChunkingPolicyRecord(
            min_tokens=1,
            max_tokens=10,
            overlap_numerator=0,
            overlap_denominator=1,
            structural_boundary_policy=ChunkStructuralBoundaryPolicy.GREEDY,
            oversize_unit_policy=ChunkOversizeUnitPolicy.TOKEN_WINDOW,
        ),
        configuration_identity=_CONFIGURATION,
        payload_digest=OutputDigest(
            DigestAlgorithm.SHA256,
            hashlib.sha256(payload).hexdigest(),
        ),
        canonical_payload=payload,
    )


def _membership(
    *,
    document_version_id: DocumentVersionId = _DOCUMENT_VERSION_ID,
) -> CorpusDocumentVersionMembership:
    """Build the single-corpus membership used by the fixture."""
    return CorpusDocumentVersionMembership(
        corpus_id=_CORPUS_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=document_version_id,
    )


def _manifest(
    corpus_version: CorpusVersionId,
    membership: CorpusDocumentVersionMembership,
    selection: ChunkSetSelectionRecord,
    *,
    generation_id: str,
) -> BaselineChunkGenerationManifest:
    """Build a complete one-entry chunks manifest."""
    return BaselineChunkGenerationManifest(
        corpus_id=_CORPUS_ID,
        corpus_version=corpus_version,
        configuration_identity=_CONFIGURATION,
        generation_id=generation_id,
        entries=(
            BaselineChunkGenerationEntry(
                membership=membership,
                selection=selection,
            ),
        ),
    )


def _seed_catalogue(
    repository: SQLiteMetadataRepository,
) -> ChunkSetSelectionRecord:
    """Create one corpus-visible document and its immutable chunk set."""
    membership = _membership()
    selection = _selection()
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(
            DocumentVersion(
                _DOCUMENT_VERSION_ID,
                _DOCUMENT_ID,
                ContentHash(HashAlgorithm.SHA256, "b" * 64),
            )
        )
        transaction.add_corpus_document_version_membership(membership)
        transaction.write_chunk_set(
            ChunkSetWriteRequest(corpus_id=_CORPUS_ID, selection=selection)
        )
    return selection


def _publish_first(
    repository: SQLiteMetadataRepository,
    selection: ChunkSetSelectionRecord,
) -> tuple[CorpusVersionId, CorpusDocumentVersionMembership]:
    """Publish the initial version through the ingest publication API."""
    corpus_version = CorpusVersionId.generate()
    membership = _membership()
    manifest = _manifest(
        corpus_version,
        membership,
        selection,
        generation_id="generation-first",
    )
    with repository.transaction() as transaction:
        begin_baseline_corpus_update(
            transaction,
            _CORPUS_ID,
            corpus_version,
            expected_corpus_version=None,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_ID,
            corpus_version,
            membership,
        )
        result = publish_baseline_chunk_generation(transaction, repository, manifest)
        assert isinstance(result, CommitCorpusUpdateOutcome)
        assert (
            result.manifest.component_generations[0].state
            is ComponentGenerationState.READY
        )
    return corpus_version, membership


def test_sqlite_publication_commits_and_reopens_exact_generation(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        selection = _seed_catalogue(repository)
        corpus_version, membership = _publish_first(repository, selection)

        with repository.open_active_corpus_snapshot(_CORPUS_ID) as snapshot:
            assert snapshot.version.corpus_version == corpus_version
            assert snapshot.manifest.document_memberships == (membership,)
            assert (
                snapshot.manifest.component_generations[0].component_kind.as_str()
                == "chunks"
            )
    finally:
        repository.close()

    reopened = _open(tmp_path / "metadata.sqlite3")
    try:
        active = reopened.get_active_corpus_version(_CORPUS_ID)
        assert active is not None
        assert active.corpus_version == corpus_version
    finally:
        reopened.close()


def test_sqlite_open_snapshot_remains_pinned_across_next_publication(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        selection = _seed_catalogue(repository)
        first_version, membership = _publish_first(repository, selection)
        changed_document_version = DocumentVersionId.generate()
        changed_membership = _membership(
            document_version_id=changed_document_version,
        )
        changed_selection = _selection(
            document_version_id=changed_document_version,
            payload=b"next-canonical-chunk-set",
        )
        with repository.transaction() as transaction:
            transaction.add_document_version(
                DocumentVersion(
                    changed_document_version,
                    _DOCUMENT_ID,
                    ContentHash(HashAlgorithm.SHA256, "c" * 64),
                )
            )
            transaction.add_corpus_document_version_membership(changed_membership)
            transaction.write_chunk_set(
                ChunkSetWriteRequest(
                    corpus_id=_CORPUS_ID,
                    selection=changed_selection,
                )
            )

        next_version = CorpusVersionId.generate()
        next_manifest = _manifest(
            next_version,
            changed_membership,
            changed_selection,
            generation_id="generation-next",
        )
        with repository.open_active_corpus_snapshot(_CORPUS_ID) as snapshot:
            assert snapshot.version.corpus_version == first_version
            with repository.transaction() as transaction:
                begin_baseline_corpus_update(
                    transaction,
                    _CORPUS_ID,
                    next_version,
                    expected_corpus_version=first_version,
                )
                transaction.replace_corpus_update_document_membership(
                    _CORPUS_ID,
                    next_version,
                    membership,
                    changed_membership,
                )
                result = publish_baseline_chunk_generation(
                    transaction,
                    repository,
                    next_manifest,
                )
                assert isinstance(result, CommitCorpusUpdateOutcome)

            assert snapshot.version.corpus_version == first_version
            assert snapshot.manifest.document_memberships == (membership,)

        active = repository.get_active_corpus_version(_CORPUS_ID)
        assert active is not None
        assert active.corpus_version == next_version
    finally:
        repository.close()


def test_sqlite_identical_reingest_returns_no_change_without_new_sequence(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        selection = _seed_catalogue(repository)
        first_version, membership = _publish_first(repository, selection)
        second_version = CorpusVersionId.generate()
        manifest = _manifest(
            second_version,
            membership,
            selection,
            generation_id="generation-replay",
        )

        result: NoCorpusChangeOutcome | CommitCorpusUpdateOutcome | None = None
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                second_version,
                expected_corpus_version=first_version,
            )
            result = publish_baseline_chunk_generation(
                transaction, repository, manifest
            )

        assert isinstance(result, NoCorpusChangeOutcome)
        assert result is not None
        active = repository.get_active_corpus_version(_CORPUS_ID)
        assert active is not None
        assert active.corpus_version == first_version
        assert len(repository.list_corpus_versions(_CORPUS_ID)) == 1
    finally:
        repository.close()


def test_sqlite_stale_parent_conflicts_without_rebase(tmp_path: Path) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    try:
        selection = _seed_catalogue(repository)
        first_version, membership = _publish_first(repository, selection)
        changed_document_version = DocumentVersionId.generate()
        changed_membership = _membership(
            document_version_id=changed_document_version,
        )
        changed_selection = _selection(
            document_version_id=changed_document_version,
            payload=b"changed-canonical-chunk-set",
        )
        with repository.transaction() as transaction:
            transaction.add_document_version(
                DocumentVersion(
                    changed_document_version,
                    _DOCUMENT_ID,
                    ContentHash(HashAlgorithm.SHA256, "c" * 64),
                )
            )
            transaction.add_corpus_document_version_membership(changed_membership)
            transaction.write_chunk_set(
                ChunkSetWriteRequest(
                    corpus_id=_CORPUS_ID,
                    selection=changed_selection,
                )
            )
        stale_version = CorpusVersionId.generate()
        winning_version = CorpusVersionId.generate()
        stale_manifest = _manifest(
            stale_version,
            changed_membership,
            changed_selection,
            generation_id="generation-stale",
        )
        winning_manifest = _manifest(
            winning_version,
            changed_membership,
            changed_selection,
            generation_id="generation-winning",
        )
        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                stale_version,
                expected_corpus_version=first_version,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_ID,
                stale_version,
                membership,
                changed_membership,
            )

        with repository.transaction() as transaction:
            begin_baseline_corpus_update(
                transaction,
                _CORPUS_ID,
                winning_version,
                expected_corpus_version=first_version,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_ID,
                winning_version,
                membership,
                changed_membership,
            )
            result = publish_baseline_chunk_generation(
                transaction,
                repository,
                winning_manifest,
            )
            assert isinstance(result, CommitCorpusUpdateOutcome)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                publish_baseline_chunk_generation(
                    transaction,
                    repository,
                    stale_manifest,
                )
    finally:
        repository.close()
