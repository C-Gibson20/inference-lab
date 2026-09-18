"""Tests for immutable baseline chunk-generation manifests."""

import hashlib
from dataclasses import replace
from typing import cast

import pytest

from inference_lab.ingest import (
    BASELINE_CHUNK_COMPONENT_KIND,
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    BaselineChunkGenerationValidator,
    BaselineFailureCategory,
    BaselineGenerationValidationError,
    begin_baseline_corpus_update,
    publish_baseline_chunk_generation,
    publish_baseline_generations,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
)
from inference_lab.storage import (
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSetReader,
    ChunkSetSelectionRecord,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
    ComponentGenerationReference,
    ComponentGenerationState,
    MetadataTransaction,
    StorageConflictError,
)


def _selection(
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    configuration_identity: ConfigurationIdentity,
    payload: bytes = b"chunk-set-payload",
) -> ChunkSetSelectionRecord:
    """Build a valid exact chunk-selection fixture."""
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
        configuration_identity=configuration_identity,
        payload_digest=OutputDigest(
            algorithm=DigestAlgorithm.SHA256,
            digest=hashlib.sha256(payload).hexdigest(),
        ),
        canonical_payload=payload,
    )


def _manifest(
    *,
    configuration_identity: ConfigurationIdentity | None = None,
) -> tuple[
    BaselineChunkGenerationManifest,
    tuple[CorpusDocumentVersionMembership, ...],
]:
    """Build a two-document manifest and its canonical membership tuple."""
    corpus_id = CorpusId.generate()
    corpus_version = CorpusVersionId.generate()
    configuration = configuration_identity or ConfigurationIdentity(
        algorithm=DigestAlgorithm.SHA256,
        digest="a" * 64,
    )
    first = CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
    )
    second = CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=DocumentId.generate(),
        document_version_id=DocumentVersionId.generate(),
    )
    memberships = tuple(
        sorted((first, second), key=lambda item: item.document_id.as_str())
    )
    manifest = BaselineChunkGenerationManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        configuration_identity=configuration,
        generation_id="generation-1",
        entries=tuple(
            BaselineChunkGenerationEntry(
                membership=membership,
                selection=_selection(
                    document_id=membership.document_id,
                    document_version_id=membership.document_version_id,
                    configuration_identity=configuration,
                ),
            )
            for membership in memberships
        ),
    )
    return manifest, memberships


def test_manifest_maps_every_membership_and_builds_generation_reference() -> None:
    manifest, memberships = _manifest()

    manifest.validate_memberships(memberships)
    reference = manifest.component_generation()

    assert reference.component_kind == BASELINE_CHUNK_COMPONENT_KIND
    assert reference.state is ComponentGenerationState.PREPARING
    assert reference.configuration_digest == "a" * 64
    assert reference.source_manifest_digest == manifest.source_manifest_digest


def test_manifest_rejects_cross_configuration_selection() -> None:
    manifest, memberships = _manifest()
    wrong_configuration = ConfigurationIdentity(
        algorithm=DigestAlgorithm.SHA256,
        digest="b" * 64,
    )
    entry = BaselineChunkGenerationEntry(
        membership=memberships[0],
        selection=_selection(
            document_id=memberships[0].document_id,
            document_version_id=memberships[0].document_version_id,
            configuration_identity=wrong_configuration,
        ),
    )

    with pytest.raises(BaselineGenerationValidationError) as error:
        BaselineChunkGenerationManifest(
            corpus_id=manifest.corpus_id,
            corpus_version=manifest.corpus_version,
            configuration_identity=manifest.configuration_identity,
            generation_id="generation-2",
            entries=(entry,),
        )

    assert error.value.category is BaselineFailureCategory.GENERATION_VALIDATION_FAILED


def test_manifest_rejects_incomplete_membership_mapping() -> None:
    manifest, memberships = _manifest()

    with pytest.raises(BaselineGenerationValidationError) as error:
        manifest.validate_memberships(memberships[:1])

    assert error.value.category is BaselineFailureCategory.GENERATION_VALIDATION_FAILED


class _FakeChunkReader:
    """Return exact corpus-scoped selections for validator tests."""

    def __init__(
        self, selections: dict[DocumentVersionId, ChunkSetSelectionRecord]
    ) -> None:
        self.selections = selections

    def get_chunk_set_selection(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        del corpus_id, configuration_identity
        return self.selections.get(document_version_id)


def test_validator_accepts_exact_durable_selections() -> None:
    manifest, memberships = _manifest()
    reader = _FakeChunkReader(
        {
            entry.membership.document_version_id: entry.selection
            for entry in manifest.entries
        }
    )
    validator = BaselineChunkGenerationValidator(reader, manifest)

    assert validator.validate_component_generation(
        manifest.component_generation(),
        memberships,
    )


def test_validator_rejects_missing_selection() -> None:
    manifest, memberships = _manifest()
    reader = _FakeChunkReader({})
    validator = BaselineChunkGenerationValidator(reader, manifest)

    assert not validator.validate_component_generation(
        manifest.component_generation(),
        memberships,
    )


def test_validator_rejects_changed_selection_payload() -> None:
    manifest, memberships = _manifest()
    first = manifest.entries[0]
    reader = _FakeChunkReader(
        {
            first.membership.document_version_id: _selection(
                document_id=first.membership.document_id,
                document_version_id=first.membership.document_version_id,
                configuration_identity=manifest.configuration_identity,
                payload=b"different-payload",
            ),
            **{
                entry.membership.document_version_id: entry.selection
                for entry in manifest.entries[1:]
            },
        }
    )
    validator = BaselineChunkGenerationValidator(reader, manifest)

    assert not validator.validate_component_generation(
        manifest.component_generation(),
        memberships,
    )


def test_validator_rejects_generation_reference_mismatch() -> None:
    manifest, memberships = _manifest()
    reader = _FakeChunkReader(
        {
            entry.membership.document_version_id: entry.selection
            for entry in manifest.entries
        }
    )
    validator = BaselineChunkGenerationValidator(reader, manifest)
    reference = manifest.component_generation()
    mismatched = ComponentGenerationReference(
        component_kind=reference.component_kind,
        generation_id="different-generation",
        implementation=reference.implementation,
        format_version=reference.format_version,
        contract_version=reference.contract_version,
        configuration_digest=reference.configuration_digest,
        source_manifest_digest=reference.source_manifest_digest,
        corpus_id=reference.corpus_id,
        corpus_version=reference.corpus_version,
        state=ComponentGenerationState.PREPARING,
    )

    assert not validator.validate_component_generation(mismatched, memberships)


class _FakePublicationTransaction:
    """Capture lifecycle calls while exposing exact selections to the validator."""

    def __init__(self, manifest: BaselineChunkGenerationManifest) -> None:
        self.manifest = manifest
        self.calls: list[str] = []
        self.selections = {
            entry.membership.document_version_id: entry.selection
            for entry in manifest.entries
        }
        self.fail_registration = False
        self.commit_result = object()

    def get_chunk_set_selection(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        del corpus_id, configuration_identity
        return self.selections.get(document_version_id)

    def register_corpus_update_component_generation(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        generation: ComponentGenerationReference,
        *,
        validator: BaselineChunkGenerationValidator,
    ) -> ComponentGenerationReference:
        del corpus_id, corpus_version
        self.calls.append("register")
        if self.fail_registration:
            raise StorageConflictError()
        assert validator.validate_component_generation(
            generation,
            tuple(entry.membership for entry in self.manifest.entries),
        )
        return replace(generation, state=ComponentGenerationState.READY)

    def seal_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> object:
        del corpus_id, corpus_version
        self.calls.append("seal")
        return object()

    def commit_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> object:
        del corpus_id, corpus_version
        self.calls.append("commit")
        return self.commit_result

    def begin_corpus_update(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        *,
        expected_corpus_version: CorpusVersionId | None,
        required_component_kinds: tuple[object, ...],
    ) -> object:
        del corpus_id, corpus_version, expected_corpus_version
        self.calls.append("begin")
        assert tuple(required_component_kinds) == (BASELINE_CHUNK_COMPONENT_KIND,)
        return object()


def test_publication_registers_seals_and_commits_in_order() -> None:
    manifest, _ = _manifest()
    transaction = _FakePublicationTransaction(manifest)

    result = publish_baseline_chunk_generation(
        cast(MetadataTransaction, transaction),
        cast(ChunkSetReader, transaction),
        manifest,
    )

    assert result is transaction.commit_result
    assert transaction.calls == ["register", "seal", "commit"]


def test_generic_publication_seam_accepts_chunk_manifest_participant() -> None:
    manifest, _ = _manifest()
    transaction = _FakePublicationTransaction(manifest)

    result = publish_baseline_generations(
        cast(MetadataTransaction, transaction),
        cast(ChunkSetReader, transaction),
        (manifest,),
    )

    assert result is transaction.commit_result
    assert transaction.calls == ["register", "seal", "commit"]


def test_publication_preserves_storage_conflict_category() -> None:
    manifest, _ = _manifest()
    transaction = _FakePublicationTransaction(manifest)
    transaction.fail_registration = True

    with pytest.raises(StorageConflictError):
        publish_baseline_chunk_generation(
            cast(MetadataTransaction, transaction),
            cast(ChunkSetReader, transaction),
            manifest,
        )

    assert transaction.calls == ["register"]


def test_begin_requires_baseline_chunks_component() -> None:
    manifest, _ = _manifest()
    transaction = _FakePublicationTransaction(manifest)

    result = begin_baseline_corpus_update(
        cast(MetadataTransaction, transaction),
        manifest.corpus_id,
        manifest.corpus_version,
        expected_corpus_version=None,
    )

    assert result is not None
    assert transaction.calls == ["begin"]
