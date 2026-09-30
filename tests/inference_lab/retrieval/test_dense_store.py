"""Persistent dense generation build, reuse, and publication seam tests."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
)
from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.retrieval import (
    DenseError,
    DenseFailureCategory,
    DenseGenerationParticipant,
    DenseIndexCompatibility,
    build_dense_generation,
    open_dense_generation,
    select_dense_parent_reference,
)
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ComponentGenerationReference,
    ComponentGenerationState,
    MetadataRepository,
)
from tests.inference_lab.retrieval.lexical_fixtures import (
    CORPUS_ID,
    selection,
)
from tests.inference_lab.retrieval.lexical_fixtures import (
    manifest as make_source_manifest,
)

_VERSION_ONE = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000101")
_VERSION_TWO = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000102")
_VERSION_THREE = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000103")
_DOC_A = DocumentId.from_str("00000000-0000-4000-8000-000000000111")
_DOC_B = DocumentId.from_str("00000000-0000-4000-8000-000000000112")
_DOC_C = DocumentId.from_str("00000000-0000-4000-8000-000000000113")
_DOC_A_V1 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000121")
_DOC_B_V1 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000122")
_DOC_B_V2 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000123")
_DOC_C_V1 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000124")


class _DeterministicEmbedder:
    """Small in-memory embedder fake with observable document batches."""

    def __init__(self) -> None:
        self.metadata = BackendMetadata(
            backend_name="fixture-embedder",
            model_name="fixture-model",
            model_revision="r1",
            supported_operations=(BackendOperation.EMBED,),
            capabilities=(),
            embedding_dimensions=3,
        )
        self.profile = EmbeddingProfile(
            profile_version="1.0.0",
            tokenizer_id="fixture-tokenizer",
            tokenizer_revision="r1",
            query_prefix="query: ",
            document_prefix="",
            pooling_strategy="mean",
            input_normalisation="identity",
            max_input_tokens=256,
            over_limit_policy="truncate_tail",
            l2_normalised=False,
            inference_dtype="float32",
            device_type="cpu",
        )
        self.batches: list[tuple[str, ...]] = []
        self.fail = False
        self.response_metadata = self.metadata
        self.is_ready = True
        self.is_closed = False

    def load(self) -> None:
        self.is_ready = True

    def close(self) -> None:
        self.is_ready = False
        self.is_closed = True

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.batches.append(request.input_texts)
        if self.fail:
            raise RuntimeError("private backend failure")
        vectors = tuple(self._vector(text) for text in request.input_texts)
        return EmbeddingResult(vectors, self.response_metadata, self.profile)

    @staticmethod
    def _vector(text: str) -> tuple[float, float, float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return float(digest[0] + 1), float(digest[1] + 1), float(digest[2] + 1)


class _ChunkReader:
    """Corpus-scoped fake reader for registration validation fixtures."""

    def __init__(self, source: BaselineChunkGenerationManifest) -> None:
        self._selections = {
            entry.membership.document_version_id: entry.selection
            for entry in source.entries
        }

    def get_chunk_set_selection(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        assert corpus_id == CORPUS_ID
        del configuration_identity
        return self._selections.get(document_version_id)


class _Snapshot:
    """Minimal active snapshot fixture for dense-parent selection."""

    def __init__(self, version: CorpusVersionId, reference) -> None:
        self.version = SimpleNamespace(corpus_version=version)
        self.manifest = SimpleNamespace(component_generations=(reference,))

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Repository:
    """Repository fixture returning one active snapshot."""

    def __init__(self, snapshot: _Snapshot) -> None:
        self._snapshot = snapshot

    def open_active_corpus_snapshot(self, corpus_id: CorpusId) -> _Snapshot:
        assert corpus_id == CORPUS_ID
        return self._snapshot


def _compatibility(embedder: _DeterministicEmbedder) -> DenseIndexCompatibility:
    return DenseIndexCompatibility(
        model_metadata=embedder.metadata,
        dimensions=3,
        embedding_profile=embedder.profile,
    )


def _source(
    selected: tuple[tuple[str, DocumentId, DocumentVersionId], ...],
    corpus_version: CorpusVersionId,
) -> BaselineChunkGenerationManifest:
    selections = tuple(
        selection(
            text,
            document_id=document_id,
            document_version_id=document_version_id,
        )[1]
        for text, document_id, document_version_id in selected
    )
    return make_source_manifest(selections, corpus_version=corpus_version)


def _build(
    tmp_path: Path,
    source: BaselineChunkGenerationManifest,
    embedder: _DeterministicEmbedder,
    *,
    parent: ComponentGenerationReference | None = None,
):
    return build_dense_generation(
        tmp_path / "index",
        source,
        embedder,
        metadata_db_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "sources",
        compatibility=_compatibility(embedder),
        parent_reference=parent,
    )


def test_full_generation_reopens_and_registration_validator_checks_membership(
    tmp_path: Path,
) -> None:
    embedder = _DeterministicEmbedder()
    source = _source(
        (
            ("alpha passage", _DOC_A, _DOC_A_V1),
            ("beta passage", _DOC_B, _DOC_B_V1),
        ),
        _VERSION_ONE,
    )

    outcome = _build(tmp_path, source, embedder)
    generation_reference = outcome.manifest.component_generation()
    participant = DenseGenerationParticipant(tmp_path / "index", outcome, source)
    ready_reference = replace(
        generation_reference,
        state=ComponentGenerationState.READY,
    )

    with open_dense_generation(
        tmp_path / "index",
        outcome.manifest.generation_id,
    ) as generation:
        entries = generation.iter_entries()
        assert len(entries) == 2
        assert generation.member_selections() == outcome.manifest.members
        assert (
            generation.manifest.source_manifest_digest == source.source_manifest_digest
        )
    assert participant.component_generation() == generation_reference
    assert participant.build_validator(
        _ChunkReader(source)
    ).validate_component_generation(
        generation_reference,
        tuple(entry.membership for entry in source.entries),
    )
    assert ready_reference.state is ComponentGenerationState.READY
    assert outcome.statistics.embedded_vectors == 2
    assert len(embedder.batches) == 1


def test_parent_selection_is_pinned_to_expected_active_corpus_version(
    tmp_path: Path,
) -> None:
    embedder = _DeterministicEmbedder()
    source = _source((("parent passage", _DOC_A, _DOC_A_V1),), _VERSION_ONE)
    outcome = _build(tmp_path, source, embedder)
    ready_reference = replace(
        outcome.manifest.component_generation(),
        state=ComponentGenerationState.READY,
    )
    repository = _Repository(_Snapshot(_VERSION_ONE, ready_reference))

    selected = select_dense_parent_reference(
        cast(MetadataRepository, repository), CORPUS_ID, _VERSION_ONE
    )

    assert selected == ready_reference
    stale_repository = _Repository(_Snapshot(_VERSION_TWO, ready_reference))
    with pytest.raises(DenseError) as raised:
        select_dense_parent_reference(
            cast(MetadataRepository, stale_repository),
            CORPUS_ID,
            _VERSION_ONE,
        )
    assert raised.value.category is DenseFailureCategory.INCOMPATIBLE_GENERATION


def test_identical_generation_is_reused_without_reembedding(tmp_path: Path) -> None:
    embedder = _DeterministicEmbedder()
    source = _source((("same passage", _DOC_A, _DOC_A_V1),), _VERSION_ONE)

    first = _build(tmp_path, source, embedder)
    prior_batches = tuple(embedder.batches)
    replay = _build(tmp_path, source, embedder)

    assert replay.manifest == first.manifest
    assert replay.statistics.generation_reused
    assert replay.statistics.reused_vectors == 1
    assert tuple(embedder.batches) == prior_batches


def test_incremental_build_reuses_matching_chunks_and_tracks_changes(
    tmp_path: Path,
) -> None:
    embedder = _DeterministicEmbedder()
    first_source = _source(
        (
            ("stable passage", _DOC_A, _DOC_A_V1),
            ("old replacement passage", _DOC_B, _DOC_B_V1),
        ),
        _VERSION_ONE,
    )
    first = _build(tmp_path, first_source, embedder)
    old_database_bytes = (
        tmp_path / "index" / "dense" / first.manifest.generation_id / "vectors.sqlite3"
    ).read_bytes()
    child_source = _source(
        (
            ("stable passage", _DOC_A, _DOC_A_V1),
            ("new replacement passage", _DOC_B, _DOC_B_V2),
            ("new member passage", _DOC_C, _DOC_C_V1),
        ),
        _VERSION_TWO,
    )
    parent = replace(
        first.manifest.component_generation(),
        state=ComponentGenerationState.READY,
    )
    before_batches = len(embedder.batches)

    child = _build(tmp_path, child_source, embedder, parent=parent)

    assert child.statistics.reused_vectors == 1
    assert child.statistics.embedded_vectors == 2
    assert child.statistics.replaced_vectors == 1
    assert child.statistics.removed_vectors == 0
    assert len(embedder.batches) == before_batches + 1
    assert (
        old_database_bytes
        == (
            tmp_path
            / "index"
            / "dense"
            / first.manifest.generation_id
            / "vectors.sqlite3"
        ).read_bytes()
    )
    with open_dense_generation(
        tmp_path / "index", child.manifest.generation_id
    ) as opened:
        assert len(opened.iter_entries()) == 3

    removal_source = _source((("stable passage", _DOC_A, _DOC_A_V1),), _VERSION_THREE)
    child_reference = replace(
        child.manifest.component_generation(),
        state=ComponentGenerationState.READY,
    )
    removal = _build(tmp_path, removal_source, embedder, parent=child_reference)
    assert removal.statistics.reused_vectors == 1
    assert removal.statistics.embedded_vectors == 0
    assert removal.statistics.removed_vectors == 2


def test_empty_generation_is_valid_and_does_not_call_embedder(tmp_path: Path) -> None:
    embedder = _DeterministicEmbedder()
    single = _source((("seed", _DOC_A, _DOC_A_V1),), _VERSION_ONE)
    empty = BaselineChunkGenerationManifest(
        corpus_id=single.corpus_id,
        corpus_version=_VERSION_TWO,
        configuration_identity=single.configuration_identity,
        generation_id="empty-fixture-generation",
        entries=(),
    )

    outcome = _build(tmp_path, empty, embedder)

    assert outcome.statistics.chunks_processed == 0
    assert outcome.statistics.embedded_vectors == 0
    assert embedder.batches == []
    with open_dense_generation(
        tmp_path / "index", outcome.manifest.generation_id
    ) as opened:
        assert opened.iter_entries() == ()


def test_embedding_failure_leaves_no_generation_and_suppresses_backend_message(
    tmp_path: Path,
) -> None:
    embedder = _DeterministicEmbedder()
    embedder.fail = True
    source = _source((("private failure marker", _DOC_A, _DOC_A_V1),), _VERSION_ONE)

    with pytest.raises(DenseError) as raised:
        _build(tmp_path, source, embedder)

    assert raised.value.category is DenseFailureCategory.EMBEDDING_FAILURE
    assert "private backend failure" not in str(raised.value)
    dense_root = tmp_path / "index" / "dense"
    assert not dense_root.exists()


def test_backend_metadata_must_match_the_full_compatibility_identity(
    tmp_path: Path,
) -> None:
    embedder = _DeterministicEmbedder()
    embedder.response_metadata = BackendMetadata(
        backend_name=embedder.metadata.backend_name,
        model_name=embedder.metadata.model_name,
        model_revision=embedder.metadata.model_revision,
        supported_operations=embedder.metadata.supported_operations,
        capabilities=(BackendCapability.BATCHING,),
        embedding_dimensions=embedder.metadata.embedding_dimensions,
    )
    source = _source((("passage", _DOC_A, _DOC_A_V1),), _VERSION_ONE)

    with pytest.raises(DenseError) as raised:
        _build(tmp_path, source, embedder)

    assert raised.value.category is DenseFailureCategory.INCOMPATIBLE_GENERATION
    assert not (tmp_path / "index" / "dense").exists()
