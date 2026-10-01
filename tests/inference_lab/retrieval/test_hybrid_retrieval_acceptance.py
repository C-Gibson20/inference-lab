"""Phase 6 acceptance for pinned lexical+dense fusion and duplicate handling."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from structlog.testing import capture_logs

import inference_lab.retrieval.hybrid as hybrid
from inference_lab.provenance import (
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    DenseGeneration,
    DenseGenerationParticipant,
    DenseIndexCompatibility,
    EvidenceHit,
    EvidenceResult,
    EvidenceSource,
    FailureCategory,
    FusionPolicy,
    LexicalGeneration,
    LexicalGenerationParticipant,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
    build_dense_generation,
    build_lexical_generation,
    execute_hybrid_retrieval,
    open_dense_generation,
    open_lexical_generation,
)
from inference_lab.storage import (
    ComponentGenerationState,
    CorpusSnapshot,
    MetadataRepository,
)
from tests.inference_lab.retrieval.lexical_fixtures import (
    CORPUS_ID,
    CORPUS_VERSION_ID,
    manifest,
    selection,
)
from tests.inference_lab.retrieval.test_dense_retrieval import _QueryEmbedder

_DOCUMENTS = tuple(
    (
        DocumentId.from_str(f"00000000-0000-4000-8000-{value:012d}"),
        DocumentVersionId.from_str(f"00000000-0000-4000-8000-{value + 10:012d}"),
    )
    for value in (401, 402, 403)
)
_FAMILY_ID = DocumentFamilyId.from_str("00000000-0000-4000-8000-000000000450")


class _Snapshot:
    """Provide the corpus-scoped source and relationship view for fixture indexes."""

    def __init__(
        self,
        lexical_reference: object,
        dense_reference: object,
        entries: tuple[Any, ...],
    ) -> None:
        self.corpus_id = CORPUS_ID
        self.version = SimpleNamespace(corpus_version=CORPUS_VERSION_ID)
        self.manifest = SimpleNamespace(
            component_generations=(lexical_reference, dense_reference)
        )
        self._versions: dict[DocumentVersionId, Any] = {}
        self._artefacts: dict[DocumentVersionId, Any] = {}
        self._spans: dict[Any, Any] = {}
        self._duplicate = DuplicateRelationship(
            min(_DOCUMENTS[0][1], _DOCUMENTS[1][1], key=lambda value: value.as_str()),
            max(_DOCUMENTS[0][1], _DOCUMENTS[1][1], key=lambda value: value.as_str()),
            DuplicateClass.BYTE_IDENTICAL,
        )
        self._family_memberships = tuple(
            DocumentFamilyMembership(_FAMILY_ID, document_id, version_id)
            for document_id, version_id in _DOCUMENTS
        )
        for item in entries:
            if hasattr(item, "candidate"):
                candidate = item.candidate
                document_id = candidate.sources[0].document_id
                version_id = candidate.sources[0].document_version_id
                content = candidate.content
            else:
                document_id = item.document_id
                version_id = item.document_version_id
                content = item.content
            self._versions[version_id] = SimpleNamespace(document_id=document_id)
            self._artefacts[version_id] = SimpleNamespace(
                document_id=document_id,
                document_version_id=version_id,
                media_type="text/plain",
                byte_count=len(content.encode("utf-8")),
            )
            for source in (
                item.candidate.sources
                if hasattr(item, "candidate")
                else item.source_links
            ):
                source_span_id = source.source_span_id
                self._spans[source_span_id] = SimpleNamespace(
                    document_id=document_id,
                    location=SimpleNamespace(
                        document_version_id=version_id,
                        source_span_id=source_span_id,
                        parsed_record_id=SimpleNamespace(),
                        start_offset=0,
                        end_offset=len(content),
                        page_number=None,
                        content_origin=SimpleNamespace(value="native"),
                    ),
                )

    def add_visible_source(
        self,
        document_id: DocumentId,
        version_id: DocumentVersionId,
        span_id: SourceSpanId,
        *,
        content_length: int,
    ) -> None:
        """Add one exact visible source link for a focused isolation fixture."""
        self._versions[version_id] = SimpleNamespace(document_id=document_id)
        self._artefacts[version_id] = SimpleNamespace(
            document_id=document_id,
            document_version_id=version_id,
            media_type="text/plain",
            byte_count=content_length,
        )
        self._spans[span_id] = SimpleNamespace(
            document_id=document_id,
            location=SimpleNamespace(
                document_version_id=version_id,
                source_span_id=span_id,
                parsed_record_id=SimpleNamespace(),
                start_offset=0,
                end_offset=content_length,
                page_number=None,
                content_origin=SimpleNamespace(value="native"),
            ),
        )

    def get_document_version(self, version_id: DocumentVersionId):
        return self._versions.get(version_id)

    def get_source_artefact(self, version_id: DocumentVersionId):
        return self._artefacts.get(version_id)

    def get_parsed_representation(self, record_id: object):
        del record_id
        return None

    def resolve_source_span(self, span_id: object):
        return self._spans.get(span_id)

    def list_duplicate_relationships(self, version_id: DocumentVersionId):
        if version_id in (_DOCUMENTS[0][1], _DOCUMENTS[1][1]):
            return (self._duplicate,)
        return ()

    def list_family_memberships_by_document_version(
        self,
        version_id: DocumentVersionId,
    ):
        return tuple(
            membership
            for membership in self._family_memberships
            if membership.document_version_id == version_id
        )

    def list_family_memberships_by_document(self, document_id: DocumentId):
        return tuple(
            membership
            for membership in self._family_memberships
            if membership.document_id == document_id
        )


class _Repository:
    """Satisfy the borrowed-snapshot composition boundary used by the executor."""

    def open_corpus_snapshot(self, *_: object) -> None:
        raise AssertionError("the caller supplied the already pinned snapshot")


def test_hybrid_fails_closed_when_pinned_snapshot_version_differs() -> None:
    query = RetrievalQuery("alpha", CORPUS_VERSION_ID, RetrievalLimit(2))
    snapshot = _Snapshot(SimpleNamespace(), SimpleNamespace(), ())
    snapshot.version = SimpleNamespace(
        corpus_version=CorpusVersionId.generate(),
    )

    result = execute_hybrid_retrieval(
        cast(MetadataRepository, _Repository()),
        Path("indexes"),
        CORPUS_ID,
        query,
        _QueryEmbedder(),
        snapshot=cast(CorpusSnapshot, snapshot),
    )

    assert result.status is ResultStatus.FAILED
    assert result.failures[0].category is FailureCategory.CORPUS_VERSION_MISMATCH


def test_hybrid_journey_fuses_channels_and_keeps_distinct_family_evidence(
    tmp_path: Path,
) -> None:
    selections = tuple(
        selection(
            content,
            document_id=document_id,
            document_version_id=version_id,
        )[1]
        for (document_id, version_id), content in zip(
            _DOCUMENTS,
            (
                "alpha common wording",
                "alpha common wording",
                "alpha distinct wording",
            ),
            strict=True,
        )
    )
    source_manifest = manifest(selections)
    embedder = _QueryEmbedder()
    index_dir = tmp_path / "indexes"
    lexical_outcome = build_lexical_generation(index_dir, source_manifest)
    dense_outcome = build_dense_generation(
        index_dir,
        source_manifest,
        embedder,
        metadata_db_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "sources",
        compatibility=DenseIndexCompatibility(
            model_metadata=embedder.metadata,
            dimensions=3,
            embedding_profile=embedder.profile,
        ),
    )
    with open_lexical_generation(
        index_dir, lexical_outcome.manifest.generation_id
    ) as lex:
        lexical_entries = lex.iter_chunks()
    with open_dense_generation(
        index_dir, dense_outcome.manifest.generation_id
    ) as dense:
        dense_entries = dense.iter_entries()

    lexical_reference = replace(
        LexicalGenerationParticipant(
            index_dir,
            lexical_outcome,
            source_manifest,
        ).component_generation(),
        state=ComponentGenerationState.READY,
    )
    dense_reference = replace(
        DenseGenerationParticipant(
            index_dir,
            dense_outcome,
            source_manifest,
        ).component_generation(),
        state=ComponentGenerationState.READY,
    )
    snapshot = _Snapshot(
        lexical_reference,
        dense_reference,
        tuple(lexical_entries) + tuple(dense_entries),
    )
    query = RetrievalQuery("alpha", CORPUS_VERSION_ID, RetrievalLimit(3))

    with capture_logs() as logs:
        with (
            open_lexical_generation(
                index_dir, lexical_outcome.manifest.generation_id
            ) as lexical_generation,
            open_dense_generation(
                index_dir, dense_outcome.manifest.generation_id
            ) as dense_generation,
        ):
            result = execute_hybrid_retrieval(
                cast(MetadataRepository, _Repository()),
                index_dir,
                CORPUS_ID,
                query,
                embedder,
                policy=FusionPolicy(),
                snapshot=cast(CorpusSnapshot, snapshot),
                lexical_generation=cast(LexicalGeneration, lexical_generation),
                dense_generation=cast(DenseGeneration, dense_generation),
            )

    assert result.status.value == "completed"
    assert len(result.hits) == 2
    assert tuple(
        score.channel for hit in result.hits for score in hit.contributions
    ) == (
        RetrievalChannel.DENSE,
        RetrievalChannel.LEXICAL,
        RetrievalChannel.DENSE,
        RetrievalChannel.LEXICAL,
    )
    returned_documents = {hit.candidate.sources[0].document_id for hit in result.hits}
    assert len(returned_documents) == 2
    assert _DOCUMENTS[2][0] in returned_documents
    assert (
        len(returned_documents.intersection({item[0] for item in _DOCUMENTS[:2]})) == 1
    )
    fusion_log = next(
        record for record in logs if record["event"] == "retrieval_fusion_completed"
    )
    assert fusion_log["fusion_policy"] == FusionPolicy().identity
    assert fusion_log["deduplication_policy"] == "1.0.0"
    assert fusion_log["pipeline_stage"] == "retrieval_fusion"
    assert "alpha" not in repr(logs)
    assert "alpha common wording" not in repr(logs)


def test_hybrid_discards_a_foreign_channel_candidate_and_keeps_safe_partial_hits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    query = RetrievalQuery(
        "private query marker",
        CORPUS_VERSION_ID,
        RetrievalLimit(2),
    )
    foreign = RetrievalCandidate(
        RecordId.generate(),
        CandidateKind.ORIGINAL_PASSAGE,
        "foreign source payload",
        (
            EvidenceSource(
                DocumentId.generate(),
                DocumentVersionId.generate(),
                SourceSpanId.generate(),
            ),
        ),
    )
    safe_document, safe_version = _DOCUMENTS[2]
    safe_span = SourceSpanId.generate()
    safe = RetrievalCandidate(
        RecordId.generate(),
        CandidateKind.ORIGINAL_PASSAGE,
        "visible source payload",
        (EvidenceSource(safe_document, safe_version, safe_span),),
    )
    snapshot = _Snapshot(SimpleNamespace(), SimpleNamespace(), ())
    snapshot.add_visible_source(
        safe_document,
        safe_version,
        safe_span,
        content_length=len(safe.content),
    )

    def channel_result(
        channel: RetrievalChannel,
        candidate: RetrievalCandidate,
    ) -> EvidenceResult:
        return EvidenceResult(
            CORPUS_VERSION_ID,
            query.limit,
            (
                EvidenceHit(
                    candidate,
                    (ChannelScore(channel, 1),),
                    1,
                ),
            ),
            ResultStatus.COMPLETED,
            (),
        )

    monkeypatch.setattr(
        hybrid,
        "execute_lexical_retrieval",
        lambda *_args, **_kwargs: channel_result(RetrievalChannel.LEXICAL, foreign),
    )
    monkeypatch.setattr(
        hybrid,
        "execute_dense_retrieval",
        lambda *_args, **_kwargs: channel_result(RetrievalChannel.DENSE, safe),
    )

    result = execute_hybrid_retrieval(
        cast(MetadataRepository, _Repository()),
        tmp_path / "indexes",
        CORPUS_ID,
        query,
        _QueryEmbedder(),
        snapshot=cast(CorpusSnapshot, snapshot),
    )

    assert result.status is ResultStatus.PARTIAL
    assert tuple(hit.candidate.record_id for hit in result.hits) == (safe.record_id,)
    assert result.failures[0].channel is RetrievalChannel.LEXICAL
    assert result.failures[0].category is FailureCategory.INVALID_CHANNEL_RESPONSE
