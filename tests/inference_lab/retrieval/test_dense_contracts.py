"""Dense compatibility, vector validation, and deterministic ranking tests."""

import math

import pytest

from inference_lab.backends.runtime import (
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    DENSE_COMPONENT_KIND,
    CandidateKind,
    DenseIndexBuildRequest,
    DenseIndexCompatibility,
    DenseIndexConfiguration,
    DenseIndexEntry,
    DenseIndexQuery,
    DenseVector,
    EvidenceSource,
    ExactDenseVectorIndex,
    InvalidDenseContractError,
    RetrievalCandidate,
    RetrievalLimit,
    RetrievalQuery,
)


def _embedding_profile(*, tokenizer_revision: str = "tokenizer-r1") -> EmbeddingProfile:
    return EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="qwen-tokenizer",
        tokenizer_revision=tokenizer_revision,
        query_prefix="query: ",
        document_prefix="",
        pooling_strategy="last-token",
        input_normalisation="identity",
        max_input_tokens=4096,
        over_limit_policy="truncate_tail",
        l2_normalised=True,
        inference_dtype="float32",
        device_type="cpu",
    )


def _metadata(*, revision: str = "model-r1") -> BackendMetadata:
    return BackendMetadata(
        backend_name="fake-embedder",
        model_name="local-embedder",
        model_revision=revision,
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=2,
    )


def _compatibility() -> DenseIndexCompatibility:
    return DenseIndexCompatibility(
        model_metadata=_metadata(),
        dimensions=2,
        embedding_profile=_embedding_profile(),
    )


def _query(vector: DenseVector) -> DenseIndexQuery:
    return DenseIndexQuery(
        query=RetrievalQuery(
            text="private query",
            corpus_version=CorpusVersionId.from_str(
                "00000000-0000-4000-8000-000000000031"
            ),
            limit=RetrievalLimit(3),
        ),
        query_vector=vector,
        compatibility=_compatibility(),
    )


def _candidate(record_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        record_id=RecordId.from_str(record_id),
        kind=CandidateKind.ORIGINAL_PASSAGE,
        content=f"passage {record_id}",
        sources=(
            EvidenceSource(
                document_id=DocumentId.from_str("00000000-0000-4000-8000-000000000001"),
                document_version_id=DocumentVersionId.from_str(
                    "00000000-0000-4000-8000-000000000002"
                ),
                source_span_id=SourceSpanId.from_str(
                    "00000000-0000-4000-8000-000000000003"
                ),
            ),
        ),
    )


def test_dense_compatibility_captures_model_profile_and_component_identity() -> None:
    compatibility = _compatibility()
    changed_tokenizer = DenseIndexCompatibility(
        model_metadata=_metadata(),
        dimensions=2,
        embedding_profile=_embedding_profile(tokenizer_revision="tokenizer-r2"),
    )
    changed_model = DenseIndexCompatibility(
        model_metadata=_metadata(revision="model-r2"),
        dimensions=2,
        embedding_profile=_embedding_profile(),
    )

    assert DENSE_COMPONENT_KIND.as_str() == "dense-index"
    assert len(compatibility.configuration_digest) == 64
    assert compatibility.configuration_digest != changed_tokenizer.configuration_digest
    assert compatibility.configuration_digest != changed_model.configuration_digest
    assert "local" not in compatibility.configuration_digest


def test_dense_compatibility_rejects_dimension_mismatch() -> None:
    with pytest.raises(InvalidDenseContractError):
        DenseIndexCompatibility(
            model_metadata=_metadata(),
            dimensions=3,
            embedding_profile=_embedding_profile(),
        )


def test_dense_build_request_binds_entries_to_one_corpus_generation() -> None:
    request = DenseIndexBuildRequest(
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000041"),
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000042"),
        source_manifest_digest="a" * 64,
        compatibility=_compatibility(),
        entries=(
            DenseIndexEntry(
                _candidate("00000000-0000-4000-8000-000000000043"),
                DenseVector((1.0, 0.0)),
            ),
        ),
    )

    assert request.corpus_id.as_str() == "00000000-0000-4000-8000-000000000041"


def test_dense_query_rejects_vector_dimension_mismatch() -> None:
    with pytest.raises(InvalidDenseContractError):
        _query(DenseVector((1.0, 0.0, 0.0)))


@pytest.mark.parametrize(
    "values",
    [(), (0.0, 0.0), (1.0, math.inf), (1.0, math.nan), (True, 1.0)],
)
def test_dense_vectors_reject_empty_zero_non_finite_and_boolean_values(
    values: tuple[float, ...],
) -> None:
    with pytest.raises((TypeError, InvalidDenseContractError)):
        DenseVector(values)


def test_exact_dense_index_orders_neighbours_and_breaks_ties_by_record_id() -> None:
    later = _candidate("00000000-0000-4000-8000-000000000012")
    earlier = _candidate("00000000-0000-4000-8000-000000000011")
    distant = _candidate("00000000-0000-4000-8000-000000000013")
    index = ExactDenseVectorIndex(
        (
            DenseIndexEntry(later, DenseVector((1.0, 1.0))),
            DenseIndexEntry(distant, DenseVector((0.0, 1.0))),
            DenseIndexEntry(earlier, DenseVector((1.0, 1.0))),
        )
    )

    matches = index.search(_query(DenseVector((1.0, 0.0))))

    assert tuple(match.candidate.record_id for match in matches) == (
        earlier.record_id,
        later.record_id,
        distant.record_id,
    )
    assert tuple(match.rank for match in matches) == (1, 2, 3)
    assert matches[0].cosine_similarity == pytest.approx(1 / math.sqrt(2))
    assert matches[2].cosine_similarity == pytest.approx(0.0)


def test_exact_dense_index_rejects_dimension_mismatch_and_duplicate_records() -> None:
    candidate = _candidate("00000000-0000-4000-8000-000000000021")
    same_id = _candidate("00000000-0000-4000-8000-000000000021")
    with pytest.raises(InvalidDenseContractError):
        ExactDenseVectorIndex(
            (DenseIndexEntry(candidate, DenseVector((1.0, 0.0))),)
        ).search(
            DenseIndexQuery(
                query=RetrievalQuery(
                    "private query",
                    CorpusVersionId.from_str("00000000-0000-4000-8000-000000000031"),
                    RetrievalLimit(1),
                ),
                query_vector=DenseVector((1.0, 0.0, 0.0)),
                compatibility=DenseIndexCompatibility(
                    _metadata(), 2, _embedding_profile()
                ),
            )
        )

    with pytest.raises(InvalidDenseContractError):
        ExactDenseVectorIndex(
            (
                DenseIndexEntry(candidate, DenseVector((1.0, 0.0))),
                DenseIndexEntry(same_id, DenseVector((0.0, 1.0))),
            )
        ).search(_query(DenseVector((1.0, 0.0))))


@pytest.mark.parametrize(
    "revision",
    ["/mnt/models/embedder", "~/models/embedder", "https://example.invalid/model"],
)
def test_dense_compatibility_rejects_a_deployment_path_in_model_identity(
    revision: str,
) -> None:
    # The configuration digest is persisted in every generation manifest, so a
    # deployment path must never reach it.
    with pytest.raises(InvalidDenseContractError):
        DenseIndexCompatibility(
            model_metadata=_metadata(revision=revision),
            dimensions=2,
            embedding_profile=_embedding_profile(),
        )


def test_dense_configuration_rejects_a_deployment_path_in_library_version() -> None:
    with pytest.raises(InvalidDenseContractError):
        DenseIndexConfiguration(library_version="/usr/lib/sqlite3")
