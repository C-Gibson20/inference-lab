"""Shared corpus identities and embedder for persistent-index acceptance."""

from inference_lab.backends.runtime import (
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
)
from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)

CORPUS_A = CorpusId.from_str("00000000-0000-4000-8000-000000000011")
CORPUS_B = CorpusId.from_str("00000000-0000-4000-8000-000000000012")
VERSION_A = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000021")
VERSION_B = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000022")
UNPUBLISHED_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000023")
VERSION_A2 = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000024")
VERSION_A3 = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000025")
DOC_SHARED = DocumentId.from_str("00000000-0000-4000-8000-000000000031")
DOC_A = DocumentId.from_str("00000000-0000-4000-8000-000000000032")
DOC_B = DocumentId.from_str("00000000-0000-4000-8000-000000000033")
DOC_STABLE = DocumentId.from_str("00000000-0000-4000-8000-000000000034")
DOC_VERSION_SHARED = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000041")
DOC_VERSION_A = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000042")
DOC_VERSION_B = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000043")
DOC_VERSION_STABLE = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000044")
DOC_VERSION_A2 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000045")
DOC_VERSION_A3 = DocumentVersionId.from_str("00000000-0000-4000-8000-000000000046")
DOCUMENTS = (
    (DOC_SHARED, DOC_VERSION_SHARED, "common assay terminology"),
    (DOC_A, DOC_VERSION_A, "orchid receptor evidence"),
    (DOC_B, DOC_VERSION_B, "tulip catalyst evidence"),
    (DOC_STABLE, DOC_VERSION_STABLE, "stable control evidence"),
)


class EmbeddingFixture:
    """Deterministic local vectors with an observable embed-work count."""

    metadata = BackendMetadata(
        backend_name="persistent-index-fixture",
        model_name="persistent-index-fixture",
        model_revision="1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(),
        embedding_dimensions=3,
    )
    profile = EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="persistent-index-tokenizer",
        tokenizer_revision="1",
        query_prefix="query: ",
        document_prefix="",
        pooling_strategy="mean",
        input_normalisation="identity",
        max_input_tokens=128,
        over_limit_policy="truncate_tail",
        l2_normalised=False,
        inference_dtype="float32",
        device_type="cpu",
    )

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def load(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def output_metadata(self) -> BackendMetadata:
        return self.metadata

    @property
    def output_profile(self) -> EmbeddingProfile:
        return self.profile

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.embedded_texts.extend(request.input_texts)

        def vector(text: str) -> tuple[float, float, float]:
            if "orchid" in text:
                return (1.0, 0.0, 0.0)
            if "tulip" in text:
                return (0.0, 1.0, 0.0)
            return (0.0, 0.0, 1.0)

        return EmbeddingResult(
            tuple(vector(text) for text in request.input_texts),
            self.metadata,
            self.profile,
        )
