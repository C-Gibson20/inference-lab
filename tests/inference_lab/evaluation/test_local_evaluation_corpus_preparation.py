"""Acceptance for private, version-pinned corpus preparation."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from runpy import run_path
from typing import cast

from inference_lab.backends.runtime import (
    BackendCapability,
    BackendMetadata,
    BackendOperation,
    EmbeddingProfile,
    EmbeddingRequest,
    EmbeddingResult,
)
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.retrieval import DENSE_COMPONENT_KIND, open_dense_generation
from inference_lab.storage import open_metadata_repository
from tests.inference_lab.serving.operator_fixtures import (  # noqa: TID252
    LOCAL_TOKENISER_FIXTURE,
    run_example,
    write_config,
)

_SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/inference_lab/evaluation/prepare_local_evaluation_corpus.py"
)


def test_preparation_ingests_an_explicit_directory_without_leaking_payloads(
    tmp_path: Path,
) -> None:
    """One private directory becomes one corpus version with bounded output."""
    write_config(tmp_path)
    source = tmp_path / "approved-public-corpus"
    source.mkdir()
    document_text = "SARM1 literature passage " * 180
    (source / "paper.txt").write_text(document_text, encoding="utf-8")
    corpus_id = CorpusId.generate().as_str()
    version = CorpusVersionId.generate().as_str()

    output, stderr = run_example(
        tmp_path,
        _SCRIPT,
        {
            "EVALUATION_CORPUS_SOURCE_DIRECTORY": str(source),
            "EVALUATION_CORPUS_TOKENISER_DIRECTORY": str(LOCAL_TOKENISER_FIXTURE),
            "EVALUATION_CORPUS_ID": corpus_id,
            "EVALUATION_CORPUS_PROPOSED_VERSION": version,
        },
    )

    assert len(output) == 1
    assert {
        key: value
        for key, value in output[0].items()
        if key != "component_generation_ids"
    } == {
        "corpus_id": corpus_id,
        "document_outcome_count": 1,
        "execution_status": "completed",
        "failure_categories": [],
        "proposed_version": version,
        "publication_status": "committed",
    }
    assert {
        item["component_kind"] for item in output[0]["component_generation_ids"]
    } == {"chunks", "lexical-index"}
    safe_output = json.dumps(output) + stderr
    assert str(source) not in safe_output
    assert document_text not in safe_output
    assert "paper.txt" not in safe_output


class _EmbeddingFixture:
    """Small deterministic embedder for atomic dense publication coverage."""

    metadata = BackendMetadata(
        backend_name="evaluation-test",
        model_name="evaluation-test-model",
        model_revision="r1",
        supported_operations=(BackendOperation.EMBED,),
        capabilities=(BackendCapability.BATCHING,),
        embedding_dimensions=3,
    )
    output_profile = EmbeddingProfile(
        profile_version="1.0.0",
        tokenizer_id="evaluation-test-tokenizer",
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

    def __init__(self) -> None:
        self.is_ready = False
        self.is_closed = False

    def load(self) -> None:
        self.is_ready = True

    def close(self) -> None:
        self.is_ready = False
        self.is_closed = True

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            tuple((1.0, 0.0, 0.0) for _ in request.input_texts),
            self.metadata,
            self.output_profile,
        )


def test_dense_index_is_published_with_the_same_immutable_corpus_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The optional local embedding stage joins the lexical publication."""
    write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "approved-public-corpus"
    source.mkdir()
    (source / "paper.txt").write_text(
        "SARM1 literature passage " * 180,
        encoding="utf-8",
    )
    corpus_id = CorpusId.generate()
    version = CorpusVersionId.generate()
    backend = _EmbeddingFixture()
    for key, value in {
        "EVALUATION_CORPUS_SOURCE_DIRECTORY": str(source),
        "EVALUATION_CORPUS_TOKENISER_DIRECTORY": str(LOCAL_TOKENISER_FIXTURE),
        "EVALUATION_CORPUS_ID": corpus_id.as_str(),
        "EVALUATION_CORPUS_PROPOSED_VERSION": version.as_str(),
        "EVALUATION_CORPUS_INCLUDE_DENSE": "true",
    }.items():
        monkeypatch.setenv(key, value)
    namespace = run_path(str(_SCRIPT), run_name="prepare_local_corpus_script")

    result = cast(Callable[..., dict[str, object]], namespace["_run"])(
        embedding_backend_factory=lambda _path: backend
    )
    assert result["publication_status"] == "committed"
    component_rows = cast(list[dict[str, str]], result["component_generation_ids"])
    components = {row["component_kind"]: row["generation_id"] for row in component_rows}
    assert set(components) == {"chunks", "lexical-index", DENSE_COMPONENT_KIND.as_str()}
    assert backend.is_closed

    settings_repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "source-data",
        apply_migrations=False,
    )
    try:
        with settings_repository.open_corpus_snapshot(corpus_id, version) as snapshot:
            assert {
                reference.component_kind.as_str()
                for reference in snapshot.manifest.component_generations
            } == {"chunks", "lexical-index", DENSE_COMPONENT_KIND.as_str()}
        with open_dense_generation(
            tmp_path / "index", components[DENSE_COMPONENT_KIND.as_str()]
        ) as generation:
            assert generation.manifest.corpus_id == corpus_id
            assert generation.manifest.corpus_version == version
            assert generation.manifest.entry_count > 0
    finally:
        settings_repository.close()


def test_preparation_fails_closed_when_config_is_absent(
    capsys,
) -> None:
    """Missing deployment configuration produces one fixed safe message."""
    namespace = run_path(str(_SCRIPT), run_name="prepare_local_corpus_script")
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])
    assert main(None) == 1
    assert capsys.readouterr().err.strip() == (
        "local evaluation corpus preparation failed"
    )
