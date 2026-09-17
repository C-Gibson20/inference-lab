"""Tests for the resumable baseline chunk-stage executor."""

import hashlib
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Self

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    BaselineChunkErrorCategory,
    BaselineChunkExecutionError,
    BaselineChunkRequest,
    BaselineChunkResult,
    BaselineParseResult,
    BaselineStage,
    BaselineStageDisposition,
    ChunkingPolicy,
    ChunkingRequest,
    FakeChunkTokeniser,
    NormalisedText,
    ParsedDocument,
    build_chunk_set,
    execute_baseline_chunk,
)
from inference_lab.ingest.baseline_checkpoints import BaselineWorkKey
from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.parse_reconstruction import ReconstructedParseOutcome
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
    RecordId,
)
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    WriteChunkSetOutcome,
)

from .fixtures import fixture_parsed_document  # noqa: TID252

_PARSE_CONFIGURATION = ConfigurationIdentity(DigestAlgorithm.SHA256, "b" * 64)


class _Repository:
    """Repository double exposing only the public chunk-set boundary."""

    def __init__(self) -> None:
        self.selection: ChunkSetSelectionRecord | None = None
        self.write_count = 0
        self.fail_write = False

    def get_chunk_set_selection(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
        configuration_identity: ConfigurationIdentity,
    ) -> ChunkSetSelectionRecord | None:
        if self.selection is not None:
            assert self.selection.document_id
            assert self.selection.document_version_id == document_version_id
            assert self.selection.configuration_identity == configuration_identity
        del corpus_id
        return self.selection

    def transaction(self) -> "_Transaction":
        return _Transaction(self)

    def _write_chunk_set(self, request: ChunkSetWriteRequest) -> WriteChunkSetOutcome:
        if self.fail_write:
            raise RuntimeError("injected chunk publication failure")
        replayed = self.selection == request.selection
        self.selection = request.selection
        self.write_count += 1
        return WriteChunkSetOutcome(selection=request.selection, replayed=replayed)


class _Transaction:
    """Minimal rollback-capable transaction double for the public boundary."""

    def __init__(self, repository: _Repository) -> None:
        self._repository = repository
        self._selection = repository.selection
        self._write_count = repository.write_count

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, traceback
        if exception is not None:
            self._repository.selection = self._selection
            self._repository.write_count = self._write_count
        return False

    def write_chunk_set(self, request: ChunkSetWriteRequest) -> WriteChunkSetOutcome:
        return self._repository._write_chunk_set(request)


@dataclass(frozen=True, slots=True)
class _Case:
    request: BaselineChunkRequest
    repository: _Repository
    parsed_document: ParsedDocument


def _case(*, include_paragraph: bool = True) -> _Case:
    corpus_id = CorpusId.generate()
    parsed_document = (
        fixture_parsed_document()
        if include_paragraph
        else ParsedDocument(
            document_id=DocumentId.generate(),
            document_version_id=DocumentVersionId.generate(),
            text=NormalisedText(source_text=""),
            blocks=(),
            pages=None,
            sections=None,
            paragraphs=(),
            page_layouts=(),
            language=None,
            page_count=None,
            character_count=0,
            token_count=None,
        )
    )
    tokeniser = FakeChunkTokeniser()
    policy = ChunkingPolicy(min_tokens=1, max_tokens=8)
    expected = build_chunk_set(
        ChunkingRequest(
            corpus_id=corpus_id,
            parsed_document=parsed_document,
            tokeniser=tokeniser,
            policy=policy,
        )
    )
    parse_result = BaselineParseResult(
        corpus_id=corpus_id,
        work_key=BaselineWorkKey(
            proposed_corpus_version=CorpusVersionId.generate(),
            document_version_id=parsed_document.document_version_id,
            stage=BaselineStage.PARSE,
            configuration_identity=_PARSE_CONFIGURATION,
        ),
        parser_version=CURRENT_PARSER_VERSION,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
        disposition=BaselineStageDisposition.COMPLETED,
        reconstructed=ReconstructedParseOutcome(
            parser_version=CURRENT_PARSER_VERSION,
            outcome=ParseOutcome(
                status="succeeded",
                document=parsed_document,
                failures=(),
            ),
        ),
    )
    request = BaselineChunkRequest(
        corpus_id=corpus_id,
        work_key=BaselineWorkKey(
            proposed_corpus_version=parse_result.work_key.proposed_corpus_version,
            document_version_id=parsed_document.document_version_id,
            stage=BaselineStage.CHUNK,
            configuration_identity=expected.configuration_identity,
        ),
        parse_result=parse_result,
        tokeniser=tokeniser,
        policy=policy,
        chunker_version=CURRENT_CHUNKER_VERSION,
    )
    return _Case(
        request=request, repository=_Repository(), parsed_document=parsed_document
    )


def test_chunk_stage_publishes_then_reuses_complete_payload() -> None:
    case = _case()

    with capture_logs() as logs:
        published = execute_baseline_chunk(case.repository, case.request)
        reused = execute_baseline_chunk(case.repository, case.request)

    assert isinstance(published, BaselineChunkResult)
    assert published.disposition is BaselineStageDisposition.COMPLETED
    assert reused.disposition is BaselineStageDisposition.REUSED
    assert published.chunk_set == reused.chunk_set
    assert published.chunk_set.chunks
    assert case.repository.write_count == 1
    assert [record["event"] for record in logs] == [
        "baseline_chunk_started",
        "chunk_started",
        "chunk_completed",
        "baseline_chunk_completed",
        "baseline_chunk_started",
        "baseline_chunk_reused",
    ]


def test_chunk_stage_persists_a_complete_zero_chunk_set() -> None:
    case = _case(include_paragraph=False)

    result = execute_baseline_chunk(case.repository, case.request)

    assert result.chunk_set.chunks == ()
    assert result.chunk_set.lineage.records == ()
    assert case.repository.write_count == 1


def test_chunk_stage_rejects_chunk_and_publication_failures_without_partial_output() -> (
    None
):
    case = _case()
    case.repository.fail_write = True

    with pytest.raises(BaselineChunkExecutionError) as publication_error:
        execute_baseline_chunk(case.repository, case.request)

    assert (
        publication_error.value.category
        is BaselineChunkErrorCategory.CHUNK_PERSISTENCE_FAILED
    )
    assert case.repository.selection is None
    assert case.repository.write_count == 0


def test_chunk_stage_maps_invalid_tokeniser_to_chunk_failure() -> None:
    case = _case()

    class _InvalidTokeniser:
        pass

    invalid_request = replace(case.request, tokeniser=_InvalidTokeniser())  # type: ignore[arg-type]

    with pytest.raises(BaselineChunkExecutionError) as error:
        execute_baseline_chunk(case.repository, invalid_request)

    assert error.value.category is BaselineChunkErrorCategory.CHUNK_FAILED
    assert case.repository.write_count == 0


def test_chunk_stage_rejects_corrupt_replay_payload() -> None:
    case = _case()
    execute_baseline_chunk(case.repository, case.request)
    assert case.repository.selection is not None
    corrupt_payload = b"{}"
    case.repository.selection = replace(
        case.repository.selection,
        canonical_payload=corrupt_payload,
        payload_digest=OutputDigest(
            algorithm=DigestAlgorithm.SHA256,
            digest=hashlib.sha256(corrupt_payload).hexdigest(),
        ),
    )

    with pytest.raises(BaselineChunkExecutionError) as error:
        execute_baseline_chunk(case.repository, case.request)

    assert error.value.category is BaselineChunkErrorCategory.CORRUPT_STATE
