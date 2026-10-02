"""RET-07 pinned snapshot integration, provenance mapping, and logging."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import Chunk, normalise_source_text
from inference_lab.provenance import (
    ContentHash,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceReductionExecutionError,
    EvidenceReductionFailureCategory,
    EvidenceResult,
    EvidenceSource,
    EvidenceTokenisation,
    EvidenceTokeniserIdentity,
    EvidenceTokenOffset,
    ResultStatus,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalLimit,
    build_lexical_generation,
    open_lexical_generation,
    reduce_retrieval_evidence,
)
from inference_lab.settings import RetrievalConfig
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusSnapshot,
    ParsedRepresentation,
    ResolvedSourceSpan,
    SourceArtefact,
)
from tests.inference_lab.retrieval.lexical_fixtures import (
    CORPUS_ID,
    CORPUS_VERSION_ID,
    manifest,
    selection,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    PARSED_RECORD_ID,
    parsed_representation,
)


@dataclass(frozen=True, slots=True)
class _WhitespaceTokeniser:
    """Local deterministic tokenizer that reports code-point offsets."""

    identity: EvidenceTokeniserIdentity = EvidenceTokeniserIdentity(
        implementation="test-whitespace",
        revision="fixture-v1",
        contract_version="1.0.0",
    )

    def tokenise(self, text: str) -> EvidenceTokenisation:
        offsets: list[EvidenceTokenOffset] = []
        start: int | None = None
        for index, character in enumerate(text):
            if character.isspace():
                if start is not None:
                    offsets.append(EvidenceTokenOffset(start, index))
                    start = None
            elif start is None:
                start = index
        if start is not None:
            offsets.append(EvidenceTokenOffset(start, len(text)))
        return EvidenceTokenisation(tuple(offsets))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenise(text).offsets)


class _Snapshot:
    """Narrow snapshot surface backed by exact parsed source locations."""

    corpus_id = CORPUS_ID

    def __init__(
        self,
        parsed: ParsedRepresentation,
        source: SourceArtefact,
        generation_manifest,
    ) -> None:
        self.version = SimpleNamespace(corpus_version=CORPUS_VERSION_ID)
        self.manifest = SimpleNamespace(
            component_generations=(
                ComponentGenerationReference(
                    component_kind=ComponentKind("lexical-index"),
                    generation_id=generation_manifest.generation_id,
                    implementation=generation_manifest.implementation,
                    format_version=generation_manifest.format_version,
                    contract_version=generation_manifest.contract_version,
                    configuration_digest=generation_manifest.configuration_digest,
                    source_manifest_digest=generation_manifest.source_manifest_digest,
                    corpus_id=CORPUS_ID,
                    corpus_version=CORPUS_VERSION_ID,
                    state=ComponentGenerationState.READY,
                ),
            )
        )
        self._parsed = parsed
        self._source = source
        self._locations = {
            item.source_span_id: item for item in parsed.source_locations
        }

    def get_source_artefact(
        self, document_version_id: DocumentVersionId
    ) -> SourceArtefact | None:
        return self._source if document_version_id == DOCUMENT_VERSION_ID else None

    def get_document_version(
        self, document_version_id: DocumentVersionId
    ) -> DocumentVersion | None:
        if document_version_id != DOCUMENT_VERSION_ID:
            return None
        return DocumentVersion(
            document_version_id=DOCUMENT_VERSION_ID,
            document_id=DOCUMENT_ID,
            content_hash=self._source.content_hash,
        )

    def get_parsed_representation(
        self, record_id: RecordId
    ) -> ParsedRepresentation | None:
        return self._parsed if record_id == self._parsed.record_id else None

    def resolve_source_span(
        self, source_span_id: SourceSpanId
    ) -> ResolvedSourceSpan | None:
        location = self._locations.get(source_span_id)
        if location is None:
            return None
        return ResolvedSourceSpan(
            document_id=DOCUMENT_ID,
            location=location,
            page_layout=None,
        )


def _parsed_for_chunk(text: str, chunk: Chunk) -> ParsedRepresentation:
    """Build the exact parsed representation named by the chunk's source link."""
    base = parsed_representation()
    source_span_id = chunk.source_references[0].source_span_id
    canonical_text = normalise_source_text(text)
    block = replace(
        base.blocks[0],
        text=canonical_text,
        source_span_id=source_span_id,
    )
    location = replace(
        base.source_locations[0],
        source_span_id=source_span_id,
        parsed_record_id=PARSED_RECORD_ID,
        start_offset=0,
        end_offset=len(text),
    )
    return replace(
        base,
        source_text=text,
        text=canonical_text,
        character_count=len(canonical_text),
        token_count=len(canonical_text.split()),
        blocks=(block,),
        source_locations=(location,),
    )


def _source_artefact(text: str) -> SourceArtefact:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceArtefact(
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        content_hash=ContentHash(HashAlgorithm.SHA256, digest),
        byte_count=len(text.encode("utf-8")),
        media_type="text/plain",
        storage_key=f"sha256/{digest[:2]}/{digest}",
    )


def test_public_reduction_maps_unicode_truncation_and_logs_privately(
    tmp_path: Path,
) -> None:
    source_text = "Cafe\u0301 alpha beta gamma"
    chunk_set, selected = selection(
        source_text,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
    )
    chunk = chunk_set.chunks[0]
    built = build_lexical_generation(tmp_path, manifest((selected,)))
    parsed = _parsed_for_chunk(source_text, chunk)
    snapshot = _Snapshot(parsed, _source_artefact(source_text), built.manifest)
    candidate = RetrievalCandidate(
        record_id=chunk.record_id,
        kind=CandidateKind.ORIGINAL_PASSAGE,
        content=chunk.text,
        sources=tuple(
            EvidenceSource(DOCUMENT_ID, link.document_version_id, link.source_span_id)
            for link in chunk.source_references
        ),
    )
    hit = EvidenceHit(
        candidate,
        (ChannelScore(RetrievalChannel.LEXICAL, 1),),
        1,
    )
    result = EvidenceResult(
        corpus_version=CORPUS_VERSION_ID,
        requested_limit=RetrievalLimit(5),
        hits=(hit,),
        status=ResultStatus.COMPLETED,
        failures=(),
    )

    with open_lexical_generation(tmp_path, built.manifest.generation_id) as generation:
        with capture_logs() as logs:
            outcome = reduce_retrieval_evidence(
                CORPUS_ID,
                cast(CorpusSnapshot, snapshot),
                generation,
                result,
                _WhitespaceTokeniser(),
                RetrievalConfig(evidence_token_budget=4),
                token_budget_override=1,
            )
        with pytest.raises(EvidenceReductionExecutionError) as mismatch:
            reduce_retrieval_evidence(
                CORPUS_ID,
                cast(CorpusSnapshot, snapshot),
                generation,
                result,
                _WhitespaceTokeniser(),
                RetrievalConfig(),
                expected_tokeniser_identity=replace(
                    _WhitespaceTokeniser().identity,
                    revision="different-revision",
                ),
            )
        snapshot.version = SimpleNamespace(
            corpus_version=type(CORPUS_VERSION_ID).generate()
        )
        with pytest.raises(EvidenceReductionExecutionError) as version_mismatch:
            reduce_retrieval_evidence(
                CORPUS_ID,
                cast(CorpusSnapshot, snapshot),
                generation,
                result,
                _WhitespaceTokeniser(),
                RetrievalConfig(),
            )

    assert len(outcome.items) == 1
    item = outcome.items[0]
    assert item.text == "Café"
    assert item.source_ranges[0].source_start == 0
    assert item.source_ranges[0].source_end == 5
    assert item.source_ranges[0].boundaries[-1].source_offset == 5
    assert (
        normalise_source_text(source_text[: item.source_ranges[0].source_end])
        == item.text
    )
    assert outcome.token_count == 1
    truncation = next(log for log in logs if log["event"] == "evidence_truncated")
    assert truncation["log_level"] == "warning"
    completion = next(
        log for log in logs if log["event"] == "evidence_reduction_completed"
    )
    assert completion["log_level"] == "info"
    assert completion["corpus_version"] == CORPUS_VERSION_ID.as_str()
    assert source_text not in repr(logs)
    assert (
        mismatch.value.category is EvidenceReductionFailureCategory.TOKENISER_MISMATCH
    )
    assert (
        version_mismatch.value.category
        is EvidenceReductionFailureCategory.CORPUS_VERSION_MISMATCH
    )
