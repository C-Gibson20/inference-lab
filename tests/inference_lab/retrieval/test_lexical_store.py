"""Tests for persistent immutable lexical generations."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    ChunkingPolicy,
    ChunkingRequest,
    ChunkSet,
    FakeChunkTokeniser,
    NormalisedText,
    ParsedDocument,
    SourceSpan,
    StructuralBlock,
    build_chunk_set,
    generate_source_span_id,
    map_chunk_set,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.retrieval import (
    LexicalError,
    LexicalFailureCategory,
    build_lexical_generation,
    open_lexical_generation,
)
from inference_lab.storage import ChunkSetSelectionRecord

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
CORPUS_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")


def _parsed_document(text: str, *, empty: bool = False) -> ParsedDocument:
    actual_document_id = DocumentId.generate()
    document_version_id = DocumentVersionId.generate()
    blocks: tuple[StructuralBlock, ...] = ()
    if not empty:
        span = SourceSpan(
            source_span_id=generate_source_span_id(
                document_version_id,
                parser_version=CURRENT_PARSER_VERSION,
                ordinal=0,
                kind="paragraph",
                start_offset=0,
                end_offset=len(text),
            ),
            document_version_id=document_version_id,
            start_offset=0,
            end_offset=len(text),
        )
        blocks = (StructuralBlock(kind="paragraph", text=text, source_span=span),)
    return ParsedDocument(
        document_id=actual_document_id,
        document_version_id=document_version_id,
        text=NormalisedText(source_text=text),
        blocks=blocks,
        pages=None,
        sections=None,
        paragraphs=blocks,
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=len(text),
        token_count=None,
    )


def _selection(
    text: str, *, empty: bool = False
) -> tuple[ChunkSet, ChunkSetSelectionRecord]:
    chunk_set = build_chunk_set(
        ChunkingRequest(
            corpus_id=CORPUS_ID,
            parsed_document=_parsed_document(text, empty=empty),
            tokeniser=FakeChunkTokeniser(),
            policy=ChunkingPolicy(min_tokens=1, max_tokens=512),
        )
    )
    return chunk_set, map_chunk_set(chunk_set)


def _source_manifest() -> tuple[BaselineChunkGenerationManifest, ChunkSet, ChunkSet]:
    first, first_selection = _selection("alpha cites 10.1234/ABC-7 and 10.1234/ABC-7")
    second, second_selection = _selection("beta has no identifier")
    entries = tuple(
        sorted(
            (
                BaselineChunkGenerationEntry(
                    membership=CorpusDocumentVersionMembership(
                        corpus_id=CORPUS_ID,
                        document_id=selection.document_id,
                        document_version_id=selection.document_version_id,
                    ),
                    selection=selection,
                )
                for selection in (first_selection, second_selection)
            ),
            key=lambda entry: entry.key(),
        )
    )
    manifest = BaselineChunkGenerationManifest(
        corpus_id=CORPUS_ID,
        corpus_version=CORPUS_VERSION_ID,
        configuration_identity=first.configuration_identity,
        generation_id="baseline-test-generation",
        entries=entries,
    )
    return manifest, first, second


def test_full_build_publishes_restartable_generation_with_source_links(
    tmp_path: Path,
) -> None:
    source_manifest, first, second = _source_manifest()

    outcome = build_lexical_generation(tmp_path, source_manifest)
    generation_dir = tmp_path / "lexical" / outcome.manifest.generation_id

    assert outcome.statistics.members_processed == 2
    assert outcome.statistics.chunks_processed == 2
    assert outcome.statistics.inserted_chunks == 2
    assert generation_dir.is_dir()
    assert {path.name for path in generation_dir.iterdir()} == {
        "index.sqlite3",
        "manifest.json",
    }

    with open_lexical_generation(
        tmp_path, outcome.manifest.generation_id
    ) as generation:
        chunks = generation.iter_chunks()
        assert tuple(chunk.content for chunk in chunks) == tuple(
            chunk.text
            for chunk in sorted(
                first.chunks + second.chunks, key=lambda item: item.document_id.as_str()
            )
        )
        assert (
            chunks[0].source_links[0].document_version_id
            == chunks[0].document_version_id
        )
        identifier_matches = generation.lookup_identifier("10.1234/ABC-7")
        assert len(identifier_matches) == 1
        assert identifier_matches[0][1] == 2

    reopened = open_lexical_generation(tmp_path, outcome.manifest.generation_id)
    reopened.close()


def test_zero_chunk_member_is_retained_and_exact_replay_reuses_database(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = _source_manifest()
    empty_set, empty_selection = _selection("", empty=True)
    entries = tuple(
        sorted(
            source_manifest.entries
            + (
                BaselineChunkGenerationEntry(
                    membership=CorpusDocumentVersionMembership(
                        corpus_id=CORPUS_ID,
                        document_id=empty_selection.document_id,
                        document_version_id=empty_selection.document_version_id,
                    ),
                    selection=empty_selection,
                ),
            ),
            key=lambda entry: entry.key(),
        )
    )
    manifest = BaselineChunkGenerationManifest(
        corpus_id=CORPUS_ID,
        corpus_version=CORPUS_VERSION_ID,
        configuration_identity=source_manifest.configuration_identity,
        generation_id="baseline-test-generation",
        entries=entries,
    )

    first = build_lexical_generation(tmp_path, manifest)
    database = tmp_path / "lexical" / first.manifest.generation_id / "index.sqlite3"
    before = hashlib.sha256(database.read_bytes()).digest()
    second = build_lexical_generation(tmp_path, manifest)
    after = hashlib.sha256(database.read_bytes()).digest()

    assert empty_set.chunks == ()
    assert first.manifest.member_count == 3
    assert second.statistics.database_reused is True
    assert second.statistics.reused_chunks == first.manifest.chunk_count
    assert before == after


def test_corrupt_database_is_rejected_without_exposing_sqlite_details(
    tmp_path: Path,
) -> None:
    source_manifest, _, _ = _source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    database = tmp_path / "lexical" / outcome.manifest.generation_id / "index.sqlite3"
    with database.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(LexicalError) as error:
        open_lexical_generation(tmp_path, outcome.manifest.generation_id)

    assert error.value.category is LexicalFailureCategory.CORRUPT_GENERATION
    assert str(error.value) == LexicalFailureCategory.CORRUPT_GENERATION.value
    assert str(tmp_path) not in str(error.value)


def test_published_database_is_read_only(tmp_path: Path) -> None:
    source_manifest, _, _ = _source_manifest()
    outcome = build_lexical_generation(tmp_path, source_manifest)
    database = tmp_path / "lexical" / outcome.manifest.generation_id / "index.sqlite3"
    connection = sqlite3.connect(f"file:{database.absolute()}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO members VALUES ('x', 'y', 0)")
    finally:
        connection.close()


def test_generation_id_cannot_escape_the_lexical_directory(tmp_path: Path) -> None:
    with pytest.raises(LexicalError) as error:
        open_lexical_generation(tmp_path, "../outside")

    assert error.value.category is LexicalFailureCategory.INVALID_REQUEST
