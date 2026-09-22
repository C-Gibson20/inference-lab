"""Reusable published-lexical-generation fixtures for cross-domain tests."""

from pathlib import Path

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
    build_and_publish_baseline_lexical_generation,
    build_chunk_set,
    generate_source_span_id,
    map_chunk_set,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
)
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    CommitCorpusUpdateOutcome,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
CORPUS_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
SECOND_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000003")


def parsed_document(
    text: str,
    *,
    empty: bool = False,
    document_id: DocumentId | None = None,
    document_version_id: DocumentVersionId | None = None,
) -> ParsedDocument:
    """Build a source-spanned parsed document for lexical fixtures."""
    actual_document_id = document_id or DocumentId.generate()
    actual_document_version_id = document_version_id or DocumentVersionId.generate()
    blocks: tuple[StructuralBlock, ...] = ()
    if not empty:
        span = SourceSpan(
            source_span_id=generate_source_span_id(
                actual_document_version_id,
                parser_version=CURRENT_PARSER_VERSION,
                ordinal=0,
                kind="paragraph",
                start_offset=0,
                end_offset=len(text),
            ),
            document_version_id=actual_document_version_id,
            start_offset=0,
            end_offset=len(text),
        )
        blocks = (StructuralBlock(kind="paragraph", text=text, source_span=span),)
    return ParsedDocument(
        document_id=actual_document_id,
        document_version_id=actual_document_version_id,
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


def selection(
    text: str,
    *,
    empty: bool = False,
    document_id: DocumentId | None = None,
    document_version_id: DocumentVersionId | None = None,
    policy: ChunkingPolicy | None = None,
) -> tuple[ChunkSet, ChunkSetSelectionRecord]:
    """Build a chunk set and its durable selection record."""
    chunk_set = build_chunk_set(
        ChunkingRequest(
            corpus_id=CORPUS_ID,
            parsed_document=parsed_document(
                text,
                empty=empty,
                document_id=document_id,
                document_version_id=document_version_id,
            ),
            tokeniser=FakeChunkTokeniser(),
            policy=policy or ChunkingPolicy(min_tokens=1, max_tokens=512),
        )
    )
    return chunk_set, map_chunk_set(chunk_set)


def manifest(
    selections: tuple[ChunkSetSelectionRecord, ...],
    *,
    corpus_version: CorpusVersionId = CORPUS_VERSION_ID,
) -> BaselineChunkGenerationManifest:
    """Describe a complete canonical chunk generation for selected members."""
    entries = tuple(
        sorted(
            tuple(
                BaselineChunkGenerationEntry(
                    membership=CorpusDocumentVersionMembership(
                        corpus_id=CORPUS_ID,
                        document_id=item.document_id,
                        document_version_id=item.document_version_id,
                    ),
                    selection=item,
                )
                for item in selections
            ),
            key=lambda entry: entry.key(),
        )
    )
    return BaselineChunkGenerationManifest(
        corpus_id=CORPUS_ID,
        corpus_version=corpus_version,
        configuration_identity=selections[0].configuration_identity,
        generation_id="baseline-test-generation",
        entries=entries,
    )


def source_manifest() -> tuple[BaselineChunkGenerationManifest, ChunkSet, ChunkSet]:
    """Return a two-document fixture with terms and repeated identifier text."""
    first, first_selection = selection("alpha cites 10.1234/ABC-7 and 10.1234/ABC-7")
    second, second_selection = selection("beta has no identifier")
    entries = tuple(
        sorted(
            (
                BaselineChunkGenerationEntry(
                    membership=CorpusDocumentVersionMembership(
                        corpus_id=CORPUS_ID,
                        document_id=item.document_id,
                        document_version_id=item.document_version_id,
                    ),
                    selection=item,
                )
                for item in (first_selection, second_selection)
            ),
            key=lambda entry: entry.key(),
        )
    )
    fixture_manifest = BaselineChunkGenerationManifest(
        corpus_id=CORPUS_ID,
        corpus_version=CORPUS_VERSION_ID,
        configuration_identity=first.configuration_identity,
        generation_id="baseline-test-generation",
        entries=entries,
    )
    return fixture_manifest, first, second


def privacy_source_manifest() -> BaselineChunkGenerationManifest:
    """Return a fixture with distinctive private query and payload markers."""
    _, first_selection = selection(
        "private lexical chunk marker with 10.1234/PRIVATE-7"
    )
    _, second_selection = selection("private secondary chunk marker")
    return manifest((first_selection, second_selection))


def seed_metadata_repository(
    repository: SQLiteMetadataRepository,
    source_manifest: BaselineChunkGenerationManifest,
    *,
    seed_catalogue: bool = True,
) -> None:
    """Persist the catalogue and exact chunk selections used by publication."""
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(source_manifest.corpus_id, "test corpus"))
        for ordinal, entry in enumerate(source_manifest.entries, start=1):
            if seed_catalogue:
                transaction.add_document(DocumentIdentity(entry.membership.document_id))
                transaction.add_document_version(
                    DocumentVersion(
                        entry.membership.document_version_id,
                        entry.membership.document_id,
                        ContentHash(HashAlgorithm.SHA256, str(ordinal).zfill(64)),
                    )
                )
            transaction.add_corpus_document_version_membership(entry.membership)
            transaction.write_chunk_set(
                ChunkSetWriteRequest(
                    corpus_id=source_manifest.corpus_id,
                    selection=entry.selection,
                )
            )


def publish_query_fixture(
    repository: SQLiteMetadataRepository,
    index_dir: Path,
    source_manifest: BaselineChunkGenerationManifest,
) -> CommitCorpusUpdateOutcome:
    """Publish one complete lexical fixture for query acceptance tests."""
    seed_metadata_repository(repository, source_manifest)
    outcome = build_and_publish_baseline_lexical_generation(
        repository,
        index_dir,
        source_manifest,
        expected_corpus_version=None,
    )
    assert isinstance(outcome, CommitCorpusUpdateOutcome)
    return outcome
