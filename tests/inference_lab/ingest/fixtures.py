"""Reusable deterministic fixtures for ingest tests."""

from pathlib import Path

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    NormalisedText,
    ParsedDocument,
    ParserInput,
    SourceSpan,
    StructuralBlock,
    generate_source_span_id,
)
from inference_lab.provenance import DocumentId, DocumentVersionId
from inference_lab.settings import (
    IngestConfig,
    ModelPaths,
    RuntimeConfig,
    Settings,
    StorageConfig,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "fixtures" / "ingest" / "text_native"
)

FIXTURE_DOCUMENT_ID = DocumentId.from_str("00000000-0000-4000-8000-000000000001")
FIXTURE_DOCUMENT_VERSION_ID = DocumentVersionId.from_str(
    "00000000-0000-4000-8000-000000000002"
)


def fixture_bytes(name: str) -> bytes:
    """Read one named fixture, decoding ``.bin`` fixtures from ASCII hex."""
    candidate = (FIXTURE_ROOT / name).resolve()
    root = FIXTURE_ROOT.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("fixture_name_must_remain_inside_fixture_root") from error

    if not candidate.is_file():
        raise FileNotFoundError("fixture_not_found")

    content = candidate.read_bytes()
    if candidate.suffix == ".bin":
        try:
            return bytes.fromhex(content.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("fixture_binary_must_be_ascii_hex") from error
    return content


def fixture_source(name: str) -> ParserInput:
    """Build a parser input with stable test provenance identifiers."""
    return ParserInput(
        document_id=FIXTURE_DOCUMENT_ID,
        document_version_id=FIXTURE_DOCUMENT_VERSION_ID,
        content=fixture_bytes(name),
        media_type="text/plain",
        encoding="utf-8",
    )


def fixture_settings(
    *,
    max_source_bytes: int = 50 * 1024 * 1024,
    max_concurrent_documents: int = 1,
) -> Settings:
    """Return validated settings for deterministic local fixture tests."""
    return Settings(
        models=ModelPaths(
            embedder_path=Path("fixture/embedder"),
            reranker_path=Path("fixture/reranker"),
            generator_path=Path("fixture/generator"),
        ),
        runtime=RuntimeConfig(vram_budget_gb=24),
        storage=StorageConfig(
            index_dir=Path("fixture/index"),
            metadata_db_path=Path("fixture/metadata.db"),
            source_dir=Path("fixture/source-data"),
        ),
        ingest=IngestConfig(
            max_source_bytes=max_source_bytes,
            max_concurrent_documents=max_concurrent_documents,
        ),
    )


def fixture_parsed_document(*, include_paragraph: bool = True) -> ParsedDocument:
    """Build a valid parsed document with optional paragraph structure."""
    document_version_id = DocumentVersionId.generate()
    span = SourceSpan(
        source_span_id=generate_source_span_id(
            document_version_id,
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        ),
        document_version_id=document_version_id,
        start_offset=0,
        end_offset=1,
    )
    block = StructuralBlock(kind="paragraph", text="A", source_span=span)
    blocks = (block,) if include_paragraph else ()

    return ParsedDocument(
        document_id=DocumentId.generate(),
        document_version_id=document_version_id,
        text=NormalisedText(source_text="A"),
        blocks=blocks,
        pages=None,
        sections=None,
        paragraphs=blocks,
        page_layouts=(),
        language=None,
        page_count=None,
        character_count=1,
        token_count=None,
    )
