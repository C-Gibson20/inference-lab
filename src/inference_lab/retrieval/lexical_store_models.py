"""Internal models and layout constants for lexical generation storage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage import ComponentKind

LEXICAL_DIRECTORY_NAME: Final[str] = "lexical"
LEXICAL_DATABASE_NAME: Final[str] = "index.sqlite3"
LEXICAL_MANIFEST_NAME: Final[str] = "manifest.json"
LEXICAL_DATABASE_USER_VERSION: Final[int] = 2
GENERATION_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
LEXICAL_COMPONENT_KIND: Final[ComponentKind] = ComponentKind("lexical-index")


@dataclass(frozen=True, slots=True)
class LexicalSourceLink:
    """One source span retained alongside an indexed chunk."""

    source_span_id: SourceSpanId
    document_version_id: DocumentVersionId
    block_ordinal: int
    block_kind: str
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class LexicalIndexedChunk:
    """One immutable chunk row reconstructed from a lexical generation."""

    record_id: RecordId
    document_id: DocumentId
    document_version_id: DocumentVersionId
    ordinal: int
    content: str
    token_count: int
    source_links: tuple[LexicalSourceLink, ...]


@dataclass(frozen=True, slots=True)
class LexicalMemberSelection:
    """Identity and payload digest for one indexed corpus member."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    selection_digest: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class BuildChunk:
    """Validated chunk data prepared before entering the SQLite transaction."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    record_id: RecordId
    ordinal: int
    content: str
    token_count: int
    source_links: tuple[LexicalSourceLink, ...]
    identifiers: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class BuildMember:
    """Validated source selection identity used by the SQLite writer."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    selection_digest: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class DiffStatistics:
    """Chunk-level changes applied to a cloned parent database."""

    reused_chunks: int
    inserted_chunks: int
    replaced_chunks: int
    removed_chunks: int
    identifiers_emitted: int
