"""Immutable SQLite storage for one corpus-scoped lexical generation."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from inference_lab.ingest import (
    BaselineChunkGenerationManifest,
    deserialise_chunk_set_selection,
)
from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.lexical import (
    LexicalError,
    LexicalFailureCategory,
    LexicalIndexBuildOutcome,
    LexicalIndexBuildStatistics,
    LexicalIndexConfiguration,
    LexicalIndexManifest,
    derive_lexical_generation_id,
    extract_identifiers,
)
from inference_lab.retrieval.lexical_serialisation import (
    deserialise_lexical_manifest,
    serialise_lexical_manifest,
)

LEXICAL_DIRECTORY_NAME: Final[str] = "lexical"
LEXICAL_DATABASE_NAME: Final[str] = "index.sqlite3"
LEXICAL_MANIFEST_NAME: Final[str] = "manifest.json"
_LEXICAL_DATABASE_USER_VERSION: Final[int] = 1
_GENERATION_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


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
class _BuildChunk:
    """Validated chunk data prepared before entering the SQLite transaction."""

    document_id: DocumentId
    document_version_id: DocumentVersionId
    record_id: RecordId
    ordinal: int
    content: str
    token_count: int
    source_links: tuple[LexicalSourceLink, ...]
    identifiers: tuple[tuple[str, int], ...]


def build_lexical_generation(
    index_dir: Path,
    source_manifest: BaselineChunkGenerationManifest,
    *,
    configuration: LexicalIndexConfiguration | None = None,
) -> LexicalIndexBuildOutcome:
    """Build or exactly reuse one immutable full lexical generation."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(source_manifest, BaselineChunkGenerationManifest):
        raise TypeError("source_manifest must be a BaselineChunkGenerationManifest.")
    if configuration is None:
        configuration = LexicalIndexConfiguration()
    if not isinstance(configuration, LexicalIndexConfiguration):
        raise TypeError("configuration must be a LexicalIndexConfiguration.")

    generation_id = derive_lexical_generation_id(
        source_manifest.corpus_id,
        source_manifest.corpus_version,
        configuration,
        source_manifest.source_manifest_digest,
    )
    lexical_dir = index_dir / LEXICAL_DIRECTORY_NAME
    target_dir = lexical_dir / generation_id
    if target_dir.exists():
        return _reuse_existing_generation(
            target_dir,
            source_manifest=source_manifest,
            configuration=configuration,
            generation_id=generation_id,
        )

    chunks, member_count = _prepare_chunks(source_manifest)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".lexical-staging-", dir=str(_ensure_directory(lexical_dir))
        )
    )
    try:
        database_path = staging_dir / LEXICAL_DATABASE_NAME
        _write_database(
            database_path,
            source_manifest=source_manifest,
            configuration=configuration,
            chunks=chunks,
            member_count=member_count,
        )
        manifest = _build_manifest(
            database_path,
            source_manifest=source_manifest,
            configuration=configuration,
            generation_id=generation_id,
            member_count=member_count,
            chunks=chunks,
        )
        (staging_dir / LEXICAL_MANIFEST_NAME).write_bytes(
            serialise_lexical_manifest(manifest)
        )
        staged_generation = _open_generation_path(
            staging_dir,
            expected_generation_id=generation_id,
        )
        staged_generation.close()
        try:
            os.rename(staging_dir, target_dir)
        except FileExistsError:
            return _reuse_existing_generation(
                target_dir,
                source_manifest=source_manifest,
                configuration=configuration,
                generation_id=generation_id,
            )
        return LexicalIndexBuildOutcome(
            manifest=manifest,
            statistics=LexicalIndexBuildStatistics(
                members_processed=member_count,
                chunks_processed=len(chunks),
                identifiers_emitted=sum(len(chunk.identifiers) for chunk in chunks),
                inserted_chunks=len(chunks),
            ),
        )
    except LexicalError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def open_lexical_generation(index_dir: Path, generation_id: str) -> LexicalGeneration:
    """Open one published lexical generation in strict read-only mode."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(generation_id, str):
        raise TypeError("generation_id must be a string.")
    if _GENERATION_ID_PATTERN.fullmatch(generation_id) is None:
        raise LexicalError(LexicalFailureCategory.INVALID_REQUEST)
    return _open_generation_path(index_dir / LEXICAL_DIRECTORY_NAME / generation_id)


class LexicalGeneration:
    """Read-only handle for a validated immutable lexical generation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        manifest: LexicalIndexManifest,
    ) -> None:
        self._connection = connection
        self.manifest = manifest
        self._closed = False

    def __enter__(self) -> LexicalGeneration:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the read-only database handle."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def iter_chunks(self) -> tuple[LexicalIndexedChunk, ...]:
        """Return all indexed chunks in deterministic document/ordinal order."""
        self._require_open()
        try:
            rows = self._connection.execute(
                """
                SELECT c.record_id, c.document_id, c.document_version_id,
                       c.ordinal, c.content, c.token_count,
                       s.source_span_id, s.document_version_id AS source_version_id,
                       s.block_ordinal, s.block_kind, s.source_start, s.source_end
                FROM chunks AS c
                LEFT JOIN source_links AS s ON s.record_id = c.record_id
                ORDER BY c.document_id, c.ordinal,
                         s.block_ordinal, s.source_span_id
                """
            ).fetchall()
            return _rows_to_chunks(rows)
        except (sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
            raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION) from None

    def get_chunk(self, record_id: RecordId) -> LexicalIndexedChunk | None:
        """Return one indexed chunk by its immutable record identity."""
        if not isinstance(record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        self._require_open()
        try:
            rows = self._connection.execute(
                """
                SELECT c.record_id, c.document_id, c.document_version_id,
                       c.ordinal, c.content, c.token_count,
                       s.source_span_id, s.document_version_id AS source_version_id,
                       s.block_ordinal, s.block_kind, s.source_start, s.source_end
                FROM chunks AS c
                LEFT JOIN source_links AS s ON s.record_id = c.record_id
                WHERE c.record_id = ?
                ORDER BY s.block_ordinal, s.source_span_id
                """,
                (record_id.as_str(),),
            ).fetchall()
            chunks = _rows_to_chunks(rows)
            return chunks[0] if chunks else None
        except (sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
            raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION) from None

    def lookup_identifier(
        self, identifier: str
    ) -> tuple[tuple[LexicalIndexedChunk, int], ...]:
        """Return exact binary identifier matches with occurrence counts."""
        if not isinstance(identifier, str) or not identifier:
            raise TypeError("identifier must be a non-empty string.")
        self._require_open()
        try:
            rows = self._connection.execute(
                """
                SELECT c.record_id, c.document_id, c.document_version_id,
                       c.ordinal, c.content, c.token_count,
                       s.source_span_id, s.document_version_id AS source_version_id,
                       s.block_ordinal, s.block_kind, s.source_start, s.source_end,
                       i.occurrence_count
                FROM identifiers AS i
                JOIN chunks AS c ON c.record_id = i.record_id
                LEFT JOIN source_links AS s ON s.record_id = c.record_id
                WHERE i.lexeme = ?
                ORDER BY c.document_id, c.ordinal,
                         s.block_ordinal, s.source_span_id
                """,
                (identifier,),
            ).fetchall()
            occurrence_by_record = {
                str(row["record_id"]): int(row["occurrence_count"]) for row in rows
            }
            grouped: list[tuple[LexicalIndexedChunk, int]] = []
            for chunk in _rows_to_chunks(rows):
                occurrence = occurrence_by_record[chunk.record_id.as_str()]
                if not grouped or grouped[-1][0].record_id != chunk.record_id:
                    grouped.append((chunk, occurrence))
            return tuple(grouped)
        except (sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
            raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION) from None

    def _require_open(self) -> None:
        if self._closed:
            raise LexicalError(LexicalFailureCategory.GENERATION_UNAVAILABLE)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_chunks(
    source_manifest: BaselineChunkGenerationManifest,
) -> tuple[tuple[_BuildChunk, ...], int]:
    prepared: list[_BuildChunk] = []
    for entry in source_manifest.entries:
        try:
            chunk_set = deserialise_chunk_set_selection(
                entry.selection.canonical_payload,
                corpus_id=source_manifest.corpus_id,
            )
        except (TypeError, ValueError):
            raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION) from None
        if (
            chunk_set.document_id != entry.membership.document_id
            or chunk_set.document_version_id != entry.membership.document_version_id
            or chunk_set.configuration_identity
            != source_manifest.configuration_identity
        ):
            raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
        for chunk in chunk_set.chunks:
            identifier_counts = Counter(extract_identifiers(chunk.text))
            prepared.append(
                _BuildChunk(
                    document_id=chunk.document_id,
                    document_version_id=chunk.document_version_id,
                    record_id=chunk.record_id,
                    ordinal=chunk.ordinal,
                    content=chunk.text,
                    token_count=chunk.token_count,
                    source_links=tuple(
                        LexicalSourceLink(
                            source_span_id=reference.source_span_id,
                            document_version_id=reference.document_version_id,
                            block_ordinal=reference.block_ordinal,
                            block_kind=reference.block_kind,
                            source_start=reference.source_start,
                            source_end=reference.source_end,
                        )
                        for reference in chunk.source_references
                    ),
                    identifiers=tuple(sorted(identifier_counts.items())),
                )
            )
    return tuple(prepared), len(source_manifest.entries)


def _write_database(
    database_path: Path,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
    chunks: tuple[_BuildChunk, ...],
    member_count: int,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        tokenizer = configuration.fts_tokenizer.replace("'", "''")
        connection.executescript(
            f"""
            PRAGMA user_version = {_LEXICAL_DATABASE_USER_VERSION};
            CREATE TABLE generation_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE members (
                document_id TEXT PRIMARY KEY,
                document_version_id TEXT NOT NULL UNIQUE,
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0)
            ) WITHOUT ROWID;
            CREATE TABLE chunks (
                record_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES members(document_id),
                document_version_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL CHECK (token_count > 0),
                UNIQUE(document_id, ordinal)
            ) WITHOUT ROWID;
            CREATE TABLE source_links (
                record_id TEXT NOT NULL REFERENCES chunks(record_id) ON DELETE CASCADE,
                source_span_id TEXT NOT NULL,
                document_version_id TEXT NOT NULL,
                block_ordinal INTEGER NOT NULL CHECK (block_ordinal >= 0),
                block_kind TEXT NOT NULL,
                source_start INTEGER NOT NULL CHECK (source_start >= 0),
                source_end INTEGER NOT NULL CHECK (source_end > source_start),
                PRIMARY KEY(record_id, source_span_id)
            ) WITHOUT ROWID;
            CREATE TABLE identifiers (
                record_id TEXT NOT NULL REFERENCES chunks(record_id) ON DELETE CASCADE,
                lexeme TEXT COLLATE BINARY NOT NULL,
                occurrence_count INTEGER NOT NULL CHECK (occurrence_count > 0),
                PRIMARY KEY(record_id, lexeme)
            ) WITHOUT ROWID;
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                record_id UNINDEXED,
                content,
                tokenize='{tokenizer}'
            );
            """
        )
        metadata = {
            "corpus_id": source_manifest.corpus_id.as_str(),
            "corpus_version": source_manifest.corpus_version.as_str(),
            "generation_id": derive_lexical_generation_id(
                source_manifest.corpus_id,
                source_manifest.corpus_version,
                configuration,
                source_manifest.source_manifest_digest,
            ),
            "configuration_digest": configuration.configuration_identity.digest,
            "source_manifest_digest": source_manifest.source_manifest_digest,
            "member_count": str(member_count),
            "chunk_count": str(len(chunks)),
            "identifier_count": str(sum(len(chunk.identifiers) for chunk in chunks)),
        }
        connection.executemany(
            "INSERT INTO generation_metadata(key, value) VALUES (?, ?)",
            tuple(metadata.items()),
        )
        connection.executemany(
            "INSERT INTO members(document_id, document_version_id, chunk_count) "
            "VALUES (?, ?, ?)",
            tuple(
                (
                    entry.membership.document_id.as_str(),
                    entry.membership.document_version_id.as_str(),
                    sum(
                        chunk.document_id == entry.membership.document_id
                        for chunk in chunks
                    ),
                )
                for entry in source_manifest.entries
            ),
        )
        connection.executemany(
            "INSERT INTO chunks(record_id, document_id, document_version_id, ordinal, "
            "content, token_count) VALUES (?, ?, ?, ?, ?, ?)",
            tuple(
                (
                    chunk.record_id.as_str(),
                    chunk.document_id.as_str(),
                    chunk.document_version_id.as_str(),
                    chunk.ordinal,
                    chunk.content,
                    chunk.token_count,
                )
                for chunk in chunks
            ),
        )
        connection.executemany(
            "INSERT INTO source_links(record_id, source_span_id, document_version_id, "
            "block_ordinal, block_kind, source_start, source_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            tuple(
                (
                    chunk.record_id.as_str(),
                    link.source_span_id.as_str(),
                    link.document_version_id.as_str(),
                    link.block_ordinal,
                    link.block_kind,
                    link.source_start,
                    link.source_end,
                )
                for chunk in chunks
                for link in chunk.source_links
            ),
        )
        connection.executemany(
            "INSERT INTO identifiers(record_id, lexeme, occurrence_count) VALUES (?, ?, ?)",
            tuple(
                (chunk.record_id.as_str(), lexeme, count)
                for chunk in chunks
                for lexeme, count in chunk.identifiers
            ),
        )
        connection.executemany(
            "INSERT INTO chunk_fts(record_id, content) VALUES (?, ?)",
            tuple((chunk.record_id.as_str(), chunk.content) for chunk in chunks),
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()


def _build_manifest(
    database_path: Path,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
    generation_id: str,
    member_count: int,
    chunks: tuple[_BuildChunk, ...],
) -> LexicalIndexManifest:
    return LexicalIndexManifest(
        corpus_id=source_manifest.corpus_id,
        corpus_version=source_manifest.corpus_version,
        generation_id=generation_id,
        configuration=configuration,
        configuration_digest=configuration.configuration_identity.digest,
        source_manifest_digest=source_manifest.source_manifest_digest,
        database_checksum=_sha256_file(database_path),
        member_count=member_count,
        chunk_count=len(chunks),
        identifier_count=sum(len(chunk.identifiers) for chunk in chunks),
    )


def _reuse_existing_generation(
    target_dir: Path,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
    generation_id: str,
) -> LexicalIndexBuildOutcome:
    generation = _open_generation_path(target_dir)
    manifest = generation.manifest
    generation.close()
    if (
        manifest.corpus_id != source_manifest.corpus_id
        or manifest.corpus_version != source_manifest.corpus_version
        or manifest.generation_id != generation_id
        or manifest.configuration_digest != configuration.configuration_identity.digest
        or manifest.source_manifest_digest != source_manifest.source_manifest_digest
    ):
        raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
    return LexicalIndexBuildOutcome(
        manifest=manifest,
        statistics=LexicalIndexBuildStatistics(
            members_processed=manifest.member_count,
            chunks_processed=manifest.chunk_count,
            identifiers_emitted=manifest.identifier_count,
            reused_chunks=manifest.chunk_count,
            database_reused=True,
        ),
    )


def _open_generation_path(
    generation_dir: Path,
    *,
    expected_generation_id: str | None = None,
) -> LexicalGeneration:
    if not generation_dir.is_dir():
        raise LexicalError(LexicalFailureCategory.GENERATION_UNAVAILABLE)
    try:
        entries = {entry.name for entry in generation_dir.iterdir()}
        if entries != {LEXICAL_DATABASE_NAME, LEXICAL_MANIFEST_NAME}:
            raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION)
        manifest = deserialise_lexical_manifest(
            (generation_dir / LEXICAL_MANIFEST_NAME).read_bytes()
        )
        expected_id = (
            generation_dir.name
            if expected_generation_id is None
            else expected_generation_id
        )
        if manifest.generation_id != expected_id:
            raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
        database_path = generation_dir / LEXICAL_DATABASE_NAME
        if _sha256_file(database_path) != manifest.database_checksum:
            raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION)
        connection = sqlite3.connect(
            f"file:{database_path.absolute().as_uri()[len('file:') :]}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        _validate_database(connection, manifest)
        return LexicalGeneration(connection, manifest)
    except LexicalError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
        raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION) from None


def _validate_database(
    connection: sqlite3.Connection,
    manifest: LexicalIndexManifest,
) -> None:
    row = connection.execute("PRAGMA integrity_check").fetchone()
    if row is None or row[0] != "ok":
        raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION)
    version = connection.execute("PRAGMA user_version").fetchone()
    if version is None or version[0] != _LEXICAL_DATABASE_USER_VERSION:
        raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
    metadata = dict(
        connection.execute("SELECT key, value FROM generation_metadata").fetchall()
    )
    expected_metadata = {
        "corpus_id": manifest.corpus_id.as_str(),
        "corpus_version": manifest.corpus_version.as_str(),
        "generation_id": manifest.generation_id,
        "configuration_digest": manifest.configuration_digest,
        "source_manifest_digest": manifest.source_manifest_digest,
        "member_count": str(manifest.member_count),
        "chunk_count": str(manifest.chunk_count),
        "identifier_count": str(manifest.identifier_count),
    }
    if metadata != expected_metadata:
        raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION)
    counts = {
        "member_count": connection.execute("SELECT COUNT(*) FROM members").fetchone()[
            0
        ],
        "chunk_count": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "identifier_count": connection.execute(
            "SELECT COUNT(*) FROM identifiers"
        ).fetchone()[0],
        "fts_count": connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0],
    }
    if counts != {
        "member_count": manifest.member_count,
        "chunk_count": manifest.chunk_count,
        "identifier_count": manifest.identifier_count,
        "fts_count": manifest.chunk_count,
    }:
        raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION)


def _rows_to_chunks(rows: list[sqlite3.Row]) -> tuple[LexicalIndexedChunk, ...]:
    if not rows:
        return ()
    chunks: list[LexicalIndexedChunk] = []
    current_record_id: str | None = None
    previous_row: sqlite3.Row | None = None
    links: list[LexicalSourceLink] = []
    for row in rows:
        record_id = str(row["record_id"])
        if current_record_id is not None and record_id != current_record_id:
            if previous_row is None:
                raise ValueError("chunk row grouping lost its first row.")
            chunks.append(_row_to_chunk(previous_row, tuple(links)))
            links = []
        current_record_id = record_id
        previous_row = row
        if row["source_span_id"] is not None:
            links.append(
                LexicalSourceLink(
                    source_span_id=SourceSpanId.from_str(row["source_span_id"]),
                    document_version_id=DocumentVersionId.from_str(
                        row["source_version_id"]
                    ),
                    block_ordinal=row["block_ordinal"],
                    block_kind=row["block_kind"],
                    source_start=row["source_start"],
                    source_end=row["source_end"],
                )
            )
    if previous_row is None:
        raise ValueError("chunk row grouping has no first row.")
    chunks.append(_row_to_chunk(previous_row, tuple(links)))
    return tuple(chunks)


def _row_to_chunk(
    row: sqlite3.Row,
    links: tuple[LexicalSourceLink, ...],
) -> LexicalIndexedChunk:
    return LexicalIndexedChunk(
        record_id=RecordId.from_str(row["record_id"]),
        document_id=DocumentId.from_str(row["document_id"]),
        document_version_id=DocumentVersionId.from_str(row["document_version_id"]),
        ordinal=row["ordinal"],
        content=row["content"],
        token_count=row["token_count"],
        source_links=links,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
