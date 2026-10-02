"""Read-only access to validated immutable lexical generations."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.lexical import (
    LEXICAL_CONTRACT_VERSION,
    LexicalError,
    LexicalFailureCategory,
    LexicalIndexManifest,
    LexicalQuery,
    expose_bm25_score,
)
from inference_lab.retrieval.lexical_serialisation import (
    deserialise_lexical_manifest,
)
from inference_lab.retrieval.lexical_store_models import (
    GENERATION_ID_PATTERN,
    LEXICAL_DATABASE_NAME,
    LEXICAL_DATABASE_USER_VERSION,
    LEXICAL_DIRECTORY_NAME,
    LEXICAL_MANIFEST_NAME,
    LexicalIndexedChunk,
    LexicalMemberSelection,
    LexicalSourceLink,
)


def open_lexical_generation(index_dir: Path, generation_id: str) -> LexicalGeneration:
    """Open one published lexical generation in strict read-only mode."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(generation_id, str):
        raise TypeError("generation_id must be a string.")
    if GENERATION_ID_PATTERN.fullmatch(generation_id) is None:
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
                       c.source_start AS chunk_source_start,
                       c.source_end AS chunk_source_end,
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

    def member_selections(self) -> tuple[LexicalMemberSelection, ...]:
        """Return complete member selection identities in canonical order."""
        self._require_open()
        try:
            rows = self._connection.execute(
                """
                SELECT document_id, document_version_id, selection_digest, chunk_count
                FROM members
                ORDER BY document_id
                """
            ).fetchall()
            return tuple(
                LexicalMemberSelection(
                    document_id=DocumentId.from_str(row["document_id"]),
                    document_version_id=DocumentVersionId.from_str(
                        row["document_version_id"]
                    ),
                    selection_digest=row["selection_digest"],
                    chunk_count=row["chunk_count"],
                )
                for row in rows
            )
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
                       c.source_start AS chunk_source_start,
                       c.source_end AS chunk_source_end,
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
                       c.source_start AS chunk_source_start,
                       c.source_end AS chunk_source_end,
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

    def search(
        self,
        request: LexicalQuery,
    ) -> tuple[tuple[LexicalIndexedChunk, int | float], ...]:
        """Return deterministic top-k matches from this immutable generation."""
        if not isinstance(request, LexicalQuery):
            raise TypeError("request must be a LexicalQuery.")
        self._require_open()
        limit = request.query.limit.value
        try:
            return tuple(
                islice(
                    self.iter_ranked_matches(request, page_size=limit),
                    limit,
                )
            )
        except LexicalError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
            raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None

    def iter_ranked_matches(
        self,
        request: LexicalQuery,
        *,
        page_size: int = 64,
    ) -> Iterator[tuple[LexicalIndexedChunk, int | float]]:
        """Iterate deterministic ranked pages without applying the request limit.

        The iterator is bounded to one SQLite page at a time.  Callers that need
        to apply an additional predicate must do so while consuming the ranked
        stream, before imposing their final result limit.
        """
        if not isinstance(request, LexicalQuery):
            raise TypeError("request must be a LexicalQuery.")
        if type(page_size) is not int:
            raise TypeError("page_size must be an int.")
        if page_size <= 0:
            raise ValueError("page_size must be positive.")
        self._require_open()
        return self._iter_ranked_matches(request, page_size=page_size)

    def _iter_ranked_matches(
        self,
        request: LexicalQuery,
        *,
        page_size: int,
    ) -> Iterator[tuple[LexicalIndexedChunk, int | float]]:
        offset = 0
        while True:
            page = self._ranked_page(request, limit=page_size, offset=offset)
            yield from page
            if len(page) < page_size:
                return
            offset += len(page)

    def _ranked_page(
        self,
        request: LexicalQuery,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[LexicalIndexedChunk, int | float], ...]:
        try:
            if request.identifier is not None:
                rows = self._connection.execute(
                    """
                    WITH ranked AS (
                        SELECT c.record_id, i.occurrence_count
                        FROM identifiers AS i
                        JOIN chunks AS c ON c.record_id = i.record_id
                        WHERE i.lexeme = ?
                        ORDER BY i.occurrence_count DESC, c.record_id
                        LIMIT ? OFFSET ?
                    )
                    SELECT c.record_id, c.document_id, c.document_version_id,
                           c.ordinal, c.content, c.token_count,
                           c.source_start AS chunk_source_start,
                           c.source_end AS chunk_source_end,
                           s.source_span_id, s.document_version_id AS source_version_id,
                           s.block_ordinal, s.block_kind, s.source_start, s.source_end,
                           ranked.occurrence_count
                    FROM ranked
                    JOIN chunks AS c ON c.record_id = ranked.record_id
                    LEFT JOIN source_links AS s ON s.record_id = c.record_id
                    ORDER BY ranked.occurrence_count DESC, c.record_id,
                             s.block_ordinal, s.source_span_id
                    """,
                    (request.identifier, limit, offset),
                ).fetchall()
                occurrence_by_record = {
                    str(row["record_id"]): int(row["occurrence_count"]) for row in rows
                }
                chunks = _rows_to_chunks(rows)
                return tuple(
                    (
                        chunk,
                        occurrence_by_record[chunk.record_id.as_str()],
                    )
                    for chunk in chunks
                )

            engine_query = request.engine_query()
            assert engine_query is not None
            rows = self._connection.execute(
                """
                WITH ranked AS (
                    SELECT c.record_id, bm25(chunk_fts, 0.0, ?) AS engine_score
                    FROM chunk_fts
                    JOIN chunks AS c ON c.record_id = chunk_fts.record_id
                    WHERE chunk_fts MATCH ?
                    ORDER BY engine_score, c.record_id
                    LIMIT ? OFFSET ?
                )
                SELECT c.record_id, c.document_id, c.document_version_id,
                       c.ordinal, c.content, c.token_count,
                       c.source_start AS chunk_source_start,
                       c.source_end AS chunk_source_end,
                       s.source_span_id, s.document_version_id AS source_version_id,
                       s.block_ordinal, s.block_kind, s.source_start, s.source_end,
                       ranked.engine_score
                FROM ranked
                JOIN chunks AS c ON c.record_id = ranked.record_id
                LEFT JOIN source_links AS s ON s.record_id = c.record_id
                ORDER BY ranked.engine_score, c.record_id,
                         s.block_ordinal, s.source_span_id
                """,
                (
                    (
                        1.0
                        if self.manifest.contract_version == "1.0.0"
                        else self.manifest.configuration.column_weights[0]
                    ),
                    engine_query,
                    limit,
                    offset,
                ),
            ).fetchall()
            score_by_record = {
                str(row["record_id"]): expose_bm25_score(
                    float(row["engine_score"] or 0)
                )
                for row in rows
            }
            chunks = _rows_to_chunks(rows)
            return tuple(
                (chunk, score_by_record[chunk.record_id.as_str()]) for chunk in chunks
            )
        except LexicalError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
            raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None

    def _require_open(self) -> None:
        if self._closed:
            raise LexicalError(LexicalFailureCategory.GENERATION_UNAVAILABLE)


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
    if manifest.contract_version not in {"1.0.0", LEXICAL_CONTRACT_VERSION}:
        raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
    row = connection.execute("PRAGMA integrity_check").fetchone()
    if row is None or row[0] != "ok":
        raise LexicalError(LexicalFailureCategory.CORRUPT_GENERATION)
    version = connection.execute("PRAGMA user_version").fetchone()
    if version is None or version[0] != LEXICAL_DATABASE_USER_VERSION:
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
        source_start=row["chunk_source_start"],
        source_end=row["chunk_source_end"],
        source_links=links,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
