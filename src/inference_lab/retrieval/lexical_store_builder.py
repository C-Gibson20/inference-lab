"""SQLite build and diff mechanics for immutable lexical generations."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from inference_lab.ingest import (
    BaselineChunkGenerationEntry,
    BaselineChunkGenerationManifest,
    deserialise_chunk_set_selection,
)
from inference_lab.provenance import DocumentId
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
from inference_lab.retrieval.lexical_store_models import (
    LEXICAL_DATABASE_USER_VERSION as _LEXICAL_DATABASE_USER_VERSION,
)
from inference_lab.retrieval.lexical_store_models import (
    BuildChunk as _BuildChunk,
)
from inference_lab.retrieval.lexical_store_models import (
    BuildMember as _BuildMember,
)
from inference_lab.retrieval.lexical_store_models import (
    DiffStatistics as _DiffStatistics,
)
from inference_lab.retrieval.lexical_store_models import (
    LexicalMemberSelection,
    LexicalSourceLink,
)
from inference_lab.retrieval.lexical_store_reader import (
    LexicalGeneration,
    _open_generation_path,
    _sha256_file,
    open_lexical_generation,
)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_chunks(
    source_manifest: BaselineChunkGenerationManifest,
) -> tuple[tuple[_BuildChunk, ...], tuple[_BuildMember, ...]]:
    prepared: list[_BuildChunk] = []
    members: list[_BuildMember] = []
    for entry in source_manifest.entries:
        entry_chunks = _prepare_entry_chunks(entry, source_manifest)
        prepared.extend(entry_chunks)
        members.append(
            _BuildMember(
                document_id=entry.membership.document_id,
                document_version_id=entry.membership.document_version_id,
                selection_digest=entry.selection.payload_digest.digest,
                chunk_count=len(entry_chunks),
            )
        )
    return tuple(prepared), tuple(members)


def _prepare_entry_chunks(
    entry: BaselineChunkGenerationEntry,
    source_manifest: BaselineChunkGenerationManifest,
) -> tuple[_BuildChunk, ...]:
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
        or chunk_set.configuration_identity != source_manifest.configuration_identity
    ):
        raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
    return tuple(
        _BuildChunk(
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            record_id=chunk.record_id,
            ordinal=chunk.ordinal,
            content=chunk.text,
            token_count=chunk.token_count,
            source_start=chunk.source_start,
            source_end=chunk.source_end,
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
            identifiers=tuple(sorted(Counter(extract_identifiers(chunk.text)).items())),
        )
        for chunk in chunk_set.chunks
    )


def _build_full_database(
    database_path: Path,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
) -> LexicalIndexBuildStatistics:
    chunks, members = _prepare_chunks(source_manifest)
    _write_database(
        database_path,
        source_manifest=source_manifest,
        configuration=configuration,
        chunks=chunks,
        members=members,
    )
    return LexicalIndexBuildStatistics(
        members_processed=len(members),
        chunks_processed=len(chunks),
        identifiers_emitted=sum(len(chunk.identifiers) for chunk in chunks),
        inserted_chunks=len(chunks),
    )


def _build_from_parent(
    database_path: Path,
    *,
    index_dir: Path,
    parent_generation_id: str,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
) -> LexicalIndexBuildStatistics:
    try:
        parent = open_lexical_generation(index_dir, parent_generation_id)
    except LexicalError as error:
        if error.category is not LexicalFailureCategory.INCOMPATIBLE_GENERATION:
            raise
        return _build_full_database(
            database_path,
            source_manifest=source_manifest,
            configuration=configuration,
        )

    try:
        if parent.manifest.corpus_id != source_manifest.corpus_id:
            raise LexicalError(LexicalFailureCategory.INCOMPATIBLE_GENERATION)
        if (
            parent.manifest.configuration_digest
            != configuration.configuration_identity.digest
        ):
            return _build_full_database(
                database_path,
                source_manifest=source_manifest,
                configuration=configuration,
            )
        parent_members = {
            member.document_id: member for member in parent.member_selections()
        }
        proposed_entries = {
            entry.membership.document_id: entry for entry in source_manifest.entries
        }
        unchanged_members = {
            document_id: member
            for document_id, member in parent_members.items()
            if (
                document_id in proposed_entries
                and member.document_version_id
                == proposed_entries[document_id].membership.document_version_id
                and member.selection_digest
                == proposed_entries[document_id].selection.payload_digest.digest
            )
        }
        removed_members = tuple(
            member
            for document_id, member in parent_members.items()
            if document_id not in proposed_entries
        )
        changed_entries = tuple(
            entry
            for document_id, entry in proposed_entries.items()
            if document_id not in unchanged_members
        )
        prepared_chunks: list[_BuildChunk] = []
        prepared_members: list[_BuildMember] = []
        for entry in changed_entries:
            entry_chunks = _prepare_entry_chunks(entry, source_manifest)
            prepared_chunks.extend(entry_chunks)
            prepared_members.append(
                _BuildMember(
                    document_id=entry.membership.document_id,
                    document_version_id=entry.membership.document_version_id,
                    selection_digest=entry.selection.payload_digest.digest,
                    chunk_count=len(entry_chunks),
                )
            )
        _clone_parent_database(parent, database_path)
        diff_statistics = _apply_parent_diff(
            database_path,
            source_manifest=source_manifest,
            configuration=configuration,
            removed_members=removed_members,
            replaced_document_ids=tuple(
                entry.membership.document_id
                for entry in changed_entries
                if entry.membership.document_id in parent_members
            ),
            unchanged_chunk_count=sum(
                member.chunk_count for member in unchanged_members.values()
            ),
            members=tuple(prepared_members),
            chunks=tuple(prepared_chunks),
        )
        return LexicalIndexBuildStatistics(
            members_processed=len(source_manifest.entries),
            chunks_processed=len(prepared_chunks),
            identifiers_emitted=diff_statistics.identifiers_emitted,
            reused_chunks=diff_statistics.reused_chunks,
            inserted_chunks=diff_statistics.inserted_chunks,
            replaced_chunks=diff_statistics.replaced_chunks,
            removed_chunks=diff_statistics.removed_chunks,
            database_reused=True,
        )
    finally:
        parent.close()


def _clone_parent_database(parent: LexicalGeneration, database_path: Path) -> None:
    parent._require_open()
    try:
        parent._connection.execute("VACUUM INTO ?", (str(database_path),))
    except sqlite3.Error:
        raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None


def _apply_parent_diff(
    database_path: Path,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
    removed_members: tuple[LexicalMemberSelection, ...],
    replaced_document_ids: tuple[DocumentId, ...],
    unchanged_chunk_count: int,
    members: tuple[_BuildMember, ...],
    chunks: tuple[_BuildChunk, ...],
) -> _DiffStatistics:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            old_record_ids: dict[DocumentId, set[str]] = {
                document_id: {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT record_id FROM chunks WHERE document_id = ?",
                        (document_id.as_str(),),
                    ).fetchall()
                }
                for document_id in replaced_document_ids
            }
            desired_chunks: dict[DocumentId, tuple[_BuildChunk, ...]] = {}
            for chunk in chunks:
                desired_chunks[chunk.document_id] = (
                    *desired_chunks.get(chunk.document_id, ()),
                    chunk,
                )
            reused_chunks = unchanged_chunk_count
            inserted_chunks = 0
            replaced_chunks = 0
            removed_chunks = sum(member.chunk_count for member in removed_members)
            identifiers_emitted = 0
            for member in removed_members:
                _delete_member(connection, member.document_id)
            members_by_id = {member.document_id: member for member in members}
            chunks_to_insert: list[_BuildChunk] = []
            for document_id in replaced_document_ids:
                old_ids = old_record_ids[document_id]
                desired = desired_chunks.get(document_id, ())
                desired_ids = {chunk.record_id.as_str() for chunk in desired}
                for record_id in old_ids - desired_ids:
                    _delete_chunk(connection, record_id)
                reused_chunks += len(old_ids & desired_ids)
                replaced_chunks += len(desired_ids - old_ids)
                removed_chunks += len(old_ids - desired_ids)
                member = members_by_id[document_id]
                connection.execute(
                    "UPDATE members SET document_version_id = ?, selection_digest = ?, "
                    "chunk_count = ? WHERE document_id = ?",
                    (
                        member.document_version_id.as_str(),
                        member.selection_digest,
                        member.chunk_count,
                        member.document_id.as_str(),
                    ),
                )
                new_chunks = tuple(
                    chunk
                    for chunk in desired
                    if chunk.record_id.as_str() not in old_ids
                )
                chunks_to_insert.extend(new_chunks)
                identifiers_emitted += sum(
                    len(chunk.identifiers) for chunk in new_chunks
                )
            new_members = tuple(
                member for member in members if member.document_id not in old_record_ids
            )
            connection.executemany(
                "INSERT INTO members(document_id, document_version_id, selection_digest, "
                "chunk_count) VALUES (?, ?, ?, ?)",
                tuple(
                    (
                        member.document_id.as_str(),
                        member.document_version_id.as_str(),
                        member.selection_digest,
                        member.chunk_count,
                    )
                    for member in new_members
                ),
            )
            new_chunks = tuple(
                chunk for chunk in chunks if chunk.document_id not in old_record_ids
            )
            chunks_to_insert.extend(new_chunks)
            inserted_chunks = len(new_chunks)
            identifiers_emitted += sum(len(chunk.identifiers) for chunk in new_chunks)
            _insert_chunks(connection, tuple(chunks_to_insert))
            _update_generation_metadata(
                connection,
                source_manifest=source_manifest,
                configuration=configuration,
            )
    except sqlite3.Error:
        raise LexicalError(LexicalFailureCategory.EXECUTION_FAILURE) from None
    finally:
        connection.close()
    return _DiffStatistics(
        reused_chunks=reused_chunks,
        inserted_chunks=inserted_chunks,
        replaced_chunks=replaced_chunks,
        removed_chunks=removed_chunks,
        identifiers_emitted=identifiers_emitted,
    )


def _delete_member(connection: sqlite3.Connection, document_id: DocumentId) -> None:
    rows = connection.execute(
        "SELECT record_id FROM chunks WHERE document_id = ?",
        (document_id.as_str(),),
    ).fetchall()
    for row in rows:
        _delete_chunk(connection, str(row[0]))
    connection.execute(
        "DELETE FROM members WHERE document_id = ?", (document_id.as_str(),)
    )


def _delete_chunk(connection: sqlite3.Connection, record_id: str) -> None:
    connection.execute("DELETE FROM chunk_fts WHERE record_id = ?", (record_id,))
    connection.execute("DELETE FROM source_links WHERE record_id = ?", (record_id,))
    connection.execute("DELETE FROM identifiers WHERE record_id = ?", (record_id,))
    connection.execute("DELETE FROM chunks WHERE record_id = ?", (record_id,))


def _insert_chunks(
    connection: sqlite3.Connection,
    chunks: tuple[_BuildChunk, ...],
) -> None:
    connection.executemany(
        "INSERT INTO chunks(record_id, document_id, document_version_id, ordinal, "
        "content, token_count, source_start, source_end) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(
            (
                chunk.record_id.as_str(),
                chunk.document_id.as_str(),
                chunk.document_version_id.as_str(),
                chunk.ordinal,
                chunk.content,
                chunk.token_count,
                chunk.source_start,
                chunk.source_end,
            )
            for chunk in chunks
        ),
    )
    connection.executemany(
        "INSERT INTO source_links(record_id, source_span_id, document_version_id, "
        "block_ordinal, block_kind, source_start, source_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
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


def _update_generation_metadata(
    connection: sqlite3.Connection,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
) -> None:
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
        "member_count": connection.execute("SELECT COUNT(*) FROM members").fetchone()[
            0
        ],
        "chunk_count": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "identifier_count": connection.execute(
            "SELECT COUNT(*) FROM identifiers"
        ).fetchone()[0],
    }
    connection.executemany(
        "UPDATE generation_metadata SET value = ? WHERE key = ?",
        tuple((str(value), key) for key, value in metadata.items()),
    )


def _write_database(
    database_path: Path,
    *,
    source_manifest: BaselineChunkGenerationManifest,
    configuration: LexicalIndexConfiguration,
    chunks: tuple[_BuildChunk, ...],
    members: tuple[_BuildMember, ...],
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
                selection_digest TEXT NOT NULL,
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0)
            ) WITHOUT ROWID;
            CREATE TABLE chunks (
                record_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES members(document_id),
                document_version_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL CHECK (token_count > 0),
                source_start INTEGER NOT NULL CHECK (source_start >= 0),
                source_end INTEGER NOT NULL CHECK (source_end > source_start),
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
            "member_count": str(len(members)),
            "chunk_count": str(len(chunks)),
            "identifier_count": str(sum(len(chunk.identifiers) for chunk in chunks)),
        }
        connection.executemany(
            "INSERT INTO generation_metadata(key, value) VALUES (?, ?)",
            tuple(metadata.items()),
        )
        connection.executemany(
            "INSERT INTO members(document_id, document_version_id, selection_digest, "
            "chunk_count) "
            "VALUES (?, ?, ?, ?)",
            tuple(
                (
                    member.document_id.as_str(),
                    member.document_version_id.as_str(),
                    member.selection_digest,
                    member.chunk_count,
                )
                for member in members
            ),
        )
        connection.executemany(
            "INSERT INTO chunks(record_id, document_id, document_version_id, ordinal, "
            "content, token_count, source_start, source_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(
                (
                    chunk.record_id.as_str(),
                    chunk.document_id.as_str(),
                    chunk.document_version_id.as_str(),
                    chunk.ordinal,
                    chunk.content,
                    chunk.token_count,
                    chunk.source_start,
                    chunk.source_end,
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
) -> LexicalIndexManifest:
    connection = sqlite3.connect(database_path)
    try:
        member_count = connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        identifier_count = connection.execute(
            "SELECT COUNT(*) FROM identifiers"
        ).fetchone()[0]
    finally:
        connection.close()
    return LexicalIndexManifest(
        corpus_id=source_manifest.corpus_id,
        corpus_version=source_manifest.corpus_version,
        generation_id=generation_id,
        configuration=configuration,
        configuration_digest=configuration.configuration_identity.digest,
        source_manifest_digest=source_manifest.source_manifest_digest,
        database_checksum=_sha256_file(database_path),
        member_count=member_count,
        chunk_count=chunk_count,
        identifier_count=identifier_count,
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
