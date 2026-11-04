"""Read-only access to validated immutable dense generations."""

from __future__ import annotations

import hashlib
import heapq
import sqlite3
import struct
from operator import mul
from pathlib import Path
from typing import Iterator, Self
from urllib.parse import quote

from inference_lab._json import load_json_object, require_json_string
from inference_lab.provenance import (
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.dense import (
    DENSE_CONTRACT_VERSION,
    DENSE_FORMAT_VERSION,
    DENSE_IMPLEMENTATION,
    DenseError,
    DenseFailureCategory,
    DenseIndexEntry,
    DenseIndexManifest,
    DenseIndexMatch,
    DenseMemberSelection,
    DenseVector,
    l2_normalised,
)
from inference_lab.retrieval.dense_serialisation import _read_manifest
from inference_lab.retrieval.dense_store_models import (
    _VECTOR_VALUE_BYTES,
    DENSE_DATABASE_NAME,
    DENSE_DATABASE_USER_VERSION,
    DENSE_DIRECTORY_NAME,
    DENSE_MANIFEST_NAME,
)
from inference_lab.retrieval.evidence import (
    CandidateKind,
    EvidenceSource,
    RetrievalCandidate,
)


def open_dense_generation(index_dir: Path, generation_id: str) -> DenseGeneration:
    """Open and validate one generation by its immutable identity."""
    if not isinstance(index_dir, Path):
        raise TypeError("index_dir must be a Path.")
    if not isinstance(generation_id, str):
        raise TypeError("generation_id must be a string.")
    if len(generation_id) != 70 or not generation_id.startswith("dense-"):
        raise DenseError(DenseFailureCategory.INVALID_REQUEST)
    root = index_dir / DENSE_DIRECTORY_NAME / generation_id
    return _open_dense_path(root, generation_id)


class DenseGeneration:
    """Read-only handle for a validated immutable dense generation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        manifest: DenseIndexManifest,
    ) -> None:
        self._connection = connection
        self._manifest = manifest
        self._closed = False

    @property
    def manifest(self) -> DenseIndexManifest:
        """Return the immutable manifest this generation was opened by."""
        return self._manifest

    def __enter__(self) -> Self:
        if self._closed:
            raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the read-only SQLite connection."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def read_entries(self) -> tuple[DenseIndexEntry, ...]:
        """Return all candidates and vectors in deterministic source order."""
        self._require_open()
        try:
            rows = self._connection.execute(
                """
                SELECT record_id, document_id, document_version_id, ordinal,
                       content, source_links, vector
                FROM entries ORDER BY document_id, ordinal, record_id
                """
            ).fetchall()
            return tuple(
                _entry_from_row(row, self.manifest.compatibility.dimensions)
                for row in rows
            )
        except DenseError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, struct.error):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def search(
        self,
        query_vector: DenseVector,
        *,
        limit: int,
    ) -> tuple[DenseIndexMatch, ...]:
        """Return the best matches without materialising the whole index."""
        if not isinstance(query_vector, DenseVector):
            raise TypeError("query_vector must be a DenseVector.")
        if type(limit) is not int:
            raise TypeError("limit must be an integer.")
        if limit <= 0:
            raise DenseError(DenseFailureCategory.INVALID_REQUEST)
        self._require_open()
        dimensions = self.manifest.compatibility.dimensions
        if len(query_vector.values) != dimensions:
            raise DenseError(DenseFailureCategory.INVALID_REQUEST)
        try:
            unit_query = l2_normalised(query_vector).values
            ranked = heapq.nsmallest(
                limit,
                self._stream_similarities(unit_query, dimensions),
                key=lambda scored: (-scored[0], scored[1]),
            )
            return tuple(
                DenseIndexMatch(
                    candidate=self._require_entry(record_id).candidate,
                    cosine_similarity=max(-1.0, min(1.0, similarity)),
                    rank=rank,
                )
                for rank, (similarity, record_id) in enumerate(ranked, start=1)
            )
        except DenseError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, struct.error):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def _stream_similarities(
        self,
        unit_query: tuple[float, ...],
        dimensions: int,
    ) -> Iterator[tuple[float, str]]:
        """Score every stored vector without holding more than one row."""
        expected_bytes = dimensions * _VECTOR_VALUE_BYTES
        for record_id, vector_bytes in self._connection.execute(
            "SELECT record_id, vector FROM entries"
        ):
            stored = bytes(vector_bytes)
            if len(stored) != expected_bytes:
                raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
            # Stored vectors are unit-length, so cosine similarity is the
            # dot product with the unit query.
            values = struct.unpack(f"<{dimensions}f", stored)
            yield sum(map(mul, unit_query, values)), record_id

    def _require_entry(self, record_id: str) -> DenseIndexEntry:
        """Hydrate one ranked record, rejecting a generation that lost it."""
        entry = self.get_entry(RecordId.from_str(record_id))
        if entry is None:
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        return entry

    def get_entry(self, record_id: RecordId) -> DenseIndexEntry | None:
        """Return one vector record by its immutable identity."""
        if not isinstance(record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        self._require_open()
        try:
            row = self._connection.execute(
                """
                SELECT record_id, document_id, document_version_id, ordinal,
                       content, source_links, vector
                FROM entries WHERE record_id = ?
                """,
                (record_id.as_str(),),
            ).fetchone()
            return (
                None
                if row is None
                else _entry_from_row(row, self.manifest.compatibility.dimensions)
            )
        except DenseError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, struct.error):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def member_selections(self) -> tuple[DenseMemberSelection, ...]:
        """Return complete staged membership represented by this generation."""
        self._require_open()
        try:
            return tuple(
                DenseMemberSelection(
                    document_id=DocumentId.from_str(row[0]),
                    document_version_id=DocumentVersionId.from_str(row[1]),
                    selection_digest=row[2],
                    chunk_count=row[3],
                )
                for row in self._connection.execute(
                    """
                    SELECT document_id, document_version_id, selection_digest,
                           chunk_count FROM members ORDER BY document_id
                    """
                ).fetchall()
            )
        except (sqlite3.Error, TypeError, ValueError, IndexError):
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None

    def _require_open(self) -> None:
        if self._closed:
            raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE)


def _open_dense_path(path: Path, expected_generation_id: str) -> DenseGeneration:
    connection: sqlite3.Connection | None = None
    try:
        if path.is_symlink() or not path.is_dir():
            raise DenseError(DenseFailureCategory.GENERATION_UNAVAILABLE)
        manifest_path = path / DENSE_MANIFEST_NAME
        database_path = path / DENSE_DATABASE_NAME
        if manifest_path.is_symlink() or database_path.is_symlink():
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        manifest = _read_manifest(manifest_path)
        if manifest.generation_id != expected_generation_id:
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if (
            manifest.compatibility.configuration.implementation != DENSE_IMPLEMENTATION
            or manifest.compatibility.configuration.format_version
            != DENSE_FORMAT_VERSION
            or manifest.compatibility.configuration.contract_version
            != DENSE_CONTRACT_VERSION
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if _sha256_file(database_path) != manifest.database_digest:
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        database_uri = f"file:{quote(str(database_path))}?mode=ro&immutable=1"
        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        if connection.execute("PRAGMA user_version").fetchone()[0] != (
            DENSE_DATABASE_USER_VERSION
        ):
            raise DenseError(DenseFailureCategory.INCOMPATIBLE_GENERATION)
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        generation = DenseGeneration(connection, manifest)
        members = generation.member_selections()
        entry_count = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if members != manifest.members or entry_count != manifest.entry_count:
            generation.close()
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        if sum(member.chunk_count for member in members) != entry_count:
            generation.close()
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        member_counts = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                """
                SELECT m.document_id, m.document_version_id, COUNT(e.record_id)
                FROM members AS m LEFT JOIN entries AS e USING (document_id)
                GROUP BY m.document_id, m.document_version_id
                """
            ).fetchall()
        }
        if any(
            member_counts.get(member.document_id.as_str())
            != (member.document_version_id.as_str(), member.chunk_count)
            for member in members
        ):
            generation.close()
            raise DenseError(DenseFailureCategory.CORRUPT_GENERATION)
        return generation
    except DenseError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
        if connection is not None:
            connection.close()
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


def _entry_from_row(row: sqlite3.Row, dimensions: int) -> DenseIndexEntry:
    # The messages below describe the defect for a maintainer reading a
    # traceback; every one is converted to a bounded corrupt-generation
    # failure before it can cross the boundary.
    try:
        source_object = load_json_object(bytes(row[5]))
        if not isinstance(source_object, dict) or set(source_object) != {"sources"}:
            raise ValueError("stored source links must hold exactly one sources key.")
        source_payload = source_object["sources"]
        if not isinstance(source_payload, list):
            raise ValueError("stored source links must be an array.")
        sources = tuple(
            EvidenceSource(
                document_id=DocumentId.from_str(
                    require_json_string(item["document_id"])
                ),
                document_version_id=DocumentVersionId.from_str(
                    require_json_string(item["document_version_id"])
                ),
                source_span_id=SourceSpanId.from_str(
                    require_json_string(item["source_span_id"])
                ),
            )
            for item in source_payload
            if isinstance(item, dict)
        )
        if len(sources) != len(source_payload):
            raise ValueError("every stored source link must be an object.")
        candidate = RetrievalCandidate(
            record_id=RecordId.from_str(row[0]),
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content=row[4],
            sources=sources,
        )
        vector_bytes = bytes(row[6])
        if len(vector_bytes) != dimensions * _VECTOR_VALUE_BYTES:
            raise ValueError("stored vector width does not match the index.")
        vector = DenseVector(struct.unpack(f"<{dimensions}f", vector_bytes))
        document_id = DocumentId.from_str(row[1])
        document_version_id = DocumentVersionId.from_str(row[2])
        if (
            document_id != sources[0].document_id
            or document_version_id != sources[0].document_version_id
            or type(row[3]) is not int
            or row[3] < 0
            or any(
                source.document_id != document_id
                or source.document_version_id != document_version_id
                for source in sources
            )
        ):
            raise ValueError("stored row lineage disagrees with its source links.")
        return DenseIndexEntry(candidate, vector)
    except (TypeError, ValueError, IndexError, KeyError, struct.error):
        raise DenseError(DenseFailureCategory.CORRUPT_GENERATION) from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
