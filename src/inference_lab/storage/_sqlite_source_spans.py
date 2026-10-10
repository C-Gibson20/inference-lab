"""Corpus-scoped SQLite source-span resolution."""

import sqlite3

from inference_lab.provenance import (
    CorpusId,
    DocumentVersionId,
    KnownSourceSpan,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_parsed_representations import (
    get_parsed_representation,
    list_parsed_representations,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
)
from inference_lab.storage.source_locations import ResolvedSourceSpan


def resolve_source_span(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    source_span_id: SourceSpanId,
) -> ResolvedSourceSpan | None:
    """Resolve one span only through its representation's corpus membership."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(source_span_id, SourceSpanId):
        raise TypeError("source_span_id must be a SourceSpanId.")
    try:
        row = connection.execute(
            """SELECT span.parsed_record_id
            FROM source_spans AS span
            JOIN parsed_representations AS representation
                ON representation.record_id = span.parsed_record_id
                AND representation.document_version_id = span.document_version_id
            JOIN corpus_document_version_memberships AS membership
                ON membership.document_version_id = representation.document_version_id
                AND membership.document_id = representation.document_id
            WHERE membership.corpus_id = ? AND span.source_span_id = ?""",
            (corpus_id.as_str(), source_span_id.as_str()),
        ).fetchone()
        if row is None:
            return None
        representation = get_parsed_representation(
            connection,
            corpus_id,
            RecordId.from_str(row["parsed_record_id"]),
        )
        if representation is None:
            raise CorruptStorageError()
        location = next(
            (
                value
                for value in representation.source_locations
                if value.source_span_id == source_span_id
            ),
            None,
        )
        if location is None:
            raise CorruptStorageError()
        page_layout = next(
            (
                value
                for value in representation.page_layouts
                if value.page_number == location.page_number
            ),
            None,
        )
        return ResolvedSourceSpan(
            document_id=representation.document_id,
            location=location,
            page_layout=page_layout,
        )
    except CorruptStorageError:
        raise
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_known_source_spans(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> tuple[KnownSourceSpan, ...]:
    """Return a validated catalogue in canonical source-span order."""
    representations = list_parsed_representations(
        connection,
        corpus_id,
        document_version_id,
    )
    spans = tuple(
        span
        for representation in representations
        for span in representation.known_source_spans()
    )
    return tuple(sorted(spans, key=lambda span: span.key()))
