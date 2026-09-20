"""Tests for the SQLite canonical chunk-set schema contract."""

import sqlite3
from contextlib import closing

import pytest

from inference_lab.storage._sqlite_chunk_schema import CHUNK_SCHEMA_STATEMENTS


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT NOT NULL PRIMARY KEY
        ) STRICT;
        CREATE TABLE document_versions (
            document_version_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            PRIMARY KEY (document_version_id, document_id),
            FOREIGN KEY (document_id) REFERENCES documents (document_id)
        ) STRICT;
        """
    )
    for statement in CHUNK_SCHEMA_STATEMENTS:
        connection.execute(statement)
    return connection


def _valid_values() -> tuple[object, ...]:
    return (
        "doc-1",
        "version-1",
        "1.0.0",
        "text-native",
        "3.0.0",
        "1.0.0",
        "structure-aware-greedy",
        "1.0.1",
        "deterministic-tokeniser",
        "1.0.0",
        "tokeniser-model",
        "2026-01",
        "normalise=unicode,provider=fixture",
        256,
        512,
        15,
        100,
        "structure-aware-greedy",
        "token-window",
        "sha256",
        "a" * 64,
        "sha256",
        "b" * 64,
        b"{}",
    )


def test_chunk_schema_exposes_complete_selection_metadata_without_corpus_id() -> None:
    with closing(_connection()) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chunk_sets)")
        }

        assert "corpus_id" not in columns
        assert {
            "document_id",
            "document_version_id",
            "payload_schema_version",
            "parser_implementation",
            "parser_contract_version",
            "parser_normalisation_version",
            "chunker_implementation",
            "chunker_contract_version",
            "tokeniser_implementation",
            "tokeniser_contract_version",
            "tokeniser_artefact_identity",
            "tokeniser_artefact_revision",
            "tokeniser_options",
            "min_tokens",
            "max_tokens",
            "overlap_numerator",
            "overlap_denominator",
            "structural_boundary_policy",
            "oversize_unit_policy",
            "configuration_algorithm",
            "configuration_digest",
            "payload_digest_algorithm",
            "payload_digest",
            "canonical_payload",
        } <= columns


def test_chunk_schema_accepts_one_valid_selection_row() -> None:
    with closing(_connection()) as connection:
        connection.execute("INSERT INTO documents (document_id) VALUES (?)", ("doc-1",))
        connection.execute(
            "INSERT INTO document_versions (document_version_id, document_id) "
            "VALUES (?, ?)",
            ("version-1", "doc-1"),
        )

        columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_info(chunk_sets)")
            if row[1] != "storage_id"
        )
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO chunk_sets ({', '.join(columns)}) VALUES ({placeholders})",
            _valid_values(),
        )

        assert connection.execute("SELECT count(*) FROM chunk_sets").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_schema_version", "2.0.0"),
        ("min_tokens", 0),
        ("max_tokens", 255),
        ("overlap_denominator", 0),
        ("structural_boundary_policy", "other"),
        ("configuration_digest", "not-a-digest"),
    ],
)
def test_chunk_schema_rejects_invalid_selection_values(
    field: str,
    value: object,
) -> None:
    with closing(_connection()) as connection:
        connection.execute("INSERT INTO documents (document_id) VALUES (?)", ("doc-1",))
        connection.execute(
            "INSERT INTO document_versions (document_version_id, document_id) "
            "VALUES (?, ?)",
            ("version-1", "doc-1"),
        )
        columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_info(chunk_sets)")
            if row[1] != "storage_id"
        )
        values = list(_valid_values())
        values[columns.index(field)] = value
        placeholders = ", ".join("?" for _ in columns)

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO chunk_sets ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
