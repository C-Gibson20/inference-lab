"""Tests for corpus-scoped exact chunk-set selection."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    DigestAlgorithm,
    DocumentVersionId,
)
from inference_lab.storage import CorruptStorageError
from inference_lab.storage._sqlite_chunk_sets import get_chunk_set_selection
from inference_lab.storage._sqlite_connection import open_sqlite_connection
from inference_lab.storage._sqlite_initialisation import initialise_empty_database

from .fixtures import fixed_applied_at as _fixed_applied_at  # noqa: TID252

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_OTHER_CORPUS_ID = CorpusId("22222222-2222-4222-8222-222222222222")
_DOCUMENT_ID = "33333333-3333-4333-8333-333333333333"
_DOCUMENT_VERSION_ID = DocumentVersionId("44444444-4444-4444-8444-444444444444")
_CONFIGURATION_IDENTITY = ConfigurationIdentity(
    algorithm=DigestAlgorithm.SHA256,
    digest="a" * 64,
)


def _open_database(tmp_path: Path) -> sqlite3.Connection:
    connection = open_sqlite_connection(tmp_path / "metadata.sqlite3")
    initialise_empty_database(connection, applied_at_factory=_fixed_applied_at)
    connection.execute(
        "INSERT INTO corpora (corpus_id, display_name) VALUES (?, ?)",
        (_CORPUS_ID.as_str(), "Primary corpus"),
    )
    connection.execute(
        "INSERT INTO corpora (corpus_id, display_name) VALUES (?, ?)",
        (_OTHER_CORPUS_ID.as_str(), "Other corpus"),
    )
    connection.execute(
        "INSERT INTO documents (document_id) VALUES (?)",
        (_DOCUMENT_ID,),
    )
    connection.execute(
        """INSERT INTO document_versions (
            document_version_id,
            document_id,
            content_hash_algorithm,
            content_hash_digest
        ) VALUES (?, ?, 'sha256', ?)""",
        (_DOCUMENT_VERSION_ID.as_str(), _DOCUMENT_ID, "b" * 64),
    )
    connection.execute(
        """INSERT INTO corpus_document_version_memberships (
            corpus_id,
            document_id,
            document_version_id
        ) VALUES (?, ?, ?)""",
        (_CORPUS_ID.as_str(), _DOCUMENT_ID, _DOCUMENT_VERSION_ID.as_str()),
    )
    connection.execute(
        """INSERT INTO chunk_sets (
            document_id,
            document_version_id,
            payload_schema_version,
            parser_implementation,
            parser_contract_version,
            parser_normalisation_version,
            chunker_implementation,
            chunker_contract_version,
            tokeniser_implementation,
            tokeniser_contract_version,
            tokeniser_artefact_identity,
            tokeniser_artefact_revision,
            tokeniser_options,
            min_tokens,
            max_tokens,
            overlap_numerator,
            overlap_denominator,
            structural_boundary_policy,
            oversize_unit_policy,
            configuration_algorithm,
            configuration_digest,
            payload_digest_algorithm,
            payload_digest,
            canonical_payload
        ) VALUES (
            ?, ?, '1.0.0', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            'sha256', ?, 'sha256', ?, ?
        )""",
        (
            _DOCUMENT_ID,
            _DOCUMENT_VERSION_ID.as_str(),
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
            "a" * 64,
            hashlib.sha256(b"canonical-payload").hexdigest(),
            b"canonical-payload",
        ),
    )
    return connection


def test_exact_selection_returns_validated_storage_metadata(tmp_path: Path) -> None:
    connection = _open_database(tmp_path)
    try:
        selection = get_chunk_set_selection(
            connection,
            _CORPUS_ID,
            _DOCUMENT_VERSION_ID,
            _CONFIGURATION_IDENTITY,
        )

        assert selection is not None
        assert selection.document_id.as_str() == _DOCUMENT_ID
        assert selection.document_version_id == _DOCUMENT_VERSION_ID
        assert selection.configuration_identity == _CONFIGURATION_IDENTITY
        assert (
            selection.payload_digest.digest
            == hashlib.sha256(b"canonical-payload").hexdigest()
        )
        assert selection.canonical_payload == b"canonical-payload"
    finally:
        connection.close()


def test_selection_hides_absent_and_other_corpus_membership(
    tmp_path: Path,
) -> None:
    connection = _open_database(tmp_path)
    try:
        assert (
            get_chunk_set_selection(
                connection,
                _OTHER_CORPUS_ID,
                _DOCUMENT_VERSION_ID,
                _CONFIGURATION_IDENTITY,
            )
            is None
        )
        assert (
            get_chunk_set_selection(
                connection,
                _CORPUS_ID,
                DocumentVersionId("55555555-5555-4555-8555-555555555555"),
                _CONFIGURATION_IDENTITY,
            )
            is None
        )
    finally:
        connection.close()


def test_selection_configuration_mismatch_is_not_a_match(tmp_path: Path) -> None:
    connection = _open_database(tmp_path)
    try:
        different_configuration = ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest="c" * 64,
        )
        assert (
            get_chunk_set_selection(
                connection,
                _CORPUS_ID,
                _DOCUMENT_VERSION_ID,
                different_configuration,
            )
            is None
        )
    finally:
        connection.close()


def test_selection_rejects_reachable_corrupt_metadata(tmp_path: Path) -> None:
    connection = _open_database(tmp_path)
    try:
        connection.execute("UPDATE chunk_sets SET tokeniser_options = 'malformed'")
        with pytest.raises(CorruptStorageError):
            get_chunk_set_selection(
                connection,
                _CORPUS_ID,
                _DOCUMENT_VERSION_ID,
                _CONFIGURATION_IDENTITY,
            )
    finally:
        connection.close()


def test_selection_rejects_payload_digest_mismatch(tmp_path: Path) -> None:
    connection = _open_database(tmp_path)
    try:
        connection.execute(
            "UPDATE chunk_sets SET canonical_payload = ?",
            (b"changed-payload",),
        )
        with pytest.raises(CorruptStorageError):
            get_chunk_set_selection(
                connection,
                _CORPUS_ID,
                _DOCUMENT_VERSION_ID,
                _CONFIGURATION_IDENTITY,
            )
    finally:
        connection.close()
