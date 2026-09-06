"""Schema tests for the source-persistence SQLite migration."""

import sqlite3
from collections.abc import Generator, Mapping

import pytest

from inference_lab.storage._sqlite_migrations import MIGRATIONS

_DOCUMENT_ID = "00000000-0000-4000-8000-000000000001"
_OTHER_DOCUMENT_ID = "00000000-0000-4000-8000-000000000002"
_VERSION_ID = "00000000-0000-4000-8000-000000000003"
_OTHER_VERSION_ID = "00000000-0000-4000-8000-000000000004"
_RECORD_ID = "00000000-0000-4000-8000-000000000005"
_OTHER_RECORD_ID = "00000000-0000-4000-8000-000000000006"
_SPAN_ID = "00000000-0000-4000-8000-000000000007"
_OTHER_SPAN_ID = "00000000-0000-4000-8000-000000000008"
_PARSE_ID = "00000000-0000-4000-8000-000000000009"
_OTHER_PARSE_ID = "00000000-0000-4000-8000-00000000000a"
_DIGEST = "a" * 64


@pytest.fixture
def database() -> Generator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS:
        for statement in migration.statements:
            connection.execute(statement)
    _insert_version(connection)
    try:
        yield connection
    finally:
        connection.close()


def _insert(
    connection: sqlite3.Connection,
    table: str,
    values: Mapping[str, object],
) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    connection.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",  # noqa: S608
        tuple(values.values()),
    )


def _insert_version(
    connection: sqlite3.Connection,
    *,
    document_id: str = _DOCUMENT_ID,
    version_id: str = _VERSION_ID,
    digest: str = _DIGEST,
) -> None:
    _insert(connection, "documents", {"document_id": document_id})
    _insert(
        connection,
        "document_versions",
        {
            "document_version_id": version_id,
            "document_id": document_id,
            "content_hash_algorithm": "sha256",
            "content_hash_digest": digest,
        },
    )


def _artefact(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "document_version_id": _VERSION_ID,
        "document_id": _DOCUMENT_ID,
        "record_schema_version": "1.0.0",
        "content_hash_algorithm": "sha256",
        "content_hash_digest": _DIGEST,
        "byte_count": 42,
        "media_type": "application/pdf",
        "storage_key": f"sha256/aa/{_DIGEST}",
    }
    values.update(overrides)
    return values


def _representation(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "record_id": _RECORD_ID,
        "document_id": _DOCUMENT_ID,
        "document_version_id": _VERSION_ID,
        "record_schema_version": "1.0.0",
        "source_text": "alpha beta",
        "canonical_text": "alpha beta",
        "parser_implementation": "test-parser",
        "parser_contract_version": "1.2.3",
        "normalisation_version": "1.0.0",
        "language": "en",
        "page_count": 1,
        "character_count": 10,
        "token_count": 2,
        "page_collection_availability": "available",
        "section_collection_availability": "available",
        "paragraph_collection_availability": "available",
        "output_digest_algorithm": "sha256",
        "output_digest": "b" * 64,
    }
    values.update(overrides)
    return values


def _layout() -> dict[str, object]:
    return {
        "parsed_record_id": _RECORD_ID,
        "document_version_id": _VERSION_ID,
        "page_number": 1,
        "coordinate_unit": "pdf_points",
        "coordinate_origin": "top_left",
        "page_width": 100.0,
        "page_height": 200.0,
        "rotation": 0,
        "content_origin": "native",
        "producer_implementation": "test-parser",
        "producer_contract_version": "1.2.3",
    }


def _span(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "source_span_id": _SPAN_ID,
        "parsed_record_id": _RECORD_ID,
        "document_version_id": _VERSION_ID,
        "start_offset": 0,
        "end_offset": 5,
        "content_origin": "native",
        "page_number": 1,
    }
    values.update(overrides)
    return values


def _outcome(parse_id: str, status: str, record_id: str | None) -> dict[str, object]:
    return {
        "parse_id": parse_id,
        "document_id": _DOCUMENT_ID,
        "document_version_id": _VERSION_ID,
        "record_schema_version": "1.0.0",
        "parse_status": status,
        "parser_implementation": "test-parser",
        "parser_contract_version": "1.2.3",
        "normalisation_version": "1.0.0",
        "parsed_record_id": record_id,
    }


def test_migration_creates_required_tables_and_query_indexes(
    database: sqlite3.Connection,
) -> None:
    schema_objects = {
        (row[0], row[1])
        for row in database.execute(
            """SELECT type, name FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'"""
        )
    }

    assert {
        ("table", name)
        for name in (
            "source_artefacts",
            "parsed_representations",
            "page_layouts",
            "source_spans",
            "structural_blocks",
            "parse_outcomes",
            "parse_failures",
        )
    } <= schema_objects
    assert {
        ("index", name)
        for name in (
            "source_artefacts_blob_reconciliation_idx",
            "parsed_representations_version_idx",
            "parsed_representations_natural_identity_idx",
            "source_spans_resolution_idx",
            "structural_blocks_traversal_idx",
            "parse_outcomes_version_idx",
            "parse_failures_traversal_idx",
        )
    } <= schema_objects


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "content_hash_digest": "c" * 64,
            "storage_key": f"sha256/cc/{'c' * 64}",
        },
        {"media_type": "Application/PDF"},
        {"media_type": "application/pdf/extra"},
        {"storage_key": f"sha256/bb/{_DIGEST}"},
    ],
)
def test_source_artefact_rejects_mismatched_or_noncanonical_metadata(
    database: sqlite3.Connection,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(database, "source_artefacts", _artefact(**overrides))


def test_source_artefact_allows_physical_blob_reuse(
    database: sqlite3.Connection,
) -> None:
    _insert(database, "source_artefacts", _artefact())
    _insert_version(
        database,
        document_id=_OTHER_DOCUMENT_ID,
        version_id=_OTHER_VERSION_ID,
    )
    _insert(
        database,
        "source_artefacts",
        _artefact(
            document_id=_OTHER_DOCUMENT_ID,
            document_version_id=_OTHER_VERSION_ID,
        ),
    )


def test_representation_preserves_availability_and_natural_identity(
    database: sqlite3.Connection,
) -> None:
    _insert(
        database,
        "parsed_representations",
        _representation(page_collection_availability="unavailable"),
    )
    assert database.execute(
        "SELECT page_collection_availability FROM parsed_representations"
    ).fetchone() == ("unavailable",)

    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            database,
            "parsed_representations",
            _representation(
                record_id=_OTHER_RECORD_ID,
                page_collection_availability="available",
            ),
        )
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            database,
            "parsed_representations",
            _representation(
                record_id=_OTHER_RECORD_ID,
                normalisation_version="2.0.0",
                ocr_implementation="test-ocr",
            ),
        )


def test_layout_span_and_block_constraints_preserve_structure(
    database: sqlite3.Connection,
) -> None:
    _insert(database, "parsed_representations", _representation())
    _insert(database, "page_layouts", _layout())
    with pytest.raises(sqlite3.IntegrityError):
        _insert(database, "page_layouts", _layout())
    with pytest.raises(sqlite3.IntegrityError):
        _insert(database, "source_spans", _span(end_offset=20))
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            database,
            "source_spans",
            _span(
                bounding_box_left=10.0,
                bounding_box_top=20.0,
                bounding_box_right=101.0,
                bounding_box_bottom=40.0,
            ),
        )

    _insert(
        database,
        "source_spans",
        _span(
            bounding_box_left=10.0,
            bounding_box_top=20.0,
            bounding_box_right=30.0,
            bounding_box_bottom=40.0,
        ),
    )
    _insert(
        database,
        "structural_blocks",
        {
            "parsed_record_id": _RECORD_ID,
            "document_version_id": _VERSION_ID,
            "ordinal": 0,
            "block_kind": "page",
            "block_text": "alpha",
            "source_span_id": _SPAN_ID,
        },
    )
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            database,
            "structural_blocks",
            {
                "parsed_record_id": _RECORD_ID,
                "document_version_id": _VERSION_ID,
                "ordinal": 2,
                "block_kind": "paragraph",
                "block_text": "beta",
                "source_span_id": _OTHER_SPAN_ID,
            },
        )


def test_unavailable_typed_collection_rejects_matching_block(
    database: sqlite3.Connection,
) -> None:
    _insert(
        database,
        "parsed_representations",
        _representation(page_collection_availability="unavailable"),
    )
    _insert(database, "source_spans", _span())
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            database,
            "structural_blocks",
            {
                "parsed_record_id": _RECORD_ID,
                "document_version_id": _VERSION_ID,
                "ordinal": 0,
                "block_kind": "page",
                "block_text": "alpha",
                "source_span_id": _SPAN_ID,
            },
        )


def test_outcomes_and_bounded_failures_enforce_status_policy_and_order(
    database: sqlite3.Connection,
) -> None:
    _insert(database, "parsed_representations", _representation())
    _insert(database, "parse_outcomes", _outcome(_PARSE_ID, "succeeded", _RECORD_ID))
    fatal_failure = {
        "parse_id": _PARSE_ID,
        "ordinal": 0,
        "failure_category": "decode_failed",
        "failure_severity": "fatal",
        "failure_stage": "decode",
    }
    with pytest.raises(sqlite3.IntegrityError):
        _insert(database, "parse_failures", fatal_failure)

    _insert(database, "parse_outcomes", _outcome(_OTHER_PARSE_ID, "failed", None))
    fatal_failure["parse_id"] = _OTHER_PARSE_ID
    _insert(database, "parse_failures", fatal_failure)
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            database,
            "parse_failures",
            {
                "parse_id": _OTHER_PARSE_ID,
                "ordinal": 2,
                "failure_category": "malformed_unit",
                "failure_severity": "recoverable",
                "failure_stage": "structure",
                "unit_kind": "page",
                "unit_index": 1,
            },
        )


@pytest.mark.parametrize(
    ("status", "record_id"),
    [("failed", _RECORD_ID), ("succeeded", None)],
)
def test_parse_outcome_requires_status_appropriate_representation_link(
    database: sqlite3.Connection,
    status: str,
    record_id: str | None,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(database, "parse_outcomes", _outcome(_PARSE_ID, status, record_id))
