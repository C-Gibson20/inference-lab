"""Internal SQLite readers for corpus-authorised canonical chunk sets."""

import sqlite3

from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.chunk_records import (
    CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION,
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSetSelectionRecord,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
)


def _parse_tokeniser_options(value: object) -> tuple[tuple[str, str], ...]:
    """Reconstruct the canonical storage representation of tokeniser options."""
    if not isinstance(value, str):
        raise TypeError("tokeniser_options must be a string.")
    if not value:
        return ()

    options: list[tuple[str, str]] = []
    for item in value.split(","):
        key, separator, option_value = item.partition("=")
        if not separator or not key or not option_value:
            raise ValueError("tokeniser_options is not canonical.")
        options.append((key, option_value))
    return tuple(options)


def _reconstruct_selection(row: sqlite3.Row) -> ChunkSetSelectionRecord:
    """Reconstruct and validate one exact chunk-set selection row."""
    return ChunkSetSelectionRecord(
        document_id=DocumentId.from_str(row["document_id"]),
        document_version_id=DocumentVersionId.from_str(row["document_version_id"]),
        parser_version=ChunkParserVersionRecord(
            implementation=row["parser_implementation"],
            contract_version=row["parser_contract_version"],
            normalisation_version=row["parser_normalisation_version"],
        ),
        chunker_version=ChunkerVersionRecord(
            implementation=row["chunker_implementation"],
            contract_version=row["chunker_contract_version"],
        ),
        tokeniser_identity=ChunkTokeniserIdentityRecord(
            implementation=row["tokeniser_implementation"],
            contract_version=row["tokeniser_contract_version"],
            artefact_identity=row["tokeniser_artefact_identity"],
            artefact_revision=row["tokeniser_artefact_revision"],
            options=_parse_tokeniser_options(row["tokeniser_options"]),
        ),
        policy=ChunkingPolicyRecord(
            min_tokens=row["min_tokens"],
            max_tokens=row["max_tokens"],
            overlap_numerator=row["overlap_numerator"],
            overlap_denominator=row["overlap_denominator"],
            structural_boundary_policy=ChunkStructuralBoundaryPolicy(
                row["structural_boundary_policy"]
            ),
            oversize_unit_policy=ChunkOversizeUnitPolicy(row["oversize_unit_policy"]),
        ),
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm(row["configuration_algorithm"]),
            digest=row["configuration_digest"],
        ),
        payload_digest=OutputDigest(
            algorithm=DigestAlgorithm(row["payload_digest_algorithm"]),
            digest=row["payload_digest"],
        ),
        canonical_payload=row["canonical_payload"],
        payload_schema_version=row["payload_schema_version"],
    )


def get_chunk_set_selection(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
    configuration_identity: ConfigurationIdentity,
) -> ChunkSetSelectionRecord | None:
    """Return one exact chunk set only when its version is corpus-visible."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection.")
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")
    if not isinstance(configuration_identity, ConfigurationIdentity):
        raise TypeError("configuration_identity must be a ConfigurationIdentity.")

    try:
        rows = connection.execute(
            """SELECT chunk_sets.*
            FROM corpus_document_version_memberships AS membership
            JOIN chunk_sets
                ON chunk_sets.document_id = membership.document_id
                AND chunk_sets.document_version_id = membership.document_version_id
            WHERE membership.corpus_id = ?
                AND membership.document_version_id = ?
                AND chunk_sets.payload_schema_version = ?
                AND chunk_sets.configuration_algorithm = ?
                AND chunk_sets.configuration_digest = ?""",
            (
                corpus_id.as_str(),
                document_version_id.as_str(),
                CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION,
                configuration_identity.algorithm.value,
                configuration_identity.digest,
            ),
        ).fetchall()
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None

    if not rows:
        return None
    if len(rows) != 1:
        raise CorruptStorageError()

    try:
        selection = _reconstruct_selection(rows[0])
    except (KeyError, IndexError, TypeError, ValueError):
        raise CorruptStorageError() from None
    if selection.configuration_identity != configuration_identity:
        raise CorruptStorageError()
    return selection
