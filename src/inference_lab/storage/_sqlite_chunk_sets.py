"""Internal SQLite readers for corpus-authorised canonical chunk sets."""

import sqlite3

from inference_lab._json import canonical_json_text, load_json_object
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    DigestAlgorithm,
    DocumentId,
    DocumentVersionId,
    OutputDigest,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage._sqlite_corpus_memberships import (
    get_corpus_document_version_membership,
)
from inference_lab.storage.chunk_records import (
    CURRENT_CHUNK_PAYLOAD_SCHEMA_VERSION,
    ChunkerVersionRecord,
    ChunkingPolicyRecord,
    ChunkOversizeUnitPolicy,
    ChunkParserVersionRecord,
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    ChunkStructuralBoundaryPolicy,
    ChunkTokeniserIdentityRecord,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
)
from inference_lab.storage.outcomes import WriteChunkSetOutcome

_SELECTION_KEY_COLUMNS = (
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
)


def _selection_key_values(
    selection: ChunkSetSelectionRecord,
) -> tuple[object, ...]:
    """Return values matching the schema's immutable selection key."""
    return (
        selection.document_version_id.as_str(),
        selection.payload_schema_version,
        selection.parser_version.implementation,
        selection.parser_version.contract_version,
        selection.parser_version.normalisation_version,
        selection.chunker_version.implementation,
        selection.chunker_version.contract_version,
        selection.tokeniser_identity.implementation,
        selection.tokeniser_identity.contract_version,
        selection.tokeniser_identity.artefact_identity,
        selection.tokeniser_identity.artefact_revision,
        _serialise_tokeniser_options(selection.tokeniser_identity.options),
        selection.policy.min_tokens,
        selection.policy.max_tokens,
        selection.policy.overlap_numerator,
        selection.policy.overlap_denominator,
        selection.policy.structural_boundary_policy.value,
        selection.policy.oversize_unit_policy.value,
        selection.configuration_identity.algorithm.value,
        selection.configuration_identity.digest,
    )


def _reconstruct_existing_selection(
    row: sqlite3.Row,
) -> ChunkSetSelectionRecord:
    """Reconstruct an existing row or fail closed as corrupt storage."""
    try:
        return _reconstruct_selection(row)
    except (KeyError, IndexError, TypeError, ValueError):
        raise CorruptStorageError() from None


def _parse_tokeniser_options(value: object) -> tuple[tuple[str, str], ...]:
    """Reconstruct the canonical storage representation of tokeniser options."""
    if not isinstance(value, str):
        raise TypeError("tokeniser_options must be a string.")
    if not value:
        return ()

    if value.startswith("{"):
        payload = load_json_object(value.encode("utf-8"))
        if canonical_json_text(payload) != value:
            raise ValueError("tokeniser_options is not canonical.")
        if any(not isinstance(option_value, str) for option_value in payload.values()):
            raise ValueError("tokeniser_options values must be strings.")
        return tuple(
            (key, option_value)
            for key, option_value in payload.items()
            if isinstance(option_value, str)
        )

    options: list[tuple[str, str]] = []
    for item in value.split(","):
        key, separator, option_value = item.partition("=")
        if not separator or not key or not option_value:
            raise ValueError("tokeniser_options is not canonical.")
        options.append((key, option_value))
    return tuple(options)


def _serialise_tokeniser_options(
    options: tuple[tuple[str, str], ...],
) -> str:
    """Encode options canonically without delimiter ambiguity."""
    return canonical_json_text(dict(options))


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


def write_chunk_set(
    connection: sqlite3.Connection,
    request: ChunkSetWriteRequest,
) -> WriteChunkSetOutcome:
    """Insert one corpus-authorised immutable chunk-set row."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection.")
    if not isinstance(request, ChunkSetWriteRequest):
        raise TypeError("request must be a ChunkSetWriteRequest.")

    selection = request.selection
    membership = get_corpus_document_version_membership(
        connection,
        request.corpus_id,
        selection.document_version_id,
    )
    if membership is None or membership.document_id != selection.document_id:
        raise StorageConflictError()

    try:
        selection_key_clause = " AND ".join(
            f"{column} = ?" for column in _SELECTION_KEY_COLUMNS
        )
        existing_rows = connection.execute(
            f"SELECT * FROM chunk_sets WHERE {selection_key_clause}",
            _selection_key_values(selection),
        ).fetchall()
        if len(existing_rows) > 1:
            raise CorruptStorageError()
        if existing_rows:
            existing = _reconstruct_existing_selection(existing_rows[0])
            if existing == selection:
                return WriteChunkSetOutcome(selection=existing, replayed=True)
            raise StorageConflictError()

        digest_rows = connection.execute(
            """SELECT * FROM chunk_sets
            WHERE payload_digest_algorithm = ? AND payload_digest = ?""",
            (
                selection.payload_digest.algorithm.value,
                selection.payload_digest.digest,
            ),
        ).fetchall()
        if len(digest_rows) > 1:
            raise CorruptStorageError()
        if digest_rows:
            _reconstruct_existing_selection(digest_rows[0])
            raise StorageConflictError()

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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                selection.document_id.as_str(),
                selection.document_version_id.as_str(),
                selection.payload_schema_version,
                selection.parser_version.implementation,
                selection.parser_version.contract_version,
                selection.parser_version.normalisation_version,
                selection.chunker_version.implementation,
                selection.chunker_version.contract_version,
                selection.tokeniser_identity.implementation,
                selection.tokeniser_identity.contract_version,
                selection.tokeniser_identity.artefact_identity,
                selection.tokeniser_identity.artefact_revision,
                _serialise_tokeniser_options(selection.tokeniser_identity.options),
                selection.policy.min_tokens,
                selection.policy.max_tokens,
                selection.policy.overlap_numerator,
                selection.policy.overlap_denominator,
                selection.policy.structural_boundary_policy.value,
                selection.policy.oversize_unit_policy.value,
                selection.configuration_identity.algorithm.value,
                selection.configuration_identity.digest,
                selection.payload_digest.algorithm.value,
                selection.payload_digest.digest,
                selection.canonical_payload,
            ),
        )
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None

    return WriteChunkSetOutcome(selection=selection)
