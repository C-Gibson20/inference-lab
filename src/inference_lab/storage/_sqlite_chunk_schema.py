"""SQLite schema statements for immutable canonical chunk-set metadata."""

from typing import Final

CHUNK_SCHEMA_STATEMENTS: Final = (
    """CREATE TABLE chunk_sets (
    storage_id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    payload_schema_version TEXT NOT NULL
        CHECK (payload_schema_version = '1.0.0'),
    parser_implementation TEXT NOT NULL
        CHECK (
            length(parser_implementation) > 0
            AND substr(parser_implementation, 1, 1) GLOB '[a-z]'
            AND parser_implementation = lower(parser_implementation)
            AND parser_implementation NOT GLOB '*[^a-z0-9-]*'
            AND parser_implementation NOT GLOB '*--*'
            AND substr(parser_implementation, -1, 1) != '-'
        ),
    parser_contract_version TEXT NOT NULL
        CHECK (length(trim(parser_contract_version)) > 0),
    parser_normalisation_version TEXT NOT NULL
        CHECK (length(trim(parser_normalisation_version)) > 0),
    chunker_implementation TEXT NOT NULL
        CHECK (
            length(chunker_implementation) > 0
            AND substr(chunker_implementation, 1, 1) GLOB '[a-z]'
            AND chunker_implementation = lower(chunker_implementation)
            AND chunker_implementation NOT GLOB '*[^a-z0-9-]*'
            AND chunker_implementation NOT GLOB '*--*'
            AND substr(chunker_implementation, -1, 1) != '-'
        ),
    chunker_contract_version TEXT NOT NULL
        CHECK (length(trim(chunker_contract_version)) > 0),
    tokeniser_implementation TEXT NOT NULL
        CHECK (
            length(tokeniser_implementation) > 0
            AND substr(tokeniser_implementation, 1, 1) GLOB '[a-z]'
            AND tokeniser_implementation = lower(tokeniser_implementation)
            AND tokeniser_implementation NOT GLOB '*[^a-z0-9-]*'
            AND tokeniser_implementation NOT GLOB '*--*'
            AND substr(tokeniser_implementation, -1, 1) != '-'
        ),
    tokeniser_contract_version TEXT NOT NULL
        CHECK (length(trim(tokeniser_contract_version)) > 0),
    tokeniser_artefact_identity TEXT NOT NULL
        CHECK (
            length(trim(tokeniser_artefact_identity)) > 0
            AND instr(tokeniser_artefact_identity, char(0)) = 0
            AND instr(tokeniser_artefact_identity, '/') = 0
            AND instr(tokeniser_artefact_identity, char(92)) = 0
        ),
    tokeniser_artefact_revision TEXT NOT NULL
        CHECK (
            length(trim(tokeniser_artefact_revision)) > 0
            AND instr(tokeniser_artefact_revision, char(0)) = 0
            AND instr(tokeniser_artefact_revision, '/') = 0
            AND instr(tokeniser_artefact_revision, char(92)) = 0
        ),
    tokeniser_options TEXT NOT NULL
        CHECK (
            instr(tokeniser_options, char(0)) = 0
            AND instr(tokeniser_options, '/') = 0
            AND instr(tokeniser_options, char(92)) = 0
        ),
    min_tokens INTEGER NOT NULL CHECK (min_tokens > 0),
    max_tokens INTEGER NOT NULL CHECK (max_tokens >= min_tokens),
    overlap_numerator INTEGER NOT NULL
        CHECK (overlap_numerator >= 0),
    overlap_denominator INTEGER NOT NULL
        CHECK (
            overlap_denominator > 0
            AND overlap_numerator <= overlap_denominator
        ),
    structural_boundary_policy TEXT NOT NULL
        CHECK (structural_boundary_policy = 'structure-aware-greedy'),
    oversize_unit_policy TEXT NOT NULL
        CHECK (oversize_unit_policy = 'token-window'),
    configuration_algorithm TEXT NOT NULL
        CHECK (configuration_algorithm = 'sha256'),
    configuration_digest TEXT NOT NULL
        CHECK (
            length(configuration_digest) = 64
            AND configuration_digest NOT GLOB '*[^0-9a-f]*'
        ),
    payload_digest_algorithm TEXT NOT NULL
        CHECK (payload_digest_algorithm = 'sha256'),
    payload_digest TEXT NOT NULL
        CHECK (
            length(payload_digest) = 64
            AND payload_digest NOT GLOB '*[^0-9a-f]*'
        ),
    canonical_payload BLOB NOT NULL CHECK (length(canonical_payload) > 0),
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    UNIQUE (
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
        configuration_digest
    ),
    UNIQUE (payload_digest_algorithm, payload_digest)
) STRICT""",
    """CREATE INDEX chunk_sets_selection_idx
ON chunk_sets (
    document_version_id,
    configuration_algorithm,
    configuration_digest,
    payload_schema_version
)""",
)
