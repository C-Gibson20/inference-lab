"""Internal SQLite metadata baseline and migration definitions."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Final

from inference_lab.storage._sqlite_version_schema import (
    CORPUS_VERSION_SCHEMA_STATEMENTS,
)

_MIGRATION_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_TRANSACTION_CONTROL_PATTERN: Final = re.compile(
    r"^(?:begin|commit|end|rollback|savepoint|release)\b",
    flags=re.IGNORECASE,
)

# ASCII "ILAB". This distinguishes metadata repositories from unrelated SQLite files.
SQLITE_APPLICATION_ID: Final = 0x494C4142


@dataclass(frozen=True, slots=True)
class MigrationDefinition:
    """Describe one deterministic, transaction-neutral schema upgrade."""

    revision: int
    migration_id: str
    statements: tuple[str, ...]
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        definition = json.dumps(
            {
                "migration_id": self.migration_id,
                "revision": self.revision,
                "statements": self.statements,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        object.__setattr__(self, "checksum", hashlib.sha256(definition).hexdigest())

    def _validate(self) -> None:
        """Validate the migration definition."""
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("revision must be an integer.")
        if self.revision <= 0:
            raise ValueError("revision must be greater than zero.")
        if not isinstance(self.migration_id, str):
            raise TypeError("migration_id must be a string.")
        if _MIGRATION_ID_PATTERN.fullmatch(self.migration_id) is None:
            raise ValueError("migration_id must be a lowercase underscore slug.")
        if not isinstance(self.statements, tuple):
            raise TypeError("statements must be a tuple.")
        if not self.statements:
            raise ValueError("statements must not be empty.")

        for statement in self.statements:
            if not isinstance(statement, str):
                raise TypeError("statements must contain only strings.")
            if not statement or statement != statement.strip():
                raise ValueError("migration statements must be non-empty and stripped.")
            if _TRANSACTION_CONTROL_PATTERN.match(statement) is not None:
                raise ValueError(
                    "migration statements must not contain transaction control."
                )


_CORE_SCHEMA_STATEMENTS: Final = (
    """CREATE TABLE metadata_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_name TEXT NOT NULL UNIQUE
        CHECK (schema_name = 'inference_lab_metadata'),
    current_revision INTEGER NOT NULL CHECK (current_revision >= 0)
) STRICT""",
    """INSERT INTO metadata_schema (singleton, schema_name, current_revision)
VALUES (1, 'inference_lab_metadata', 0)""",
    """CREATE TABLE migration_history (
    revision INTEGER PRIMARY KEY CHECK (revision > 0),
    migration_id TEXT NOT NULL UNIQUE
        CHECK (
            length(migration_id) > 0
            AND substr(migration_id, 1, 1) GLOB '[a-z]'
            AND migration_id = lower(migration_id)
            AND migration_id NOT GLOB '*[^a-z0-9_]*'
        ),
    checksum TEXT NOT NULL
        CHECK (
            length(checksum) = 64
            AND checksum NOT GLOB '*[^0-9a-f]*'
        ),
    applied_at_utc TEXT NOT NULL CHECK (length(applied_at_utc) > 0)
) STRICT""",
    """CREATE TABLE corpora (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL UNIQUE
        CHECK (
            length(corpus_id) = 36
            AND corpus_id = lower(corpus_id)
            AND substr(corpus_id, 9, 1) = '-'
            AND substr(corpus_id, 14, 1) = '-'
            AND substr(corpus_id, 19, 1) = '-'
            AND substr(corpus_id, 24, 1) = '-'
            AND length(replace(corpus_id, '-', '')) = 32
            AND corpus_id NOT GLOB '*[^0-9a-f-]*'
        ),
    display_name TEXT NOT NULL
        CHECK (
            length(trim(display_name)) > 0
            AND instr(display_name, char(0)) = 0
        )
) STRICT""",
    """CREATE TABLE documents (
    storage_id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE
        CHECK (
            length(document_id) = 36
            AND document_id = lower(document_id)
            AND substr(document_id, 9, 1) = '-'
            AND substr(document_id, 14, 1) = '-'
            AND substr(document_id, 19, 1) = '-'
            AND substr(document_id, 24, 1) = '-'
            AND length(replace(document_id, '-', '')) = 32
            AND document_id NOT GLOB '*[^0-9a-f-]*'
        )
) STRICT""",
    """CREATE TABLE document_versions (
    storage_id INTEGER PRIMARY KEY,
    document_version_id TEXT NOT NULL UNIQUE
        CHECK (
            length(document_version_id) = 36
            AND document_version_id = lower(document_version_id)
            AND substr(document_version_id, 9, 1) = '-'
            AND substr(document_version_id, 14, 1) = '-'
            AND substr(document_version_id, 19, 1) = '-'
            AND substr(document_version_id, 24, 1) = '-'
            AND length(replace(document_version_id, '-', '')) = 32
            AND document_version_id NOT GLOB '*[^0-9a-f-]*'
        ),
    document_id TEXT NOT NULL,
    content_hash_algorithm TEXT NOT NULL
        CHECK (content_hash_algorithm = 'sha256'),
    content_hash_digest TEXT NOT NULL
        CHECK (
            length(content_hash_digest) = 64
            AND content_hash_digest NOT GLOB '*[^0-9a-f]*'
        ),
    FOREIGN KEY (document_id) REFERENCES documents (document_id),
    UNIQUE (document_version_id, document_id)
) STRICT""",
    """CREATE INDEX document_versions_content_hash_idx
ON document_versions (content_hash_algorithm, content_hash_digest)""",
    """CREATE TABLE external_identifiers (
    storage_id INTEGER PRIMARY KEY,
    document_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    namespace TEXT NOT NULL
        CHECK (
            length(namespace) > 0
            AND substr(namespace, 1, 1) GLOB '[a-z]'
            AND namespace NOT GLOB '*[^a-z0-9_-]*'
        ),
    identifier_value TEXT NOT NULL CHECK (length(trim(identifier_value)) > 0),
    FOREIGN KEY (document_version_id)
        REFERENCES document_versions (document_version_id),
    UNIQUE (document_version_id, ordinal),
    UNIQUE (document_version_id, namespace, identifier_value)
) STRICT""",
    """CREATE INDEX external_identifiers_lookup_idx
ON external_identifiers (namespace, identifier_value, document_version_id)""",
    """CREATE TABLE duplicate_relationships (
    storage_id INTEGER PRIMARY KEY,
    duplicate_class TEXT NOT NULL CHECK (duplicate_class = 'byte_identical'),
    first_version_id TEXT NOT NULL,
    second_version_id TEXT NOT NULL,
    FOREIGN KEY (first_version_id)
        REFERENCES document_versions (document_version_id),
    FOREIGN KEY (second_version_id)
        REFERENCES document_versions (document_version_id),
    CHECK (first_version_id < second_version_id),
    UNIQUE (duplicate_class, first_version_id, second_version_id)
) STRICT""",
    """CREATE INDEX duplicate_relationships_second_version_idx
ON duplicate_relationships (second_version_id, duplicate_class, first_version_id)""",
    """CREATE TABLE document_families (
    storage_id INTEGER PRIMARY KEY,
    family_id TEXT NOT NULL UNIQUE
        CHECK (
            length(family_id) = 36
            AND family_id = lower(family_id)
            AND substr(family_id, 9, 1) = '-'
            AND substr(family_id, 14, 1) = '-'
            AND substr(family_id, 19, 1) = '-'
            AND substr(family_id, 24, 1) = '-'
            AND length(replace(family_id, '-', '')) = 32
            AND family_id NOT GLOB '*[^0-9a-f-]*'
        )
) STRICT""",
    """CREATE TABLE document_family_memberships (
    storage_id INTEGER PRIMARY KEY,
    family_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_version_id TEXT,
    FOREIGN KEY (family_id) REFERENCES document_families (family_id),
    FOREIGN KEY (document_id) REFERENCES documents (document_id),
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id)
) STRICT""",
    """CREATE UNIQUE INDEX document_family_memberships_key_idx
ON document_family_memberships (
    family_id,
    document_id,
    coalesce(document_version_id, '')
)""",
    """CREATE INDEX document_family_memberships_document_idx
ON document_family_memberships (document_id, family_id, document_version_id)""",
    """CREATE INDEX document_family_memberships_version_idx
ON document_family_memberships (document_version_id, family_id, document_id)
WHERE document_version_id IS NOT NULL""",
)

_CORPUS_MEMBERSHIP_STATEMENTS: Final = (
    """CREATE TABLE corpus_document_version_memberships (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    FOREIGN KEY (corpus_id) REFERENCES corpora (corpus_id),
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    UNIQUE (corpus_id, document_version_id)
) STRICT""",
    """CREATE INDEX corpus_document_version_memberships_document_idx
ON corpus_document_version_memberships (
    corpus_id,
    document_id,
    document_version_id
)""",
    """CREATE INDEX corpus_document_version_memberships_version_idx
ON corpus_document_version_memberships (document_version_id, corpus_id)""",
)

_SOURCE_OBSERVATION_STATEMENTS: Final = (
    """CREATE TABLE source_observations (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    current_document_version_id TEXT NOT NULL,
    current_revision INTEGER NOT NULL CHECK (current_revision >= 1),
    FOREIGN KEY (corpus_id) REFERENCES corpora (corpus_id),
    FOREIGN KEY (current_document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    FOREIGN KEY (corpus_id, current_document_version_id)
        REFERENCES corpus_document_version_memberships (
            corpus_id,
            document_version_id
        ),
    UNIQUE (source_observation_id, corpus_id, document_id),
    CHECK (
        length(source_observation_id) = 36
        AND source_observation_id = lower(source_observation_id)
        AND substr(source_observation_id, 9, 1) = '-'
        AND substr(source_observation_id, 14, 1) = '-'
        AND substr(source_observation_id, 19, 1) = '-'
        AND substr(source_observation_id, 24, 1) = '-'
        AND length(replace(source_observation_id, '-', '')) = 32
        AND source_observation_id NOT GLOB '*[^0-9a-f-]*'
    )
) STRICT""",
    """CREATE INDEX source_observations_corpus_idx
ON source_observations (corpus_id, source_observation_id)""",
    """CREATE TABLE source_observation_history (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL,
    observation_revision INTEGER NOT NULL CHECK (observation_revision >= 1),
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    FOREIGN KEY (source_observation_id, corpus_id, document_id)
        REFERENCES source_observations (
            source_observation_id,
            corpus_id,
            document_id
        ),
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    FOREIGN KEY (corpus_id, document_version_id)
        REFERENCES corpus_document_version_memberships (
            corpus_id,
            document_version_id
        ),
    UNIQUE (source_observation_id, observation_revision),
    UNIQUE (source_observation_id, document_version_id),
    UNIQUE (source_observation_id, corpus_id, document_version_id, document_id)
) STRICT""",
    """CREATE INDEX source_observation_history_version_idx
ON source_observation_history (
    corpus_id,
    document_version_id,
    source_observation_id
)""",
    """CREATE TABLE external_identifier_assertions (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    namespace TEXT NOT NULL
        CHECK (
            length(namespace) > 0
            AND substr(namespace, 1, 1) GLOB '[a-z]'
            AND namespace NOT GLOB '*[^a-z0-9_-]*'
        ),
    identifier_value TEXT NOT NULL CHECK (length(trim(identifier_value)) > 0),
    FOREIGN KEY (
        source_observation_id,
        corpus_id,
        document_version_id,
        document_id
    ) REFERENCES source_observation_history (
        source_observation_id,
        corpus_id,
        document_version_id,
        document_id
    ),
    UNIQUE (
        source_observation_id,
        document_version_id,
        namespace,
        identifier_value
    )
) STRICT""",
    """CREATE INDEX external_identifier_assertions_lookup_idx
ON external_identifier_assertions (
    corpus_id,
    namespace,
    identifier_value,
    document_version_id,
    source_observation_id
)""",
)

_DUPLICATE_INTEGRITY_STATEMENTS: Final = (
    """CREATE TRIGGER duplicate_relationships_hash_insert
BEFORE INSERT ON duplicate_relationships
WHEN NEW.duplicate_class = 'byte_identical'
    AND NOT EXISTS (
        SELECT 1
        FROM document_versions AS first
        JOIN document_versions AS second
            ON second.document_version_id = NEW.second_version_id
        WHERE first.document_version_id = NEW.first_version_id
            AND first.content_hash_algorithm = second.content_hash_algorithm
            AND first.content_hash_digest = second.content_hash_digest
    )
BEGIN
    SELECT RAISE(ABORT, 'byte-identical duplicate hashes must match');
END""",
    """CREATE TRIGGER duplicate_relationships_hash_update
BEFORE UPDATE ON duplicate_relationships
WHEN NEW.duplicate_class = 'byte_identical'
    AND NOT EXISTS (
        SELECT 1
        FROM document_versions AS first
        JOIN document_versions AS second
            ON second.document_version_id = NEW.second_version_id
        WHERE first.document_version_id = NEW.first_version_id
            AND first.content_hash_algorithm = second.content_hash_algorithm
            AND first.content_hash_digest = second.content_hash_digest
    )
BEGIN
    SELECT RAISE(ABORT, 'byte-identical duplicate hashes must match');
END""",
    """CREATE TRIGGER document_versions_duplicate_hash_update
BEFORE UPDATE OF content_hash_algorithm, content_hash_digest ON document_versions
WHEN EXISTS (
    SELECT 1
    FROM duplicate_relationships AS relationship
    JOIN document_versions AS other
        ON other.document_version_id = CASE
            WHEN relationship.first_version_id = OLD.document_version_id
                THEN relationship.second_version_id
            ELSE relationship.first_version_id
        END
    WHERE OLD.document_version_id IN (
        relationship.first_version_id,
        relationship.second_version_id
    )
        AND (
            NEW.content_hash_algorithm != other.content_hash_algorithm
            OR NEW.content_hash_digest != other.content_hash_digest
        )
)
BEGIN
    SELECT RAISE(ABORT, 'byte-identical duplicate hashes must match');
END""",
)

_SOURCE_LOCATOR_CONTINUITY_STATEMENTS: Final = (
    """CREATE UNIQUE INDEX source_observations_owner_key_idx
ON source_observations (corpus_id, source_observation_id)""",
    """CREATE TABLE source_locator_revisions (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL,
    locator_revision INTEGER NOT NULL CHECK (locator_revision >= 1),
    locator_path TEXT NOT NULL
        CHECK (
            length(locator_path) > 0
            AND instr(locator_path, char(0)) = 0
        ),
    device_id INTEGER CHECK (device_id >= 0),
    file_id INTEGER CHECK (file_id >= 0),
    content_hash_algorithm TEXT NOT NULL
        CHECK (content_hash_algorithm = 'sha256'),
    content_hash_digest TEXT NOT NULL
        CHECK (
            length(content_hash_digest) = 64
            AND content_hash_digest NOT GLOB '*[^0-9a-f]*'
        ),
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    modified_time_ns INTEGER NOT NULL CHECK (modified_time_ns >= 0),
    FOREIGN KEY (corpus_id, source_observation_id)
        REFERENCES source_observations (corpus_id, source_observation_id),
    CHECK (
        (device_id IS NULL AND file_id IS NULL)
        OR (device_id IS NOT NULL AND file_id IS NOT NULL)
    ),
    UNIQUE (source_observation_id, locator_revision),
    UNIQUE (
        corpus_id,
        source_observation_id,
        locator_revision,
        locator_path
    )
) STRICT""",
    """CREATE TRIGGER source_locator_revisions_sequential_insert
BEFORE INSERT ON source_locator_revisions
WHEN NEW.locator_revision != coalesce(
    (
        SELECT max(existing.locator_revision) + 1
        FROM source_locator_revisions AS existing
        WHERE existing.source_observation_id = NEW.source_observation_id
    ),
    1
)
BEGIN
    SELECT RAISE(ABORT, 'source locator revisions must be sequential');
END""",
    """CREATE TRIGGER source_locator_revisions_no_update
BEFORE UPDATE ON source_locator_revisions
BEGIN
    SELECT RAISE(ABORT, 'source locator revisions are append-only');
END""",
    """CREATE TRIGGER source_locator_revisions_no_delete
BEFORE DELETE ON source_locator_revisions
BEGIN
    SELECT RAISE(ABORT, 'source locator revisions are append-only');
END""",
    """CREATE INDEX source_locator_revisions_path_idx
ON source_locator_revisions (
    corpus_id,
    locator_path,
    source_observation_id,
    locator_revision
)""",
    """CREATE INDEX source_locator_revisions_hint_idx
ON source_locator_revisions (
    corpus_id,
    device_id,
    file_id,
    source_observation_id,
    locator_revision
)
WHERE device_id IS NOT NULL""",
    """CREATE TABLE current_source_locators (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL UNIQUE,
    locator_revision INTEGER NOT NULL CHECK (locator_revision >= 1),
    locator_path TEXT NOT NULL
        CHECK (
            length(locator_path) > 0
            AND instr(locator_path, char(0)) = 0
        ),
    FOREIGN KEY (
        corpus_id,
        source_observation_id,
        locator_revision,
        locator_path
    ) REFERENCES source_locator_revisions (
        corpus_id,
        source_observation_id,
        locator_revision,
        locator_path
    ),
    UNIQUE (corpus_id, locator_path),
    UNIQUE (corpus_id, source_observation_id)
) STRICT""",
)

_BASELINE_STATEMENTS: Final = (
    *_CORE_SCHEMA_STATEMENTS,
    *_CORPUS_MEMBERSHIP_STATEMENTS,
    *_SOURCE_OBSERVATION_STATEMENTS,
    *_DUPLICATE_INTEGRITY_STATEMENTS,
)

_SOURCE_PERSISTENCE_STATEMENTS: Final = (
    """CREATE UNIQUE INDEX document_versions_source_artefact_owner_idx
ON document_versions (
    document_version_id,
    document_id,
    content_hash_algorithm,
    content_hash_digest
)""",
    """CREATE TABLE source_artefacts (
    storage_id INTEGER PRIMARY KEY,
    document_version_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    record_schema_version TEXT NOT NULL
        CHECK (record_schema_version = '1.0.0'),
    content_hash_algorithm TEXT NOT NULL
        CHECK (content_hash_algorithm = 'sha256'),
    content_hash_digest TEXT NOT NULL
        CHECK (
            length(content_hash_digest) = 64
            AND content_hash_digest NOT GLOB '*[^0-9a-f]*'
        ),
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    media_type TEXT NOT NULL
        CHECK (
            instr(media_type, '/') > 1
            AND instr(substr(media_type, instr(media_type, '/') + 1), '/') = 0
            AND media_type = lower(media_type)
            AND substr(media_type, 1, 1) GLOB '[a-z0-9]'
            AND substr(media_type, instr(media_type, '/') + 1, 1)
                GLOB '[a-z0-9]'
            AND media_type NOT GLOB '*[^a-z0-9!#$&^_.+/-]*'
        ),
    storage_key TEXT NOT NULL
        CHECK (
            storage_key = 'sha256/'
                || substr(content_hash_digest, 1, 2)
                || '/'
                || content_hash_digest
        ),
    FOREIGN KEY (
        document_version_id,
        document_id,
        content_hash_algorithm,
        content_hash_digest
    ) REFERENCES document_versions (
        document_version_id,
        document_id,
        content_hash_algorithm,
        content_hash_digest
    )
) STRICT""",
    """CREATE INDEX source_artefacts_blob_reconciliation_idx
ON source_artefacts (storage_key, document_version_id)""",
    """CREATE TABLE parsed_representations (
    storage_id INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE
        CHECK (
            length(record_id) = 36
            AND record_id = lower(record_id)
            AND substr(record_id, 9, 1) = '-'
            AND substr(record_id, 14, 1) = '-'
            AND substr(record_id, 19, 1) = '-'
            AND substr(record_id, 24, 1) = '-'
            AND length(replace(record_id, '-', '')) = 32
            AND record_id NOT GLOB '*[^0-9a-f-]*'
        ),
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    record_schema_version TEXT NOT NULL
        CHECK (record_schema_version = '1.0.0'),
    source_text TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
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
        CHECK (
            length(trim(parser_contract_version)) > 0
            AND instr(parser_contract_version, char(0)) = 0
        ),
    parser_artefact_identity TEXT,
    parser_artefact_revision TEXT,
    normalisation_version TEXT NOT NULL
        CHECK (
            length(trim(normalisation_version)) > 0
            AND instr(normalisation_version, char(0)) = 0
        ),
    ocr_implementation TEXT,
    ocr_contract_version TEXT,
    ocr_artefact_identity TEXT,
    ocr_artefact_revision TEXT,
    language TEXT
        CHECK (
            language IS NULL
            OR (
                length(trim(language)) > 0
                AND instr(language, char(0)) = 0
            )
        ),
    page_count INTEGER CHECK (page_count IS NULL OR page_count >= 0),
    character_count INTEGER NOT NULL CHECK (character_count >= 0),
    token_count INTEGER CHECK (token_count IS NULL OR token_count >= 0),
    page_collection_availability TEXT NOT NULL
        CHECK (page_collection_availability IN ('available', 'unavailable')),
    section_collection_availability TEXT NOT NULL
        CHECK (section_collection_availability IN ('available', 'unavailable')),
    paragraph_collection_availability TEXT NOT NULL
        CHECK (paragraph_collection_availability IN ('available', 'unavailable')),
    output_digest_algorithm TEXT NOT NULL
        CHECK (output_digest_algorithm = 'sha256'),
    output_digest TEXT NOT NULL
        CHECK (
            length(output_digest) = 64
            AND output_digest NOT GLOB '*[^0-9a-f]*'
        ),
    supersedes_record_id TEXT,
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    FOREIGN KEY (supersedes_record_id)
        REFERENCES parsed_representations (record_id),
    UNIQUE (record_id, document_version_id),
    CHECK (supersedes_record_id IS NULL OR supersedes_record_id != record_id),
    CHECK (
        (parser_artefact_identity IS NULL)
            = (parser_artefact_revision IS NULL)
    ),
    CHECK (
        parser_artefact_identity IS NULL
        OR (
            length(trim(parser_artefact_identity)) > 0
            AND instr(parser_artefact_identity, char(0)) = 0
            AND length(trim(parser_artefact_revision)) > 0
            AND instr(parser_artefact_revision, char(0)) = 0
        )
    ),
    CHECK (
        (ocr_implementation IS NULL)
            = (ocr_contract_version IS NULL)
        AND (ocr_artefact_identity IS NULL)
            = (ocr_artefact_revision IS NULL)
        AND (ocr_implementation IS NOT NULL OR ocr_artefact_identity IS NULL)
    ),
    CHECK (
        ocr_implementation IS NULL
        OR (
            length(ocr_implementation) > 0
            AND substr(ocr_implementation, 1, 1) GLOB '[a-z]'
            AND ocr_implementation = lower(ocr_implementation)
            AND ocr_implementation NOT GLOB '*[^a-z0-9-]*'
            AND ocr_implementation NOT GLOB '*--*'
            AND substr(ocr_implementation, -1, 1) != '-'
            AND length(trim(ocr_contract_version)) > 0
            AND instr(ocr_contract_version, char(0)) = 0
        )
    ),
    CHECK (
        ocr_artefact_identity IS NULL
        OR (
            length(trim(ocr_artefact_identity)) > 0
            AND instr(ocr_artefact_identity, char(0)) = 0
            AND length(trim(ocr_artefact_revision)) > 0
            AND instr(ocr_artefact_revision, char(0)) = 0
        )
    )
) STRICT""",
    """CREATE UNIQUE INDEX parsed_representations_natural_identity_idx
ON parsed_representations (
    document_version_id,
    parser_implementation,
    parser_contract_version,
    coalesce(parser_artefact_identity, char(0)),
    coalesce(parser_artefact_revision, char(0)),
    normalisation_version,
    coalesce(ocr_implementation, char(0)),
    coalesce(ocr_contract_version, char(0)),
    coalesce(ocr_artefact_identity, char(0)),
    coalesce(ocr_artefact_revision, char(0))
)""",
    """CREATE INDEX parsed_representations_version_idx
ON parsed_representations (document_version_id, record_id)""",
    """CREATE TABLE page_layouts (
    storage_id INTEGER PRIMARY KEY,
    parsed_record_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number >= 1),
    coordinate_unit TEXT NOT NULL
        CHECK (coordinate_unit IN ('pdf_points', 'image_pixels')),
    coordinate_origin TEXT NOT NULL
        CHECK (coordinate_origin IN ('top_left', 'bottom_left')),
    page_width REAL NOT NULL
        CHECK (page_width > 0 AND page_width <= 1.7976931348623157e308),
    page_height REAL NOT NULL
        CHECK (page_height > 0 AND page_height <= 1.7976931348623157e308),
    rotation INTEGER NOT NULL CHECK (rotation IN (0, 90, 180, 270)),
    content_origin TEXT NOT NULL CHECK (content_origin IN ('native', 'ocr')),
    producer_implementation TEXT NOT NULL,
    producer_contract_version TEXT NOT NULL,
    producer_artefact_identity TEXT,
    producer_artefact_revision TEXT,
    FOREIGN KEY (parsed_record_id, document_version_id)
        REFERENCES parsed_representations (record_id, document_version_id),
    UNIQUE (parsed_record_id, page_number),
    CHECK (
        length(producer_implementation) > 0
        AND substr(producer_implementation, 1, 1) GLOB '[a-z]'
        AND producer_implementation = lower(producer_implementation)
        AND producer_implementation NOT GLOB '*[^a-z0-9-]*'
        AND producer_implementation NOT GLOB '*--*'
        AND substr(producer_implementation, -1, 1) != '-'
    ),
    CHECK (
        length(trim(producer_contract_version)) > 0
        AND instr(producer_contract_version, char(0)) = 0
    ),
    CHECK (
        (producer_artefact_identity IS NULL)
            = (producer_artefact_revision IS NULL)
    ),
    CHECK (
        producer_artefact_identity IS NULL
        OR (
            length(trim(producer_artefact_identity)) > 0
            AND instr(producer_artefact_identity, char(0)) = 0
            AND length(trim(producer_artefact_revision)) > 0
            AND instr(producer_artefact_revision, char(0)) = 0
        )
    )
) STRICT""",
    """CREATE INDEX page_layouts_traversal_idx
ON page_layouts (parsed_record_id, page_number)""",
    """CREATE TABLE source_spans (
    storage_id INTEGER PRIMARY KEY,
    source_span_id TEXT NOT NULL UNIQUE
        CHECK (
            length(source_span_id) = 36
            AND source_span_id = lower(source_span_id)
            AND substr(source_span_id, 9, 1) = '-'
            AND substr(source_span_id, 14, 1) = '-'
            AND substr(source_span_id, 19, 1) = '-'
            AND substr(source_span_id, 24, 1) = '-'
            AND length(replace(source_span_id, '-', '')) = 32
            AND source_span_id NOT GLOB '*[^0-9a-f-]*'
        ),
    parsed_record_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    content_origin TEXT NOT NULL CHECK (content_origin IN ('native', 'ocr')),
    page_number INTEGER CHECK (page_number IS NULL OR page_number >= 1),
    bounding_box_left REAL,
    bounding_box_top REAL,
    bounding_box_right REAL,
    bounding_box_bottom REAL,
    FOREIGN KEY (parsed_record_id, document_version_id)
        REFERENCES parsed_representations (record_id, document_version_id),
    UNIQUE (parsed_record_id, source_span_id),
    CHECK (
        (bounding_box_left IS NULL)
            = (bounding_box_top IS NULL)
        AND (bounding_box_left IS NULL)
            = (bounding_box_right IS NULL)
        AND (bounding_box_left IS NULL)
            = (bounding_box_bottom IS NULL)
    ),
    CHECK (bounding_box_left IS NULL OR page_number IS NOT NULL),
    CHECK (
        bounding_box_left IS NULL
        OR (
            bounding_box_left >= 0
            AND bounding_box_top >= 0
            AND bounding_box_right >= bounding_box_left
            AND bounding_box_bottom >= bounding_box_top
            AND bounding_box_right <= 1.7976931348623157e308
            AND bounding_box_bottom <= 1.7976931348623157e308
        )
    )
) STRICT""",
    """CREATE INDEX source_spans_resolution_idx
ON source_spans (document_version_id, source_span_id)""",
    """CREATE TABLE structural_blocks (
    storage_id INTEGER PRIMARY KEY,
    parsed_record_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    block_kind TEXT NOT NULL
        CHECK (
            block_kind IN (
                'page',
                'section',
                'paragraph',
                'table',
                'figure_caption',
                'other'
            )
        ),
    block_text TEXT NOT NULL,
    source_span_id TEXT NOT NULL,
    FOREIGN KEY (parsed_record_id, document_version_id)
        REFERENCES parsed_representations (record_id, document_version_id),
    FOREIGN KEY (parsed_record_id, source_span_id)
        REFERENCES source_spans (parsed_record_id, source_span_id),
    UNIQUE (parsed_record_id, ordinal),
    UNIQUE (parsed_record_id, source_span_id)
) STRICT""",
    """CREATE INDEX structural_blocks_traversal_idx
ON structural_blocks (parsed_record_id, ordinal)""",
    """CREATE TABLE parse_outcomes (
    storage_id INTEGER PRIMARY KEY,
    parse_id TEXT NOT NULL UNIQUE
        CHECK (
            length(parse_id) = 36
            AND parse_id = lower(parse_id)
            AND substr(parse_id, 9, 1) = '-'
            AND substr(parse_id, 14, 1) = '-'
            AND substr(parse_id, 19, 1) = '-'
            AND substr(parse_id, 24, 1) = '-'
            AND length(replace(parse_id, '-', '')) = 32
            AND parse_id NOT GLOB '*[^0-9a-f-]*'
        ),
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    record_schema_version TEXT NOT NULL
        CHECK (record_schema_version = '1.0.0'),
    parse_status TEXT NOT NULL
        CHECK (
            parse_status IN (
                'succeeded',
                'completed_with_recoverable_failures',
                'failed'
            )
        ),
    parser_implementation TEXT NOT NULL,
    parser_contract_version TEXT NOT NULL,
    parser_artefact_identity TEXT,
    parser_artefact_revision TEXT,
    normalisation_version TEXT NOT NULL,
    ocr_implementation TEXT,
    ocr_contract_version TEXT,
    ocr_artefact_identity TEXT,
    ocr_artefact_revision TEXT,
    parsed_record_id TEXT,
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    FOREIGN KEY (parsed_record_id, document_version_id)
        REFERENCES parsed_representations (record_id, document_version_id),
    CHECK (
        (parse_status = 'failed' AND parsed_record_id IS NULL)
        OR (
            parse_status IN (
                'succeeded',
                'completed_with_recoverable_failures'
            )
            AND parsed_record_id IS NOT NULL
        )
    ),
    CHECK (
        length(parser_implementation) > 0
        AND substr(parser_implementation, 1, 1) GLOB '[a-z]'
        AND parser_implementation = lower(parser_implementation)
        AND parser_implementation NOT GLOB '*[^a-z0-9-]*'
        AND parser_implementation NOT GLOB '*--*'
        AND substr(parser_implementation, -1, 1) != '-'
    ),
    CHECK (
        length(trim(parser_contract_version)) > 0
        AND instr(parser_contract_version, char(0)) = 0
        AND length(trim(normalisation_version)) > 0
        AND instr(normalisation_version, char(0)) = 0
    ),
    CHECK (
        (parser_artefact_identity IS NULL)
            = (parser_artefact_revision IS NULL)
    ),
    CHECK (
        parser_artefact_identity IS NULL
        OR (
            length(trim(parser_artefact_identity)) > 0
            AND instr(parser_artefact_identity, char(0)) = 0
            AND length(trim(parser_artefact_revision)) > 0
            AND instr(parser_artefact_revision, char(0)) = 0
        )
    ),
    CHECK (
        (ocr_implementation IS NULL)
            = (ocr_contract_version IS NULL)
        AND (ocr_artefact_identity IS NULL)
            = (ocr_artefact_revision IS NULL)
        AND (ocr_implementation IS NOT NULL OR ocr_artefact_identity IS NULL)
    ),
    CHECK (
        ocr_implementation IS NULL
        OR (
            length(ocr_implementation) > 0
            AND substr(ocr_implementation, 1, 1) GLOB '[a-z]'
            AND ocr_implementation = lower(ocr_implementation)
            AND ocr_implementation NOT GLOB '*[^a-z0-9-]*'
            AND ocr_implementation NOT GLOB '*--*'
            AND substr(ocr_implementation, -1, 1) != '-'
            AND length(trim(ocr_contract_version)) > 0
            AND instr(ocr_contract_version, char(0)) = 0
        )
    ),
    CHECK (
        ocr_artefact_identity IS NULL
        OR (
            length(trim(ocr_artefact_identity)) > 0
            AND instr(ocr_artefact_identity, char(0)) = 0
            AND length(trim(ocr_artefact_revision)) > 0
            AND instr(ocr_artefact_revision, char(0)) = 0
        )
    )
) STRICT""",
    """CREATE INDEX parse_outcomes_version_idx
ON parse_outcomes (document_version_id, parse_id)""",
    """CREATE TABLE parse_failures (
    storage_id INTEGER PRIMARY KEY,
    parse_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    failure_category TEXT NOT NULL
        CHECK (
            failure_category IN (
                'invalid_input',
                'unsupported_format',
                'invalid_encoding',
                'decode_failed',
                'malformed_source',
                'malformed_unit',
                'unsupported_structure',
                'invalid_output',
                'source_too_large'
            )
        ),
    failure_severity TEXT NOT NULL
        CHECK (failure_severity IN ('fatal', 'recoverable')),
    failure_stage TEXT NOT NULL
        CHECK (failure_stage IN ('input', 'decode', 'structure', 'output')),
    unit_kind TEXT
        CHECK (
            unit_kind IS NULL
            OR unit_kind IN (
                'page',
                'section',
                'paragraph',
                'table',
                'figure_caption',
                'other'
            )
        ),
    unit_index INTEGER CHECK (unit_index IS NULL OR unit_index >= 0),
    FOREIGN KEY (parse_id) REFERENCES parse_outcomes (parse_id),
    UNIQUE (parse_id, ordinal),
    CHECK ((unit_kind IS NULL) = (unit_index IS NULL)),
    CHECK (
        (failure_severity = 'fatal' AND unit_kind IS NULL)
        OR (failure_severity = 'recoverable' AND unit_kind IS NOT NULL)
    ),
    CHECK (
        (failure_category = 'invalid_input'
            AND failure_severity = 'fatal'
            AND failure_stage = 'input')
        OR (failure_category = 'unsupported_format'
            AND failure_severity = 'fatal'
            AND failure_stage = 'input')
        OR (failure_category = 'invalid_encoding'
            AND failure_severity = 'fatal'
            AND failure_stage = 'input')
        OR (failure_category = 'source_too_large'
            AND failure_severity = 'fatal'
            AND failure_stage = 'input')
        OR (failure_category = 'decode_failed'
            AND failure_severity = 'fatal'
            AND failure_stage = 'decode')
        OR (failure_category = 'malformed_source'
            AND failure_severity = 'fatal'
            AND failure_stage = 'structure')
        OR (failure_category = 'malformed_unit'
            AND failure_severity = 'recoverable'
            AND failure_stage = 'structure')
        OR (failure_category = 'unsupported_structure'
            AND failure_severity = 'recoverable'
            AND failure_stage = 'structure')
        OR (failure_category = 'invalid_output'
            AND failure_severity = 'fatal'
            AND failure_stage = 'output')
    )
) STRICT""",
    """CREATE INDEX parse_failures_traversal_idx
ON parse_failures (parse_id, ordinal)""",
    """CREATE TRIGGER page_layouts_representation_insert
BEFORE INSERT ON page_layouts
WHEN EXISTS (
    SELECT 1
    FROM parsed_representations AS representation
    WHERE representation.record_id = NEW.parsed_record_id
        AND (
            (representation.page_count IS NOT NULL
                AND NEW.page_number > representation.page_count)
            OR (NEW.content_origin = 'ocr'
                AND (
                    representation.ocr_implementation IS NULL
                    OR NEW.producer_implementation
                        != representation.ocr_implementation
                    OR NEW.producer_contract_version
                        != representation.ocr_contract_version
                    OR NEW.producer_artefact_identity IS NOT
                        representation.ocr_artefact_identity
                    OR NEW.producer_artefact_revision IS NOT
                        representation.ocr_artefact_revision
                ))
        )
)
BEGIN
    SELECT RAISE(ABORT, 'page layout conflicts with parsed representation');
END""",
    """CREATE TRIGGER source_spans_representation_insert
BEFORE INSERT ON source_spans
WHEN EXISTS (
    SELECT 1
    FROM parsed_representations AS representation
    WHERE representation.record_id = NEW.parsed_record_id
        AND (
            NEW.end_offset > length(representation.source_text)
            OR (NEW.content_origin = 'ocr'
                AND representation.ocr_implementation IS NULL)
            OR (representation.page_count IS NOT NULL
                AND NEW.page_number > representation.page_count)
        )
)
BEGIN
    SELECT RAISE(ABORT, 'source span conflicts with parsed representation');
END""",
    """CREATE TRIGGER source_spans_layout_insert
BEFORE INSERT ON source_spans
WHEN NEW.bounding_box_left IS NOT NULL
    AND NOT EXISTS (
        SELECT 1
        FROM page_layouts AS layout
        WHERE layout.parsed_record_id = NEW.parsed_record_id
            AND layout.page_number = NEW.page_number
            AND NEW.bounding_box_right <= layout.page_width
            AND NEW.bounding_box_bottom <= layout.page_height
    )
BEGIN
    SELECT RAISE(ABORT, 'source span bounding box requires matching layout');
END""",
    """CREATE TRIGGER structural_blocks_contiguous_insert
BEFORE INSERT ON structural_blocks
WHEN NEW.ordinal != (
    SELECT count(*)
    FROM structural_blocks
    WHERE parsed_record_id = NEW.parsed_record_id
)
BEGIN
    SELECT RAISE(ABORT, 'structural block ordinals must be contiguous');
END""",
    """CREATE TRIGGER structural_blocks_availability_insert
BEFORE INSERT ON structural_blocks
WHEN EXISTS (
    SELECT 1
    FROM parsed_representations AS representation
    WHERE representation.record_id = NEW.parsed_record_id
        AND (
            (NEW.block_kind = 'page'
                AND representation.page_collection_availability = 'unavailable')
            OR (NEW.block_kind = 'section'
                AND representation.section_collection_availability = 'unavailable')
            OR (NEW.block_kind = 'paragraph'
                AND representation.paragraph_collection_availability = 'unavailable')
        )
)
BEGIN
    SELECT RAISE(ABORT, 'structural block collection is unavailable');
END""",
    """CREATE TRIGGER structural_blocks_page_insert
BEFORE INSERT ON structural_blocks
WHEN NEW.block_kind = 'page'
    AND NOT EXISTS (
        SELECT 1
        FROM source_spans AS span
        WHERE span.parsed_record_id = NEW.parsed_record_id
            AND span.source_span_id = NEW.source_span_id
            AND span.page_number IS NOT NULL
    )
BEGIN
    SELECT RAISE(ABORT, 'page block requires a page source span');
END""",
    """CREATE TRIGGER parse_failures_contiguous_insert
BEFORE INSERT ON parse_failures
WHEN NEW.ordinal != (
    SELECT count(*) FROM parse_failures WHERE parse_id = NEW.parse_id
)
BEGIN
    SELECT RAISE(ABORT, 'parse failure ordinals must be contiguous');
END""",
    """CREATE TRIGGER parse_failures_status_insert
BEFORE INSERT ON parse_failures
WHEN EXISTS (
    SELECT 1
    FROM parse_outcomes AS outcome
    WHERE outcome.parse_id = NEW.parse_id
        AND (
            outcome.parse_status = 'succeeded'
            OR (outcome.parse_status = 'completed_with_recoverable_failures'
                AND NEW.failure_severity != 'recoverable')
        )
)
BEGIN
    SELECT RAISE(ABORT, 'parse failure conflicts with outcome status');
END""",
)

MIGRATIONS: Final = (
    MigrationDefinition(
        revision=1,
        migration_id="multi_corpus_metadata_baseline",
        statements=_BASELINE_STATEMENTS,
    ),
    MigrationDefinition(
        revision=2,
        migration_id="source_locator_continuity",
        statements=_SOURCE_LOCATOR_CONTINUITY_STATEMENTS,
    ),
    MigrationDefinition(
        revision=3,
        migration_id="source_persistence",
        statements=_SOURCE_PERSISTENCE_STATEMENTS,
    ),
    MigrationDefinition(
        revision=4,
        migration_id="corpus_version_publication",
        statements=CORPUS_VERSION_SCHEMA_STATEMENTS,
    ),
)


def validate_migration_registry(
    migrations: tuple[MigrationDefinition, ...],
) -> tuple[MigrationDefinition, ...]:
    """Validate an ordered, gap-free migration registry."""
    if not isinstance(migrations, tuple):
        raise TypeError("migrations must be a tuple.")
    if not migrations:
        raise ValueError("migrations must not be empty.")

    expected_revision = 1
    migration_ids: set[str] = set()
    checksums: set[str] = set()
    for migration in migrations:
        if not isinstance(migration, MigrationDefinition):
            raise TypeError("migrations must contain only MigrationDefinition values.")
        if migration.revision != expected_revision:
            raise ValueError("migration revisions must be contiguous and start at one.")
        if migration.migration_id in migration_ids:
            raise ValueError("migration identifiers must be unique.")
        if migration.checksum in checksums:
            raise ValueError("migration checksums must be unique.")

        migration_ids.add(migration.migration_id)
        checksums.add(migration.checksum)
        expected_revision += 1

    return migrations


validate_migration_registry(MIGRATIONS)
