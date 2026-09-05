"""Internal SQLite metadata baseline and migration definitions."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Final

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
