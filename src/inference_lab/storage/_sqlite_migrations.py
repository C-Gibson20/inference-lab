"""Internal, append-only SQLite metadata migration definitions."""

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


_REVISION_1_STATEMENTS: Final = (
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

MIGRATIONS: Final = (
    MigrationDefinition(
        revision=1,
        migration_id="initial_metadata",
        statements=_REVISION_1_STATEMENTS,
    ),
)


def validate_migration_registry(
    migrations: tuple[MigrationDefinition, ...],
) -> tuple[MigrationDefinition, ...]:
    """Validate an ordered, gap-free migration registry."""
    if not isinstance(migrations, tuple):
        raise TypeError("migrations must be a tuple.")

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
