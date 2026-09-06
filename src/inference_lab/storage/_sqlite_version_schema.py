"""SQLite schema statements for durable corpus-version publication."""

from typing import Final

CORPUS_VERSION_SCHEMA_STATEMENTS: Final = (
    """CREATE TABLE corpus_updates (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL UNIQUE
        CHECK (
            length(corpus_version) = 36
            AND corpus_version = lower(corpus_version)
            AND substr(corpus_version, 9, 1) = '-'
            AND substr(corpus_version, 14, 1) = '-'
            AND substr(corpus_version, 19, 1) = '-'
            AND substr(corpus_version, 24, 1) = '-'
            AND length(replace(corpus_version, '-', '')) = 32
            AND corpus_version NOT GLOB '*[^0-9a-f-]*'
        ),
    expected_corpus_version TEXT,
    state TEXT NOT NULL
        CHECK (state IN ('preparing', 'ready', 'committed', 'aborted')),
    manifest_digest TEXT
        CHECK (
            manifest_digest IS NULL
            OR (
                length(manifest_digest) = 64
                AND manifest_digest NOT GLOB '*[^0-9a-f]*'
            )
        ),
    created_at TEXT NOT NULL CHECK (length(created_at) > 0),
    restores_corpus_version TEXT,
    FOREIGN KEY (corpus_id) REFERENCES corpora (corpus_id),
    FOREIGN KEY (corpus_id, expected_corpus_version)
        REFERENCES corpus_versions (corpus_id, corpus_version),
    FOREIGN KEY (corpus_id, restores_corpus_version)
        REFERENCES corpus_versions (corpus_id, corpus_version),
    UNIQUE (corpus_id, corpus_version),
    CHECK (
        (state = 'preparing' AND manifest_digest IS NULL)
        OR state = 'aborted'
        OR (state IN ('ready', 'committed') AND manifest_digest IS NOT NULL)
    ),
    CHECK (
        expected_corpus_version IS NULL
        OR expected_corpus_version != corpus_version
    ),
    CHECK (
        restores_corpus_version IS NULL
        OR (
            expected_corpus_version IS NOT NULL
            AND restores_corpus_version != corpus_version
        )
    )
) STRICT""",
    """CREATE INDEX corpus_updates_recovery_idx
ON corpus_updates (corpus_id, state, created_at, corpus_version)""",
    """CREATE INDEX corpus_updates_expected_version_idx
ON corpus_updates (corpus_id, expected_corpus_version, state)
WHERE expected_corpus_version IS NOT NULL""",
    """CREATE TABLE corpus_update_required_components (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    component_kind TEXT NOT NULL
        CHECK (
            length(component_kind) > 0
            AND substr(component_kind, 1, 1) GLOB '[a-z]'
            AND component_kind = lower(component_kind)
            AND component_kind NOT GLOB '*[^a-z0-9-]*'
            AND component_kind NOT GLOB '*--*'
            AND substr(component_kind, -1, 1) GLOB '[a-z0-9]'
        ),
    FOREIGN KEY (corpus_id, corpus_version)
        REFERENCES corpus_updates (corpus_id, corpus_version),
    UNIQUE (corpus_id, corpus_version, component_kind)
) STRICT""",
    """CREATE TABLE corpus_update_component_generations (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    component_kind TEXT NOT NULL,
    generation_id TEXT NOT NULL
        CHECK (
            length(generation_id) > 0
            AND length(trim(generation_id)) = length(generation_id)
            AND instr(generation_id, '/') = 0
            AND instr(generation_id, char(92)) = 0
            AND instr(generation_id, char(0)) = 0
        ),
    implementation TEXT NOT NULL
        CHECK (
            length(implementation) > 0
            AND substr(implementation, 1, 1) GLOB '[a-z]'
            AND implementation = lower(implementation)
            AND implementation NOT GLOB '*[^a-z0-9-]*'
            AND implementation NOT GLOB '*--*'
            AND substr(implementation, -1, 1) GLOB '[a-z0-9]'
        ),
    format_version TEXT NOT NULL CHECK (length(format_version) > 0),
    contract_version TEXT NOT NULL CHECK (length(contract_version) > 0),
    configuration_digest TEXT NOT NULL
        CHECK (
            length(configuration_digest) = 64
            AND configuration_digest NOT GLOB '*[^0-9a-f]*'
        ),
    source_manifest_digest TEXT NOT NULL
        CHECK (
            length(source_manifest_digest) = 64
            AND source_manifest_digest NOT GLOB '*[^0-9a-f]*'
        ),
    state TEXT NOT NULL CHECK (state IN ('preparing', 'ready')),
    FOREIGN KEY (corpus_id, corpus_version, component_kind)
        REFERENCES corpus_update_required_components (
            corpus_id,
            corpus_version,
            component_kind
        ),
    UNIQUE (corpus_id, corpus_version, component_kind),
    UNIQUE (component_kind, generation_id)
) STRICT""",
    """CREATE INDEX corpus_update_component_generations_state_idx
ON corpus_update_component_generations (
    corpus_id,
    corpus_version,
    state,
    component_kind
)""",
    """CREATE TABLE corpus_update_document_memberships (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    FOREIGN KEY (corpus_id, corpus_version)
        REFERENCES corpus_updates (corpus_id, corpus_version),
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    FOREIGN KEY (corpus_id, document_version_id)
        REFERENCES corpus_document_version_memberships (
            corpus_id,
            document_version_id
        ),
    UNIQUE (corpus_id, corpus_version, document_id)
) STRICT""",
    """CREATE INDEX corpus_update_document_memberships_version_idx
ON corpus_update_document_memberships (
    corpus_id,
    corpus_version,
    document_version_id
)""",
    """CREATE TABLE corpus_versions (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL UNIQUE,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    parent_corpus_version TEXT,
    manifest_digest TEXT NOT NULL
        CHECK (
            length(manifest_digest) = 64
            AND manifest_digest NOT GLOB '*[^0-9a-f]*'
        ),
    created_at TEXT NOT NULL CHECK (length(created_at) > 0),
    restores_corpus_version TEXT,
    FOREIGN KEY (corpus_id) REFERENCES corpora (corpus_id),
    FOREIGN KEY (corpus_id, corpus_version)
        REFERENCES corpus_updates (corpus_id, corpus_version),
    FOREIGN KEY (corpus_id, parent_corpus_version)
        REFERENCES corpus_versions (corpus_id, corpus_version),
    FOREIGN KEY (corpus_id, restores_corpus_version)
        REFERENCES corpus_versions (corpus_id, corpus_version),
    UNIQUE (corpus_id, corpus_version),
    UNIQUE (corpus_id, sequence),
    CHECK (
        (sequence = 1 AND parent_corpus_version IS NULL)
        OR (sequence > 1 AND parent_corpus_version IS NOT NULL)
    ),
    CHECK (
        parent_corpus_version IS NULL
        OR parent_corpus_version != corpus_version
    ),
    CHECK (
        restores_corpus_version IS NULL
        OR restores_corpus_version != corpus_version
    )
) STRICT""",
    """CREATE INDEX corpus_versions_history_idx
ON corpus_versions (corpus_id, sequence DESC, corpus_version)""",
    """CREATE INDEX corpus_versions_parent_idx
ON corpus_versions (corpus_id, parent_corpus_version)
WHERE parent_corpus_version IS NOT NULL""",
    """CREATE TABLE corpus_version_document_memberships (
    storage_id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    FOREIGN KEY (corpus_id, corpus_version)
        REFERENCES corpus_versions (corpus_id, corpus_version),
    FOREIGN KEY (document_version_id, document_id)
        REFERENCES document_versions (document_version_id, document_id),
    FOREIGN KEY (corpus_id, document_version_id)
        REFERENCES corpus_document_version_memberships (
            corpus_id,
            document_version_id
        ),
    UNIQUE (corpus_id, corpus_version, document_id)
) STRICT""",
    """CREATE INDEX corpus_version_document_memberships_version_idx
ON corpus_version_document_memberships (
    corpus_id,
    corpus_version,
    document_version_id
)""",
    """CREATE INDEX corpus_version_document_memberships_document_idx
ON corpus_version_document_memberships (
    corpus_id,
    document_id,
    corpus_version,
    document_version_id
)""",
    """CREATE TABLE corpus_active_versions (
    corpus_id TEXT PRIMARY KEY,
    corpus_version TEXT,
    FOREIGN KEY (corpus_id) REFERENCES corpora (corpus_id),
    FOREIGN KEY (corpus_id, corpus_version)
        REFERENCES corpus_versions (corpus_id, corpus_version)
) WITHOUT ROWID, STRICT""",
    """CREATE INDEX corpus_active_versions_version_idx
ON corpus_active_versions (corpus_version, corpus_id)
WHERE corpus_version IS NOT NULL""",
    """INSERT INTO corpus_active_versions (corpus_id, corpus_version)
SELECT corpus_id, NULL FROM corpora""",
    """CREATE TRIGGER corpora_initialise_active_version
AFTER INSERT ON corpora
BEGIN
    INSERT INTO corpus_active_versions (corpus_id, corpus_version)
    VALUES (NEW.corpus_id, NULL);
END""",
    """CREATE TRIGGER corpus_updates_lifecycle_update
BEFORE UPDATE ON corpus_updates
WHEN NEW.corpus_id IS NOT OLD.corpus_id
    OR NEW.corpus_version IS NOT OLD.corpus_version
    OR NEW.expected_corpus_version IS NOT OLD.expected_corpus_version
    OR NEW.created_at IS NOT OLD.created_at
    OR NEW.restores_corpus_version IS NOT OLD.restores_corpus_version
    OR NOT (
        (NEW.state = OLD.state AND NEW.manifest_digest IS OLD.manifest_digest)
        OR (
            OLD.state = 'preparing'
            AND NEW.state = 'ready'
            AND NEW.manifest_digest IS NOT NULL
        )
        OR (OLD.state = 'preparing' AND NEW.state = 'aborted')
        OR (
            OLD.state = 'ready'
            AND NEW.state IN ('committed', 'aborted')
            AND NEW.manifest_digest IS OLD.manifest_digest
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'invalid corpus update mutation');
END""",
    """CREATE TRIGGER corpus_updates_ready_components
BEFORE UPDATE OF state ON corpus_updates
WHEN NEW.state = 'ready'
    AND OLD.state = 'preparing'
    AND (
        EXISTS (
            SELECT 1
            FROM corpus_update_required_components AS required
            WHERE required.corpus_id = OLD.corpus_id
                AND required.corpus_version = OLD.corpus_version
                AND NOT EXISTS (
                    SELECT 1
                    FROM corpus_update_component_generations AS generation
                    WHERE generation.corpus_id = required.corpus_id
                        AND generation.corpus_version = required.corpus_version
                        AND generation.component_kind = required.component_kind
                        AND generation.state = 'ready'
                )
        )
        OR EXISTS (
            SELECT 1
            FROM corpus_update_component_generations AS generation
            WHERE generation.corpus_id = OLD.corpus_id
                AND generation.corpus_version = OLD.corpus_version
                AND generation.state != 'ready'
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'required corpus update components are incomplete');
END""",
    """CREATE TRIGGER corpus_updates_committed_active_version
BEFORE UPDATE OF state ON corpus_updates
WHEN NEW.state = 'committed'
    AND OLD.state = 'ready'
    AND NOT EXISTS (
        SELECT 1
        FROM corpus_active_versions AS active
        WHERE active.corpus_id = OLD.corpus_id
            AND active.corpus_version = OLD.corpus_version
    )
BEGIN
    SELECT RAISE(ABORT, 'committed update must be the active corpus version');
END""",
    """CREATE TRIGGER corpus_updates_delete_immutable
BEFORE DELETE ON corpus_updates
BEGIN
    SELECT RAISE(ABORT, 'corpus updates cannot be deleted');
END""",
    """CREATE TRIGGER corpus_versions_validate_insert
BEFORE INSERT ON corpus_versions
WHEN NOT EXISTS (
        SELECT 1
        FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = NEW.corpus_id
            AND update_record.corpus_version = NEW.corpus_version
            AND update_record.state = 'ready'
            AND update_record.manifest_digest = NEW.manifest_digest
            AND update_record.expected_corpus_version
                IS NEW.parent_corpus_version
            AND update_record.restores_corpus_version
                IS NEW.restores_corpus_version
    )
    OR (
        NEW.sequence = 1
        AND EXISTS (
            SELECT 1
            FROM corpus_versions AS existing
            WHERE existing.corpus_id = NEW.corpus_id
        )
    )
    OR (
        NEW.sequence > 1
        AND NOT EXISTS (
            SELECT 1
            FROM corpus_versions AS parent
            WHERE parent.corpus_id = NEW.corpus_id
                AND parent.corpus_version = NEW.parent_corpus_version
                AND parent.sequence = NEW.sequence - 1
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'invalid corpus version publication');
END""",
    """CREATE TRIGGER corpus_versions_update_immutable
BEFORE UPDATE ON corpus_versions
BEGIN
    SELECT RAISE(ABORT, 'corpus versions are immutable');
END""",
    """CREATE TRIGGER corpus_versions_delete_immutable
BEFORE DELETE ON corpus_versions
BEGIN
    SELECT RAISE(ABORT, 'corpus versions are immutable');
END""",
    """CREATE TRIGGER corpus_active_versions_advance
BEFORE UPDATE OF corpus_version ON corpus_active_versions
WHEN NEW.corpus_id IS NOT OLD.corpus_id
    OR (
        OLD.corpus_version IS NOT NULL
        AND NEW.corpus_version IS NULL
    )
    OR (
        NEW.corpus_version IS NOT NULL
        AND NOT EXISTS (
            SELECT 1
            FROM corpus_versions AS proposed
            JOIN corpus_updates AS update_record
                ON update_record.corpus_id = proposed.corpus_id
                AND update_record.corpus_version = proposed.corpus_version
            WHERE proposed.corpus_id = NEW.corpus_id
                AND proposed.corpus_version = NEW.corpus_version
                AND update_record.state IN ('ready', 'committed')
                AND (
                    OLD.corpus_version IS NULL
                    OR NEW.corpus_version = OLD.corpus_version
                    OR proposed.sequence > (
                        SELECT active_version.sequence
                        FROM corpus_versions AS active_version
                        WHERE active_version.corpus_id = OLD.corpus_id
                            AND active_version.corpus_version = OLD.corpus_version
                    )
                )
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'active corpus version must advance monotonically');
END""",
    """CREATE TRIGGER corpus_active_versions_insert_guard
BEFORE INSERT ON corpus_active_versions
WHEN NEW.corpus_version IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'a new active-version pointer must be empty');
END""",
    """CREATE TRIGGER corpus_active_versions_delete_immutable
BEFORE DELETE ON corpus_active_versions
BEGIN
    SELECT RAISE(ABORT, 'active-version pointers cannot be deleted');
END""",
    """CREATE TRIGGER corpus_update_required_components_insert_guard
BEFORE INSERT ON corpus_update_required_components
WHEN NOT EXISTS (
    SELECT 1 FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = NEW.corpus_id
        AND update_record.corpus_version = NEW.corpus_version
        AND update_record.state = 'preparing'
)
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_required_components_update_guard
BEFORE UPDATE ON corpus_update_required_components
WHEN NOT EXISTS (
        SELECT 1 FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = OLD.corpus_id
            AND update_record.corpus_version = OLD.corpus_version
            AND update_record.state = 'preparing'
    )
    OR NOT EXISTS (
        SELECT 1 FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = NEW.corpus_id
            AND update_record.corpus_version = NEW.corpus_version
            AND update_record.state = 'preparing'
    )
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_required_components_delete_guard
BEFORE DELETE ON corpus_update_required_components
WHEN NOT EXISTS (
    SELECT 1 FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = OLD.corpus_id
        AND update_record.corpus_version = OLD.corpus_version
        AND update_record.state = 'preparing'
)
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_component_generations_insert_guard
BEFORE INSERT ON corpus_update_component_generations
WHEN NOT EXISTS (
    SELECT 1 FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = NEW.corpus_id
        AND update_record.corpus_version = NEW.corpus_version
        AND update_record.state = 'preparing'
)
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_component_generations_update_guard
BEFORE UPDATE ON corpus_update_component_generations
WHEN NOT EXISTS (
        SELECT 1 FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = OLD.corpus_id
            AND update_record.corpus_version = OLD.corpus_version
            AND update_record.state = 'preparing'
    )
    OR NOT EXISTS (
        SELECT 1 FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = NEW.corpus_id
            AND update_record.corpus_version = NEW.corpus_version
            AND update_record.state = 'preparing'
    )
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_component_generations_delete_guard
BEFORE DELETE ON corpus_update_component_generations
WHEN NOT EXISTS (
    SELECT 1 FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = OLD.corpus_id
        AND update_record.corpus_version = OLD.corpus_version
        AND update_record.state = 'preparing'
)
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_document_memberships_insert_guard
BEFORE INSERT ON corpus_update_document_memberships
WHEN NOT EXISTS (
    SELECT 1 FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = NEW.corpus_id
        AND update_record.corpus_version = NEW.corpus_version
        AND update_record.state = 'preparing'
)
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_document_memberships_update_guard
BEFORE UPDATE ON corpus_update_document_memberships
WHEN NOT EXISTS (
        SELECT 1 FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = OLD.corpus_id
            AND update_record.corpus_version = OLD.corpus_version
            AND update_record.state = 'preparing'
    )
    OR NOT EXISTS (
        SELECT 1 FROM corpus_updates AS update_record
        WHERE update_record.corpus_id = NEW.corpus_id
            AND update_record.corpus_version = NEW.corpus_version
            AND update_record.state = 'preparing'
    )
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_update_document_memberships_delete_guard
BEFORE DELETE ON corpus_update_document_memberships
WHEN NOT EXISTS (
    SELECT 1 FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = OLD.corpus_id
        AND update_record.corpus_version = OLD.corpus_version
        AND update_record.state = 'preparing'
)
BEGIN
    SELECT RAISE(ABORT, 'sealed corpus update state is immutable');
END""",
    """CREATE TRIGGER corpus_version_document_memberships_insert_guard
BEFORE INSERT ON corpus_version_document_memberships
WHEN NOT EXISTS (
    SELECT 1
    FROM corpus_updates AS update_record
    WHERE update_record.corpus_id = NEW.corpus_id
        AND update_record.corpus_version = NEW.corpus_version
        AND update_record.state = 'ready'
)
BEGIN
    SELECT RAISE(ABORT, 'committed corpus membership is immutable');
END""",
    """CREATE TRIGGER corpus_version_document_memberships_update_immutable
BEFORE UPDATE ON corpus_version_document_memberships
BEGIN
    SELECT RAISE(ABORT, 'committed corpus membership is immutable');
END""",
    """CREATE TRIGGER corpus_version_document_memberships_delete_immutable
BEFORE DELETE ON corpus_version_document_memberships
BEGIN
    SELECT RAISE(ABORT, 'committed corpus membership is immutable');
END""",
)
