"""Read-only compatibility inspection for SQLite metadata repositories."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from inference_lab.storage._sqlite_connection import (
    DEFAULT_BUSY_TIMEOUT_MS,
    enable_durable_wal,
    open_sqlite_connection,
    translate_sqlite_error,
)
from inference_lab.storage._sqlite_migrations import (
    MIGRATIONS,
    SQLITE_APPLICATION_ID,
    MigrationDefinition,
    validate_migration_registry,
)
from inference_lab.storage.errors import (
    CorruptStorageError,
    IncompatibleSchemaError,
    MigrationRequiredError,
    StorageError,
    StorageMigrationError,
    StorageTransactionError,
)


@dataclass(frozen=True, slots=True)
class SchemaInspection:
    """Validated metadata schema state for later recovery manifests."""

    current_revision: int
    latest_supported_revision: int
    applied_migration_count: int


@dataclass(frozen=True, slots=True)
class _SchemaObject:
    object_type: str
    name: str
    table_name: str
    definition: str


def _read_pragma_integer(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None or not isinstance(row[0], int) or isinstance(row[0], bool):
        raise IncompatibleSchemaError()
    return row[0]


def _has_schema_objects(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            """SELECT 1
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            LIMIT 1"""
        ).fetchone()
        is not None
    )


def _read_schema_manifest(
    connection: sqlite3.Connection,
) -> tuple[_SchemaObject, ...]:
    rows = connection.execute(
        """SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
            AND sql IS NOT NULL
        ORDER BY type, name, tbl_name"""
    ).fetchall()
    manifest: list[_SchemaObject] = []
    for row in rows:
        object_type, name, table_name, definition = row
        if not all(
            isinstance(value, str)
            for value in (object_type, name, table_name, definition)
        ):
            raise IncompatibleSchemaError()
        manifest.append(
            _SchemaObject(
                object_type=object_type,
                name=name,
                table_name=table_name,
                definition=definition,
            )
        )
    return tuple(manifest)


@lru_cache(maxsize=32)
def _expected_schema_manifest(
    migrations: tuple[MigrationDefinition, ...],
    current_revision: int,
) -> tuple[_SchemaObject, ...]:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        for migration in migrations[:current_revision]:
            for statement in migration.statements:
                connection.execute(statement)
        return _read_schema_manifest(connection)
    except StorageError:
        raise
    except sqlite3.Error:
        raise StorageMigrationError() from None
    finally:
        if connection is not None:
            _close_preserving_primary(connection)


def _validate_schema_manifest(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...],
    current_revision: int,
) -> None:
    if _read_schema_manifest(connection) != _expected_schema_manifest(
        migrations,
        current_revision,
    ):
        raise IncompatibleSchemaError()


def _validate_integrity(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA quick_check").fetchall()
    if len(rows) != 1 or rows[0][0] != "ok":
        raise CorruptStorageError()
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise CorruptStorageError()
    mismatched_duplicate = connection.execute(
        """SELECT 1
        FROM duplicate_relationships AS relationship
        JOIN document_versions AS first
            ON first.document_version_id = relationship.first_version_id
        JOIN document_versions AS second
            ON second.document_version_id = relationship.second_version_id
        WHERE relationship.duplicate_class = 'byte_identical'
            AND (
                first.content_hash_algorithm != second.content_hash_algorithm
                OR first.content_hash_digest != second.content_hash_digest
            )
        LIMIT 1"""
    ).fetchone()
    if mismatched_duplicate is not None:
        raise CorruptStorageError()


def _validate_applied_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise IncompatibleSchemaError()
    try:
        parsed = datetime.fromisoformat(value)
        offset = parsed.utcoffset()
    except (OverflowError, TypeError, ValueError):
        raise IncompatibleSchemaError() from None
    if parsed.tzinfo is None or offset != timedelta(0):
        raise IncompatibleSchemaError()
    canonical = parsed.astimezone(UTC).isoformat(timespec="microseconds")
    if value != canonical:
        raise IncompatibleSchemaError()
    return parsed


def _validate_schema_metadata(connection: sqlite3.Connection) -> int:
    rows = connection.execute(
        """SELECT singleton, schema_name, current_revision
        FROM metadata_schema"""
    ).fetchall()
    if len(rows) != 1:
        raise IncompatibleSchemaError()
    singleton, schema_name, current_revision = rows[0]
    if (
        singleton != 1
        or schema_name != "inference_lab_metadata"
        or not isinstance(current_revision, int)
        or isinstance(current_revision, bool)
        or current_revision <= 0
    ):
        raise IncompatibleSchemaError()
    return current_revision


def _validate_migration_history(
    connection: sqlite3.Connection,
    *,
    current_revision: int,
    migrations: tuple[MigrationDefinition, ...],
) -> None:
    rows = connection.execute(
        """SELECT revision, migration_id, checksum, applied_at_utc
        FROM migration_history
        ORDER BY revision"""
    ).fetchall()
    if len(rows) != current_revision:
        raise IncompatibleSchemaError()

    previous_applied_at: datetime | None = None
    for expected, row in zip(migrations[:current_revision], rows, strict=True):
        revision, migration_id, checksum, applied_at_utc = row
        if (
            revision != expected.revision
            or migration_id != expected.migration_id
            or checksum != expected.checksum
        ):
            raise IncompatibleSchemaError()
        applied_at = _validate_applied_at(applied_at_utc)
        if previous_applied_at is not None and applied_at < previous_applied_at:
            raise IncompatibleSchemaError()
        previous_applied_at = applied_at


def _inspect_database_schema(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...],
) -> SchemaInspection:
    latest_revision = migrations[-1].revision
    try:
        application_id = _read_pragma_integer(connection, "application_id")
        user_version = _read_pragma_integer(connection, "user_version")
        has_schema_objects = _has_schema_objects(connection)

        if application_id == 0 and user_version == 0 and not has_schema_objects:
            raise MigrationRequiredError()
        if application_id != SQLITE_APPLICATION_ID:
            raise IncompatibleSchemaError()
        if user_version <= 0 or user_version > latest_revision:
            raise IncompatibleSchemaError()

        current_revision = _validate_schema_metadata(connection)
        if current_revision != user_version or current_revision > latest_revision:
            raise IncompatibleSchemaError()
        _validate_migration_history(
            connection,
            current_revision=current_revision,
            migrations=migrations,
        )
        _validate_integrity(connection)
        _validate_schema_manifest(
            connection,
            migrations=migrations,
            current_revision=current_revision,
        )

        return SchemaInspection(
            current_revision=current_revision,
            latest_supported_revision=latest_revision,
            applied_migration_count=current_revision,
        )
    except StorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=IncompatibleSchemaError,
            constraint=IncompatibleSchemaError,
        ) from None


def inspect_database_schema(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
) -> SchemaInspection:
    """Inspect a compatible current or older schema without changing the file."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection.")
    try:
        validate_migration_registry(migrations)
    except (TypeError, ValueError):
        raise StorageMigrationError() from None
    if connection.in_transaction:
        raise StorageTransactionError()
    return _inspect_database_schema(connection, migrations=migrations)


def validate_database_schema(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
) -> SchemaInspection:
    """Validate that a repository is at the latest supported revision."""
    inspection = inspect_database_schema(connection, migrations=migrations)
    if inspection.current_revision < inspection.latest_supported_revision:
        raise MigrationRequiredError()
    return inspection


def _close_preserving_primary(connection: sqlite3.Connection) -> None:
    try:
        connection.close()
    except sqlite3.Error:
        pass


def open_current_database(
    path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    migrations: tuple[MigrationDefinition, ...] = MIGRATIONS,
) -> sqlite3.Connection:
    """Open a current compatible repository and then enable durable WAL mode."""
    connection = open_sqlite_connection(path, busy_timeout_ms=busy_timeout_ms)
    try:
        validate_database_schema(connection, migrations=migrations)
        enable_durable_wal(connection)
    except BaseException:
        _close_preserving_primary(connection)
        raise
    return connection
