"""Typed failures for derived-record lineage contracts."""


class InvalidLineageError(ValueError):
    """Raised when correctly typed lineage data violates its invariants."""


class InvalidLineageMetadataError(InvalidLineageError):
    """Raised when producing-operation metadata violates lineage invariants."""


class InvalidLineageReferenceError(InvalidLineageError):
    """Raised when a lineage input or edge violates local invariants."""


class InvalidDerivedRecordError(InvalidLineageError):
    """Raised when derived-record metadata violates local invariants."""


class InvalidLineageGraphError(InvalidLineageError):
    """Raised when a complete lineage bundle violates graph invariants."""


class InvalidLineageSerialisationError(InvalidLineageError):
    """Raised when serialised lineage data violates its canonical contract."""


class UnsupportedLineageSchemaVersionError(InvalidLineageError):
    """Raised when lineage data names a schema this build cannot interpret."""
