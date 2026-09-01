"""Typed failures for derived-record lineage contracts."""


class InvalidLineageError(ValueError):
    """Raised when correctly typed lineage data violates its invariants."""


class UnsupportedLineageSchemaVersionError(InvalidLineageError):
    """Raised when lineage data names a schema this build cannot interpret."""
