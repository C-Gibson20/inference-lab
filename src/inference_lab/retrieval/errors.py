"""Errors raised by the in-memory retrieval contract values."""


class InvalidRetrievalContractError(ValueError):
    """Raised when correctly typed retrieval data violates its contract."""


class InvalidRetrievalResultError(InvalidRetrievalContractError):
    """Raised when an evidence result violates its status or ordering rules."""


class InvalidRetrievalSerialisationError(InvalidRetrievalContractError):
    """Raised when serialised retrieval data violates its strict contract."""


class UnsupportedRetrievalSchemaVersionError(InvalidRetrievalContractError):
    """Raised when retrieval data names an unsupported schema version."""
