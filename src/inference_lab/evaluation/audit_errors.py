"""Bounded errors for the corpus-audit contract boundary."""

from enum import StrEnum


class AuditErrorCategory(StrEnum):
    """Closed categories for invalid corpus-audit inputs."""

    INVALID_INPUT = "invalid_input"
    CORPUS_MISMATCH = "corpus_mismatch"
    MEMBERSHIP_MISMATCH = "membership_mismatch"
    OBSERVATION_MISMATCH = "observation_mismatch"
    UNKNOWN_PARSER_CAPABILITY = "unknown_parser_capability"
    UNSUPPORTED_OBSERVATION = "unsupported_observation"
    RELATIONSHIP_OUTSIDE_SCOPE = "relationship_outside_scope"
    INVALID_POLICY = "invalid_policy"
    INVALID_ORDERING = "invalid_ordering"


class AuditContractError(ValueError):
    """Reject a corpus-audit value without retaining private payloads."""

    def __init__(self, category: AuditErrorCategory) -> None:
        """Create a fixed-message, non-retryable contract error."""
        if not isinstance(category, AuditErrorCategory):
            raise TypeError("category must be an AuditErrorCategory.")
        super().__init__(category.value)
        self.category = category
        self.stage = "contract"
        self.retryable = False


class AuditPublicationError(OSError):
    """Report a local publication failure without exposing filesystem details."""

    def __init__(self) -> None:
        """Create a fixed-message publication error."""
        super().__init__("audit report publication failed")
