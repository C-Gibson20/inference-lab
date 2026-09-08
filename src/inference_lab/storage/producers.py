"""Immutable producer identities for persisted parser and layout records."""

from dataclasses import dataclass

from inference_lab._validation import is_canonical_semantic_version, is_lowercase_slug


def _validate_optional_identity(value: object, field_name: str) -> None:
    """Validate optional bounded identity text without exposing its value."""
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None.")
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-blank and contain no NUL.")


@dataclass(frozen=True, slots=True)
class ProducerIdentity:
    """Identify a parser, OCR engine, or layout producer without its payload."""

    implementation: str
    contract_version: str
    artefact_identity: str | None = None
    artefact_revision: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the producer identity is self-consistent."""
        if not isinstance(self.implementation, str):
            raise TypeError("implementation must be a string.")
        if not is_lowercase_slug(self.implementation):
            raise ValueError("implementation must be a lowercase hyphenated slug.")
        if not isinstance(self.contract_version, str):
            raise TypeError("contract_version must be a string.")
        if not is_canonical_semantic_version(self.contract_version):
            raise ValueError("contract_version must be a canonical semantic version.")
        _validate_optional_identity(self.artefact_identity, "artefact_identity")
        _validate_optional_identity(self.artefact_revision, "artefact_revision")
        if (self.artefact_identity is None) != (self.artefact_revision is None):
            raise ValueError(
                "artefact identity and revision must be supplied together."
            )
