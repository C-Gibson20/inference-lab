"""Shared, side-effect-free validation helpers for provenance values."""

from inference_lab.provenance.values import ExternalIdentifier


def validate_external_identifiers(identifiers: object) -> None:
    """Validate an immutable, unique, canonically ordered identifier tuple."""
    if not isinstance(identifiers, tuple):
        raise TypeError("external_identifiers must be a tuple.")

    if any(
        not isinstance(identifier, ExternalIdentifier) for identifier in identifiers
    ):
        raise TypeError(
            "external_identifiers must contain only ExternalIdentifier values."
        )

    keys = tuple(identifier.as_tuple() for identifier in identifiers)

    if len(keys) != len(set(keys)):
        raise ValueError("external_identifiers must not contain duplicates.")

    if keys != tuple(sorted(keys)):
        raise ValueError("external_identifiers must be in canonical order.")
