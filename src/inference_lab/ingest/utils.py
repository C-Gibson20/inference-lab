"""Utilities for parsing and ingesting documents."""

import unicodedata


def is_strict_integer(value: object) -> bool:
    """Return whether a runtime value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def normalise_source_text(source_text: str) -> str:
    """Normalise decoded source text without changing its whitespace semantics."""
    newline_normalised = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", newline_normalised)
