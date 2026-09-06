"""Canonical text transformations shared across source-data boundaries."""

import unicodedata


def normalise_source_text(source_text: str) -> str:
    """Normalise decoded source text without changing whitespace semantics."""
    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string.")
    newline_normalised = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", newline_normalised)
