"""Canonical text transformations for ingest source content."""

import unicodedata


def normalise_source_text(source_text: str) -> str:
    """Normalise decoded source text without changing its whitespace semantics."""
    newline_normalised = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", newline_normalised)
