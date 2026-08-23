"""Utilities for parsing and ingesting documents."""

import unicodedata


def normalise_source_text(source_text: str) -> str:
    """Normalize decoded source text without changing its whitespace semantics."""
    newline_normalised = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", newline_normalised)
