"""Utilities for parsing and ingesting documents."""

import unicodedata


def normalise_text(source_text: str) -> str:
    """Apply the deterministic canonical text-normalisation policy."""
    newline_normalised = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", newline_normalised)
