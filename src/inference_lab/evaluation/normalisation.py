"""Lossy text transformations used only by evaluation comparisons."""

import re
import unicodedata
from typing import Final

_WHITESPACE_PATTERN: Final = re.compile(r"\s+")


def normalise_for_duplicate_detection(value: str) -> str:
    """Normalise text only for deterministic duplicate detection."""
    normalised = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE_PATTERN.sub(" ", normalised).strip()
