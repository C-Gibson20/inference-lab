"""Error classes for handling parsing and decoding failures."""

from typing import Literal


class ParserDecodeError(UnicodeError):
    """Represent a source decoding failure without exposing source content."""

    category: Literal["decode_failed"] = "decode_failed"

    def __init__(self, encoding: str) -> None:
        """Create a safe decoding error identified by its encoding."""
        super().__init__("decode_failed")
        self.encoding = encoding
