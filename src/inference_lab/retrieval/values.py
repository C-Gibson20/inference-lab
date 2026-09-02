"""Immutable primitive values shared by retrieval channels."""

import math
import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

from inference_lab.retrieval.errors import InvalidRetrievalContractError


class RetrievalChannel(StrEnum):
    """Initial retrieval channels supported by the common contract."""

    LEXICAL = "lexical"
    DENSE = "dense"
    METADATA = "metadata"
    STRUCTURED = "structured"
    SUMMARY = "summary"


@dataclass(frozen=True, slots=True)
class RetrievalLimit:
    """A strict positive result limit."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError("value must be an integer.")
        if self.value <= 0:
            raise InvalidRetrievalContractError("value must be positive.")


_FIELD_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_-]*$")


def validate_field_name(field: object) -> str:
    """Validate and return a canonical retrieval field name."""
    if not isinstance(field, str):
        raise TypeError("field must be a string.")
    if _FIELD_PATTERN.fullmatch(field) is None:
        raise InvalidRetrievalContractError(
            "field must be a lowercase project-owned slug."
        )
    return field


def validate_filter_scalar(value: object, *, name: str = "value") -> None:
    """Validate one supported, strictly typed filter scalar."""
    if isinstance(value, bool) or isinstance(value, str) or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidRetrievalContractError(f"{name} must be finite.")
        return

    if type(value) is date:
        return
    raise TypeError(f"{name} must be a string, integer, float, boolean, or date.")
