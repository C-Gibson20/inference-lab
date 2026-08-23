"""Shared validation types for evaluation contracts."""

import re
import unicodedata
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, Field

from inference_lab.evaluation.versioning import (
    CURRENT_SCHEMA_VERSION,
    validate_schema_version,
)

CaseId = DatasetId = Annotated[
    str,
    Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9-]{2,63}$",
    ),
]

EvidenceId = Annotated[
    str,
    Field(
        min_length=3,
        max_length=64,
        pattern=r"^evidence-[a-z0-9][a-z0-9-]{0,54}$",
    ),
]

ExpectationKey = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    ),
]

SplitName = Literal["train", "development", "held_out"]
HELD_OUT_SPLIT: Final[SplitName] = "held_out"

SchemaVersion = Annotated[
    str,
    Field(
        validate_default=True,
        json_schema_extra={"const": CURRENT_SCHEMA_VERSION},
    ),
    AfterValidator(validate_schema_version),
]

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_WHITESPACE = re.compile(r"\s+")


def is_sha256_pattern(s: str) -> bool:
    """Return True if the string matches the SHA256 pattern, False otherwise."""
    return bool(_SHA256_PATTERN.fullmatch(s))


def normalise_for_duplicate_detection(value: str) -> str:
    """Normalise text only for deterministic duplicate detection."""
    normalised = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE.sub(" ", normalised).strip()
