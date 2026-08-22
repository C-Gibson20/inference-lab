"""Shared validation types for evaluation contracts."""

import re
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, Field

from inference_lab.evaluation.versioning import validate_schema_version


CaseId = DatasetId = Annotated[
    str,
    Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9-]{2,63}$",
    ),
]

SplitName = Literal["train", "development", "held_out"]
HELD_OUT_SPLIT: Final[SplitName] = "held_out"

SchemaVersion = Annotated[
    str,
    Field(validate_default=True),
    AfterValidator(validate_schema_version),
]

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_sha256_pattern(s: str) -> bool:
    """Return True if the string matches the SHA256 pattern, False otherwise."""
    return bool(_SHA256_PATTERN.fullmatch(s))
