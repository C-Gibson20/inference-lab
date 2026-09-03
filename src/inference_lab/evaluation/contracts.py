"""Annotated scalar types and constants for evaluation contracts."""

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
