"""Shared validation types for evaluation contracts."""

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
