"""Privacy and local-operation controls for evaluation workflows."""

from pathlib import Path
import re


_URI_LOCATION = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def validate_local_path(path: str | Path, *, field_name: str = "path") -> Path:
    """Return a resolved local path and reject URI-shaped locations.

    Evaluation data and artifacts are deliberately filesystem-only. This
    validation is a boundary check; it does not create directories or touch
    the filesystem.
    """

    raw_path = str(path)
    if _URI_LOCATION.match(raw_path):
        raise ValueError(f"{field_name} must be a local filesystem path")
    return Path(path).expanduser().resolve(strict=False)
