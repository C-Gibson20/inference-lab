"""Privacy and local-operation controls for evaluation workflows."""

from pathlib import Path

from inference_lab._validation import is_local_path_shape


def validate_local_path(path: str | Path, *, field_name: str = "path") -> Path:
    """Return a resolved local path after rejecting remote-shaped inputs.

    The check deliberately recognises both ``scheme://`` strings and the
    ``scheme:/`` form produced when a URI has already been coerced to ``Path``.
    It does not create directories or otherwise touch the filesystem.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{field_name} must be a string or Path")

    if not is_local_path_shape(path):
        raise ValueError(f"{field_name} must be a local filesystem path")

    return Path(path).expanduser().resolve(strict=False)
