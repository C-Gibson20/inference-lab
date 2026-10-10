"""Durable local publication primitives shared across persistence boundaries."""

import ctypes
import errno
import os
import sys
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX filesystems."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree_directories(root: Path) -> None:
    """Persist every directory created inside a staged tree, deepest first."""
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        fsync_directory(directory)


def rename_directory_without_replacement(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when its destination is absent.

    Raises ``FileExistsError`` when the destination already exists, including
    when it is an empty directory. A plain rename would replace an empty
    destination silently, so callers cannot distinguish a first publication
    from one that overwrote a partially removed generation.
    """
    if sys.platform == "darwin":
        _call_libc_rename(
            "renamex_np",
            os.fsencode(source),
            os.fsencode(destination),
            _RENAME_EXCL,
            destination=destination,
        )
        return
    if sys.platform.startswith("linux"):
        _call_libc_rename(
            "renameat2",
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
            destination=destination,
        )
        return
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory rename is unsupported",
        str(destination),
    )


def _call_libc_rename(
    function_name: str,
    *arguments: bytes | int,
    destination: Path,
) -> None:
    """Call a platform rename primitive and normalise an occupied destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = getattr(libc, function_name)
    except AttributeError:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory rename is unsupported",
            str(destination),
        ) from None
    if function_name == "renamex_np":
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    else:
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
    rename.restype = ctypes.c_int
    if rename(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    # Both spellings mean the destination is occupied; callers branch on one.
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), str(destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))
