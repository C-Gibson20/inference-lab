"""Deterministic serialisation utilities for evaluation contracts."""

import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import cast

from pydantic import BaseModel

from inference_lab._json import JsonObject, JsonValue, canonical_json_text
from inference_lab.evaluation.cases import EvaluationCase
from inference_lab.evaluation.datasets import DatasetManifest
from inference_lab.evaluation.loader import _validate_dataset_consistency, load_dataset
from inference_lab.evaluation.privacy import validate_local_path
from inference_lab.evaluation.splitting import validate_no_leakage
from inference_lab.evaluation.versioning import CURRENT_SCHEMA_VERSION

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


def _canonical_payload(model: BaseModel) -> JsonObject:
    """Convert a model to a JSON-compatible canonical payload."""
    payload = cast(JsonObject, model.model_dump(mode="json"))

    tags = payload.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise RuntimeError("Canonical evaluation tags must be a string array")
        payload["tags"] = cast(JsonValue, sorted(cast(list[str], tags)))

    return payload


def _human_readable_json_bytes(payload: JsonObject) -> bytes:
    """Encode a payload as deterministic human-readable UTF-8 JSON."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{text}\n".encode("utf-8")


def serialise_cases(cases: Iterable[EvaluationCase]) -> bytes:
    """Serialise cases as deterministic JSONL in the supplied order."""
    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise ValueError("Evaluation datasets must contain at least one case")

    lines = [canonical_json_text(_canonical_payload(case)) for case in ordered_cases]
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialise_manifest(manifest: DatasetManifest) -> bytes:
    """Serialise a manifest as human-reviewable deterministic JSON."""
    return _human_readable_json_bytes(_canonical_payload(manifest))


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX filesystems."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    """Persist every directory created inside a staged dataset."""
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        _fsync_directory(directory)


def _rename_directory_without_replacement(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when its destination is absent."""
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
        destination,
    )


def _call_libc_rename(
    function_name: str,
    *arguments: bytes | int,
    destination: Path,
) -> None:
    """Call a platform rename primitive and preserve its errno semantics."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = getattr(libc, function_name)
    except AttributeError:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory rename is unsupported",
            destination,
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
    raise OSError(error_number, os.strerror(error_number), destination)


def atomic_write(path: str | Path, content: bytes) -> None:
    """Write bytes atomically within the destination directory."""
    path = validate_local_path(path, field_name="path")
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _output_case_path(destination: Path, case_file: str) -> Path:
    """Resolve a case output path below the dataset destination."""
    if Path(case_file).is_absolute():
        raise ValueError("case_file must be relative to the dataset destination")
    validate_local_path(case_file, field_name="case_file")

    root = destination.resolve()
    resolved = (root / case_file).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            "case_file must remain inside the dataset destination"
        ) from None

    manifest_path = root / "manifest.json"
    if resolved == manifest_path:
        raise ValueError("case_file cannot be manifest.json")

    return resolved


def write_dataset(
    destination: str | Path,
    manifest: DatasetManifest,
    cases: tuple[EvaluationCase, ...],
) -> DatasetManifest:
    """Stage, validate, and atomically publish a new immutable dataset."""
    ordered_cases = tuple(cases)
    _validate_dataset_consistency(manifest, ordered_cases)
    validate_no_leakage(ordered_cases)

    destination = validate_local_path(destination, field_name="destination")
    _output_case_path(destination, manifest.case_file)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Dataset destination already exists")

    case_bytes = serialise_cases(ordered_cases)
    checksum = f"sha256:{hashlib.sha256(case_bytes).hexdigest()}"

    manifest_payload = manifest.model_dump(mode="python")
    manifest_payload["checksums"] = {manifest.case_file: checksum}
    final_manifest = DatasetManifest.model_validate(manifest_payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".staging",
        )
    )

    try:
        case_path = _output_case_path(staging_path, final_manifest.case_file)
        atomic_write(case_path, case_bytes)
        atomic_write(
            staging_path / "manifest.json",
            serialise_manifest(final_manifest),
        )

        loaded = load_dataset(staging_path / "manifest.json")
        _fsync_tree_directories(staging_path)

        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Dataset destination already exists")
        os.replace(staging_path, destination)
        staging_path = None
        _fsync_directory(destination.parent)
        return loaded.manifest
    except BaseException as error:
        if staging_path is not None:
            try:
                shutil.rmtree(staging_path)
            except OSError as cleanup_error:
                error.add_note(
                    f"Staging cleanup also failed: {type(cleanup_error).__name__}"
                )
        raise


def evaluation_json_schema() -> JsonObject:
    """Return one Draft 2020-12 schema for manifests and cases."""
    definitions: JsonObject = {}
    for model in (DatasetManifest, EvaluationCase):
        schema = cast(
            JsonObject,
            model.model_json_schema(ref_template="#/$defs/{model}"),
        )
        model_definitions = schema.pop("$defs", {})
        if not isinstance(model_definitions, dict):
            raise RuntimeError("JSON Schema definitions must be an object")
        for name, definition in model_definitions.items():
            existing = definitions.get(name)
            if existing is not None and existing != definition:
                raise RuntimeError(f"Conflicting JSON Schema definition: {name}")
            definitions[name] = definition
        definitions[model.__name__] = schema

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:inference-lab:evaluation-schema:{CURRENT_SCHEMA_VERSION}",
        "title": "Inference Lab evaluation dataset contract",
        "description": (
            "A versioned evaluation dataset manifest or evaluation case record."
        ),
        "type": "object",
        "oneOf": [
            {"$ref": "#/$defs/DatasetManifest"},
            {"$ref": "#/$defs/EvaluationCase"},
        ],
        "$defs": definitions,
    }


def write_json_schema(path: str | Path) -> None:
    """Write the published evaluation contract schema atomically."""
    atomic_write(path, _human_readable_json_bytes(evaluation_json_schema()))
