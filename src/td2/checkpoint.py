"""Crash-safe checkpoints and completion markers.

Checkpoint blobs are content-addressed.  An atomic JSON pointer is replaced
only after the new blob and its directory entry have been fsynced, so a killed
Slurm job leaves either the old or new checkpoint loadable.  One previous
pointer is retained as a corruption fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, TypeVar

from flax import serialization


T = TypeVar("T")
SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_pointer(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"invalid checkpoint pointer {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint pointer is not an object: {path}")
    return value


def save_checkpoint(
    directory: str | Path,
    state: Any,
    *,
    metadata: Mapping[str, Any],
    immutable: bool = False,
) -> Path:
    """Serialize state and atomically publish a verified pointer."""

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = serialization.to_bytes(state)
    digest = hashlib.sha256(payload).hexdigest()
    blob_name = f"checkpoint-{digest[:20]}.msgpack"
    blob_path = root / blob_name
    if not blob_path.exists():
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".checkpoint.", suffix=".tmp", dir=root
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, blob_path)
            _fsync_directory(root)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    elif hashlib.sha256(blob_path.read_bytes()).hexdigest() != digest:
        raise CheckpointError(f"content-address collision/corruption at {blob_path}")

    pointer = {
        "schema_version": SCHEMA_VERSION,
        "blob": blob_name,
        "sha256": digest,
        "bytes": len(payload),
        "metadata": dict(metadata),
    }
    latest_path = root / "latest.json"
    previous_path = root / "previous.json"
    old_pointer = _read_pointer(latest_path)
    if immutable and old_pointer is not None and old_pointer != pointer:
        raise CheckpointError(f"refusing to replace immutable checkpoint in {root}")
    if old_pointer is not None and old_pointer != pointer and not immutable:
        _atomic_json(previous_path, old_pointer)
    _atomic_json(latest_path, pointer)

    if not immutable:
        keep = {pointer["blob"]}
        previous = _read_pointer(previous_path)
        if previous is not None and isinstance(previous.get("blob"), str):
            keep.add(previous["blob"])
        for candidate in root.glob("checkpoint-*.msgpack"):
            if candidate.name not in keep:
                candidate.unlink(missing_ok=True)
        _fsync_directory(root)
    return blob_path


def _load_from_pointer(root: Path, pointer_name: str, template: T) -> T:
    pointer_path = root / pointer_name
    pointer = _read_pointer(pointer_path)
    if pointer is None:
        raise CheckpointError(f"missing checkpoint pointer: {pointer_path}")
    if int(pointer.get("schema_version", -1)) != SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema in {pointer_path}: "
            f"{pointer.get('schema_version')!r}"
        )
    blob_name = pointer.get("blob")
    expected = pointer.get("sha256")
    if not isinstance(blob_name, str) or not isinstance(expected, str):
        raise CheckpointError(f"malformed checkpoint pointer: {pointer_path}")
    blob_path = root / blob_name
    try:
        payload = blob_path.read_bytes()
    except OSError as error:
        raise CheckpointError(f"cannot read checkpoint blob {blob_path}: {error}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise CheckpointError(
            f"checkpoint checksum mismatch for {blob_path}: {actual} != {expected}"
        )
    try:
        return serialization.from_bytes(template, payload)
    except Exception as error:  # Flax raises several version-dependent types.
        raise CheckpointError(f"cannot deserialize {blob_path}: {error}") from error


def load_checkpoint(directory: str | Path, template: T) -> T:
    """Load latest, falling back once to the retained previous checkpoint."""

    state, _ = load_checkpoint_with_metadata(directory, template)
    return state


def load_checkpoint_with_metadata(
    directory: str | Path, template: T
) -> tuple[T, dict[str, Any]]:
    """Load a state and metadata from the same verified pointer.

    Returning them together is essential at distribution boundaries: if the
    newest blob is corrupt and loading falls back to ``previous.json``, callers
    must also use the previous pointer's distribution metadata.
    """

    root = Path(directory).expanduser().resolve()
    first_error: CheckpointError | None = None
    for pointer_name in ("latest.json", "previous.json"):
        try:
            state = _load_from_pointer(root, pointer_name, template)
            pointer = _read_pointer(root / pointer_name)
            assert pointer is not None  # _load_from_pointer already verified it.
            metadata = pointer.get("metadata", {})
            if not isinstance(metadata, dict):
                raise CheckpointError(
                    f"invalid metadata in {root / pointer_name}"
                )
            return state, dict(metadata)
        except CheckpointError as error:
            if first_error is None:
                first_error = error
    assert first_error is not None
    raise first_error


def checkpoint_metadata(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    first_error: CheckpointError | None = None
    for pointer_name in ("latest.json", "previous.json"):
        try:
            pointer = _read_pointer(root / pointer_name)
            if pointer is None:
                raise CheckpointError(f"missing checkpoint pointer: {root / pointer_name}")
            metadata = pointer.get("metadata", {})
            if not isinstance(metadata, dict):
                raise CheckpointError(f"invalid metadata in {root / pointer_name}")
            return dict(metadata)
        except CheckpointError as error:
            if first_error is None:
                first_error = error
    assert first_error is not None
    raise first_error


def atomic_marker(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish a marker only after all prerequisite files are durable."""

    marker = Path(path).expanduser().resolve()
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise CheckpointError(f"marker already exists with different data: {marker}")
        return
    _atomic_json(marker, dict(payload))


def clone_checkpoint(source: str | Path, destination: str | Path) -> None:
    """Copy a verified checkpoint without mutating the parent directory."""

    source_root = Path(source).expanduser().resolve()
    destination_root = Path(destination).expanduser().resolve()
    pointer = _read_pointer(source_root / "latest.json")
    if pointer is None:
        raise CheckpointError(f"missing source checkpoint: {source_root}")
    destination_root.mkdir(parents=True, exist_ok=True)
    blob = str(pointer["blob"])
    source_blob = source_root / blob
    destination_blob = destination_root / blob
    if not destination_blob.exists():
        shutil.copy2(source_blob, destination_blob)
        _fsync_directory(destination_root)
    _atomic_json(destination_root / "latest.json", pointer)
