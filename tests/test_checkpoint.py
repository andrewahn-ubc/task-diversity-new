from __future__ import annotations

import json

from flax import struct
import jax.numpy as jnp
import numpy as np
import pytest

from td2.checkpoint import (
    CheckpointError,
    checkpoint_metadata,
    load_checkpoint,
    load_checkpoint_with_metadata,
    save_checkpoint,
)


@struct.dataclass
class _ToyState:
    step: jnp.ndarray
    weights: jnp.ndarray


def _state(step: int) -> _ToyState:
    return _ToyState(
        step=jnp.asarray(step, dtype=jnp.int32),
        weights=jnp.asarray([step + 0.25, step + 0.5], dtype=jnp.float32),
    )


def _assert_state(actual: _ToyState, expected: _ToyState) -> None:
    np.testing.assert_array_equal(actual.step, expected.step)
    np.testing.assert_array_equal(actual.weights, expected.weights)


def test_checkpoint_roundtrip_and_metadata(tmp_path) -> None:
    expected = _state(3)
    blob = save_checkpoint(tmp_path, expected, metadata={"distribution": 2})

    assert blob.is_file()
    restored = load_checkpoint(tmp_path, _state(0))
    _assert_state(restored, expected)
    assert checkpoint_metadata(tmp_path) == {"distribution": 2}


def test_checksum_failure_falls_back_to_previous_checkpoint(tmp_path) -> None:
    previous = _state(1)
    latest = _state(2)
    save_checkpoint(tmp_path, previous, metadata={"generation": "previous"})
    save_checkpoint(tmp_path, latest, metadata={"generation": "latest"})
    pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    (tmp_path / pointer["blob"]).write_bytes(b"deliberately corrupted")

    restored, metadata = load_checkpoint_with_metadata(tmp_path, _state(0))

    _assert_state(restored, previous)
    assert metadata == {"generation": "previous"}


def test_invalid_latest_pointer_falls_back_with_previous_metadata(tmp_path) -> None:
    previous = _state(1)
    latest = _state(2)
    save_checkpoint(tmp_path, previous, metadata={"distribution": 1})
    save_checkpoint(tmp_path, latest, metadata={"distribution": 2})
    (tmp_path / "latest.json").write_text("not json", encoding="utf-8")

    restored, metadata = load_checkpoint_with_metadata(tmp_path, _state(0))

    _assert_state(restored, previous)
    assert metadata == {"distribution": 1}
    assert checkpoint_metadata(tmp_path) == {"distribution": 1}


def test_immutable_checkpoint_rejects_replacement_and_remains_loadable(tmp_path) -> None:
    original = _state(4)
    save_checkpoint(tmp_path, original, metadata={"kind": "heldout"}, immutable=True)
    # Idempotent publication of exactly the same immutable pointer is allowed.
    save_checkpoint(tmp_path, original, metadata={"kind": "heldout"}, immutable=True)

    with pytest.raises(CheckpointError, match="immutable"):
        save_checkpoint(
            tmp_path,
            _state(5),
            metadata={"kind": "heldout"},
            immutable=True,
        )

    _assert_state(load_checkpoint(tmp_path, _state(0)), original)
