from dataclasses import replace
import json

import numpy as np
import pytest

import td2_kinetix.data as data
from td2_kinetix.config import PROTOCOL as REAL_PROTOCOL


def _fake_candidates(kind, seed, distribution, start, count):
    indexes = np.arange(start, start + count, dtype=np.uint32)
    return np.stack((indexes, indexes ^ np.uint32(0xA5A5)), axis=1)


def test_restartable_bank_generation_and_validation(tmp_path, monkeypatch):
    protocol = replace(REAL_PROTOCOL, diversity_levels=(1, 4))
    monkeypatch.setattr(data, "PROTOCOL", protocol)
    monkeypatch.setattr(data, "candidate_keys", _fake_candidates)

    calls = 0

    def filter_even(keys):
        nonlocal calls
        calls += 1
        return keys[:, 0] % 2 == 0

    assert data.generate_bank(
        tmp_path,
        kind="train",
        seed=0,
        distribution=0,
        candidate_batch_size=4,
        soft_limit_seconds=10,
        filter_batch=filter_even,
    )
    root = data.train_bank_dir(tmp_path, 0, 0)
    manifest = data.validate_bank(root, expected_kind="train", expected_seed=0, expected_distribution=0)
    assert manifest["accepted_count"] == 4
    assert calls == 2
    np.testing.assert_array_equal(data.load_bank(tmp_path, seed=0, distribution=0)[:, 0], [0, 2, 4, 6])

    # A complete bank is validated and reused rather than generated again.
    assert data.generate_bank(
        tmp_path,
        kind="train",
        seed=0,
        distribution=0,
        candidate_batch_size=4,
        soft_limit_seconds=10,
        filter_batch=lambda _: (_ for _ in ()).throw(AssertionError("must not run")),
    )


def test_heldout_bank_is_global_and_corruption_is_detected(tmp_path, monkeypatch):
    protocol = replace(REAL_PROTOCOL, heldout_tasks=3)
    monkeypatch.setattr(data, "PROTOCOL", protocol)
    monkeypatch.setattr(data, "candidate_keys", _fake_candidates)
    assert data.generate_bank(
        tmp_path,
        kind="heldout",
        candidate_batch_size=3,
        soft_limit_seconds=10,
        filter_batch=lambda keys: np.ones(len(keys), dtype=bool),
    )
    assert data.heldout_bank_dir(tmp_path).name == "heldout"
    assert data.load_bank(tmp_path, heldout=True).shape == (3, 2)
    with pytest.raises(data.DataError, match="shared"):
        data.load_bank(tmp_path, heldout=True, seed=0)

    root = data.heldout_bank_dir(tmp_path)
    values = np.load(root / "keys.npy", allow_pickle=False)
    values[0, 0] ^= np.uint32(1)
    with (root / "keys.npy").open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    with pytest.raises(data.DataError, match="checksum"):
        data.validate_bank(root, expected_kind="heldout")


def test_partial_identity_mismatch_fails_closed(tmp_path):
    root = data.train_bank_dir(tmp_path, 0, 0) / ".partial"
    root.mkdir(parents=True)
    with (root / "keys.npy").open("wb") as handle:
        np.save(handle, np.empty((0, 2), dtype=np.uint32), allow_pickle=False)
    (root / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "wrong",
                "kind": "train",
                "seed": 0,
                "distribution": 0,
                "accepted_count": 0,
                "candidate_cursor": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(data.DataError, match="protocol_id"):
        data._load_progress(data.train_bank_dir(tmp_path, 0, 0), kind="train", seed=0, distribution=0)


def test_array_index_cli_maps_canonical_bank(monkeypatch, tmp_path):
    seen = {}

    def fake_generate(root, **kwargs):
        seen.update(kwargs)
        assert root == str(tmp_path)
        return True

    monkeypatch.setattr(data, "generate_bank", fake_generate)
    assert data.main(
        [
            "--data-root",
            str(tmp_path),
            "--index",
            "20",
            "--soft-time-limit-seconds",
            "123",
        ]
    ) == 0
    assert seen == {
        "kind": "train",
        "seed": 2,
        "distribution": 6,
        "candidate_batch_size": 512,
        "soft_limit_seconds": 123,
    }
