from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import td2.prepare_data as prepare_data_module
from td2.prepare_data import (
    BANYAN_COLOR_STRIDE,
    DataPreparationError,
    GENERATOR_MODULE,
    HELDOUT_GENERATOR_API,
    MAX_NEW_UNARY_SIGNATURES_PER_TRAIN_ROUND,
    MAX_TOPOLOGY_ATTEMPTS,
    PACKED_PAD,
    PAPER_NUM_COLORS,
    PAPER_NUM_ITEMS,
    SCHEMA_VERSION,
    _generation_spec,
    _heldout_round_seed,
    _load_packed,
    _normalize_and_validate_rounds,
    _ruleset_generator_command,
    _validate_logical_round,
    _verify_existing,
    _write_json,
    _write_packed,
    generate_layout_masks,
    paper_leaf_pool_uids,
)


def test_paper_leaf_pool_is_seeded_unique_and_inside_8_by_10_vocabulary() -> None:
    selected = paper_leaf_pool_uids(0)
    assert selected == paper_leaf_pool_uids(0)
    assert len(selected) == 40
    assert len(set(selected)) == 40
    valid = {
        item * BANYAN_COLOR_STRIDE + color
        for item in range(PAPER_NUM_ITEMS)
        for color in range(PAPER_NUM_COLORS)
    }
    assert set(selected) <= valid


def test_layout_generation_is_deterministic_unique_and_valid() -> None:
    masks_a, reachable_a = generate_layout_masks(128, seed=1729)
    masks_b, reachable_b = generate_layout_masks(128, seed=1729)

    assert np.array_equal(masks_a, masks_b)
    assert np.array_equal(reachable_a, reachable_b)
    assert masks_a.shape == (128, 6, 6)
    assert reachable_a.shape == masks_a.shape
    assert len({np.packbits(mask).tobytes() for mask in masks_a}) == 128

    wall_counts = masks_a.sum(axis=(1, 2))
    assert np.all((wall_counts >= 2) & (wall_counts <= 10))
    assert not masks_a[:, 0, 0].any()
    assert not masks_a[:, 1, 0].any()
    open_counts = (~masks_a).sum(axis=(1, 2))
    assert np.all(open_counts / 36 >= 0.30)
    assert np.all(reachable_a.sum(axis=(1, 2)) / open_counts >= 0.55)
    assert not np.any(reachable_a & masks_a)


def test_existing_bundle_is_accepted_only_when_spec_and_hashes_match(
    tmp_path: Path,
) -> None:
    output = tmp_path / "data"
    output.mkdir()
    payload = output / "payload.bin"
    payload.write_bytes(b"immutable")
    spec = {"schema_version": 1, "seed": 3}
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {"generation_spec": spec, "file_sha256": {"payload.bin": digest}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output / "manifest.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="ascii"
    )

    assert _verify_existing(output, spec) == manifest

    payload.write_bytes(b"changed")
    with pytest.raises(DataPreparationError, match="failed checksum"):
        _verify_existing(output, spec)


def test_existing_bundle_with_different_spec_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "data"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"generation_spec": {"seed": 1}, "file_sha256": {"x": "bad"}}),
        encoding="utf-8",
    )
    (output / "manifest.sha256").write_text("bad  manifest.json\n", encoding="ascii")

    with pytest.raises(DataPreparationError, match="different specification"):
        _verify_existing(output, {"seed": 2})


def test_reserved_empty_output_directory_is_allowed(tmp_path: Path) -> None:
    output = tmp_path / "reserved"
    output.mkdir()
    assert _verify_existing(output, {"schema_version": 1}) is None


def test_production_generation_spec_and_training_command_reserve_heldout_capacity(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        experiment=SimpleNamespace(
            diversity_levels=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536),
            num_distributions=10,
            adaptation=SimpleNamespace(heldout_tasks=20, heldout_task_seed=991),
        ),
        ruleset=SimpleNamespace(ruleset_base_seed=0, tree_depth=6),
        runtime=SimpleNamespace(seed=0),
    )
    spec = _generation_spec(config)
    command = _ruleset_generator_command(tmp_path, spec)

    def argument_after(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert SCHEMA_VERSION == 3
    assert spec["num_train_distributions"] == 10
    assert spec["num_rounds"] == 11
    assert spec["max_n"] == 256
    assert spec["diversity_widths"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert spec["heldout_n_per_depth"] == 20
    assert spec["ruleset_generation"]["heldout_interface"] == HELDOUT_GENERATOR_API
    assert argument_after("-m") == GENERATOR_MODULE
    assert argument_after("--n") == "256"
    assert argument_after("--rounds") == "10"
    assert argument_after("--max-topology-attempts") == str(
        MAX_TOPOLOGY_ATTEMPTS
    )
    assert argument_after("--max-new-unary-signatures-per-round") == str(
        MAX_NEW_UNARY_SIGNATURES_PER_TRAIN_ROUND
    )
    assert MAX_NEW_UNARY_SIGNATURES_PER_TRAIN_ROUND == 10
    assert _heldout_round_seed(master_seed=0, max_n=256, round_index=10) == 1_263_838_259


def _round_metadata(round_index: int, logical_n: int, prefix: str) -> dict:
    topologies = [f"{prefix}-{index}" for index in range(logical_n)]
    return {
        "round_index": round_index,
        "n_per_depth": logical_n,
        "structure": {str(depth): logical_n for depth in range(1, 7)},
        "round_depth6_topology_signatures": topologies,
        "round_unique_depth6_topology_count": len(topologies),
        "round_unique_producer_signature_count": 0,
    }


def test_mixed_size_rounds_are_padded_depth_by_depth_to_the_max_n_stride(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_data_module, "_producer_signatures", lambda _: set())
    run_dir = tmp_path / "rulesets" / "n0002"
    run_dir.mkdir(parents=True)

    train = (np.arange(24, dtype=np.uint32) + 100).reshape(12, 2)
    heldout = (np.arange(18, dtype=np.uint32) + 1_000).reshape(6, 3)
    train_paths = (run_dir / "train.npy.bz2", run_dir / "train_meta.json")
    heldout_paths = (run_dir / "heldout.npy.bz2", run_dir / "heldout_meta.json")
    _write_packed(train_paths[0], train)
    _write_json(train_paths[1], _round_metadata(0, 2, "train"))
    _write_packed(heldout_paths[0], heldout)
    _write_json(heldout_paths[1], _round_metadata(1, 1, "heldout"))

    records = _normalize_and_validate_rounds(
        [train_paths, heldout_paths],
        max_n=2,
        num_depths=6,
        logical_n_per_depth=[2, 1],
    )

    normalized_train = _load_packed(train_paths[0])
    normalized_heldout = _load_packed(heldout_paths[0])
    assert normalized_train.shape == (12, 3)
    assert np.array_equal(normalized_train[:, :2], train)
    assert np.all(normalized_train[:, 2] == PACKED_PAD)
    assert normalized_heldout.shape == (12, 3)
    for depth_index in range(6):
        assert np.array_equal(normalized_heldout[depth_index * 2], heldout[depth_index])
        assert np.all(normalized_heldout[depth_index * 2 + 1] == PACKED_PAD)

    heldout_meta = json.loads(heldout_paths[1].read_text(encoding="utf-8"))
    assert heldout_meta["td2_logical_n_per_depth"] == 1
    assert heldout_meta["td2_storage_n_per_depth"] == 2
    assert heldout_meta["packed_shape"] == [12, 3]
    assert records[1]["logical_n_per_depth"] == 1
    assert records[1]["storage_n_per_depth"] == 2


def test_logical_round_validation_rejects_cross_round_topology_and_producer_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = np.full((6, 1), PACKED_PAD, dtype=np.uint32)
    metadata = _round_metadata(1, 1, "heldout")

    with pytest.raises(DataPreparationError, match="topology repeated"):
        _validate_logical_round(
            packed,
            metadata,
            round_index=1,
            logical_n=1,
            num_depths=6,
            blocked_topologies={"heldout-0"},
            blocked_producers=set(),
        )

    monkeypatch.setattr(
        prepare_data_module,
        "_producer_signatures",
        lambda _: {("U", 7)},
    )
    metadata["round_unique_producer_signature_count"] = 1
    with pytest.raises(DataPreparationError, match="producer signature repeated"):
        _validate_logical_round(
            packed,
            metadata,
            round_index=1,
            logical_n=1,
            num_depths=6,
            blocked_topologies=set(),
            blocked_producers={("U", 7)},
        )
