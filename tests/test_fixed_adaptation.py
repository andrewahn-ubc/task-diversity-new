from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from td2.config import load_config
from td2.fixed_adaptation import SourceAuditError, audit_sources
from td2.plan import main_work_items, plan_identity


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path):
    data_root = tmp_path / "data"
    manifest_digests = {}
    for seed in range(5):
        manifest = data_root / f"seed_{seed}" / "manifest.json"
        _write_json(manifest, {"data_seed": seed})
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest.with_name("manifest.sha256").write_text(
            f"{digest}  manifest.json\n", encoding="ascii"
        )
        manifest_digests[seed] = digest

    specs = (
        (1, 256, "fixed", ROOT / "configs/experiment.yaml", tmp_path / "one"),
        (
            2,
            256,
            "fixed_2b_fresh",
            ROOT / "configs/experiment.yaml",
            tmp_path / "two256",
        ),
        (
            2,
            512,
            "fixed_2b_fresh",
            ROOT / "configs/experiment_gru512.yaml",
            tmp_path / "two512",
        ),
        (
            2,
            1024,
            "fixed_2b_fresh",
            ROOT / "configs/experiment_gru1024.yaml",
            tmp_path / "two1024",
        ),
    )
    for horizon, _width, mode, config_path, source_root in specs:
        config = load_config(config_path)
        target_updates = (
            horizon * config.derived.total_training_updates_per_stream
        )
        target_optimizer_steps = (
            target_updates
            * config.derived.training_per_distribution.optimizer_steps_per_update
        )
        for item in main_work_items(config):
            stream = (
                source_root
                / mode
                / item.algorithm
                / f"diversity_{item.diversity_n2}"
                / f"seed_{item.seed}"
            )
            identity = {
                "config_sha256": plan_identity(config),
                "data_manifest_sha256": manifest_digests[item.seed],
                "data_seed": item.seed,
            }
            if horizon == 2:
                identity["fixed_2b_fresh"] = {
                    "target_updates": target_updates
                }
            metadata = {
                "mode": mode,
                "algorithm": item.algorithm,
                "diversity_n": int(item.diversity_n2**0.5),
                "diversity_n2": item.diversity_n2,
                "seed": item.seed,
                "distribution": 1,
                "source_distribution": 1,
                "update_in_phase": target_updates,
                "global_update": target_updates,
                "optimizer_steps": target_optimizer_steps,
                "schedule_step": target_optimizer_steps,
                "protocol_identity": identity,
            }
            blob = stream / "resume/checkpoint-00000000000000000000.msgpack"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"x")
            _write_json(
                stream / "resume/latest.json",
                {
                    "blob": blob.name,
                    "sha256": "0" * 64,
                    "bytes": 1,
                    "metadata": metadata,
                },
            )
            _write_json(stream / "DONE", metadata)
    return data_root, specs


def test_audit_accepts_all_four_exact_terminal_checkpoint_families(
    tmp_path: Path,
) -> None:
    data_root, specs = _fixture(tmp_path)
    assert audit_sources(data_root, specs) == 180


def test_audit_rejects_a_missing_checkpoint_blob(tmp_path: Path) -> None:
    data_root, specs = _fixture(tmp_path)
    missing = (
        specs[0][4]
        / "fixed/ppo_cbp/diversity_1/seed_0/resume/"
        / "checkpoint-00000000000000000000.msgpack"
    )
    missing.unlink()
    with pytest.raises(SourceAuditError, match="checkpoint blob is missing"):
        audit_sources(data_root, specs)
