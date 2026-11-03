import argparse
import csv

import pytest

from td2_kinetix.ppo import SETTINGS, official_config
from td2_kinetix.train import (
    DIVERSITIES,
    EVAL_INTERVAL_UPDATES,
    UPDATES_PER_PHASE,
    _existing_metric_keys,
    _validate_args,
    source_stage_root,
    stream_root,
)


def _args(**overrides):
    values = {
        "mode": "main",
        "diversity": 16,
        "seed": 1,
        "stage": 3,
        "soft_time_limit_seconds": 2700,
        "max_updates": None,
        "checkpoint_every_updates": 10,
        "checkpoint_every_seconds": 300,
        "eval_attempts": 10,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exact_ppo_schedule_and_paper_dense_reward_setting():
    assert SETTINGS.environment_steps_per_update == 131_072
    assert SETTINGS.optimizer_steps_per_update == 256
    assert UPDATES_PER_PHASE == 762
    assert EVAL_INTERVAL_UPDATES == 76
    config = official_config()
    assert config["dense_reward_scale"] == 2.0
    assert config["recurrent_model"] is False
    assert DIVERSITIES == (1, 16, 64, 256, 65_536)


def test_stream_layout_matches_slurm_contract(tmp_path):
    assert stream_root(
        tmp_path, mode="main", diversity=64, seed=2, stage=6
    ) == tmp_path / "main/diversity_64/seed_2/stages/stage_6"
    assert stream_root(
        tmp_path, mode="heldout", diversity=64, seed=2, stage=6
    ) == tmp_path / "heldout/diversity_64/seed_2/stage_6"
    assert source_stage_root(
        tmp_path, diversity=64, seed=2, stage=6
    ) == tmp_path / "main/diversity_64/seed_2/stages/stage_6"


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "bad"},
        {"diversity": 4},
        {"seed": 3},
        {"stage": 7},
        {"soft_time_limit_seconds": 0},
        {"max_updates": 0},
        {"checkpoint_every_updates": 0},
        {"checkpoint_every_seconds": 0},
        {"eval_attempts": 0},
    ],
)
def test_invalid_training_coordinates_fail_closed(overrides):
    with pytest.raises(ValueError):
        _validate_args(_args(**overrides))


def test_metric_restart_keying_avoids_duplicate_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_type", "update_in_phase"])
        writer.writeheader()
        writer.writerow({"row_type": "train", "update_in_phase": 1})
        writer.writerow({"row_type": "eval", "update_in_phase": 0})
    assert _existing_metric_keys(path) == {("train", 1), ("eval", 0)}
