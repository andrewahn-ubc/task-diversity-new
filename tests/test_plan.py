from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

from td2.config import load_config
from td2.plan import adaptation_work_items, main_work_items, plan_identity


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "configs" / "experiment.yaml"


def test_production_work_plan_is_complete_unique_and_stably_ordered() -> None:
    config = load_config(EXPERIMENT_CONFIG)
    mains = main_work_items(config)
    adaptations = adaptation_work_items(config)

    assert len(mains) == 45
    assert len(set(mains)) == 45
    assert astuple(mains[0]) == ("ppo_cbp", 1, 0)
    assert astuple(mains[4]) == ("ppo_cbp", 1, 4)
    assert astuple(mains[5]) == ("ppo_cbp", 4, 0)
    assert astuple(mains[-1]) == ("ppo_cbp", 65_536, 4)

    assert len(adaptations) == 450
    assert len(set(adaptations)) == 450
    assert astuple(adaptations[0]) == ("ppo_cbp", 1, 0, 1)
    assert astuple(adaptations[44]) == ("ppo_cbp", 65_536, 4, 1)
    assert astuple(adaptations[45]) == ("ppo_cbp", 1, 0, 2)
    assert astuple(adaptations[-1]) == ("ppo_cbp", 65_536, 4, 10)
    assert len(plan_identity(config)) == 64
