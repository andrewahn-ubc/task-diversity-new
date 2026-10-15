from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

from td2.config import load_config
from td2.plan import (
    adaptation_work_items,
    continuation_work_items,
    data_seeds,
    fixed_work_items,
    main,
    main_work_items,
    plan_identity,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "configs" / "experiment.yaml"


def test_production_work_plan_is_complete_unique_and_stably_ordered() -> None:
    config = load_config(EXPERIMENT_CONFIG)
    seeds = data_seeds(config)
    mains = main_work_items(config)
    fixed = fixed_work_items(config)
    continuations = continuation_work_items(config)
    adaptations = adaptation_work_items(config)

    assert seeds == (0, 1, 2, 3, 4)
    assert len(mains) == 45
    assert len(set(mains)) == 45
    assert astuple(mains[0]) == ("ppo_cbp", 1, 0)
    assert astuple(mains[4]) == ("ppo_cbp", 1, 4)
    assert astuple(mains[5]) == ("ppo_cbp", 4, 0)
    assert astuple(mains[-1]) == ("ppo_cbp", 65_536, 4)

    assert len(fixed) == 45
    assert len(set(fixed)) == 45
    assert astuple(fixed[0]) == ("ppo_cbp", 1, 0)
    assert astuple(fixed[-1]) == ("ppo_cbp", 65_536, 4)

    assert len(continuations) == 90
    assert len(set(continuations)) == 90
    assert astuple(continuations[0]) == ("ppo_cbp", 1, 0, "main")
    assert astuple(continuations[44]) == ("ppo_cbp", 65_536, 4, "main")
    assert astuple(continuations[45]) == ("ppo_cbp", 1, 0, "fixed")
    assert astuple(continuations[-1]) == ("ppo_cbp", 65_536, 4, "fixed")

    assert len(adaptations) == 450
    assert len(set(adaptations)) == 450
    assert astuple(adaptations[0]) == ("ppo_cbp", 1, 0, 1)
    assert astuple(adaptations[44]) == ("ppo_cbp", 65_536, 4, 1)
    assert astuple(adaptations[45]) == ("ppo_cbp", 1, 0, 2)
    assert astuple(adaptations[-1]) == ("ppo_cbp", 65_536, 4, 10)
    assert len(plan_identity(config)) == 64


def test_continuation_cli_preserves_existing_counts_output(capsys) -> None:
    assert main(["--config", str(EXPERIMENT_CONFIG), "counts"]) == 0
    assert capsys.readouterr().out == "45\t450\n"

    assert main(["--config", str(EXPERIMENT_CONFIG), "continuation-count"]) == 0
    assert capsys.readouterr().out == "90\n"

    assert main(
        ["--config", str(EXPERIMENT_CONFIG), "continuation", "--index", "0"]
    ) == 0
    assert capsys.readouterr().out == "main\tppo_cbp\t1\t0\n"

    assert main(
        ["--config", str(EXPERIMENT_CONFIG), "continuation", "--index", "44"]
    ) == 0
    assert capsys.readouterr().out == "main\tppo_cbp\t65536\t4\n"

    assert main(
        ["--config", str(EXPERIMENT_CONFIG), "continuation", "--index", "45"]
    ) == 0
    assert capsys.readouterr().out == "fixed\tppo_cbp\t1\t0\n"

    assert main(
        ["--config", str(EXPERIMENT_CONFIG), "continuation", "--index", "89"]
    ) == 0
    assert capsys.readouterr().out == "fixed\tppo_cbp\t65536\t4\n"
