from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from td2.config import ConfigurationError, load_config


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "configs" / "experiment.yaml"
SMOKE_CONFIG = ROOT / "configs" / "smoke.yaml"
SOURCE_CONFIG = ROOT / "configs" / "source_hyperparameters.yaml"


def test_scientific_notation_strings_are_normalized_to_integers() -> None:
    with SOURCE_CONFIG.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)

    # Retain the supplied spelling in the source fixture. PyYAML deliberately
    # treats these YAML scalars as strings, so the application must normalize.
    assert source["TOTAL_TIMESTEPS"] == "1e8"
    assert source["REW_SHAPING_HORIZON"] == "4e7"

    config = load_config(EXPERIMENT_CONFIG)
    assert config.ppo.total_timesteps == 100_000_000
    assert isinstance(config.ppo.total_timesteps, int)
    assert config.ppo.reward_shaping_horizon == 40_000_000
    assert isinstance(config.ppo.reward_shaping_horizon, int)


def test_default_experiment_contract_and_derived_schedule() -> None:
    config = load_config(EXPERIMENT_CONFIG)

    assert config.experiment.num_distributions == 7
    assert config.experiment.diversity_levels == (1, 16, 256, 4096, 65_536)
    assert config.experiment.adaptation.after_distributions == tuple(range(1, 8))
    assert "reference to 10" in config.experiment.ambiguity_resolution
    assert config.runtime.seeds == (0, 1, 2, 3, 4)

    assert config.architecture.shared_pre_gru_fc_layers == 1
    assert config.architecture.actor_head_fc_layers == 1
    assert config.architecture.critic_head_fc_layers == 1

    schedule = config.derived.training_per_distribution
    assert schedule.num_actors == 2048
    assert schedule.environment_steps_per_update == 262_144
    assert schedule.num_updates == 381
    assert schedule.consumed_timesteps == 99_876_864
    assert schedule.timestep_remainder == 123_136
    assert schedule.actor_sequences_per_minibatch == 512
    assert schedule.actor_transitions_per_minibatch == 65_536
    assert schedule.optimizer_steps_per_update == 8
    assert schedule.total_optimizer_steps == 3048
    assert schedule.warmup_optimizer_steps == 153
    assert config.derived.total_training_updates_per_stream == 2667
    assert config.derived.total_training_optimizer_steps_per_stream == 21_336
    assert config.derived.adaptation_probe_count == 7


def test_config_and_nested_values_are_immutable() -> None:
    config = load_config(EXPERIMENT_CONFIG)

    with pytest.raises(FrozenInstanceError):
        config.ppo.total_timesteps = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.experiment.num_distributions = 10  # type: ignore[misc]


def test_smoke_config_inherits_and_recomputes_schedule() -> None:
    config = load_config(SMOKE_CONFIG)

    assert config.ppo.num_envs == 4
    assert config.ppo.num_steps == 8
    assert config.network.fc_dim == 32
    assert config.experiment.num_distributions == 2
    assert config.experiment.adaptation.after_distributions == (1, 2)
    assert config.runtime.num_seeds == 1

    training = config.derived.training_per_distribution
    adaptation = config.derived.adaptation_per_probe
    assert training.num_actors == 4
    assert training.actor_sequences_per_minibatch == 2
    assert training.actor_transitions_per_minibatch == 16
    assert training.num_updates == 8
    assert training.total_optimizer_steps == 16
    assert config.derived.total_training_updates_per_stream == 16
    assert config.derived.total_training_optimizer_steps_per_stream == 32
    assert adaptation is not None
    assert adaptation.num_updates == 4
    assert adaptation.total_optimizer_steps == 8


def test_nonintegral_step_count_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"extends: {EXPERIMENT_CONFIG}",
                'TOTAL_TIMESTEPS: "1.5e-1"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="TOTAL_TIMESTEPS must be an integer"):
        load_config(config_path)


def test_stale_derived_expectation_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "stale.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"extends: {EXPERIMENT_CONFIG}",
                "derived_expectations:",
                "  updates_per_distribution: 999",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="derives 381"):
        load_config(config_path)
