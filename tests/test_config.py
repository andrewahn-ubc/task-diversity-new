from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from td2.config import ConfigurationError, load_config
from td2.ppo import scheduled_learning_rate
from td2.train import (
    _core_aux_2b_protocol_identity,
    _fixed_2b_schedule,
    _fixed_axis_steps,
    _stream_root,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "configs" / "experiment.yaml"
GRU512_CONFIG = ROOT / "configs" / "experiment_gru512.yaml"
GRU1024_CONFIG = ROOT / "configs" / "experiment_gru1024.yaml"
SMOKE_CONFIG = ROOT / "configs" / "smoke.yaml"
SOURCE_CONFIG = ROOT / "configs" / "source_hyperparameters.yaml"
CORE_AUX_CONFIG = ROOT / "configs" / "core_aux_2b.yaml"


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

    assert config.experiment.num_distributions == 10
    assert config.experiment.diversity_levels == (
        1,
        4,
        16,
        64,
        256,
        1_024,
        4_096,
        16_384,
        65_536,
    )
    assert config.experiment.methods == ("ppo_cbp",)
    assert config.experiment.adaptation.after_distributions == tuple(range(1, 11))
    assert config.experiment.adaptation.heldout_tasks == 100
    assert "full ten-distribution sequence" in config.experiment.ambiguity_resolution
    assert "fixed distribution" in config.experiment.ambiguity_resolution
    assert config.runtime.seeds == (0, 1, 2, 3, 4)

    assert config.architecture.shared_pre_gru_fc_layers == 2
    assert config.architecture.actor_head_fc_layers == 2
    assert config.architecture.critic_head_fc_layers == 2
    assert config.network.embedding_dim == 512

    schedule = config.derived.training_per_distribution
    adaptation = config.derived.adaptation_per_probe
    assert schedule.num_actors == 2048
    assert schedule.environment_steps_per_update == 262_144
    assert schedule.num_updates == 381
    assert schedule.consumed_timesteps == 99_876_864
    assert schedule.timestep_remainder == 123_136
    assert schedule.actor_sequences_per_minibatch == 512
    assert schedule.actor_transitions_per_minibatch == 65_536
    assert schedule.optimizer_steps_per_update == 8
    assert schedule.total_optimizer_steps == 3048
    assert schedule.warmup_optimizer_steps == 152
    assert config.derived.total_training_updates_per_stream == 3810
    assert config.derived.total_training_optimizer_steps_per_stream == 30_480
    assert (
        config.derived.total_training_warmup_optimizer_steps_per_stream == 1_520
    )
    assert config.derived.adaptation_probe_count == 10
    assert adaptation is not None
    assert adaptation.total_optimizer_steps == 3_048
    assert adaptation.warmup_optimizer_steps == 152


@pytest.mark.parametrize(
    ("config_path", "hidden_dim"),
    ((GRU512_CONFIG, 512), (GRU1024_CONFIG, 1024)),
)
def test_wider_gru_experiment_changes_only_recurrent_width(
    config_path: Path, hidden_dim: int
) -> None:
    baseline = load_config(EXPERIMENT_CONFIG)
    widened = load_config(config_path)

    expected = replace(
        baseline,
        network=replace(baseline.network, recurrent_hidden_dim=hidden_dim),
        source_files=widened.source_files,
    )
    assert widened == expected


def test_fixed_branch_uses_the_shared_prefix_and_nominal_one_billion_axis() -> None:
    config = load_config(EXPERIMENT_CONFIG)
    schedule = config.derived.training_per_distribution

    assert _fixed_axis_steps(
        schedule, update_in_phase=schedule.num_updates
    ) == (99_876_864, 99_876_864)
    assert _fixed_axis_steps(
        schedule, update_in_phase=schedule.num_updates + 1
    ) == (100_262_144, 100_139_008)
    assert _fixed_axis_steps(
        schedule,
        update_in_phase=config.derived.total_training_updates_per_stream,
    ) == (999_876_864, 998_768_640)

    target_updates, target_optimizer_steps, warmup_optimizer_steps = (
        _fixed_2b_schedule(config)
    )
    assert target_updates == 7_620
    assert target_optimizer_steps == 60_960
    assert warmup_optimizer_steps == 3_048
    assert _fixed_axis_steps(
        schedule, update_in_phase=target_updates
    ) == (1_999_876_864, 1_997_537_280)

    lr_kwargs = {
        "base_learning_rate": config.ppo.learning_rate,
        "total_optimizer_steps": target_optimizer_steps,
        "warmup_optimizer_steps": warmup_optimizer_steps,
        "anneal": config.ppo.anneal_learning_rate,
    }
    assert float(scheduled_learning_rate(jnp.asarray(0), **lr_kwargs)) == 0.0
    assert np.isclose(
        float(
            scheduled_learning_rate(
                jnp.asarray(warmup_optimizer_steps), **lr_kwargs
            )
        ),
        config.ppo.learning_rate,
    )
    before_1b = float(
        scheduled_learning_rate(jnp.asarray(30_479), **lr_kwargs)
    )
    at_1b = float(scheduled_learning_rate(jnp.asarray(30_480), **lr_kwargs))
    assert abs(at_1b - before_1b) < 1e-7
    assert 0.0 < at_1b < config.ppo.learning_rate
    assert float(
        scheduled_learning_rate(jnp.asarray(target_optimizer_steps), **lr_kwargs)
    ) == 0.0


def test_core_aux_config_and_checkpoint_identity_are_exact() -> None:
    config = load_config(CORE_AUX_CONFIG)
    assert config.network.recurrent_hidden_dim == 256
    assert config.experiment.methods == ("ppo_cbp",)
    assert config.experiment.diversity_levels == (
        256,
        1_024,
        4_096,
        16_384,
        65_536,
    )
    assert config.experiment.adaptation.enabled is False
    assert config.derived.adaptation_probe_count == 0
    assert _fixed_2b_schedule(config) == (7_620, 60_960, 3_048)

    root = _stream_root(
        Path("/results"),
        mode="core_aux_2b",
        algorithm="ppo_cbp",
        diversity_n2=65_536,
        seed=4,
        stage=None,
    )
    assert root == Path(
        "/results/core_aux_2b/ppo_cbp/aux_diversity_65536/seed_4"
    )

    identity = _core_aux_2b_protocol_identity(
        {"data_manifest_sha256": "abc"},
        core_n=16,
        auxiliary_n=256,
        target_updates=7_620,
        target_optimizer_steps=60_960,
        warmup_optimizer_steps=3_048,
    )
    protocol = identity["core_aux_2b"]
    assert protocol["core_n"] == 16
    assert protocol["auxiliary_n"] == 256
    assert protocol["core_transition_fraction"] == "1/2"
    assert protocol["target_updates"] == 7_620


def test_config_and_nested_values_are_immutable() -> None:
    config = load_config(EXPERIMENT_CONFIG)

    with pytest.raises(FrozenInstanceError):
        config.ppo.total_timesteps = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.experiment.num_distributions = 11  # type: ignore[misc]


def test_smoke_config_inherits_and_recomputes_schedule() -> None:
    config = load_config(SMOKE_CONFIG)

    assert config.ppo.num_envs == 4
    assert config.ppo.num_steps == 8
    assert config.network.fc_dim == 32
    assert config.network.embedding_dim == 16
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
    assert config.derived.total_training_warmup_optimizer_steps_per_stream == 0
    assert training.warmup_optimizer_steps == 0
    assert adaptation is not None
    assert adaptation.num_updates == 4
    assert adaptation.total_optimizer_steps == 8
    assert adaptation.warmup_optimizer_steps == 0


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
