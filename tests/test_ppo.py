from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import td2.ppo as ppo_module
from td2.config import load_config
from td2.network import initial_carry
from td2.ppo import (
    ElementwiseAdamState,
    RunnerState,
    adam_step,
    build_components,
    initialize_learner,
    make_update_fn,
)


ROOT = Path(__file__).resolve().parents[1]


def test_adam_bias_correction_uses_elementwise_counters() -> None:
    params = {"weight": jnp.asarray([1.0, 2.0], dtype=jnp.float32)}
    grads = {"weight": jnp.ones((2,), dtype=jnp.float32)}
    state = ElementwiseAdamState(
        count={"weight": jnp.asarray([0, 4], dtype=jnp.int32)},
        mu={"weight": jnp.zeros((2,), dtype=jnp.float32)},
        nu={"weight": jnp.zeros((2,), dtype=jnp.float32)},
    )

    updated, next_state, grad_norm = adam_step(
        params,
        grads,
        state,
        learning_rate=jnp.asarray(0.1, dtype=jnp.float32),
        max_gradient_norm=100.0,
        epsilon=0.0,
    )

    np.testing.assert_array_equal(next_state.count["weight"], np.asarray([1, 5]))
    expected_mu = np.asarray([0.1, 0.1], dtype=np.float32)
    expected_nu = np.asarray([0.001, 0.001], dtype=np.float32)
    np.testing.assert_allclose(next_state.mu["weight"], expected_mu, rtol=1e-6)
    np.testing.assert_allclose(next_state.nu["weight"], expected_nu, rtol=1e-6)
    ages = np.asarray([1.0, 5.0], dtype=np.float32)
    delta = (expected_mu / (1.0 - 0.9**ages)) / np.sqrt(
        expected_nu / (1.0 - 0.999**ages)
    )
    np.testing.assert_allclose(
        updated["weight"], np.asarray([1.0, 2.0]) - 0.1 * delta, rtol=1e-5
    )
    np.testing.assert_allclose(float(grad_norm), np.sqrt(2.0), rtol=1e-6)
    assert not np.isclose(float(updated["weight"][0] - 1.0), float(updated["weight"][1] - 2.0))


class _TinyEnv:
    num_envs = 2
    obs_dim = 3
    action_dim = 2
    rule_obs_dim = 0

    def step(self, key, state, action):
        del key, action
        next_state = state + jnp.int32(1)
        obs = jnp.full(
            (self.num_envs, self.obs_dim),
            next_state.astype(jnp.float32) + 1.0,
            dtype=jnp.float32,
        )
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        done = jnp.zeros((self.num_envs,), dtype=jnp.bool_)
        info = {
            "episode_complete": zeros,
            "episode_return": zeros,
            "goal_achieved": zeros,
            "depth": jnp.ones((self.num_envs,), dtype=jnp.int32),
        }
        return obs, next_state, zeros, done, info


def test_cbp_replaced_gru_unit_is_masked_from_live_rollout_carry(monkeypatch) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config = replace(
        config,
        ppo=replace(
            config.ppo,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
        ),
        network=replace(config.network, fc_dim=4, recurrent_hidden_dim=3),
    )
    env = _TinyEnv()
    components = build_components(config, env, algorithm="ppo_cbp")
    learner = initialize_learner(
        jax.random.key(10), config, components, obs_dim=env.obs_dim
    )
    runner = RunnerState(
        learner=learner,
        env_state=jnp.asarray(0, dtype=jnp.int32),
        obs=jnp.ones((env.num_envs, env.obs_dim), dtype=jnp.float32),
        last_done=jnp.zeros((env.num_envs,), dtype=jnp.bool_),
        previous_action=jnp.zeros(
            (env.num_envs, env.action_dim), dtype=jnp.float32
        ),
        hidden=jnp.ones((env.num_envs, 3), dtype=jnp.float32),
        rng=jax.random.key(11),
        update_in_phase=jnp.asarray(0, dtype=jnp.int32),
        global_update=jnp.asarray(0, dtype=jnp.int32),
    )

    def fake_cbp(params, adam, state, activation_stats, key, cbp_config):
        del activation_stats, key, cbp_config
        replaced = jnp.asarray([False, True, False], dtype=jnp.bool_)
        return params, adam, state, jnp.asarray(1, jnp.int32), {"gru": replaced}

    monkeypatch.setattr(ppo_module, "apply_continual_backprop", fake_cbp)
    update = make_update_fn(
        config,
        components,
        env,
        total_optimizer_steps=1,
        warmup_optimizer_steps=1,
    )

    next_runner, metrics = update(runner)

    np.testing.assert_array_equal(next_runner.hidden[:, 1], np.zeros((env.num_envs,)))
    assert np.any(np.abs(np.asarray(next_runner.hidden)[:, [0, 2]]) > 1e-7)
    np.testing.assert_array_equal(
        np.asarray(next_runner.previous_action).sum(axis=-1),
        np.ones((env.num_envs,)),
    )
    assert int(metrics["gru_units_replaced"]) == 1
