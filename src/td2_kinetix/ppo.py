"""Kinetix v0.1.0 entity-transformer PPO, split into restartable updates.

This module deliberately imports Kinetix lazily.  The repository's default
environment is the newer Banyan/JAX runtime, while the Kinetix jobs use their
own Python 3.11/JAX 0.4.35 virtual environment on the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from td2_kinetix.config import PROTOCOL


class Transition(NamedTuple):
    done: Any
    action: Any
    value: Any
    reward: Any
    log_prob: Any
    obs: Any
    info: Any


@dataclass(frozen=True)
class PPOSettings:
    num_envs: int = 2048
    rollout_steps: int = 64
    num_minibatches: int = 32
    update_epochs: int = 8
    learning_rate: float = 5e-5
    gamma: float = 0.995
    gae_lambda: float = 0.9
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 1.0

    @property
    def environment_steps_per_update(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def optimizer_steps_per_update(self) -> int:
        return self.num_minibatches * self.update_epochs


SETTINGS = PPOSettings()


def official_config() -> dict[str, Any]:
    """Return the relevant official v0.1.0 small-entity PPO configuration."""

    config = PROTOCOL.upstream_config()
    config["permute_state_during_training"] = False
    return config


def _runtime_imports() -> dict[str, Any]:
    """Import packages that exist only in the isolated Kinetix runtime."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import struct
    from flax.serialization import to_state_dict
    from flax.training.train_state import TrainState
    from kinetix.environment.env import make_kinetix_env_from_name
    from kinetix.environment.ued.distributions import sample_kinetix_level
    from kinetix.environment.ued.ued_state import UEDParams
    from kinetix.environment.wrappers import (
        AutoReplayWrapper,
        AutoResetWrapper,
        BatchEnvWrapper,
        DenseRewardWrapper,
        LogWrapper,
        UnderspecifiedToGymnaxWrapper,
    )
    from kinetix.models import make_network_from_config
    from kinetix.models.actor_critic import ScannedRNN
    from kinetix.pcg.pcg import env_state_to_pcg_state
    from kinetix.util.config import generate_params_from_config, generate_ued_params_from_config
    from kinetix.util.learning import general_eval

    return locals()


def build_components(level_keys: Any):
    """Build the exact fixed-pool training environment and native policy.

    ``level_keys`` is a ``uint32[N, 2]`` array.  Each key deterministically
    regenerates one official procedural level; a reset samples its index
    uniformly, so full physics states never need to be stored on disk or GPU.
    """

    r = _runtime_imports()
    jax, jnp, optax = r["jax"], r["jnp"], r["optax"]
    cfg = official_config()
    env_params, static_env_params = r["generate_params_from_config"](cfg)
    cfg["static_env_params"] = r["to_state_dict"](static_env_params)
    ued_params = r["generate_ued_params_from_config"](cfg)
    raw_env = r["make_kinetix_env_from_name"](
        cfg["env_name"], static_env_params=static_env_params
    )
    keys = jnp.asarray(level_keys, dtype=jnp.uint32)
    if keys.ndim != 2 or keys.shape[1] != 2 or keys.shape[0] < 1:
        raise ValueError(f"level key bank must have shape (N, 2), got {keys.shape}")

    def sample_level(reset_rng):
        choose_rng, placement_rng = jax.random.split(reset_rng)
        index = jax.random.randint(choose_rng, (), 0, keys.shape[0])
        # The bank key fixes level structure.  The independent reset RNG is
        # still used by reset_to_pcg_level, as in the official list mode.
        level = r["sample_kinetix_level"](
            keys[index],
            raw_env.physics_engine,
            env_params,
            static_env_params,
            ued_params,
            env_size_name=cfg["env_size_name"],
        )
        del placement_rng
        return r["env_state_to_pcg_state"](level)

    train_env = r["UnderspecifiedToGymnaxWrapper"](
        r["AutoResetWrapper"](raw_env, sample_level)
    )
    train_env = r["DenseRewardWrapper"](
        train_env, dense_reward_scale=cfg["dense_reward_scale"]
    )
    train_env = r["LogWrapper"](train_env)
    train_env = r["BatchEnvWrapper"](train_env, num_envs=SETTINGS.num_envs)

    eval_raw_env = r["make_kinetix_env_from_name"](
        cfg["env_name"], static_env_params=static_env_params
    )
    eval_env = r["UnderspecifiedToGymnaxWrapper"](
        r["AutoReplayWrapper"](eval_raw_env)
    )
    eval_env = r["LogWrapper"](
        r["DenseRewardWrapper"](
            eval_env, dense_reward_scale=cfg["dense_reward_scale"]
        )
    )

    network = r["make_network_from_config"](train_env, env_params, cfg)
    tx = optax.chain(
        optax.clip_by_global_norm(SETTINGS.max_gradient_norm),
        optax.adam(SETTINGS.learning_rate, eps=1e-5),
    )
    return {
        "config": cfg,
        "env_params": env_params,
        "static_env_params": static_env_params,
        "ued_params": ued_params,
        "raw_env": raw_env,
        "train_env": train_env,
        "eval_env": eval_env,
        "network": network,
        "tx": tx,
        "TrainState": r["TrainState"],
        "ScannedRNN": r["ScannedRNN"],
        "sample_kinetix_level": r["sample_kinetix_level"],
        "general_eval": r["general_eval"],
        "jax": jax,
        "jnp": jnp,
        "np": r["np"],
    }


def initialize_runner(components: dict[str, Any], rng: Any, *, global_update: int = 0):
    """Initialize policy, optimizer, and a fresh vector environment."""

    jax, jnp = components["jax"], components["jnp"]
    env, env_params = components["train_env"], components["env_params"]
    rng, reset_rng, init_rng, runner_rng = jax.random.split(rng, 4)
    obs, env_state = env.reset(reset_rng, env_params)
    done = jnp.zeros((SETTINGS.num_envs,), dtype=jnp.bool_)
    hstate = components["ScannedRNN"].initialize_carry(SETTINGS.num_envs)
    init_input = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
    params = components["network"].init(init_rng, hstate, init_input)
    train_state = components["TrainState"].create(
        apply_fn=components["network"].apply,
        params=params,
        tx=components["tx"],
    )
    return train_state, env_state, obs, done, hstate, runner_rng, jnp.asarray(
        global_update, dtype=jnp.int32
    )


def fresh_runner_from_learner(
    components: dict[str, Any], train_state: Any, rng: Any, *, global_update: int
):
    """Start a new task distribution without resetting learner/Adam state."""

    jax, jnp = components["jax"], components["jnp"]
    rng, reset_rng, runner_rng = jax.random.split(rng, 3)
    obs, env_state = components["train_env"].reset(
        reset_rng, components["env_params"]
    )
    done = jnp.zeros((SETTINGS.num_envs,), dtype=jnp.bool_)
    hstate = components["ScannedRNN"].initialize_carry(SETTINGS.num_envs)
    return train_state, env_state, obs, done, hstate, runner_rng, jnp.asarray(
        global_update, dtype=jnp.int32
    )


def make_update_fn(components: dict[str, Any]):
    """Return one JIT-friendly PPO update, matching upstream v0.1.0."""

    jax, jnp = components["jax"], components["jnp"]
    env, env_params = components["train_env"], components["env_params"]
    network = components["network"]

    def update(runner_state):
        def env_step(state, _):
            train_state, env_state, obs, done, hstate, rng, global_update = state
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            ac_input = jax.tree_util.tree_map(
                lambda x: x[None, ...], (obs, done)
            )
            hstate, pi, value = network.apply(
                train_state.params, hstate, ac_input
            )
            action = pi.sample(seed=action_rng).squeeze(0)
            log_prob = pi.log_prob(action).squeeze(0)
            value = value.squeeze(0)
            obs_next, env_state, reward, done_next, info = env.step(
                step_rng, env_state, action, env_params
            )
            transition = Transition(
                done, action, value, reward, log_prob, obs, info
            )
            return (
                train_state,
                env_state,
                obs_next,
                done_next,
                hstate,
                rng,
                global_update,
            ), transition

        initial_hstate = runner_state[4]
        runner_state, trajectory = jax.lax.scan(
            env_step, runner_state, None, length=SETTINGS.rollout_steps
        )
        train_state, env_state, obs, done, hstate, rng, global_update = runner_state
        ac_input = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        _, _, last_value = network.apply(train_state.params, hstate, ac_input)
        last_value = last_value.squeeze(0)

        def gae_step(carry, transition):
            gae, next_value, next_done = carry
            delta = (
                transition.reward
                + SETTINGS.gamma * next_value * (1.0 - next_done)
                - transition.value
            )
            gae = (
                delta
                + SETTINGS.gamma
                * SETTINGS.gae_lambda
                * (1.0 - next_done)
                * gae
            )
            return (gae, transition.value, transition.done), gae

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(last_value), last_value, done),
            trajectory,
            reverse=True,
            unroll=16,
        )
        targets = advantages + trajectory.value

        def update_epoch(epoch_state, _):
            train_state, rng = epoch_state
            rng, permutation_rng = jax.random.split(rng)
            permutation = jax.random.permutation(
                permutation_rng, SETTINGS.num_envs
            )
            batch = (initial_hstate[None, :], trajectory, advantages, targets)
            shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], SETTINGS.num_minibatches, -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled,
            )

            def update_minibatch(current_state, batch_info):
                init_hstate, traj, advantage, target = batch_info

                def loss_fn(params):
                    _, pi, value = network.apply(
                        params,
                        init_hstate[0],
                        (traj.obs, traj.done),
                    )
                    log_prob = pi.log_prob(traj.action)
                    value_clipped = traj.value + jnp.clip(
                        value - traj.value,
                        -SETTINGS.clip_epsilon,
                        SETTINGS.clip_epsilon,
                    )
                    value_loss = 0.5 * jnp.maximum(
                        jnp.square(value - target),
                        jnp.square(value_clipped - target),
                    ).mean()
                    normalized = (advantage - advantage.mean()) / (
                        advantage.std() + 1e-8
                    )
                    ratio = jnp.exp(log_prob - traj.log_prob)
                    actor_loss = -jnp.minimum(
                        ratio * normalized,
                        jnp.clip(
                            ratio,
                            1.0 - SETTINGS.clip_epsilon,
                            1.0 + SETTINGS.clip_epsilon,
                        )
                        * normalized,
                    ).mean()
                    entropy = pi.entropy().mean()
                    total = (
                        actor_loss
                        + SETTINGS.value_coefficient * value_loss
                        - SETTINGS.entropy_coefficient * entropy
                    )
                    return total, (value_loss, actor_loss, entropy)

                (loss, aux), gradients = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(current_state.params)
                current_state = current_state.apply_gradients(grads=gradients)
                return current_state, (loss, *aux)

            train_state, losses = jax.lax.scan(
                update_minibatch, train_state, minibatches
            )
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            update_epoch,
            (train_state, rng),
            None,
            length=SETTINGS.update_epochs,
        )
        episode_mask = trajectory.info["returned_episode"]
        episode_count = episode_mask.sum()
        denominator = jnp.maximum(episode_count, 1)
        success_rate = (
            trajectory.info["returned_episode_solved"] * episode_mask
        ).sum() / denominator
        episodic_return = (
            trajectory.info["returned_episode_returns"] * episode_mask
        ).sum() / denominator
        metrics = {
            "success_rate": success_rate,
            "episodic_return": episodic_return,
            "episode_count": episode_count,
            "transition_reward": trajectory.reward.mean(),
            "loss": losses[0].mean(),
            "value_loss": losses[1].mean(),
            "actor_loss": losses[2].mean(),
            "entropy": losses[3].mean(),
        }
        next_state = (
            train_state,
            env_state,
            obs,
            done,
            hstate,
            rng,
            global_update + 1,
        )
        return next_state, metrics

    return jax.jit(update)


def make_heldout_evaluator(
    components: dict[str, Any], heldout_keys: Any, *, attempts: int = 10
):
    """Return a JITted fixed-100-level stochastic-policy success evaluator."""

    jax, jnp = components["jax"], components["jnp"]
    keys = jnp.asarray(heldout_keys, dtype=jnp.uint32)
    if keys.shape != (100, 2):
        raise ValueError(f"heldout key bank must have shape (100, 2), got {keys.shape}")
    cfg = components["config"]
    raw_env = components["raw_env"]
    levels = jax.vmap(
        lambda key: components["sample_kinetix_level"](
            key,
            raw_env.physics_engine,
            components["env_params"],
            components["static_env_params"],
            components["ued_params"],
            env_size_name=cfg["env_size_name"],
        )
    )(keys)
    levels = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x, repeats=attempts, axis=0), levels
    )
    num_levels = 100 * attempts

    def evaluate(train_state, rng):
        result, (dones, _) = components["general_eval"](
            rng,
            components["eval_env"],
            components["env_params"],
            train_state,
            levels,
            components["env_params"].max_timesteps,
            num_levels,
            keep_states=False,
            return_trajectories=True,
        )
        infos = result[4]
        episodes_per_attempt = dones.sum(axis=0)
        solves_per_attempt = (
            infos["returned_episode_solved"] * dones
        ).sum(axis=0)
        # Match Kinetix's native evaluator: each stochastic attempt reports
        # the fraction of its completed episodes that solved the level, then
        # attempts and levels are averaged with equal weight.
        per_attempt_success = solves_per_attempt / jnp.maximum(
            episodes_per_attempt, 1
        )
        per_task_success = jnp.reshape(
            per_attempt_success, (100, attempts)
        ).mean(axis=1)
        return {
            "success_rate": per_attempt_success.mean(),
            "task_success_rate": per_task_success.mean(),
            "completed_episodes": episodes_per_attempt.sum(),
            "unfinished_attempts": (episodes_per_attempt == 0).sum(),
        }

    return jax.jit(evaluate)
