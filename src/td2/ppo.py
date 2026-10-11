"""PureJaxRL-style recurrent PPO with optional Continual Backpropagation.

The rollout, reverse-time GAE, environment-sequence minibatching and clipped
losses follow ``purejaxrl/ppo_rnn.py`` at the pinned upstream commit.  The
optimizer is written explicitly so CBP can reset Adam's first moment, second
moment, and *elementwise* step counter for regenerated parameter slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

from flax import struct
from flax.core import freeze
import jax
import jax.numpy as jnp

from td2.cbp import CBPConfig, CBPState, apply_continual_backprop, init_cbp_state
from td2.config import TD2Config
from td2.environment import PoolSpec, VectorBanyan
from td2.network import (
    ActorCriticRNN,
    activation_statistics,
    categorical_entropy,
    categorical_log_prob,
    categorical_mode,
    categorical_sample,
    initial_carry,
)


Array = jax.Array
PyTree = Any


@struct.dataclass
class ElementwiseAdamState:
    """Adam state with per-parameter counters, as required by CBP."""

    count: PyTree
    mu: PyTree
    nu: PyTree


@struct.dataclass
class LearnerState:
    params: PyTree
    adam: ElementwiseAdamState
    cbp: CBPState | None
    optimizer_steps: Array
    schedule_step: Array


@struct.dataclass
class RunnerState:
    learner: LearnerState
    env_state: Any
    obs: Array
    last_done: Array
    previous_action: Array
    hidden: Array
    rng: Array
    update_in_phase: Array
    global_update: Array


@struct.dataclass
class Transition:
    reset: Array
    action: Array
    value: Array
    reward: Array
    next_done: Array
    log_prob: Array
    obs: Array
    previous_action: Array
    episode_complete: Array
    episode_return: Array
    success: Array
    depth: Array


@dataclass(frozen=True)
class PPOComponents:
    model: ActorCriticRNN
    cbp_config: CBPConfig
    use_cbp: bool


def build_components(
    config: TD2Config,
    env: VectorBanyan,
    *,
    algorithm: str,
) -> PPOComponents:
    if algorithm not in config.experiment.methods:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; configured methods are "
            f"{config.experiment.methods}"
        )
    if config.network.recurrent_type != "GRU":
        raise ValueError("The pinned PureJaxRL recurrent implementation requires GRU")
    model = ActorCriticRNN(
        action_dim=env.action_dim,
        fc_dim=config.network.fc_dim,
        gru_hidden_dim=config.network.recurrent_hidden_dim,
        activation=config.network.activation,
        rule_obs_dim=env.rule_obs_dim,
        ruleset_embed_dim=config.ruleset.ruleset_embed_dim,
    )
    return PPOComponents(
        model=model,
        cbp_config=CBPConfig(),
        use_cbp=algorithm == "ppo_cbp",
    )


def _zeros_like_tree(tree: PyTree, *, dtype=None) -> PyTree:
    return jax.tree.map(
        lambda value: jnp.zeros_like(value, dtype=dtype),
        tree,
    )


def init_adam(params: PyTree) -> ElementwiseAdamState:
    return ElementwiseAdamState(
        count=_zeros_like_tree(params, dtype=jnp.int32),
        mu=_zeros_like_tree(params),
        nu=_zeros_like_tree(params),
    )


def _global_norm(tree: PyTree) -> Array:
    squared = jax.tree.leaves(
        jax.tree.map(lambda value: jnp.sum(jnp.square(value)), tree)
    )
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def adam_step(
    params: PyTree,
    grads: PyTree,
    state: ElementwiseAdamState,
    *,
    learning_rate: Array,
    max_gradient_norm: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-5,
) -> tuple[PyTree, ElementwiseAdamState, Array]:
    """Apply clipped Adam using an independent age for every scalar parameter."""

    grad_norm = _global_norm(grads)
    scale = jnp.minimum(
        jnp.asarray(1.0, jnp.float32),
        jnp.asarray(max_gradient_norm, jnp.float32) / (grad_norm + 1e-8),
    )
    grads = jax.tree.map(lambda grad: grad * scale, grads)
    count = jax.tree.map(lambda old: old + jnp.int32(1), state.count)
    mu = jax.tree.map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state.mu,
        grads,
    )
    nu = jax.tree.map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state.nu,
        grads,
    )

    def update(param, mean, variance, age):
        age_f = age.astype(param.dtype)
        corrected_mean = mean / (1.0 - jnp.power(beta1, age_f))
        corrected_variance = variance / (1.0 - jnp.power(beta2, age_f))
        delta = corrected_mean / (jnp.sqrt(corrected_variance) + epsilon)
        return param - learning_rate.astype(param.dtype) * delta

    new_params = jax.tree.map(update, params, mu, nu, count)
    return new_params, ElementwiseAdamState(count=count, mu=mu, nu=nu), grad_norm


def phase_learning_rate(
    schedule_step: Array,
    *,
    base_learning_rate: float,
    total_optimizer_steps: int,
    warmup_optimizer_steps: int,
    anneal: bool,
) -> Array:
    """Warm up then linearly anneal within one distribution/adaptation phase."""

    step = schedule_step.astype(jnp.float32)
    base = jnp.asarray(base_learning_rate, jnp.float32)
    warmup = max(1, int(warmup_optimizer_steps))
    warm = base * jnp.minimum((step + 1.0) / float(warmup), 1.0)
    if not anneal:
        return jnp.where(step < warmup, warm, base)
    decay_denominator = max(1, int(total_optimizer_steps) - warmup)
    decay = base * jnp.maximum(
        0.0,
        (float(total_optimizer_steps) - step) / float(decay_denominator),
    )
    return jnp.where(step < warmup, warm, decay)


def initialize_learner(
    key: Array,
    config: TD2Config,
    components: PPOComponents,
    *,
    obs_dim: int,
) -> LearnerState:
    dummy_obs = jnp.zeros((1, 1, obs_dim), dtype=jnp.float32)
    dummy_previous_action = jnp.zeros(
        (1, 1, components.model.action_dim), dtype=jnp.float32
    )
    dummy_reset = jnp.zeros((1, 1), dtype=jnp.bool_)
    dummy_hidden = initial_carry(1, config.network.recurrent_hidden_dim)
    variables = components.model.init(
        key, dummy_hidden, dummy_obs, dummy_previous_action, dummy_reset
    )
    # Flax 0.12 returns a mutable dict here, while older releases returned a
    # FrozenDict.  Canonicalize it so JAX scan carries and CBP masks have one
    # stable PyTree node type across every optimizer step.
    params = freeze(variables["params"])
    return LearnerState(
        params=params,
        adam=init_adam(params),
        cbp=init_cbp_state(params) if components.use_cbp else None,
        optimizer_steps=jnp.asarray(0, dtype=jnp.int32),
        schedule_step=jnp.asarray(0, dtype=jnp.int32),
    )


def initialize_runner(
    key: Array,
    config: TD2Config,
    components: PPOComponents,
    env: VectorBanyan,
    *,
    learner: LearnerState | None = None,
    global_update: int | Array = 0,
) -> RunnerState:
    key_model, key_env, key_run = jax.random.split(key, 3)
    if learner is None:
        learner = initialize_learner(
            key_model,
            config,
            components,
            obs_dim=env.obs_dim,
        )
    else:
        learner = learner.replace(schedule_step=jnp.asarray(0, dtype=jnp.int32))
    obs, env_state = env.reset(key_env)
    return RunnerState(
        learner=learner,
        env_state=env_state,
        obs=obs,
        last_done=jnp.zeros((env.num_envs,), dtype=jnp.bool_),
        previous_action=jnp.zeros(
            (env.num_envs, env.action_dim), dtype=jnp.float32
        ),
        hidden=initial_carry(env.num_envs, config.network.recurrent_hidden_dim),
        rng=key_run,
        update_in_phase=jnp.asarray(0, dtype=jnp.int32),
        global_update=jnp.asarray(global_update, dtype=jnp.int32),
    )


def _loss_and_activations(
    params: PyTree,
    *,
    model: ActorCriticRNN,
    hidden: Array,
    obs: Array,
    previous_actions: Array,
    resets: Array,
    actions: Array,
    old_log_probs: Array,
    old_values: Array,
    advantages: Array,
    targets: Array,
    actor_clip: float,
    value_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
):
    _, logits, values, activations = model.apply(
        {"params": params}, hidden, obs, previous_actions, resets
    )
    new_log_probs = categorical_log_prob(logits, actions)
    normalized_advantages = (
        advantages - jnp.mean(advantages)
    ) / (jnp.std(advantages) + 1e-8)
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    unclipped_actor = ratio * normalized_advantages
    clipped_actor = jnp.clip(
        ratio, 1.0 - actor_clip, 1.0 + actor_clip
    ) * normalized_advantages
    actor_loss = -jnp.mean(jnp.minimum(unclipped_actor, clipped_actor))

    value_clipped = old_values + jnp.clip(
        values - old_values, -value_clip, value_clip
    )
    value_loss_unclipped = jnp.square(values - targets)
    value_loss_clipped = jnp.square(value_clipped - targets)
    value_loss = 0.5 * jnp.mean(
        jnp.maximum(value_loss_unclipped, value_loss_clipped)
    )
    entropy = jnp.mean(categorical_entropy(logits))
    total_loss = (
        actor_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > actor_clip).astype(jnp.float32))
    metrics = {
        "loss": total_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
    return total_loss, (metrics, activation_statistics(activations))


def _compute_gae(
    transitions: Transition,
    last_value: Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    def step(carry, transition: Transition):
        advantage, next_value = carry
        not_done = 1.0 - transition.next_done.astype(jnp.float32)
        delta = (
            transition.reward
            + gamma * next_value * not_done
            - transition.value
        )
        advantage = delta + gamma * gae_lambda * not_done * advantage
        return (advantage, transition.value), advantage

    initial = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(step, initial, transitions, reverse=True)
    return advantages, advantages + transitions.value


def make_update_fn(
    config: TD2Config,
    components: PPOComponents,
    env: VectorBanyan,
    *,
    total_optimizer_steps: int,
    warmup_optimizer_steps: int,
):
    """Build one compiled PPO-update function for a fixed environment shape."""

    ppo = config.ppo
    model = components.model
    num_minibatches = ppo.num_minibatches
    batch_size = env.num_envs
    if batch_size % num_minibatches:
        raise ValueError("num_envs must divide evenly into recurrent minibatches")
    minibatch_size = batch_size // num_minibatches

    def update(runner: RunnerState):
        rollout_initial_hidden = runner.hidden

        def collect_step(carry, _):
            obs, env_state, last_done, previous_action, hidden, rng = carry
            rng, key_action, key_env = jax.random.split(rng, 3)
            next_hidden, logits, value, _ = model.apply(
                {"params": runner.learner.params},
                hidden,
                obs[None, ...],
                previous_action[None, ...],
                last_done[None, ...],
            )
            next_hidden = next_hidden
            logits = logits[0]
            value = value[0]
            action = categorical_sample(logits, key_action)
            log_prob = categorical_log_prob(logits, action)
            next_obs, next_env_state, reward, next_done, info = env.step(
                key_env, env_state, action
            )
            next_previous_action = jax.nn.one_hot(
                action, model.action_dim, dtype=jnp.float32
            )
            next_previous_action = jnp.where(
                next_done[:, None],
                jnp.zeros_like(next_previous_action),
                next_previous_action,
            )
            transition = Transition(
                reset=last_done,
                action=action,
                value=value,
                reward=reward,
                next_done=next_done,
                log_prob=log_prob,
                obs=obs,
                previous_action=previous_action,
                episode_complete=info["episode_complete"],
                episode_return=info["episode_return"],
                success=info["goal_achieved"],
                depth=info["depth"],
            )
            return (
                next_obs,
                next_env_state,
                next_done,
                next_previous_action,
                next_hidden,
                rng,
            ), transition

        rollout_carry = (
            runner.obs,
            runner.env_state,
            runner.last_done,
            runner.previous_action,
            runner.hidden,
            runner.rng,
        )
        (
            next_obs,
            next_env_state,
            next_done,
            next_previous_action,
            rollout_hidden,
            rng,
        ), transitions = jax.lax.scan(
            collect_step, rollout_carry, None, length=ppo.num_steps
        )
        _, _, last_value, _ = model.apply(
            {"params": runner.learner.params},
            rollout_hidden,
            next_obs[None, ...],
            next_previous_action[None, ...],
            next_done[None, ...],
        )
        advantages, targets = _compute_gae(
            transitions,
            last_value[0],
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
        )

        def epoch_step(epoch_carry, _):
            learner, rng, hidden_bank, live_hidden, replaced_union = epoch_carry
            rng, key_permutation = jax.random.split(rng)
            permutation = jax.random.permutation(key_permutation, batch_size)
            index_batches = permutation.reshape((num_minibatches, minibatch_size))

            def minibatch_step(mb_carry, indices):
                learner, hidden_bank, live_hidden, replaced_union, rng = mb_carry
                rng, key_cbp = jax.random.split(rng)
                hidden = hidden_bank[indices]

                def select(values):
                    return values[:, indices]

                loss_fn = partial(
                    _loss_and_activations,
                    model=model,
                    hidden=hidden,
                    obs=select(transitions.obs),
                    previous_actions=select(transitions.previous_action),
                    resets=select(transitions.reset),
                    actions=select(transitions.action),
                    old_log_probs=select(transitions.log_prob),
                    old_values=select(transitions.value),
                    advantages=select(advantages),
                    targets=select(targets),
                    actor_clip=ppo.actor_clip_epsilon,
                    value_clip=ppo.value_clip_epsilon,
                    value_coefficient=ppo.value_coefficient,
                    entropy_coefficient=ppo.entropy_coefficient,
                )
                (_, (loss_metrics, activation_stats)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(learner.params)
                learning_rate = phase_learning_rate(
                    learner.schedule_step,
                    base_learning_rate=ppo.learning_rate,
                    total_optimizer_steps=total_optimizer_steps,
                    warmup_optimizer_steps=warmup_optimizer_steps,
                    anneal=ppo.anneal_learning_rate,
                )
                params, adam, grad_norm = adam_step(
                    learner.params,
                    grads,
                    learner.adam,
                    learning_rate=learning_rate,
                    max_gradient_norm=ppo.max_gradient_norm,
                )
                replacements = jnp.asarray(0, dtype=jnp.int32)
                gru_replaced = jnp.zeros(
                    (config.network.recurrent_hidden_dim,), dtype=jnp.bool_
                )
                cbp_state = learner.cbp
                if components.use_cbp:
                    assert cbp_state is not None
                    (
                        params,
                        adam,
                        cbp_state,
                        replacements,
                        unit_replacements,
                    ) = apply_continual_backprop(
                        params,
                        adam,
                        cbp_state,
                        activation_stats,
                        key_cbp,
                        components.cbp_config,
                    )
                    gru_replaced = unit_replacements["gru"]
                hidden_bank = jnp.where(
                    gru_replaced[None, :], jnp.zeros_like(hidden_bank), hidden_bank
                )
                live_hidden = jnp.where(
                    gru_replaced[None, :], jnp.zeros_like(live_hidden), live_hidden
                )
                replaced_union = replaced_union | gru_replaced
                learner = learner.replace(
                    params=params,
                    adam=adam,
                    cbp=cbp_state,
                    optimizer_steps=learner.optimizer_steps + jnp.int32(1),
                    schedule_step=learner.schedule_step + jnp.int32(1),
                )
                metrics = {
                    **loss_metrics,
                    "learning_rate": learning_rate,
                    "gradient_norm": grad_norm,
                    "cbp_replacements": replacements.astype(jnp.float32),
                }
                return (
                    learner,
                    hidden_bank,
                    live_hidden,
                    replaced_union,
                    rng,
                ), metrics

            initial_mb = (learner, hidden_bank, live_hidden, replaced_union, rng)
            (
                learner,
                hidden_bank,
                live_hidden,
                replaced_union,
                rng,
            ), minibatch_metrics = jax.lax.scan(
                minibatch_step, initial_mb, index_batches
            )
            return (
                learner,
                rng,
                hidden_bank,
                live_hidden,
                replaced_union,
            ), minibatch_metrics

        initial_epoch = (
            runner.learner,
            rng,
            rollout_initial_hidden,
            rollout_hidden,
            jnp.zeros((config.network.recurrent_hidden_dim,), dtype=jnp.bool_),
        )
        (
            learner,
            rng,
            _,
            rollout_hidden,
            replaced_union,
        ), optimization_metrics = jax.lax.scan(
            epoch_step, initial_epoch, None, length=ppo.update_epochs
        )

        completed = transitions.episode_complete.astype(jnp.float32)
        completed_count = jnp.sum(completed)
        success_count = jnp.sum(transitions.success * completed)
        episodic_return_sum = jnp.sum(transitions.episode_return * completed)
        completed_denominator = jnp.maximum(completed_count, 1.0)
        depth_index = jnp.clip(transitions.depth.astype(jnp.int32) - 1, 0, 5)
        depth_oh = jax.nn.one_hot(depth_index, 6)
        depth_completed = jnp.sum(depth_oh * completed[..., None], axis=(0, 1))
        depth_success = jnp.sum(
            depth_oh * transitions.success[..., None] * completed[..., None],
            axis=(0, 1),
        )
        optimizer_summary = jax.tree.map(jnp.mean, optimization_metrics)
        optimizer_summary["cbp_replacements"] = jnp.sum(
            optimization_metrics["cbp_replacements"]
        )
        metrics = {
            **optimizer_summary,
            "mean_transition_reward": jnp.mean(transitions.reward),
            "episode_count": completed_count,
            "success_rate": jnp.where(
                completed_count > 0,
                success_count / completed_denominator,
                jnp.nan,
            ),
            "episodic_return": jnp.where(
                completed_count > 0,
                episodic_return_sum / completed_denominator,
                jnp.nan,
            ),
            "depth_completed": depth_completed,
            "depth_success": depth_success,
            "gru_units_replaced": jnp.sum(replaced_union.astype(jnp.float32)),
        }
        next_runner = RunnerState(
            learner=learner,
            env_state=next_env_state,
            obs=next_obs,
            last_done=next_done,
            previous_action=next_previous_action,
            hidden=rollout_hidden,
            rng=rng,
            update_in_phase=runner.update_in_phase + jnp.int32(1),
            global_update=runner.global_update + jnp.int32(1),
        )
        return next_runner, metrics

    return jax.jit(update)


def balanced_evaluation_indices(
    pool: PoolSpec,
    *,
    num_episodes: int,
) -> tuple[Array, Array, Array]:
    """Create deterministic, depth-balanced task/layout indices for evaluation."""

    ids = jnp.arange(num_episodes, dtype=jnp.int32)
    if pool.fixed_pairs:
        assert pool.fixed_pair_depths is not None
        assert pool.fixed_pair_locals is not None
        assert pool.fixed_pair_layouts is not None
        pair = ids % pool.fixed_pair_depths.shape[0]
        depth = pool.fixed_pair_depths[pair]
        local = pool.fixed_pair_locals[pair]
        layout = pool.fixed_pair_layouts[pair]
    else:
        depth = ids % pool.num_depths
        local = (ids // pool.num_depths) % pool.active_n
        # A coprime affine map avoids always pairing task i with layout i.
        layout_local = (ids * 73 + local * 19 + depth * 7) % pool.active_n
        layout = pool.distribution_index * pool.max_n + layout_local
    task = depth * pool.max_n + local
    return task, layout, depth + 1


def make_evaluation_fn(
    config: TD2Config,
    components: PPOComponents,
    env: VectorBanyan,
    *,
    task_indices: Array,
    layout_indices: Array,
    depths: Array,
):
    """Build deterministic argmax-policy evaluation over one episode per slot."""

    model = components.model
    max_steps = config.ppo.max_steps

    def evaluate(params: PyTree, key: Array) -> Mapping[str, Array]:
        key_reset, key_steps = jax.random.split(key)
        obs, env_state = env.reset_with_indices(
            key_reset, task_indices, layout_indices
        )
        hidden = initial_carry(env.num_envs, config.network.recurrent_hidden_dim)
        last_done = jnp.zeros((env.num_envs,), dtype=jnp.bool_)
        previous_action = jnp.zeros(
            (env.num_envs, env.action_dim), dtype=jnp.float32
        )
        active = jnp.ones((env.num_envs,), dtype=jnp.bool_)
        success = jnp.zeros((env.num_envs,), dtype=jnp.float32)
        episode_return = jnp.zeros((env.num_envs,), dtype=jnp.float32)
        step_keys = jax.random.split(key_steps, max_steps)

        def step(carry, key_step):
            (
                obs,
                env_state,
                hidden,
                last_done,
                previous_action,
                active,
                success,
                returns,
            ) = carry
            hidden, logits, _, _ = model.apply(
                {"params": params},
                hidden,
                obs[None, ...],
                previous_action[None, ...],
                last_done[None, ...],
            )
            action = categorical_mode(logits[0])
            obs, env_state, _, done, info = env.step(
                key_step, env_state, action
            )
            first_terminal = active & done
            previous_action = jax.nn.one_hot(
                action, model.action_dim, dtype=jnp.float32
            )
            previous_action = jnp.where(
                done[:, None], jnp.zeros_like(previous_action), previous_action
            )
            success = jnp.where(
                first_terminal, info["goal_achieved"], success
            )
            returns = jnp.where(
                first_terminal, info["episode_return"], returns
            )
            active = active & (~done)
            return (
                obs,
                env_state,
                hidden,
                done,
                previous_action,
                active,
                success,
                returns,
            ), None

        initial = (
            obs,
            env_state,
            hidden,
            last_done,
            previous_action,
            active,
            success,
            episode_return,
        )
        final, _ = jax.lax.scan(step, initial, step_keys)
        active, success, episode_return = final[5], final[6], final[7]
        depth_oh = jax.nn.one_hot(jnp.clip(depths - 1, 0, 5), 6)
        depth_count = jnp.sum(depth_oh, axis=0)
        depth_success = jnp.sum(depth_oh * success[:, None], axis=0)
        return {
            "success_rate": jnp.mean(success),
            "episodic_return": jnp.mean(episode_return),
            "unfinished": jnp.sum(active.astype(jnp.int32)),
            "depth_success_rate": depth_success / jnp.maximum(depth_count, 1.0),
            "episodes": jnp.asarray(env.num_envs, dtype=jnp.int32),
        }

    return jax.jit(evaluate)
