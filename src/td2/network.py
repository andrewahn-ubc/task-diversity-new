"""Recurrent actor-critic used by the continual Banyan experiments.

This is a forward port of the network shape in PureJaxRL's ``ppo_rnn.py``
(commit 31756b197773a52db763fdbe6d635e4b46522a73).  The implementation is
deliberately explicit rather than using ``flax.linen.GRUCell`` so continual
backprop can replace well-defined rows and columns of every recurrent matrix.
"""

from __future__ import annotations

from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, lecun_normal, orthogonal


Array = jax.Array


def _activation(name: str):
    if name == "relu":
        return nn.relu
    if name == "tanh":
        return nn.tanh
    raise ValueError(f"Unsupported activation: {name!r}")


class ManualGRU(nn.Module):
    """A scan-friendly GRU with stable, named parameter matrices.

    ``resets[t, b]`` indicates that the carry must be zeroed *before*
    processing ``inputs[t, b]``.  This matches PureJaxRL's previous-done
    convention and keeps recurrent state across rollout boundaries.
    """

    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        carry: Array,
        inputs: Array,
        resets: Array,
    ) -> tuple[Array, Array]:
        in_dim = int(inputs.shape[-1])
        h = self.hidden_dim
        # Match the defaults of the Flax GRUCell used by PureJaxRL: LeCun
        # normal input kernels, orthogonal recurrent kernels, input biases for
        # all gates, and a separate recurrent bias for the candidate gate.
        x_init = lecun_normal()
        h_init = orthogonal(1.0)

        x_r = self.param("x_r", x_init, (in_dim, h))
        x_z = self.param("x_z", x_init, (in_dim, h))
        x_n = self.param("x_n", x_init, (in_dim, h))
        h_r = self.param("h_r", h_init, (h, h))
        h_z = self.param("h_z", h_init, (h, h))
        h_n = self.param("h_n", h_init, (h, h))
        b_r = self.param("b_r", constant(0.0), (h,))
        b_z = self.param("b_z", constant(0.0), (h,))
        b_n = self.param("b_n", constant(0.0), (h,))
        b_hn = self.param("b_hn", constant(0.0), (h,))

        def step(prev: Array, item: tuple[Array, Array]) -> tuple[Array, Array]:
            x_t, reset_t = item
            prev = jnp.where(reset_t[:, None], jnp.zeros_like(prev), prev)
            reset_gate = jax.nn.sigmoid(x_t @ x_r + prev @ h_r + b_r)
            update_gate = jax.nn.sigmoid(x_t @ x_z + prev @ h_z + b_z)
            candidate = jnp.tanh(
                x_t @ x_n + b_n + reset_gate * (prev @ h_n + b_hn)
            )
            new = (1.0 - update_gate) * candidate + update_gate * prev
            return new, new

        return jax.lax.scan(step, carry, (inputs, resets))


class ActorCriticRNN(nn.Module):
    """Shared recurrent body with one hidden layer in each output head.

    The FC depth is the closest faithful generalization of PureJaxRL: one
    observation embedding before the GRU and one hidden FC in each actor and
    critic head.  The previous-action one-hot is appended after that embedding,
    following the recurrent-state definition in the paper.  A separate rule
    encoder is used only when the YAML-requested rule observations are present.
    """

    action_dim: int
    fc_dim: int
    gru_hidden_dim: int
    activation: str = "relu"
    rule_obs_dim: int = 0
    ruleset_embed_dim: int = 128

    @nn.compact
    def __call__(
        self,
        hidden: Array,
        obs: Array,
        previous_actions: Array,
        resets: Array,
    ) -> tuple[Array, Array, Array, Mapping[str, Array]]:
        activate = _activation(self.activation)
        activations: dict[str, Array] = {}

        if self.rule_obs_dim:
            base_obs = obs[..., : -self.rule_obs_dim]
            rule_obs = obs[..., -self.rule_obs_dim :]
            rule_embedding = nn.Dense(
                self.ruleset_embed_dim,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="rule_encoder",
            )(rule_obs)
            rule_embedding = activate(rule_embedding)
            activations["rule_encoder"] = rule_embedding
            obs = jnp.concatenate((base_obs, rule_embedding), axis=-1)

        embedding = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="obs_embed",
        )(obs)
        embedding = activate(embedding)
        activations["obs_embed"] = embedding

        # The paper's recurrent policy conditions on the current observation
        # and previous action.  A zero vector is the episode-start sentinel;
        # mask here as well as in the runner so replayed trajectories cannot
        # accidentally carry a terminal action across an episode boundary.
        previous_actions = jnp.where(
            resets[..., None], jnp.zeros_like(previous_actions), previous_actions
        )
        recurrent_input = jnp.concatenate((embedding, previous_actions), axis=-1)

        hidden, memory = ManualGRU(self.gru_hidden_dim, name="gru")(
            hidden, recurrent_input, resets
        )
        activations["gru"] = memory

        actor_hidden = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="actor_hidden",
        )(memory)
        actor_hidden = activate(actor_hidden)
        activations["actor_hidden"] = actor_hidden
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor_hidden)

        critic_hidden = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="critic_hidden",
        )(memory)
        critic_hidden = activate(critic_hidden)
        activations["critic_hidden"] = critic_hidden
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic_hidden)

        return hidden, logits, jnp.squeeze(value, axis=-1), activations


def initial_carry(batch_size: int, hidden_dim: int) -> Array:
    return jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)


def categorical_sample(logits: Array, key: Array) -> Array:
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def categorical_mode(logits: Array) -> Array:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def categorical_log_prob(logits: Array, actions: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]


def categorical_entropy(logits: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def activation_statistics(activations: Mapping[str, Array]) -> dict[str, dict[str, Array]]:
    """Reduce sequence activations to CBP's per-unit mean and MAD."""

    result: dict[str, dict[str, Array]] = {}
    for name, values in activations.items():
        axes = tuple(range(values.ndim - 1))
        mean = jnp.mean(values, axis=axes)
        mad = jnp.mean(jnp.abs(values - mean), axis=axes)
        result[name] = {"mean": mean, "mad": mad}
    return result
