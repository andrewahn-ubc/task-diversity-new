"""Recurrent actor-critic used by the continual Banyan experiments.

The topology matches the supplied authors' implementation: a 512-dimensional
float embedding, a linear projection to the recurrent width, a GRU, and two
1024-dimensional hidden layers in each output head.  The implementation is
deliberately explicit rather than using ``flax.linen.GRUCell`` so continual
backprop can replace well-defined rows and columns of every recurrent matrix.
"""

from __future__ import annotations

from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import (
    constant,
    lecun_normal,
    orthogonal,
    variance_scaling,
)


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
    """Authors' recurrent actor-critic topology, adapted to one agent.

    A separate rule encoder remains available only when rule observations are
    enabled.  The production experiment hides rules, as specified by its
    single-agent Figure-6 protocol.
    """

    action_dim: int
    embedding_dim: int
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
            self.embedding_dim,
            kernel_init=variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            ),
            bias_init=constant(0.0),
            name="obs_embed",
        )(obs)
        embedding = activate(embedding)
        activations["obs_embed"] = embedding

        recurrent_input = nn.Dense(
            self.gru_hidden_dim,
            kernel_init=lecun_normal(),
            bias_init=constant(0.0),
            name="rnn_input",
        )(embedding)
        activations["rnn_input"] = recurrent_input

        hidden, memory = ManualGRU(self.gru_hidden_dim, name="gru")(
            hidden, recurrent_input, resets
        )
        activations["gru"] = memory

        actor_hidden_1 = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="actor_hidden_1",
        )(memory)
        actor_hidden_1 = activate(actor_hidden_1)
        activations["actor_hidden_1"] = actor_hidden_1
        actor_hidden_2 = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="actor_hidden_2",
        )(actor_hidden_1)
        actor_hidden_2 = activate(actor_hidden_2)
        activations["actor_hidden_2"] = actor_hidden_2
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor_hidden_2)

        critic_hidden_1 = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="critic_hidden_1",
        )(memory)
        critic_hidden_1 = activate(critic_hidden_1)
        activations["critic_hidden_1"] = critic_hidden_1
        critic_hidden_2 = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="critic_hidden_2",
        )(critic_hidden_1)
        critic_hidden_2 = activate(critic_hidden_2)
        activations["critic_hidden_2"] = critic_hidden_2
        value = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic_hidden_2)

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
