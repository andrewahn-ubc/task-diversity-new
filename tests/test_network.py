from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from td2.network import ActorCriticRNN, ManualGRU, initial_carry


def test_manual_gru_matches_purejaxrl_flax_cell_equations() -> None:
    time, batch, input_dim, hidden_dim = 4, 2, 5, 3
    inputs = jax.random.normal(
        jax.random.key(30), (time, batch, input_dim), dtype=jnp.float32
    )
    carry = jax.random.normal(
        jax.random.key(31), (batch, hidden_dim), dtype=jnp.float32
    )
    reference = nn.GRUCell(hidden_dim)
    reference_variables = reference.init(jax.random.key(32), carry, inputs[0])
    reference_params = reference_variables["params"]
    manual_variables = {
        "params": {
            "x_r": reference_params["ir"]["kernel"],
            "x_z": reference_params["iz"]["kernel"],
            "x_n": reference_params["in"]["kernel"],
            "h_r": reference_params["hr"]["kernel"],
            "h_z": reference_params["hz"]["kernel"],
            "h_n": reference_params["hn"]["kernel"],
            "b_r": reference_params["ir"]["bias"],
            "b_z": reference_params["iz"]["bias"],
            "b_n": reference_params["in"]["bias"],
            "b_hn": reference_params["hn"]["bias"],
        }
    }

    def reference_step(previous, item):
        return reference.apply(reference_variables, previous, item)

    expected_final, expected_history = jax.lax.scan(
        reference_step, carry, inputs
    )
    actual_final, actual_history = ManualGRU(hidden_dim).apply(
        manual_variables,
        carry,
        inputs,
        jnp.zeros((time, batch), dtype=jnp.bool_),
    )
    np.testing.assert_allclose(actual_history, expected_history, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_final, expected_final, rtol=1e-6, atol=1e-6)


def test_actor_critic_forward_shapes_include_all_cbp_activations() -> None:
    time, batch, obs_dim = 4, 2, 8
    model = ActorCriticRNN(
        action_dim=5,
        embedding_dim=6,
        fc_dim=7,
        gru_hidden_dim=3,
        activation="relu",
        rule_obs_dim=2,
        ruleset_embed_dim=4,
    )
    hidden = initial_carry(batch, 3)
    obs = jax.random.normal(jax.random.key(1), (time, batch, obs_dim))
    resets = jnp.zeros((time, batch), dtype=jnp.bool_)
    variables = model.init(jax.random.key(2), hidden, obs, resets)

    final_hidden, logits, values, activations = model.apply(
        variables, hidden, obs, resets
    )

    assert final_hidden.shape == (batch, 3)
    assert logits.shape == (time, batch, 5)
    assert values.shape == (time, batch)
    assert {name: value.shape for name, value in activations.items()} == {
        "rule_encoder": (time, batch, 4),
        "obs_embed": (time, batch, 6),
        "rnn_input": (time, batch, 3),
        "gru": (time, batch, 3),
        "actor_hidden_1": (time, batch, 7),
        "actor_hidden_2": (time, batch, 7),
        "critic_hidden_1": (time, batch, 7),
        "critic_hidden_2": (time, batch, 7),
    }
    params = variables["params"]
    assert params["obs_embed"]["kernel"].shape == (
        obs_dim - model.rule_obs_dim + model.ruleset_embed_dim,
        6,
    )
    assert params["rnn_input"]["kernel"].shape == (6, 3)
    assert params["gru"]["x_r"].shape == (3, 3)
    assert params["actor_hidden_1"]["kernel"].shape == (3, 7)
    assert params["actor_hidden_2"]["kernel"].shape == (7, 7)
    assert params["critic_hidden_1"]["kernel"].shape == (3, 7)
    assert params["critic_hidden_2"]["kernel"].shape == (7, 7)


def test_author_output_initializers_use_scale_point_zero_one() -> None:
    model = ActorCriticRNN(
        action_dim=3,
        embedding_dim=6,
        fc_dim=5,
        gru_hidden_dim=4,
        activation="relu",
    )
    hidden = initial_carry(2, 4)
    obs = jax.random.normal(jax.random.key(40), (1, 2, 6))
    reset = jnp.ones((1, 2), dtype=jnp.bool_)
    params = model.init(jax.random.key(41), hidden, obs, reset)["params"]

    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(params["actor_out"]["kernel"]), axis=0),
        np.full((3,), 0.01),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(params["critic_out"]["kernel"]), axis=0),
        np.asarray([0.01]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gru_reset_makes_suffix_independent_of_pre_reset_carry() -> None:
    time, batch, input_dim, hidden_dim = 5, 2, 4, 3
    cell = ManualGRU(hidden_dim=hidden_dim)
    inputs = jax.random.normal(
        jax.random.key(3), (time, batch, input_dim), dtype=jnp.float32
    )
    initial = jnp.arange(1, batch * hidden_dim + 1, dtype=jnp.float32).reshape(
        batch, hidden_dim
    )
    resets = jnp.zeros((time, batch), dtype=jnp.bool_).at[2, :].set(True)
    variables = cell.init(jax.random.key(4), initial, inputs, resets)

    full_final, full_history = cell.apply(variables, initial, inputs, resets)
    suffix_resets = jnp.zeros((time - 2, batch), dtype=jnp.bool_)
    suffix_final, suffix_history = cell.apply(
        variables,
        initial_carry(batch, hidden_dim),
        inputs[2:],
        suffix_resets,
    )

    np.testing.assert_allclose(full_history[2:], suffix_history, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(full_final, suffix_final, rtol=1e-6, atol=1e-6)
