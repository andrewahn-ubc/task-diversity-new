from __future__ import annotations

from collections import namedtuple

from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from td2.cbp import (
    CBPConfig,
    LayerCBPState,
    _select_replacements,
    apply_continual_backprop,
    init_cbp_state,
    outgoing_weight_magnitude,
)


AdamState = namedtuple("AdamState", ("count", "mu", "nu"))


def _values(shape: tuple[int, ...], offset: float) -> jax.Array:
    size = int(np.prod(shape))
    return jnp.arange(size, dtype=jnp.float32).reshape(shape) + offset


def _params():
    return freeze(
        {
            "obs_embed": {
                "kernel": _values((3, 3), 1.0),
                "bias": _values((3,), 0.1),
            },
            "rnn_input": {
                "kernel": _values((3, 3), 5.0),
                "bias": _values((3,), 0.15),
            },
            "gru": {
                "x_r": _values((3, 3), 10.0),
                "x_z": _values((3, 3), 20.0),
                "x_n": _values((3, 3), 30.0),
                "h_r": _values((3, 3), 40.0),
                "h_z": _values((3, 3), 50.0),
                "h_n": _values((3, 3), 60.0),
                "b_r": _values((3,), 0.2),
                "b_z": _values((3,), 0.3),
                "b_n": _values((3,), 0.4),
                "b_hn": _values((3,), 0.45),
            },
            "actor_hidden_1": {
                "kernel": _values((3, 3), 70.0),
                "bias": _values((3,), 0.5),
            },
            "actor_hidden_2": {
                "kernel": _values((3, 3), 75.0),
                "bias": _values((3,), 0.55),
            },
            "actor_out": {
                "kernel": _values((3, 2), 80.0),
                "bias": _values((2,), 0.6),
            },
            "critic_hidden_1": {
                "kernel": _values((3, 3), 90.0),
                "bias": _values((3,), 0.7),
            },
            "critic_hidden_2": {
                "kernel": _values((3, 3), 95.0),
                "bias": _values((3,), 0.75),
            },
            "critic_out": {
                "kernel": _values((3, 1), 100.0),
                "bias": _values((1,), 0.8),
            },
        }
    )


def _adam(params) -> AdamState:
    return AdamState(
        count=jax.tree.map(
            lambda value: jnp.full(value.shape, 7, dtype=jnp.int32), params
        ),
        mu=jax.tree.map(lambda value: jnp.full_like(value, 2.0), params),
        nu=jax.tree.map(lambda value: jnp.full_like(value, 3.0), params),
    )


def _stats(state, *, target: str | None = None, target_mean: float = 0.0):
    result = {}
    for name, layer in state.layers.items():
        mean = jnp.zeros_like(layer.utility)
        if name == target:
            mean = mean.at[1].set(target_mean)
        result[name] = {"mean": mean, "mad": jnp.ones_like(layer.utility)}
    return freeze(result)


def _make_only_unit_one_mature(state, target: str):
    layers = dict(state.layers)
    target_state = layers[target]
    ages = jnp.zeros_like(target_state.age).at[1].set(1)
    layers[target] = target_state.replace(age=ages)
    return state.replace(layers=freeze(layers))


def _assert_tree_equal(left, right) -> None:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def test_no_replacements_or_mutations_before_maturity() -> None:
    params = _params()
    adam = _adam(params)
    state = init_cbp_state(params)
    stats = _stats(state)
    config = CBPConfig(utility_decay=0.5, maturity_steps=3, replacement_rate=1.0)

    current_params, current_adam, current_state = params, adam, state
    for step in range(2):
        (
            current_params,
            current_adam,
            current_state,
            replacements,
            unit_replacements,
        ) = apply_continual_backprop(
            current_params,
            current_adam,
            current_state,
            stats,
            jax.random.key(step),
            config,
        )
        assert int(replacements) == 0
        assert all(not np.asarray(mask).any() for mask in unit_replacements.values())
        _assert_tree_equal(current_params, params)
        _assert_tree_equal(current_adam, adam)
    assert all(
        np.all(np.asarray(layer.age) == 2)
        for layer in current_state.layers.values()
    )


def test_expanded_cbp_state_and_outgoing_graph() -> None:
    params = _params()
    state = init_cbp_state(params)
    assert set(state.layers) == {
        "actor_hidden_1",
        "actor_hidden_2",
        "critic_hidden_1",
        "critic_hidden_2",
        "gru",
        "obs_embed",
        "rnn_input",
    }
    assert all(layer.utility.shape == (3,) for layer in state.layers.values())

    p = unfreeze(params)
    np.testing.assert_allclose(
        outgoing_weight_magnitude(params, "obs_embed"),
        np.mean(np.abs(p["rnn_input"]["kernel"]), axis=1),
    )
    np.testing.assert_allclose(
        outgoing_weight_magnitude(params, "rnn_input"),
        sum(
            np.mean(np.abs(p["gru"][f"x_{gate}"]), axis=1)
            for gate in ("r", "z", "n")
        ),
    )
    np.testing.assert_allclose(
        outgoing_weight_magnitude(params, "actor_hidden_1"),
        np.mean(np.abs(p["actor_hidden_2"]["kernel"]), axis=1),
    )
    np.testing.assert_allclose(
        outgoing_weight_magnitude(params, "actor_hidden_2"),
        np.mean(np.abs(p["actor_out"]["kernel"]), axis=1),
    )


def test_fractional_budget_accumulates_and_selects_lowest_utility() -> None:
    utilities = jnp.asarray([0.4, 0.1, 0.3, 0.2], dtype=jnp.float32)
    state = LayerCBPState(
        utility=utilities,
        mean_activation=jnp.zeros((4,), dtype=jnp.float32),
        age=jnp.zeros((4,), dtype=jnp.int32),
        fractional_count=jnp.asarray(0.0, dtype=jnp.float32),
    )
    config = CBPConfig(utility_decay=0.5, maturity_steps=1, replacement_rate=0.2)

    state, first, _ = _select_replacements(
        state,
        jnp.zeros((4,), dtype=jnp.float32),
        utilities,
        jnp.ones((4,), dtype=jnp.float32),
        config,
    )
    np.testing.assert_array_equal(first, np.zeros((4,), dtype=np.bool_))
    assert float(state.fractional_count) == pytest.approx(0.8)

    state, second, _ = _select_replacements(
        state,
        jnp.zeros((4,), dtype=jnp.float32),
        utilities,
        jnp.ones((4,), dtype=jnp.float32),
        config,
    )
    np.testing.assert_array_equal(second, np.asarray([False, True, False, False]))
    assert float(state.fractional_count) == pytest.approx(0.6)
    assert int(state.age[1]) == 0


def test_dense_replacement_resets_exact_parameter_and_adam_slices() -> None:
    params = _params()
    old = unfreeze(params)
    adam = _adam(params)
    state = _make_only_unit_one_mature(init_cbp_state(params), "actor_hidden_1")
    config = CBPConfig(utility_decay=0.0, maturity_steps=2, replacement_rate=1.0)

    new_params, new_adam, _, replacements, masks = apply_continual_backprop(
        params,
        adam,
        state,
        _stats(state, target="actor_hidden_1", target_mean=2.0),
        jax.random.key(20),
        config,
    )
    new = unfreeze(new_params)

    assert int(replacements) == 1
    np.testing.assert_array_equal(masks["actor_hidden_1"], [False, True, False])
    np.testing.assert_array_equal(
        np.asarray(new["actor_hidden_1"]["kernel"])[:, [0, 2]],
        np.asarray(old["actor_hidden_1"]["kernel"])[:, [0, 2]],
    )
    assert float(new["actor_hidden_1"]["bias"][1]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(new["actor_hidden_1"]["bias"])[[0, 2]],
        np.asarray(old["actor_hidden_1"]["bias"])[[0, 2]],
    )
    np.testing.assert_array_equal(
        new["actor_hidden_2"]["kernel"][1], np.zeros((3,))
    )
    np.testing.assert_array_equal(
        np.asarray(new["actor_hidden_2"]["kernel"])[[0, 2]],
        np.asarray(old["actor_hidden_2"]["kernel"])[[0, 2]],
    )
    np.testing.assert_allclose(
        new["actor_hidden_2"]["bias"],
        old["actor_hidden_2"]["bias"]
        + 2.0 * old["actor_hidden_2"]["kernel"][1],
    )

    for field, untouched in (("count", 7), ("mu", 2.0), ("nu", 3.0)):
        tree = unfreeze(getattr(new_adam, field))
        np.testing.assert_array_equal(tree["actor_hidden_1"]["kernel"][:, 1], 0)
        np.testing.assert_array_equal(
            np.asarray(tree["actor_hidden_1"]["kernel"])[:, [0, 2]], untouched
        )
        assert float(tree["actor_hidden_1"]["bias"][1]) == 0.0
        np.testing.assert_array_equal(
            np.asarray(tree["actor_hidden_1"]["bias"])[[0, 2]], untouched
        )
        np.testing.assert_array_equal(tree["actor_hidden_2"]["kernel"][1], 0)
        np.testing.assert_array_equal(
            np.asarray(tree["actor_hidden_2"]["kernel"])[[0, 2]], untouched
        )
        np.testing.assert_array_equal(tree["actor_hidden_2"]["bias"], 0)


def test_gru_replacement_resets_incoming_columns_and_all_outgoing_rows() -> None:
    params = _params()
    old = unfreeze(params)
    adam = _adam(params)
    state = _make_only_unit_one_mature(init_cbp_state(params), "gru")
    config = CBPConfig(utility_decay=0.0, maturity_steps=2, replacement_rate=1.0)
    mean = 1.5

    new_params, new_adam, _, replacements, masks = apply_continual_backprop(
        params,
        adam,
        state,
        _stats(state, target="gru", target_mean=mean),
        jax.random.key(21),
        config,
    )
    new = unfreeze(new_params)

    assert int(replacements) == 1
    np.testing.assert_array_equal(masks["gru"], [False, True, False])
    keep = np.ix_([0, 2], [0, 2])
    for gate in ("r", "z", "n"):
        x_name, h_name = f"x_{gate}", f"h_{gate}"
        input_bias = f"b_{gate}"
        compensation_bias = input_bias if gate != "n" else "b_hn"
        np.testing.assert_array_equal(
            np.asarray(new["gru"][x_name])[:, [0, 2]],
            np.asarray(old["gru"][x_name])[:, [0, 2]],
        )
        np.testing.assert_array_equal(new["gru"][h_name][1], np.zeros((3,)))
        np.testing.assert_array_equal(
            np.asarray(new["gru"][h_name])[keep],
            np.asarray(old["gru"][h_name])[keep],
        )
        expected_compensation_bias = np.asarray(
            old["gru"][compensation_bias]
        ).copy()
        expected_compensation_bias[1] = 0.0
        expected_compensation_bias += mean * np.asarray(old["gru"][h_name][1])
        expected_compensation_bias[1] = 0.0
        np.testing.assert_allclose(
            new["gru"][compensation_bias], expected_compensation_bias
        )
        if gate == "n":
            expected_input_bias = np.asarray(old["gru"][input_bias]).copy()
            expected_input_bias[1] = 0.0
            np.testing.assert_allclose(new["gru"][input_bias], expected_input_bias)

    for destination in ("actor_hidden_1", "critic_hidden_1"):
        np.testing.assert_array_equal(new[destination]["kernel"][1], np.zeros((3,)))
        np.testing.assert_array_equal(
            np.asarray(new[destination]["kernel"])[[0, 2]],
            np.asarray(old[destination]["kernel"])[[0, 2]],
        )
        np.testing.assert_allclose(
            new[destination]["bias"],
            old[destination]["bias"] + mean * old[destination]["kernel"][1],
        )

    for field, untouched in (("count", 7), ("mu", 2.0), ("nu", 3.0)):
        tree = unfreeze(getattr(new_adam, field))
        for gate in ("r", "z", "n"):
            x_name, h_name = f"x_{gate}", f"h_{gate}"
            np.testing.assert_array_equal(tree["gru"][x_name][:, 1], 0)
            np.testing.assert_array_equal(
                np.asarray(tree["gru"][x_name])[:, [0, 2]], untouched
            )
            np.testing.assert_array_equal(tree["gru"][h_name][1], 0)
            np.testing.assert_array_equal(tree["gru"][h_name][:, 1], 0)
            np.testing.assert_array_equal(
                np.asarray(tree["gru"][h_name])[keep], untouched
            )
        for bias in ("b_r", "b_z", "b_hn"):
            np.testing.assert_array_equal(tree["gru"][bias], 0)
        np.testing.assert_array_equal(tree["gru"]["b_n"][1], 0)
        np.testing.assert_array_equal(
            np.asarray(tree["gru"]["b_n"])[[0, 2]], untouched
        )
        for destination in ("actor_hidden_1", "critic_hidden_1"):
            np.testing.assert_array_equal(tree[destination]["kernel"][1], 0)
            np.testing.assert_array_equal(
                np.asarray(tree[destination]["kernel"])[[0, 2]], untouched
            )
            np.testing.assert_array_equal(tree[destination]["bias"], 0)


def test_simultaneous_replacements_keep_every_new_unit_disconnected() -> None:
    params = _params()
    adam = _adam(params)
    state = init_cbp_state(params)
    layers = {
        name: layer.replace(age=jnp.ones_like(layer.age))
        for name, layer in state.layers.items()
    }
    state = state.replace(layers=freeze(layers))
    config = CBPConfig(utility_decay=0.0, maturity_steps=2, replacement_rate=1.0)

    new_params, _, _, replacements, masks = apply_continual_backprop(
        params,
        adam,
        state,
        _stats(state),
        jax.random.key(22),
        config,
    )
    new = unfreeze(new_params)
    assert int(replacements) == 21
    for replace in masks.values():
        np.testing.assert_array_equal(replace, np.ones((3,), dtype=np.bool_))
    for gate in ("r", "z", "n"):
        np.testing.assert_array_equal(new["gru"][f"x_{gate}"], 0)
        np.testing.assert_array_equal(new["gru"][f"h_{gate}"], 0)
    np.testing.assert_array_equal(new["rnn_input"]["kernel"], 0)
    np.testing.assert_array_equal(new["actor_hidden_1"]["kernel"], 0)
    np.testing.assert_array_equal(new["actor_hidden_2"]["kernel"], 0)
    np.testing.assert_array_equal(new["critic_hidden_1"]["kernel"], 0)
    np.testing.assert_array_equal(new["critic_hidden_2"]["kernel"], 0)
    np.testing.assert_array_equal(new["actor_out"]["kernel"], 0)
    np.testing.assert_array_equal(new["critic_out"]["kernel"], 0)
    np.testing.assert_array_equal(new["obs_embed"]["bias"], 0)
    for bias in ("b_r", "b_z", "b_n", "b_hn"):
        np.testing.assert_array_equal(new["gru"][bias], 0)
    np.testing.assert_array_equal(new["rnn_input"]["bias"], 0)
    np.testing.assert_array_equal(new["actor_hidden_1"]["bias"], 0)
    np.testing.assert_array_equal(new["actor_hidden_2"]["bias"], 0)
    np.testing.assert_array_equal(new["critic_hidden_1"]["bias"], 0)
    np.testing.assert_array_equal(new["critic_hidden_2"]["bias"], 0)


def test_rnn_input_replacement_resets_all_gru_input_edges() -> None:
    params = _params()
    old = unfreeze(params)
    adam = _adam(params)
    state = _make_only_unit_one_mature(init_cbp_state(params), "rnn_input")
    config = CBPConfig(utility_decay=0.0, maturity_steps=2, replacement_rate=1.0)

    new_params, new_adam, _, replacements, masks = apply_continual_backprop(
        params,
        adam,
        state,
        _stats(state, target="rnn_input", target_mean=2.0),
        jax.random.key(23),
        config,
    )
    new = unfreeze(new_params)

    assert int(replacements) == 1
    np.testing.assert_array_equal(masks["rnn_input"], [False, True, False])
    assert float(new["rnn_input"]["bias"][1]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(new["rnn_input"]["kernel"])[:, [0, 2]],
        np.asarray(old["rnn_input"]["kernel"])[:, [0, 2]],
    )
    for gate in ("r", "z", "n"):
        name = f"x_{gate}"
        np.testing.assert_array_equal(new["gru"][name][1], np.zeros((3,)))
        np.testing.assert_array_equal(
            np.asarray(new["gru"][name])[[0, 2]],
            np.asarray(old["gru"][name])[[0, 2]],
        )
        np.testing.assert_allclose(
            new["gru"][f"b_{gate}"],
            old["gru"][f"b_{gate}"] + 2.0 * old["gru"][name][1],
        )

    for field, untouched in (("count", 7), ("mu", 2.0), ("nu", 3.0)):
        tree = unfreeze(getattr(new_adam, field))
        np.testing.assert_array_equal(tree["rnn_input"]["kernel"][:, 1], 0)
        np.testing.assert_array_equal(
            np.asarray(tree["rnn_input"]["kernel"])[:, [0, 2]], untouched
        )
        assert float(tree["rnn_input"]["bias"][1]) == 0.0
        for gate in ("r", "z", "n"):
            np.testing.assert_array_equal(tree["gru"][f"x_{gate}"][1], 0)
            np.testing.assert_array_equal(tree["gru"][f"b_{gate}"], 0)
