"""Continual Backpropagation (CBP) for the recurrent PPO network.

The utility/replacement rule follows the attached Task Diversity paper.  The
generate-and-test mechanics (bias compensation, zero outgoing weights, and
elementwise Adam-state resets) are adapted from ``shibhansh/loss-of-plasticity``
at commit ``a6b79580d85f3025bdb601566d3627c5f489f13b``.  This is a JAX
implementation; no PyTorch code is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import flax
from flax import struct
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class CBPConfig:
    utility_decay: float = 0.986
    maturity_steps: int = 5000
    replacement_rate: float = 5e-5


@struct.dataclass
class LayerCBPState:
    utility: Array
    mean_activation: Array
    age: Array
    fractional_count: Array


@struct.dataclass
class CBPState:
    layers: Mapping[str, LayerCBPState]
    replacements: Array


def _layer_widths(params) -> dict[str, int]:
    p = unfreeze(params)
    widths = {
        "obs_embed": int(p["obs_embed"]["bias"].shape[0]),
        "rnn_input": int(p["rnn_input"]["bias"].shape[0]),
        "gru": int(p["gru"]["b_r"].shape[0]),
        "actor_hidden_1": int(p["actor_hidden_1"]["bias"].shape[0]),
        "actor_hidden_2": int(p["actor_hidden_2"]["bias"].shape[0]),
        "critic_hidden_1": int(p["critic_hidden_1"]["bias"].shape[0]),
        "critic_hidden_2": int(p["critic_hidden_2"]["bias"].shape[0]),
    }
    if "rule_encoder" in p:
        widths["rule_encoder"] = int(p["rule_encoder"]["bias"].shape[0])
    return widths


def init_cbp_state(params) -> CBPState:
    layers = {
        name: LayerCBPState(
            utility=jnp.zeros((width,), dtype=jnp.float32),
            mean_activation=jnp.zeros((width,), dtype=jnp.float32),
            age=jnp.zeros((width,), dtype=jnp.int32),
            fractional_count=jnp.asarray(0.0, dtype=jnp.float32),
        )
        for name, width in _layer_widths(params).items()
    }
    return CBPState(layers=freeze(layers), replacements=jnp.asarray(0, dtype=jnp.int32))


def _mean_abs_rows(kernel: Array) -> Array:
    return jnp.mean(jnp.abs(kernel), axis=1)


def outgoing_weight_magnitude(params, layer_name: str) -> Array:
    """Sum mean absolute outgoing weights over all destination matrices."""

    p = unfreeze(params)
    if layer_name == "rule_encoder":
        width = p["rule_encoder"]["bias"].shape[0]
        start = p["obs_embed"]["kernel"].shape[0] - width
        return _mean_abs_rows(p["obs_embed"]["kernel"][start:, :])
    if layer_name == "obs_embed":
        return _mean_abs_rows(p["rnn_input"]["kernel"])
    if layer_name == "rnn_input":
        return sum(
            _mean_abs_rows(p["gru"][name])
            for name in ("x_r", "x_z", "x_n")
        )
    if layer_name == "gru":
        recurrent = sum(
            _mean_abs_rows(p["gru"][name]) for name in ("h_r", "h_z", "h_n")
        )
        return (
            recurrent
            + _mean_abs_rows(p["actor_hidden_1"]["kernel"])
            + _mean_abs_rows(p["critic_hidden_1"]["kernel"])
        )
    if layer_name == "actor_hidden_1":
        return _mean_abs_rows(p["actor_hidden_2"]["kernel"])
    if layer_name == "actor_hidden_2":
        return _mean_abs_rows(p["actor_out"]["kernel"])
    if layer_name == "critic_hidden_1":
        return _mean_abs_rows(p["critic_hidden_2"]["kernel"])
    if layer_name == "critic_hidden_2":
        return _mean_abs_rows(p["critic_out"]["kernel"])
    raise KeyError(layer_name)


def _select_replacements(
    state: LayerCBPState,
    activation_mean: Array,
    activation_mad: Array,
    outgoing_magnitude: Array,
    config: CBPConfig,
) -> tuple[LayerCBPState, Array, Array]:
    age = state.age + 1
    mean_activation = (
        config.utility_decay * state.mean_activation
        + (1.0 - config.utility_decay) * activation_mean
    )
    instantaneous = activation_mad * outgoing_magnitude
    utility = (
        config.utility_decay * state.utility
        + (1.0 - config.utility_decay) * instantaneous
    )
    eligible = age >= config.maturity_steps
    fractional = state.fractional_count + config.replacement_rate * jnp.sum(eligible)
    count = jnp.minimum(
        jnp.floor(fractional).astype(jnp.int32),
        jnp.sum(eligible).astype(jnp.int32),
    )
    fractional = fractional - count.astype(jnp.float32)

    # A dynamic top-k has a dynamic output shape and cannot be JIT compiled.
    # Ranking all units and masking ranks below ``count`` is shape-stable.
    scores = jnp.where(eligible, utility, jnp.inf)
    order = jnp.argsort(scores)
    ranks = jnp.empty_like(order).at[order].set(jnp.arange(order.shape[0]))
    replace = eligible & (ranks < count)

    # Bias correction is used only for the downstream-bias compensation.
    correction = 1.0 - config.utility_decay ** jnp.maximum(age, 1)
    corrected_mean = mean_activation / jnp.maximum(correction, 1e-8)
    new_state = LayerCBPState(
        utility=jnp.where(replace, 0.0, utility),
        mean_activation=jnp.where(replace, 0.0, mean_activation),
        age=jnp.where(replace, 0, age),
        fractional_count=fractional,
    )
    return new_state, replace, corrected_mean


def _orthogonal(
    key: Array, shape: tuple[int, ...], scale: float, active: Array
) -> Array:
    return jax.lax.cond(
        jnp.any(active),
        lambda _: jax.nn.initializers.orthogonal(scale)(key, shape, jnp.float32),
        lambda _: jnp.zeros(shape, dtype=jnp.float32),
        operand=None,
    )


def _lecun_normal(key: Array, shape: tuple[int, ...], active: Array) -> Array:
    return jax.lax.cond(
        jnp.any(active),
        lambda _: jax.nn.initializers.lecun_normal()(key, shape, jnp.float32),
        lambda _: jnp.zeros(shape, dtype=jnp.float32),
        operand=None,
    )


def _kaiming_uniform(key: Array, shape: tuple[int, ...], active: Array) -> Array:
    """Repository-compatible ReLU Kaiming uniform for [fan_in, fan_out]."""

    bound = jnp.sqrt(jnp.asarray(6.0 / shape[0], dtype=jnp.float32))
    return jax.lax.cond(
        jnp.any(active),
        lambda _: jax.random.uniform(
            key, shape, jnp.float32, minval=-bound, maxval=bound
        ),
        lambda _: jnp.zeros(shape, dtype=jnp.float32),
        operand=None,
    )


def _mark(mask_tree, module: str, name: str, mask: Array) -> None:
    mask_tree[module][name] = mask_tree[module][name] | mask


def _replace_dense_hidden(
    p,
    masks,
    *,
    module: str,
    outgoing_module: str,
    replace: Array,
    mean: Array,
    key: Array,
) -> None:
    incoming = p[module]["kernel"]
    fresh = _kaiming_uniform(key, incoming.shape, replace)
    in_mask = jnp.broadcast_to(replace[None, :], incoming.shape)
    p[module]["kernel"] = jnp.where(in_mask, fresh, incoming)
    p[module]["bias"] = jnp.where(replace, 0.0, p[module]["bias"])
    _mark(masks, module, "kernel", in_mask)
    _mark(masks, module, "bias", replace)

    outgoing = p[outgoing_module]["kernel"]
    out_mask = jnp.broadcast_to(replace[:, None], outgoing.shape)
    compensation = jnp.sum(jnp.where(out_mask, outgoing * mean[:, None], 0.0), axis=0)
    p[outgoing_module]["bias"] = p[outgoing_module]["bias"] + compensation
    p[outgoing_module]["kernel"] = jnp.where(out_mask, 0.0, outgoing)
    _mark(masks, outgoing_module, "kernel", out_mask)
    _mark(masks, outgoing_module, "bias", jnp.any(out_mask, axis=0))


def _replace_rule_encoder(p, masks, replace: Array, mean: Array, key: Array) -> None:
    incoming = p["rule_encoder"]["kernel"]
    fresh = _kaiming_uniform(key, incoming.shape, replace)
    in_mask = jnp.broadcast_to(replace[None, :], incoming.shape)
    p["rule_encoder"]["kernel"] = jnp.where(in_mask, fresh, incoming)
    p["rule_encoder"]["bias"] = jnp.where(replace, 0.0, p["rule_encoder"]["bias"])
    _mark(masks, "rule_encoder", "kernel", in_mask)
    _mark(masks, "rule_encoder", "bias", replace)

    outgoing = p["obs_embed"]["kernel"]
    width = replace.shape[0]
    start = outgoing.shape[0] - width
    # Build explicitly to avoid shape-polymorphic scatter behavior.
    local_mask = jnp.broadcast_to(replace[:, None], (width, outgoing.shape[1]))
    out_mask = jnp.zeros_like(outgoing, dtype=jnp.bool_).at[start:, :].set(local_mask)
    local_outgoing = outgoing[start:, :]
    compensation = jnp.sum(
        jnp.where(local_mask, local_outgoing * mean[:, None], 0.0), axis=0
    )
    p["obs_embed"]["bias"] = p["obs_embed"]["bias"] + compensation
    p["obs_embed"]["kernel"] = jnp.where(out_mask, 0.0, outgoing)
    _mark(masks, "obs_embed", "kernel", out_mask)
    _mark(masks, "obs_embed", "bias", jnp.any(local_mask, axis=0))


def _replace_rnn_input(
    p, masks, replace: Array, mean: Array, keys: list[Array]
) -> None:
    """Replace linear pre-GRU features and disconnect their GRU outputs."""

    incoming = p["rnn_input"]["kernel"]
    fresh = _lecun_normal(keys[0], incoming.shape, replace)
    in_mask = jnp.broadcast_to(replace[None, :], incoming.shape)
    p["rnn_input"]["kernel"] = jnp.where(in_mask, fresh, incoming)
    p["rnn_input"]["bias"] = jnp.where(
        replace, 0.0, p["rnn_input"]["bias"]
    )
    _mark(masks, "rnn_input", "kernel", in_mask)
    _mark(masks, "rnn_input", "bias", replace)

    for gate in ("r", "z", "n"):
        name = f"x_{gate}"
        outgoing = p["gru"][name]
        out_mask = jnp.broadcast_to(replace[:, None], outgoing.shape)
        compensation = jnp.sum(
            jnp.where(out_mask, outgoing * mean[:, None], 0.0), axis=0
        )
        p["gru"][f"b_{gate}"] = p["gru"][f"b_{gate}"] + compensation
        p["gru"][name] = jnp.where(out_mask, 0.0, outgoing)
        _mark(masks, "gru", name, out_mask)
        _mark(masks, "gru", f"b_{gate}", jnp.any(out_mask, axis=0))


def _replace_gru(p, masks, replace: Array, mean: Array, keys: list[Array]) -> None:
    # Incoming columns for all gates are freshly initialized.  Recurrent rows
    # are outgoing connections and are zeroed after column initialization.
    for idx, gate in enumerate(("r", "z", "n")):
        x_name, h_name = f"x_{gate}", f"h_{gate}"
        compensation_bias = f"b_{gate}" if gate != "n" else "b_hn"
        x_kernel = p["gru"][x_name]
        h_kernel = p["gru"][h_name]
        x_fresh = _lecun_normal(keys[idx * 2], x_kernel.shape, replace)
        h_fresh = _orthogonal(
            keys[idx * 2 + 1], h_kernel.shape, 1.0, replace
        )
        x_in_mask = jnp.broadcast_to(replace[None, :], x_kernel.shape)
        h_in_mask = jnp.broadcast_to(replace[None, :], h_kernel.shape)
        old_h = h_kernel
        h_after_in = jnp.where(h_in_mask, h_fresh, h_kernel)

        h_out_mask = jnp.broadcast_to(replace[:, None], h_kernel.shape)
        compensation = jnp.sum(
            jnp.where(h_out_mask, old_h * mean[:, None], 0.0), axis=0
        )
        p["gru"][x_name] = jnp.where(x_in_mask, x_fresh, x_kernel)
        p["gru"][h_name] = jnp.where(h_out_mask, 0.0, h_after_in)
        p["gru"][compensation_bias] = (
            jnp.where(replace, 0.0, p["gru"][compensation_bias]) + compensation
        )
        _mark(masks, "gru", x_name, x_in_mask)
        _mark(masks, "gru", h_name, h_in_mask | h_out_mask)
        _mark(
            masks,
            "gru",
            compensation_bias,
            replace | jnp.any(h_out_mask, axis=0),
        )
        if gate == "n":
            p["gru"]["b_n"] = jnp.where(replace, 0.0, p["gru"]["b_n"])
            _mark(masks, "gru", "b_n", replace)

    for dest in ("actor_hidden_1", "critic_hidden_1"):
        outgoing = p[dest]["kernel"]
        out_mask = jnp.broadcast_to(replace[:, None], outgoing.shape)
        compensation = jnp.sum(
            jnp.where(out_mask, outgoing * mean[:, None], 0.0), axis=0
        )
        p[dest]["bias"] = p[dest]["bias"] + compensation
        p[dest]["kernel"] = jnp.where(out_mask, 0.0, outgoing)
        _mark(masks, dest, "kernel", out_mask)
        _mark(masks, dest, "bias", jnp.any(out_mask, axis=0))


def _zero_moments(moment_tree, reset_masks):
    return jax.tree.map(
        lambda value, mask: jnp.where(mask, jnp.zeros_like(value), value),
        moment_tree,
        freeze(reset_masks),
    )


def _enforce_replacement_invariants(
    p, unit_replacements: Mapping[str, Array]
) -> None:
    """Restore CBP invariants after simultaneous adjacent replacements.

    A destination layer's incoming-column regeneration can otherwise overwrite
    the zero at the intersection with a source unit replaced in the same
    gradient step.  Bias compensation was already computed from the old
    network and can likewise make a simultaneously replaced destination's
    incoming bias nonzero.  These regenerated-unit outgoing coordinates and
    incoming biases contributed nothing to the old network, so enforce both
    required zeros again after all layers have been processed.
    """

    if "rule_encoder" in unit_replacements:
        replace = unit_replacements["rule_encoder"]
        p["rule_encoder"]["bias"] = jnp.where(
            replace, 0.0, p["rule_encoder"]["bias"]
        )
    for module in (
        "obs_embed",
        "rnn_input",
        "actor_hidden_1",
        "actor_hidden_2",
        "critic_hidden_1",
        "critic_hidden_2",
    ):
        p[module]["bias"] = jnp.where(
            unit_replacements[module], 0.0, p[module]["bias"]
        )
    for name in ("b_r", "b_z", "b_n", "b_hn"):
        p["gru"][name] = jnp.where(
            unit_replacements["gru"], 0.0, p["gru"][name]
        )

    if "rule_encoder" in unit_replacements:
        replace = unit_replacements["rule_encoder"]
        width = replace.shape[0]
        start = p["obs_embed"]["kernel"].shape[0] - width
        local = p["obs_embed"]["kernel"][start:, :]
        p["obs_embed"]["kernel"] = p["obs_embed"]["kernel"].at[start:, :].set(
            jnp.where(replace[:, None], 0.0, local)
        )

    for source, destination in (
        ("obs_embed", "rnn_input"),
        ("actor_hidden_1", "actor_hidden_2"),
        ("actor_hidden_2", "actor_out"),
        ("critic_hidden_1", "critic_hidden_2"),
        ("critic_hidden_2", "critic_out"),
    ):
        replace = unit_replacements[source]
        p[destination]["kernel"] = jnp.where(
            replace[:, None], 0.0, p[destination]["kernel"]
        )

    replace = unit_replacements["rnn_input"]
    for gate in ("r", "z", "n"):
        name = f"x_{gate}"
        p["gru"][name] = jnp.where(
            replace[:, None], 0.0, p["gru"][name]
        )

    replace = unit_replacements["gru"]
    for gate in ("r", "z", "n"):
        name = f"h_{gate}"
        p["gru"][name] = jnp.where(
            replace[:, None], 0.0, p["gru"][name]
        )
    for destination in ("actor_hidden_1", "critic_hidden_1"):
        p[destination]["kernel"] = jnp.where(
            replace[:, None], 0.0, p[destination]["kernel"]
        )


def apply_continual_backprop(
    params,
    adam_state,
    state: CBPState,
    activation_stats: Mapping[str, Mapping[str, Array]],
    key: Array,
    config: CBPConfig,
):
    """Update utilities and replace mature low-utility units after one Adam step.

    ``adam_state`` must expose PyTrees named ``mu``, ``nu`` and optionally
    elementwise ``count``. Associated first/second moments and counters are
    zeroed at every regenerated incoming/outgoing parameter.
    """

    if not hasattr(adam_state, "mu") or not hasattr(adam_state, "nu"):
        raise TypeError("CBP requires an Optax ScaleByAdamState with mu/nu moments")

    p = unfreeze(params)
    masks = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bool_), p)
    layer_names = tuple(state.layers.keys())
    layer_keys = jax.random.split(key, len(layer_names) + 1)
    new_layers: dict[str, LayerCBPState] = {}
    total_replaced = jnp.asarray(0, dtype=jnp.int32)
    # Utilities use a coherent post-gradient, pre-replacement view of every
    # outgoing matrix. Earlier replacements must not change later rankings.
    outgoing_magnitudes = {
        name: outgoing_weight_magnitude(params, name) for name in layer_names
    }
    unit_replacements: dict[str, Array] = {}

    for layer_idx, name in enumerate(layer_names):
        stats = activation_stats[name]
        next_state, replace, corrected_mean = _select_replacements(
            state.layers[name],
            stats["mean"],
            stats["mad"],
            outgoing_magnitudes[name],
            config,
        )
        new_layers[name] = next_state
        unit_replacements[name] = replace
        total_replaced = total_replaced + jnp.sum(replace.astype(jnp.int32))
        keys = list(jax.random.split(layer_keys[layer_idx], 8))

        if name == "rule_encoder":
            _replace_rule_encoder(p, masks, replace, corrected_mean, keys[0])
        elif name == "obs_embed":
            _replace_dense_hidden(
                p,
                masks,
                module="obs_embed",
                outgoing_module="rnn_input",
                replace=replace,
                mean=corrected_mean,
                key=keys[0],
            )
        elif name == "rnn_input":
            _replace_rnn_input(p, masks, replace, corrected_mean, keys)
        elif name == "gru":
            _replace_gru(p, masks, replace, corrected_mean, keys)
        elif name == "actor_hidden_1":
            _replace_dense_hidden(
                p,
                masks,
                module="actor_hidden_1",
                outgoing_module="actor_hidden_2",
                replace=replace,
                mean=corrected_mean,
                key=keys[0],
            )
        elif name == "actor_hidden_2":
            _replace_dense_hidden(
                p,
                masks,
                module="actor_hidden_2",
                outgoing_module="actor_out",
                replace=replace,
                mean=corrected_mean,
                key=keys[0],
            )
        elif name == "critic_hidden_1":
            _replace_dense_hidden(
                p,
                masks,
                module="critic_hidden_1",
                outgoing_module="critic_hidden_2",
                replace=replace,
                mean=corrected_mean,
                key=keys[0],
            )
        elif name == "critic_hidden_2":
            _replace_dense_hidden(
                p,
                masks,
                module="critic_hidden_2",
                outgoing_module="critic_out",
                replace=replace,
                mean=corrected_mean,
                key=keys[0],
            )
        else:  # pragma: no cover - guarded by init_cbp_state
            raise KeyError(name)

    _enforce_replacement_invariants(p, unit_replacements)

    reset_masks = freeze(masks)
    replacements = {
        "mu": _zero_moments(adam_state.mu, reset_masks),
        "nu": _zero_moments(adam_state.nu, reset_masks),
    }
    if hasattr(adam_state, "count"):
        replacements["count"] = _zero_moments(adam_state.count, reset_masks)
    if hasattr(adam_state, "replace"):
        new_adam_state = adam_state.replace(**replacements)
    else:  # namedtuple compatibility
        new_adam_state = adam_state._replace(**replacements)
    new_state = CBPState(
        layers=flax.core.freeze(new_layers),
        replacements=state.replacements + total_replaced,
    )
    return (
        freeze(p),
        new_adam_state,
        new_state,
        total_replaced,
        freeze(unit_replacements),
    )
