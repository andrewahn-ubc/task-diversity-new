from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from td2.core_aux_data import (
    AUXILIARY_WIDTHS,
    CORE_N,
    STORAGE_N,
    build_core_aux_partition,
    validate_core_aux_partition,
)
from td2.environment import (
    CoreAuxPoolSpec,
    core_aux_slot_assignments,
    exhaustive_core_evaluation_indices,
)


def test_core_aux_partition_is_deterministic_disjoint_and_nested() -> None:
    partition = build_core_aux_partition(partition_seed=17)
    assert partition == build_core_aux_partition(partition_seed=17)
    assert partition != build_core_aux_partition(partition_seed=18)
    validate_core_aux_partition(partition)

    core_rules = np.asarray(partition["core_rule_local_indices_by_depth"])
    auxiliary_rules = np.asarray(
        partition["auxiliary_rule_local_indices_by_depth"]
    )
    core_layouts = np.asarray(partition["core_layout_indices"])
    auxiliary_layouts = np.asarray(partition["auxiliary_layout_indices"])
    assert core_rules.shape == (6, CORE_N)
    assert auxiliary_rules.shape == (6, 256)
    np.testing.assert_array_equal(core_rules, np.repeat(core_rules[:1], 6, axis=0))
    np.testing.assert_array_equal(
        auxiliary_rules, np.repeat(auxiliary_rules[:1], 6, axis=0)
    )
    for depth in range(6):
        assert set(core_rules[depth]).isdisjoint(auxiliary_rules[depth])
        assert set(np.concatenate((core_rules[depth], auxiliary_rules[depth]))) == set(
            range(STORAGE_N)
        )
    assert set(core_layouts).isdisjoint(auxiliary_layouts)

    for smaller, larger in zip(AUXILIARY_WIDTHS, AUXILIARY_WIDTHS[1:]):
        np.testing.assert_array_equal(
            auxiliary_rules[:, :smaller], auxiliary_rules[:, :larger][:, :smaller]
        )
        np.testing.assert_array_equal(
            auxiliary_layouts[:smaller], auxiliary_layouts[:larger][:smaller]
        )


def test_core_aux_slot_assignments_are_exactly_half_and_fixed() -> None:
    assignments = np.asarray(core_aux_slot_assignments(2048))
    assert assignments.dtype == np.bool_
    assert int(assignments.sum()) == 1024
    np.testing.assert_array_equal(assignments[:8], [False, True] * 4)
    np.testing.assert_array_equal(assignments, core_aux_slot_assignments(2048))

    with pytest.raises(ValueError, match="positive even"):
        core_aux_slot_assignments(2047)


def _test_pool() -> CoreAuxPoolSpec:
    core_rules = np.stack(
        [np.arange(CORE_N, dtype=np.int32) + depth * CORE_N for depth in range(6)]
    )
    auxiliary_rules = np.stack(
        [
            (np.arange(256, dtype=np.int32) + CORE_N + depth * 7) % STORAGE_N
            for depth in range(6)
        ]
    )
    return CoreAuxPoolSpec(
        packed_rulesets=jnp.zeros((6 * STORAGE_N, 1), dtype=jnp.uint32),
        obstacle_masks=jnp.zeros((STORAGE_N, 6, 6), dtype=jnp.bool_),
        reachable_masks=jnp.ones((STORAGE_N, 6, 6), dtype=jnp.bool_),
        core_rule_local_indices_by_depth=jnp.asarray(core_rules),
        auxiliary_rule_local_indices_by_depth=jnp.asarray(auxiliary_rules),
        core_layout_indices=jnp.arange(100, 100 + CORE_N, dtype=jnp.int32),
        auxiliary_layout_indices=jnp.arange(256, dtype=jnp.int32),
        auxiliary_n=32,
        max_n=STORAGE_N,
        num_depths=6,
    )


def test_exhaustive_core_indices_cover_every_pair_at_every_depth_once() -> None:
    pool = _test_pool()
    tasks, layouts, depths = exhaustive_core_evaluation_indices(pool)
    tasks_np = np.asarray(tasks)
    layouts_np = np.asarray(layouts)
    depths_np = np.asarray(depths)

    assert tasks_np.shape == (6 * CORE_N * CORE_N,)
    assert layouts_np.shape == tasks_np.shape
    assert depths_np.shape == tasks_np.shape
    for depth in range(1, 7):
        selected = depths_np == depth
        assert int(selected.sum()) == CORE_N * CORE_N
        local_tasks = tasks_np[selected] - (depth - 1) * STORAGE_N
        pairs = set(zip(local_tasks.tolist(), layouts_np[selected].tolist()))
        assert len(pairs) == CORE_N * CORE_N
        assert set(local_tasks) == set(
            np.asarray(pool.core_rule_local_indices_by_depth[depth - 1]).tolist()
        )
        assert set(layouts_np[selected]) == set(
            np.asarray(pool.core_layout_indices).tolist()
        )
