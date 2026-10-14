"""Resumable main and held-out adaptation training entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from td2.checkpoint import (
    CheckpointError,
    atomic_marker,
    checkpoint_metadata,
    load_checkpoint_with_metadata,
    save_checkpoint,
)
from td2.config import RunSchedule, TD2Config, load_config
from td2.environment import VectorBanyan, load_pool_spec
from td2.ppo import (
    RunnerState,
    balanced_evaluation_indices,
    build_components,
    initialize_runner,
    make_evaluation_fn,
    make_update_fn,
)


METRIC_FIELDS = (
    "method",
    "diversity_n",
    "diversity_n2",
    "seed",
    "mode",
    "distribution",
    "source_distribution",
    "update_in_phase",
    "global_update",
    "environment_steps",
    "actual_environment_steps",
    "nominal_environment_steps",
    "adaptation_steps",
    "success_rate",
    "episodic_return",
    "mean_transition_reward",
    "episode_count",
    "loss",
    "actor_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "learning_rate",
    "gradient_norm",
    "cbp_replacements",
    "gru_units_replaced",
    "eval_episodes",
    "eval_unfinished",
    "depth_1_success",
    "depth_2_success",
    "depth_3_success",
    "depth_4_success",
    "depth_5_success",
    "depth_6_success",
)


_STOP_REQUESTED = False
PROTOCOL_VERSION = 6


def _request_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _stable_fold(key: jax.Array, *values: int) -> jax.Array:
    for value in values:
        key = jax.random.fold_in(key, int(value) & 0xFFFFFFFF)
    return key


def _as_float(value: Any) -> float:
    array = np.asarray(jax.device_get(value))
    return float(array.reshape(()))


def _as_int(value: Any) -> int:
    array = np.asarray(jax.device_get(value))
    return int(array.reshape(()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_identity(config: TD2Config, data_root: Path) -> dict[str, Any]:
    config_digest = hashlib.sha256()
    for source in config.source_files:
        config_digest.update(source.name.encode("utf-8"))
        config_digest.update(b"\0")
        config_digest.update(source.read_bytes())
        config_digest.update(b"\0")

    manifest_path = data_root / "manifest.json"
    sidecar_path = data_root / "manifest.sha256"
    manifest_digest = _sha256_file(manifest_path)
    expected_digest = sidecar_path.read_text(encoding="ascii").split()[0]
    if manifest_digest != expected_digest:
        raise CheckpointError(
            f"data manifest checksum mismatch: {manifest_digest} != {expected_digest}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": config_digest.hexdigest(),
        "data_manifest_sha256": manifest_digest,
        "banyan_commit": manifest.get("banyan_commit"),
    }


def _append_metric(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _stream_root(
    output_root: Path,
    *,
    mode: str,
    algorithm: str,
    diversity_n2: int,
    seed: int,
    stage: int | None,
) -> Path:
    if mode == "main":
        return (
            output_root
            / algorithm
            / f"diversity_{diversity_n2}"
            / f"seed_{seed}"
        )
    if stage is None:
        raise ValueError("adaptation mode requires --stage")
    return (
        output_root
        / "adapt"
        / algorithm
        / f"diversity_{diversity_n2}"
        / f"seed_{seed}"
        / f"stage_{stage:02d}"
    )


def _source_stage_root(
    output_root: Path,
    *,
    algorithm: str,
    diversity_n2: int,
    seed: int,
    stage: int,
) -> Path:
    return (
        output_root
        / algorithm
        / f"diversity_{diversity_n2}"
        / f"seed_{seed}"
        / "stages"
        / f"stage_{stage:02d}"
    )


def _environment(
    config: TD2Config,
    data_root: Path,
    *,
    active_n: int,
    distribution_index: int,
    num_envs: int,
    heldout: bool = False,
) -> tuple[VectorBanyan, Any]:
    pool = load_pool_spec(
        data_root,
        distribution_index=distribution_index,
        active_n=active_n,
        heldout=heldout,
    )
    env = VectorBanyan(
        pool,
        num_envs=num_envs,
        grid_size=6,
        max_steps=config.ppo.max_steps,
        include_rules_in_obs=config.ruleset.include_rules_in_obs,
        distractor_combine_penalty=config.ruleset.distractor_combine_penalty,
    )
    return env, pool


def _runner_template(
    config: TD2Config,
    data_root: Path,
    *,
    algorithm: str,
    diversity_n: int,
    distribution: int,
    seed: int,
) -> tuple[RunnerState, VectorBanyan, Any, Any]:
    env, pool = _environment(
        config,
        data_root,
        active_n=diversity_n,
        distribution_index=distribution - 1,
        num_envs=config.ppo.num_envs,
    )
    components = build_components(config, env, algorithm=algorithm)
    key = _stable_fold(jax.random.PRNGKey(seed), 0x544432, distribution)
    runner = initialize_runner(key, config, components, env)
    return runner, env, pool, components


def _main_axis_steps(
    schedule: RunSchedule, *, distribution: int, update_in_phase: int
) -> tuple[int, int, int]:
    within = update_in_phase * schedule.environment_steps_per_update
    nominal = (distribution - 1) * schedule.requested_timesteps + within
    actual = (
        (distribution - 1) * schedule.consumed_timesteps + within
    )
    boundary = distribution * schedule.requested_timesteps
    return nominal, actual, boundary


def _metric_base(
    *,
    algorithm: str,
    diversity_n: int,
    seed: int,
    mode: str,
    distribution: int,
    update_in_phase: int,
    global_update: int,
    schedule: RunSchedule,
    source_distribution: int | str = "",
) -> dict[str, Any]:
    diversity_n2 = diversity_n * diversity_n
    if mode.startswith("adapt"):
        adaptation_steps = update_in_phase * schedule.environment_steps_per_update
        environment_steps = int(source_distribution) * 100_000_000
        actual_environment_steps = global_update * schedule.environment_steps_per_update
        nominal_environment_steps = environment_steps
    else:
        environment_steps, actual_environment_steps, _ = _main_axis_steps(
            schedule,
            distribution=distribution,
            update_in_phase=update_in_phase,
        )
        nominal_environment_steps = environment_steps
        adaptation_steps = ""
    return {
        "method": algorithm,
        "diversity_n": diversity_n,
        "diversity_n2": diversity_n2,
        "seed": seed,
        "mode": mode,
        "distribution": distribution,
        "source_distribution": source_distribution,
        "update_in_phase": update_in_phase,
        "global_update": global_update,
        "environment_steps": environment_steps,
        "actual_environment_steps": actual_environment_steps,
        "nominal_environment_steps": nominal_environment_steps,
        "adaptation_steps": adaptation_steps,
    }


def _write_training_metric(
    path: Path,
    metrics: Mapping[str, Any],
    *,
    algorithm: str,
    diversity_n: int,
    seed: int,
    mode: str,
    distribution: int,
    runner: RunnerState,
    schedule: RunSchedule,
    source_distribution: int | str = "",
) -> None:
    update_in_phase = _as_int(runner.update_in_phase)
    global_update = _as_int(runner.global_update)
    row = _metric_base(
        algorithm=algorithm,
        diversity_n=diversity_n,
        seed=seed,
        mode=mode,
        distribution=distribution,
        update_in_phase=update_in_phase,
        global_update=global_update,
        schedule=schedule,
        source_distribution=source_distribution,
    )
    scalar_fields = (
        "success_rate",
        "episodic_return",
        "mean_transition_reward",
        "episode_count",
        "loss",
        "actor_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "learning_rate",
        "gradient_norm",
        "cbp_replacements",
        "gru_units_replaced",
    )
    for field in scalar_fields:
        if field in metrics:
            row[field] = _as_float(metrics[field])
    _append_metric(path, row)


def _write_evaluation_metric(
    path: Path,
    evaluation: Mapping[str, Any],
    *,
    algorithm: str,
    diversity_n: int,
    seed: int,
    mode: str,
    distribution: int,
    runner: RunnerState,
    schedule: RunSchedule,
    source_distribution: int | str = "",
) -> None:
    row = _metric_base(
        algorithm=algorithm,
        diversity_n=diversity_n,
        seed=seed,
        mode=mode,
        distribution=distribution,
        update_in_phase=_as_int(runner.update_in_phase),
        global_update=_as_int(runner.global_update),
        schedule=schedule,
        source_distribution=source_distribution,
    )
    row.update(
        success_rate=_as_float(evaluation["success_rate"]),
        episodic_return=_as_float(evaluation["episodic_return"]),
        eval_episodes=_as_int(evaluation["episodes"]),
        eval_unfinished=_as_int(evaluation["unfinished"]),
    )
    depth = np.asarray(jax.device_get(evaluation["depth_success_rate"]))
    for index in range(6):
        row[f"depth_{index + 1}_success"] = float(depth[index])
    _append_metric(path, row)


def _evaluation_bundle(
    config: TD2Config,
    components,
    data_root: Path,
    *,
    diversity_n: int,
    distribution: int,
    heldout: bool,
):
    if heldout:
        eval_episodes = max(
            config.experiment.adaptation.heldout_tasks,
            min(1000, config.ppo.num_envs),
        )
    else:
        eval_episodes = max(6, min(1536, config.ppo.num_envs))
        eval_episodes = int(math.ceil(eval_episodes / 6) * 6)
    eval_env, eval_pool = _environment(
        config,
        data_root,
        active_n=(
            config.experiment.adaptation.heldout_tasks if heldout else diversity_n
        ),
        distribution_index=(
            config.experiment.num_distributions if heldout else distribution - 1
        ),
        num_envs=eval_episodes,
        heldout=heldout,
    )
    task_indices, layout_indices, depths = balanced_evaluation_indices(
        eval_pool, num_episodes=eval_episodes
    )
    return make_evaluation_fn(
        config,
        components,
        eval_env,
        task_indices=task_indices,
        layout_indices=layout_indices,
        depths=depths,
    )


def _eval_key(
    *, seed: int, diversity_n2: int, distribution: int, update: int, heldout: bool
) -> jax.Array:
    return _stable_fold(
        jax.random.PRNGKey(seed),
        0x4556414C,
        diversity_n2,
        distribution,
        update,
        int(heldout),
    )


def _checkpoint_meta(
    *, mode: str, algorithm: str, diversity_n: int, seed: int, distribution: int,
    runner: RunnerState, protocol_identity: Mapping[str, Any],
    source_distribution: int | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "algorithm": algorithm,
        "diversity_n": diversity_n,
        "diversity_n2": diversity_n * diversity_n,
        "seed": seed,
        "distribution": distribution,
        "source_distribution": source_distribution,
        "update_in_phase": _as_int(runner.update_in_phase),
        "global_update": _as_int(runner.global_update),
        "optimizer_steps": _as_int(runner.learner.optimizer_steps),
        "schedule_step": _as_int(runner.learner.schedule_step),
        "protocol_identity": dict(protocol_identity),
    }


def _save_resume(
    stream_root: Path,
    runner: RunnerState,
    *,
    mode: str,
    algorithm: str,
    diversity_n: int,
    seed: int,
    distribution: int,
    protocol_identity: Mapping[str, Any],
    source_distribution: int | None = None,
) -> None:
    save_checkpoint(
        stream_root / "resume",
        runner,
        metadata=_checkpoint_meta(
            mode=mode,
            algorithm=algorithm,
            diversity_n=diversity_n,
            seed=seed,
            distribution=distribution,
            runner=runner,
            protocol_identity=protocol_identity,
            source_distribution=source_distribution,
        ),
    )


def _validate_identity(
    metadata: Mapping[str, Any],
    *,
    mode: str,
    algorithm: str,
    diversity_n: int,
    seed: int,
    protocol_identity: Mapping[str, Any],
) -> None:
    expected = {
        "mode": mode,
        "algorithm": algorithm,
        "diversity_n": diversity_n,
        "seed": seed,
        "protocol_identity": dict(protocol_identity),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise CheckpointError(f"checkpoint identity mismatch: {mismatches}")


def _validate_checkpoint_counters(
    metadata: Mapping[str, Any], runner: RunnerState
) -> None:
    expected = {
        "update_in_phase": _as_int(runner.update_in_phase),
        "global_update": _as_int(runner.global_update),
        "optimizer_steps": _as_int(runner.learner.optimizer_steps),
        "schedule_step": _as_int(runner.learner.schedule_step),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise CheckpointError(f"checkpoint counter mismatch: {mismatches}")


def _tree_values_equal(left: Any, right: Any) -> bool:
    """Compare checkpoint PyTrees independent of mapping serialization order."""

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        if np.issubdtype(left_array.dtype, np.inexact):
            if not np.array_equal(left_array, right_array, equal_nan=True):
                return False
        elif not np.array_equal(left_array, right_array):
            return False
    return True


def _stage_complete(
    stream_root: Path,
    runner: RunnerState,
    *,
    algorithm: str,
    diversity_n: int,
    seed: int,
    distribution: int,
    protocol_identity: Mapping[str, Any],
) -> None:
    stage_root = stream_root / "stages" / f"stage_{distribution:02d}"
    metadata = _checkpoint_meta(
        mode="main-stage",
        algorithm=algorithm,
        diversity_n=diversity_n,
        seed=seed,
        distribution=distribution,
        runner=runner,
        protocol_identity=protocol_identity,
    )
    checkpoint_root = stage_root / "checkpoint"
    if (checkpoint_root / "latest.json").exists() or (
        checkpoint_root / "previous.json"
    ).exists():
        existing, existing_metadata = load_checkpoint_with_metadata(
            checkpoint_root, runner
        )
        if existing_metadata != metadata or not _tree_values_equal(existing, runner):
            raise CheckpointError(
                f"existing immutable stage checkpoint differs at {checkpoint_root}"
            )
    else:
        save_checkpoint(
            checkpoint_root,
            runner,
            metadata=metadata,
            immutable=True,
        )
    atomic_marker(stage_root / "DONE", metadata)


def _run_main(args, config: TD2Config, diversity_n: int, stream_root: Path) -> bool:
    schedule = config.derived.training_per_distribution
    protocol_identity = _protocol_identity(config, args.data_root)
    resume_root = stream_root / "resume"
    if (resume_root / "latest.json").exists() or (resume_root / "previous.json").exists():
        pointer_metadata = checkpoint_metadata(resume_root)
        _validate_identity(
            pointer_metadata,
            mode="main",
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            protocol_identity=protocol_identity,
        )
        # Runner PyTree shapes are invariant across distributions because data
        # generation pads every ruleset round to one global width.  A stage-1
        # template therefore lets the checkpoint loader atomically choose the
        # latest valid pointer and its matching metadata.
        template, _, _, _ = _runner_template(
            config,
            args.data_root,
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            distribution=1,
            seed=args.seed,
        )
        runner, metadata = load_checkpoint_with_metadata(resume_root, template)
        _validate_identity(
            metadata,
            mode="main",
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            protocol_identity=protocol_identity,
        )
        _validate_checkpoint_counters(metadata, runner)
        distribution = int(metadata["distribution"])
        env, pool = _environment(
            config,
            args.data_root,
            active_n=diversity_n,
            distribution_index=distribution - 1,
            num_envs=config.ppo.num_envs,
        )
        components = build_components(config, env, algorithm=args.algorithm)
    else:
        distribution = 1
        runner, env, pool, components = _runner_template(
            config,
            args.data_root,
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            distribution=distribution,
            seed=args.seed,
        )

    metrics_path = stream_root / "metrics.csv"
    evaluation_interval = max(
        1, round(10_000_000 / schedule.environment_steps_per_update)
    )
    update_fn = make_update_fn(
        config,
        components,
        env,
        total_optimizer_steps=(
            config.derived.total_training_optimizer_steps_per_stream
        ),
        warmup_optimizer_steps=(
            config.derived.total_training_warmup_optimizer_steps_per_stream
        ),
    )
    evaluate = (
        None
        if config.profiling.disable_evaluation
        else _evaluation_bundle(
            config,
            components,
            args.data_root,
            diversity_n=diversity_n,
            distribution=distribution,
            heldout=False,
        )
    )
    started = time.monotonic()
    last_checkpoint = started
    recent_durations: list[float] = []

    while distribution <= config.experiment.num_distributions:
        update_in_phase = _as_int(runner.update_in_phase)
        if update_in_phase >= schedule.num_updates:
            if evaluate is not None:
                evaluation = evaluate(
                    runner.learner.params,
                    _eval_key(
                        seed=args.seed,
                        diversity_n2=diversity_n * diversity_n,
                        distribution=distribution,
                        update=update_in_phase,
                        heldout=False,
                    ),
                )
                jax.block_until_ready(evaluation)
                _write_evaluation_metric(
                    metrics_path,
                    evaluation,
                    algorithm=args.algorithm,
                    diversity_n=diversity_n,
                    seed=args.seed,
                    mode="eval",
                    distribution=distribution,
                    runner=runner,
                    schedule=schedule,
                )
            _save_resume(
                stream_root,
                runner,
                mode="main",
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=distribution,
                protocol_identity=protocol_identity,
            )
            _stage_complete(
                stream_root,
                runner,
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=distribution,
                protocol_identity=protocol_identity,
            )
            if distribution == config.experiment.num_distributions:
                metadata = _checkpoint_meta(
                    mode="main",
                    algorithm=args.algorithm,
                    diversity_n=diversity_n,
                    seed=args.seed,
                    distribution=distribution,
                    runner=runner,
                    protocol_identity=protocol_identity,
                )
                atomic_marker(stream_root / "DONE", metadata)
                return True

            distribution += 1
            env, pool = _environment(
                config,
                args.data_root,
                active_n=diversity_n,
                distribution_index=distribution - 1,
                num_envs=config.ppo.num_envs,
            )
            components = build_components(config, env, algorithm=args.algorithm)
            key_next, key_keep = jax.random.split(runner.rng)
            runner = initialize_runner(
                key_next,
                config,
                components,
                env,
                learner=runner.learner,
                global_update=runner.global_update,
            ).replace(rng=jax.random.fold_in(key_keep, distribution))
            _save_resume(
                stream_root,
                runner,
                mode="main",
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=distribution,
                protocol_identity=protocol_identity,
            )
            update_fn = make_update_fn(
                config,
                components,
                env,
                total_optimizer_steps=(
                    config.derived.total_training_optimizer_steps_per_stream
                ),
                warmup_optimizer_steps=(
                    config.derived.total_training_warmup_optimizer_steps_per_stream
                ),
            )
            evaluate = (
                None
                if config.profiling.disable_evaluation
                else _evaluation_bundle(
                    config,
                    components,
                    args.data_root,
                    diversity_n=diversity_n,
                    distribution=distribution,
                    heldout=False,
                )
            )
            last_checkpoint = time.monotonic()
            continue

        if update_in_phase == 0 and evaluate is not None:
            evaluation = evaluate(
                runner.learner.params,
                _eval_key(
                    seed=args.seed,
                    diversity_n2=diversity_n * diversity_n,
                    distribution=distribution,
                    update=0,
                    heldout=False,
                ),
            )
            jax.block_until_ready(evaluation)
            _write_evaluation_metric(
                metrics_path,
                evaluation,
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                mode="eval",
                distribution=distribution,
                runner=runner,
                schedule=schedule,
            )

        update_started = time.monotonic()
        runner, metrics = update_fn(runner)
        jax.block_until_ready(runner)
        duration = time.monotonic() - update_started
        recent_durations = (recent_durations + [duration])[-5:]
        _write_training_metric(
            metrics_path,
            metrics,
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            mode="train",
            distribution=distribution,
            runner=runner,
            schedule=schedule,
        )

        update_in_phase = _as_int(runner.update_in_phase)
        if evaluate is not None and update_in_phase % evaluation_interval == 0:
            evaluation = evaluate(
                runner.learner.params,
                _eval_key(
                    seed=args.seed,
                    diversity_n2=diversity_n * diversity_n,
                    distribution=distribution,
                    update=update_in_phase,
                    heldout=False,
                ),
            )
            jax.block_until_ready(evaluation)
            _write_evaluation_metric(
                metrics_path,
                evaluation,
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                mode="eval",
                distribution=distribution,
                runner=runner,
                schedule=schedule,
            )

        now = time.monotonic()
        if now - last_checkpoint >= args.checkpoint_interval_seconds:
            _save_resume(
                stream_root,
                runner,
                mode="main",
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=distribution,
                protocol_identity=protocol_identity,
            )
            last_checkpoint = now
        predicted = max(recent_durations, default=60.0) * 1.25 + 120.0
        if (
            _STOP_REQUESTED
            or now - started + predicted >= args.soft_time_limit_seconds
            or (args.max_updates is not None and _as_int(runner.global_update) >= args.max_updates)
        ):
            _save_resume(
                stream_root,
                runner,
                mode="main",
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=distribution,
                protocol_identity=protocol_identity,
            )
            return False
    return False


def _load_parent_learner(
    args,
    config: TD2Config,
    diversity_n: int,
    stage: int,
    protocol_identity: Mapping[str, Any],
):
    source_root = _source_stage_root(
        args.output_root,
        algorithm=args.algorithm,
        diversity_n2=diversity_n * diversity_n,
        seed=args.seed,
        stage=stage,
    )
    if not (source_root / "DONE").is_file():
        raise CheckpointError(f"source stage is not complete: {source_root}")
    template, _, _, _ = _runner_template(
        config,
        args.data_root,
        algorithm=args.algorithm,
        diversity_n=diversity_n,
        distribution=stage,
        seed=args.seed,
    )
    checkpoint_root = source_root / "checkpoint"
    pointer_metadata = checkpoint_metadata(checkpoint_root)
    _validate_identity(
        pointer_metadata,
        mode="main-stage",
        algorithm=args.algorithm,
        diversity_n=diversity_n,
        seed=args.seed,
        protocol_identity=protocol_identity,
    )
    parent, metadata = load_checkpoint_with_metadata(checkpoint_root, template)
    _validate_identity(
        metadata,
        mode="main-stage",
        algorithm=args.algorithm,
        diversity_n=diversity_n,
        seed=args.seed,
        protocol_identity=protocol_identity,
    )
    if int(metadata.get("distribution", -1)) != stage:
        raise CheckpointError("parent checkpoint stage mismatch")
    _validate_checkpoint_counters(metadata, parent)
    return parent.learner, parent.global_update


def _run_adapt(args, config: TD2Config, diversity_n: int, stream_root: Path) -> bool:
    assert args.stage is not None
    protocol_identity = _protocol_identity(config, args.data_root)
    schedule = config.derived.adaptation_per_probe
    if schedule is None:
        raise ValueError("adaptation is disabled in the configuration")
    heldout_env, _ = _environment(
        config,
        args.data_root,
        active_n=config.experiment.adaptation.heldout_tasks,
        distribution_index=config.experiment.num_distributions,
        num_envs=config.ppo.num_envs,
        heldout=True,
    )
    components = build_components(config, heldout_env, algorithm=args.algorithm)
    parent_learner, parent_global = _load_parent_learner(
        args, config, diversity_n, args.stage, protocol_identity
    )
    # Each heldout branch is a separate, fixed-budget adaptation run.  Give all
    # stages the same authors-style 100M-step schedule while leaving the main
    # continual trajectory's schedule uninterrupted across distributions.
    parent_learner = parent_learner.replace(
        schedule_step=jnp.asarray(0, dtype=jnp.int32)
    )
    key = _stable_fold(
        jax.random.PRNGKey(args.seed),
        0x41444150,
        diversity_n * diversity_n,
        args.stage,
    )
    template = initialize_runner(
        key,
        config,
        components,
        heldout_env,
        learner=parent_learner,
        global_update=parent_global,
    )
    resume_root = stream_root / "resume"
    if (resume_root / "latest.json").exists() or (resume_root / "previous.json").exists():
        pointer_metadata = checkpoint_metadata(resume_root)
        _validate_identity(
            pointer_metadata,
            mode="adapt",
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            protocol_identity=protocol_identity,
        )
        runner, metadata = load_checkpoint_with_metadata(resume_root, template)
        _validate_identity(
            metadata,
            mode="adapt",
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            protocol_identity=protocol_identity,
        )
        _validate_checkpoint_counters(metadata, runner)
        if int(metadata.get("source_distribution", -1)) != args.stage:
            raise CheckpointError("adaptation source stage mismatch")
    else:
        runner = template

    metrics_path = stream_root / "metrics.csv"
    evaluation_interval = max(
        1, round(10_000_000 / schedule.environment_steps_per_update)
    )
    evaluate = (
        None
        if config.profiling.disable_evaluation
        else _evaluation_bundle(
            config,
            components,
            args.data_root,
            diversity_n=diversity_n,
            distribution=args.stage,
            heldout=True,
        )
    )
    started = time.monotonic()
    last_checkpoint = started
    recent_durations: list[float] = []
    update_fn = make_update_fn(
        config,
        components,
        heldout_env,
        total_optimizer_steps=schedule.total_optimizer_steps,
        warmup_optimizer_steps=schedule.warmup_optimizer_steps,
    )

    while _as_int(runner.update_in_phase) < schedule.num_updates:
        update_in_phase = _as_int(runner.update_in_phase)
        if update_in_phase == 0 and evaluate is not None:
            evaluation = evaluate(
                runner.learner.params,
                _eval_key(
                    seed=args.seed,
                    diversity_n2=diversity_n * diversity_n,
                    distribution=args.stage,
                    update=0,
                    heldout=True,
                ),
            )
            jax.block_until_ready(evaluation)
            _write_evaluation_metric(
                metrics_path,
                evaluation,
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                mode="adapt_eval",
                distribution=args.stage,
                runner=runner,
                schedule=schedule,
                source_distribution=args.stage,
            )

        update_started = time.monotonic()
        runner, metrics = update_fn(runner)
        jax.block_until_ready(runner)
        duration = time.monotonic() - update_started
        recent_durations = (recent_durations + [duration])[-5:]
        _write_training_metric(
            metrics_path,
            metrics,
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            mode="adapt_train",
            distribution=args.stage,
            runner=runner,
            schedule=schedule,
            source_distribution=args.stage,
        )

        update_in_phase = _as_int(runner.update_in_phase)
        if evaluate is not None and update_in_phase % evaluation_interval == 0:
            evaluation = evaluate(
                runner.learner.params,
                _eval_key(
                    seed=args.seed,
                    diversity_n2=diversity_n * diversity_n,
                    distribution=args.stage,
                    update=update_in_phase,
                    heldout=True,
                ),
            )
            jax.block_until_ready(evaluation)
            _write_evaluation_metric(
                metrics_path,
                evaluation,
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                mode="adapt_eval",
                distribution=args.stage,
                runner=runner,
                schedule=schedule,
                source_distribution=args.stage,
            )

        now = time.monotonic()
        if now - last_checkpoint >= args.checkpoint_interval_seconds:
            _save_resume(
                stream_root,
                runner,
                mode="adapt",
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=args.stage,
                protocol_identity=protocol_identity,
                source_distribution=args.stage,
            )
            last_checkpoint = now
        predicted = max(recent_durations, default=60.0) * 1.25 + 120.0
        if (
            _STOP_REQUESTED
            or now - started + predicted >= args.soft_time_limit_seconds
            or (args.max_updates is not None and update_in_phase >= args.max_updates)
        ):
            _save_resume(
                stream_root,
                runner,
                mode="adapt",
                algorithm=args.algorithm,
                diversity_n=diversity_n,
                seed=args.seed,
                distribution=args.stage,
                protocol_identity=protocol_identity,
                source_distribution=args.stage,
            )
            return False

    update_in_phase = _as_int(runner.update_in_phase)
    if evaluate is not None:
        evaluation = evaluate(
            runner.learner.params,
            _eval_key(
                seed=args.seed,
                diversity_n2=diversity_n * diversity_n,
                distribution=args.stage,
                update=update_in_phase,
                heldout=True,
            ),
        )
        jax.block_until_ready(evaluation)
        _write_evaluation_metric(
            metrics_path,
            evaluation,
            algorithm=args.algorithm,
            diversity_n=diversity_n,
            seed=args.seed,
            mode="adapt_eval",
            distribution=args.stage,
            runner=runner,
            schedule=schedule,
            source_distribution=args.stage,
        )
    _save_resume(
        stream_root,
        runner,
        mode="adapt",
        algorithm=args.algorithm,
        diversity_n=diversity_n,
        seed=args.seed,
        distribution=args.stage,
        protocol_identity=protocol_identity,
        source_distribution=args.stage,
    )
    metadata = _checkpoint_meta(
        mode="adapt",
        algorithm=args.algorithm,
        diversity_n=diversity_n,
        seed=args.seed,
        distribution=args.stage,
        runner=runner,
        protocol_identity=protocol_identity,
        source_distribution=args.stage,
    )
    atomic_marker(stream_root / "DONE", metadata)
    return True


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "adapt"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("ppo", "ppo_cbp"), required=True)
    parser.add_argument("--diversity", type=int, required=True, help="n^2 diversity")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", type=int)
    parser.add_argument("--soft-time-limit-seconds", type=float, default=2700.0)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=600.0)
    parser.add_argument("--max-updates", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.config = args.config.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    config = load_config(args.config)
    if args.diversity not in config.experiment.diversity_levels:
        raise ValueError(
            f"diversity {args.diversity} is not configured: "
            f"{config.experiment.diversity_levels}"
        )
    diversity_n = math.isqrt(args.diversity)
    if diversity_n * diversity_n != args.diversity:
        raise ValueError("--diversity must be a perfect-square n^2 value")
    if args.seed not in config.runtime.seeds:
        raise ValueError(f"seed {args.seed} is not configured: {config.runtime.seeds}")
    if args.mode == "adapt":
        if args.stage not in config.experiment.adaptation.after_distributions:
            raise ValueError(
                f"adaptation stage must be one of "
                f"{config.experiment.adaptation.after_distributions}"
            )
    elif args.stage is not None:
        raise ValueError("--stage is valid only with --mode adapt")
    if args.soft_time_limit_seconds <= 180:
        raise ValueError("soft time limit must leave more than 180 seconds")

    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    stream_root = _stream_root(
        args.output_root,
        mode=args.mode,
        algorithm=args.algorithm,
        diversity_n2=args.diversity,
        seed=args.seed,
        stage=args.stage,
    )
    stream_root.mkdir(parents=True, exist_ok=True)
    completed = (
        _run_main(args, config, diversity_n, stream_root)
        if args.mode == "main"
        else _run_adapt(args, config, diversity_n, stream_root)
    )
    print(
        json.dumps(
            {
                "completed": completed,
                "mode": args.mode,
                "algorithm": args.algorithm,
                "diversity_n2": args.diversity,
                "seed": args.seed,
                "stage": args.stage,
                "stream_root": str(stream_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
