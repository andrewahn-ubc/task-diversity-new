"""Restart-safe continual and heldout Kinetix training entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping, NamedTuple

import numpy as np

from td2.checkpoint import (
    CheckpointError,
    atomic_marker,
    load_checkpoint_with_metadata,
    save_checkpoint,
)
from td2_kinetix.config import KINETIX_COMMIT, PROTOCOL
from td2_kinetix.data import load_bank
from td2_kinetix.ppo import (
    SETTINGS,
    build_components,
    fresh_runner_from_learner,
    initialize_runner,
    make_heldout_evaluator,
    make_update_fn,
)


PROTOCOL_VERSION = 1
TIMESTEPS_PER_PHASE = PROTOCOL.timesteps_per_distribution
UPDATES_PER_PHASE = TIMESTEPS_PER_PHASE // SETTINGS.environment_steps_per_update
DIVERSITIES = PROTOCOL.diversity_levels
NUM_SEEDS = len(PROTOCOL.learner_seeds)
NUM_DISTRIBUTIONS = PROTOCOL.num_distributions
EVAL_INTERVAL_STEPS = 10_000_000
EVAL_INTERVAL_UPDATES = max(
    1, round(EVAL_INTERVAL_STEPS / SETTINGS.environment_steps_per_update)
)

METRIC_FIELDS = (
    "row_type",
    "mode",
    "diversity",
    "seed",
    "stage",
    "update_in_phase",
    "global_update",
    "main_environment_steps",
    "adaptation_steps",
    "success_rate",
    "task_success_rate",
    "episodic_return",
    "mean_transition_reward",
    "episode_count",
    "completed_eval_episodes",
    "unfinished_eval_attempts",
    "loss",
    "actor_loss",
    "value_loss",
    "entropy",
)

_STOP_REQUESTED = False


class LearnerSnapshot(NamedTuple):
    train_state: Any
    rng: Any
    global_update: Any


def _request_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def stream_root(
    output_root: Path,
    *,
    mode: str,
    diversity: int,
    seed: int,
    stage: int,
) -> Path:
    if mode == "main":
        return (
            output_root
            / "main"
            / f"diversity_{diversity}"
            / f"seed_{seed}"
            / "stages"
            / f"stage_{stage}"
        )
    if mode == "heldout":
        return (
            output_root
            / "heldout"
            / f"diversity_{diversity}"
            / f"seed_{seed}"
            / f"stage_{stage}"
        )
    raise ValueError(f"unknown mode: {mode!r}")


def source_stage_root(
    output_root: Path, *, diversity: int, seed: int, stage: int
) -> Path:
    return stream_root(
        output_root,
        mode="main",
        diversity=diversity,
        seed=seed,
        stage=stage,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bank_identity(data_root: Path, *, mode: str, seed: int, stage: int) -> str:
    if mode == "heldout":
        candidates = (data_root / "heldout" / "keys.npy",)
    else:
        candidates = (
            data_root / f"seed_{seed}" / f"distribution_{stage}" / "keys.npy",
        )
    for path in candidates:
        if path.is_file():
            return _sha256_file(path)
    # ``load_bank`` will produce the more descriptive protocol error.  This
    # fallback merely keeps identity construction independent of path aliases.
    return "missing"


def protocol_identity(
    data_root: Path, *, mode: str, diversity: int, seed: int, stage: int
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kinetix_commit": KINETIX_COMMIT,
        "mode": mode,
        "diversity": diversity,
        "seed": seed,
        "stage": stage,
        "bank_sha256": _bank_identity(
            data_root, mode=mode, seed=seed, stage=stage
        ),
        "num_envs": SETTINGS.num_envs,
        "rollout_steps": SETTINGS.rollout_steps,
        "updates_per_phase": UPDATES_PER_PHASE,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode not in {"main", "heldout"}:
        raise ValueError("--mode must be main or heldout")
    if args.diversity not in DIVERSITIES:
        raise ValueError(
            f"--diversity must be one of {DIVERSITIES}, got {args.diversity}"
        )
    if not 0 <= args.seed < NUM_SEEDS:
        raise ValueError(f"--seed must be in [0, {NUM_SEEDS}), got {args.seed}")
    if not 0 <= args.stage < NUM_DISTRIBUTIONS:
        raise ValueError(
            f"--stage must be in [0, {NUM_DISTRIBUTIONS}), got {args.stage}"
        )
    if args.soft_time_limit_seconds <= 0:
        raise ValueError("--soft-time-limit-seconds must be positive")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    if args.checkpoint_every_updates <= 0 or args.checkpoint_every_seconds <= 0:
        raise ValueError("checkpoint intervals must be positive")
    if args.eval_attempts <= 0:
        raise ValueError("--eval-attempts must be positive")


def _append_metric(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _existing_metric_keys(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                keys.add((row["row_type"], int(row["update_in_phase"])))
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def _as_number(value: Any, cast):
    try:
        import jax

        value = jax.device_get(value)
    except ImportError:
        pass
    return cast(np.asarray(value).reshape(()))


def _write_training_metric(
    path: Path,
    existing: set[tuple[str, int]],
    *,
    args: argparse.Namespace,
    local_update: int,
    global_update: int,
    source_global_update: int,
    metric: Mapping[str, Any],
) -> None:
    key = ("train", local_update)
    if key in existing:
        return
    _append_metric(
        path,
        {
            "row_type": "train",
            "mode": args.mode,
            "diversity": args.diversity,
            "seed": args.seed,
            "stage": args.stage,
            "update_in_phase": local_update,
            "global_update": global_update,
            "main_environment_steps": (
                global_update
                if args.mode == "main"
                else source_global_update
            )
            * SETTINGS.environment_steps_per_update,
            "adaptation_steps": (
                local_update * SETTINGS.environment_steps_per_update
                if args.mode == "heldout"
                else 0
            ),
            "success_rate": _as_number(metric["success_rate"], float),
            "episodic_return": _as_number(metric["episodic_return"], float),
            "mean_transition_reward": _as_number(
                metric["transition_reward"], float
            ),
            "episode_count": _as_number(metric["episode_count"], int),
            "loss": _as_number(metric["loss"], float),
            "actor_loss": _as_number(metric["actor_loss"], float),
            "value_loss": _as_number(metric["value_loss"], float),
            "entropy": _as_number(metric["entropy"], float),
        },
    )
    existing.add(key)


def _write_evaluation_metric(
    path: Path,
    existing: set[tuple[str, int]],
    *,
    args: argparse.Namespace,
    local_update: int,
    source_global_update: int,
    metric: Mapping[str, Any],
) -> None:
    key = ("eval", local_update)
    if key in existing:
        return
    _append_metric(
        path,
        {
            "row_type": "eval",
            "mode": args.mode,
            "diversity": args.diversity,
            "seed": args.seed,
            "stage": args.stage,
            "update_in_phase": local_update,
            "global_update": source_global_update,
            "main_environment_steps": source_global_update
            * SETTINGS.environment_steps_per_update,
            "adaptation_steps": local_update
            * SETTINGS.environment_steps_per_update,
            "success_rate": _as_number(metric["success_rate"], float),
            "task_success_rate": _as_number(
                metric["task_success_rate"], float
            ),
            "completed_eval_episodes": _as_number(
                metric["completed_episodes"], int
            ),
            "unfinished_eval_attempts": _as_number(
                metric["unfinished_attempts"], int
            ),
        },
    )
    existing.add(key)


def _load_source_snapshot(
    data_root: Path,
    output_root: Path,
    *,
    diversity: int,
    seed: int,
    stage: int,
    template_runner: tuple[Any, ...],
) -> tuple[LearnerSnapshot, dict[str, Any]]:
    source_root = source_stage_root(
        output_root, diversity=diversity, seed=seed, stage=stage
    )
    if not (source_root / "DONE").is_file():
        raise CheckpointError(f"source stage is incomplete: {source_root}")
    train_state, _, _, _, _, rng, global_update = template_runner
    template = LearnerSnapshot(train_state, rng, global_update)
    snapshot, metadata = load_checkpoint_with_metadata(
        source_root / "learner", template
    )
    expected = {
        **protocol_identity(
            data_root,
            mode="main",
            diversity=diversity,
            seed=seed,
            stage=stage,
        ),
        "target_updates": UPDATES_PER_PHASE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise CheckpointError(
                f"source checkpoint mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    return snapshot, metadata


def run(args: argparse.Namespace) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    _validate_args(args)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    root = stream_root(
        output_root,
        mode=args.mode,
        diversity=args.diversity,
        seed=args.seed,
        stage=args.stage,
    )
    done_path = root / "DONE"
    if done_path.is_file():
        try:
            completed = json.loads(done_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"invalid completion marker {done_path}: {error}") from error
        expected_done = {
            "status": "complete",
            "kinetix_commit": KINETIX_COMMIT,
            "mode": args.mode,
            "diversity": args.diversity,
            "seed": args.seed,
            "stage": args.stage,
        }
        for key, value in expected_done.items():
            if completed.get(key) != value:
                raise CheckpointError(
                    f"completion marker mismatch for {key}: "
                    f"{completed.get(key)!r} != {value!r}"
                )
        if not (root / "learner" / "latest.json").is_file():
            raise CheckpointError(f"completed stream is missing learner checkpoint: {root}")
        print(f"[td2-kinetix] already complete: {root}", flush=True)
        return 0
    root.mkdir(parents=True, exist_ok=True)

    if args.mode == "main":
        bank = load_bank(
            data_root,
            seed=args.seed,
            distribution=args.stage,
            heldout=False,
        )
        if len(bank) < args.diversity:
            raise RuntimeError(
                f"training bank has {len(bank)} keys, needs {args.diversity}"
            )
        active_keys = bank[: args.diversity]
    else:
        active_keys = load_bank(
            data_root, seed=None, distribution=None, heldout=True
        )
        if np.asarray(active_keys).shape != (100, 2):
            raise RuntimeError(
                f"heldout bank must be (100, 2), got {np.asarray(active_keys).shape}"
            )

    components = build_components(active_keys)
    jax = components["jax"]
    base_rng = jax.random.PRNGKey(args.seed)
    base_rng = jax.random.fold_in(base_rng, 0x4B494E45)
    template_runner = initialize_runner(components, base_rng, global_update=0)

    identity = protocol_identity(
        data_root,
        mode=args.mode,
        diversity=args.diversity,
        seed=args.seed,
        stage=args.stage,
    )
    target_updates = min(
        UPDATES_PER_PHASE,
        args.max_updates if args.max_updates is not None else UPDATES_PER_PHASE,
    )
    identity["target_updates"] = target_updates
    resume_root = root / "resume"
    metrics_path = root / "metrics.csv"
    existing_metrics = _existing_metric_keys(metrics_path)
    source_global_update = 0

    if (resume_root / "latest.json").is_file():
        runner, resume_metadata = load_checkpoint_with_metadata(
            resume_root, template_runner
        )
        for key, value in identity.items():
            if resume_metadata.get(key) != value:
                raise CheckpointError(
                    f"resume identity mismatch for {key}: "
                    f"{resume_metadata.get(key)!r} != {value!r}"
                )
        local_update = int(resume_metadata["local_update"])
        source_global_update = int(resume_metadata["source_global_update"])
        print(
            f"[td2-kinetix] resumed {args.mode} stage {args.stage} "
            f"at update {local_update}/{target_updates}",
            flush=True,
        )
    elif args.mode == "main" and args.stage == 0:
        runner = template_runner
        local_update = 0
    else:
        source_stage = args.stage - 1 if args.mode == "main" else args.stage
        snapshot, _ = _load_source_snapshot(
            data_root,
            output_root,
            diversity=args.diversity,
            seed=args.seed,
            stage=source_stage,
            template_runner=template_runner,
        )
        source_global_update = _as_number(snapshot.global_update, int)
        source_rng = snapshot.rng
        if args.mode == "heldout":
            source_rng = jax.random.fold_in(source_rng, 0x48454C44)
            source_rng = jax.random.fold_in(source_rng, args.stage)
        runner = fresh_runner_from_learner(
            components,
            snapshot.train_state,
            source_rng,
            global_update=source_global_update,
        )
        local_update = 0

    if args.mode == "main" and args.stage == 0:
        source_global_update = 0
    elif args.mode == "main" and source_global_update == 0:
        source_global_update = args.stage * UPDATES_PER_PHASE

    update_fn = make_update_fn(components)
    evaluator = None
    heldout_keys = None
    if args.mode == "heldout":
        heldout_keys = np.asarray(active_keys, dtype=np.uint32)
        evaluator = make_heldout_evaluator(
            components, heldout_keys, attempts=args.eval_attempts
        )

    def evaluate_if_needed() -> None:
        if evaluator is None:
            return
        should_evaluate = (
            local_update == 0
            or local_update == target_updates
            or local_update % EVAL_INTERVAL_UPDATES == 0
        )
        if not should_evaluate or ("eval", local_update) in existing_metrics:
            return
        eval_rng = jax.random.PRNGKey(args.seed)
        eval_rng = jax.random.fold_in(eval_rng, args.diversity)
        eval_rng = jax.random.fold_in(eval_rng, args.stage)
        eval_rng = jax.random.fold_in(eval_rng, local_update)
        metric = evaluator(runner[0], eval_rng)
        _write_evaluation_metric(
            metrics_path,
            existing_metrics,
            args=args,
            local_update=local_update,
            source_global_update=source_global_update,
            metric=metric,
        )
        print(
            f"[td2-kinetix] heldout eval stage={args.stage} "
            f"adapt_updates={local_update} success="
            f"{_as_number(metric['success_rate'], float):.4f}",
            flush=True,
        )

    def checkpoint() -> None:
        metadata = {
            **identity,
            "local_update": local_update,
            "source_global_update": source_global_update,
            "target_updates": target_updates,
            "actual_environment_steps": local_update
            * SETTINGS.environment_steps_per_update,
        }
        save_checkpoint(resume_root, runner, metadata=metadata)

    # On resume, do not create an evaluation at an arbitrary checkpoint
    # update: all seeds must share the same heldout x-coordinates for a valid
    # mean/SEM. Fresh branches still evaluate at update zero through the
    # normal schedule condition.
    evaluate_if_needed()
    started = time.monotonic()
    last_checkpoint = started
    while local_update < target_updates:
        runner, metric = update_fn(runner)
        # Force completion before recording/checkpointing so the durable update
        # counter can never get ahead of device computation.
        metric["loss"].block_until_ready()
        local_update += 1
        global_update = _as_number(runner[6], int)
        _write_training_metric(
            metrics_path,
            existing_metrics,
            args=args,
            local_update=local_update,
            global_update=global_update,
            source_global_update=source_global_update,
            metric=metric,
        )
        evaluate_if_needed()
        now = time.monotonic()
        should_checkpoint = (
            local_update % args.checkpoint_every_updates == 0
            or now - last_checkpoint >= args.checkpoint_every_seconds
            or _STOP_REQUESTED
            or now - started >= args.soft_time_limit_seconds
            or local_update == target_updates
        )
        if should_checkpoint:
            checkpoint()
            last_checkpoint = now
        if _STOP_REQUESTED or now - started >= args.soft_time_limit_seconds:
            print(
                f"[td2-kinetix] checkpointed for continuation at "
                f"{local_update}/{target_updates}",
                flush=True,
            )
            return 75

    # target_updates is part of the normal evaluation schedule, so this
    # records the exact terminal point without introducing off-grid rows.
    evaluate_if_needed()
    train_state, _, _, _, _, rng, global_update_array = runner
    global_update = _as_number(global_update_array, int)
    learner = LearnerSnapshot(train_state, rng, global_update_array)
    learner_metadata = {
        **identity,
        "local_update": local_update,
        "source_global_update": source_global_update,
        "global_update": global_update,
        "actual_environment_steps": local_update
        * SETTINGS.environment_steps_per_update,
    }
    save_checkpoint(root / "learner", learner, metadata=learner_metadata, immutable=True)
    atomic_marker(
        done_path,
        {
            **learner_metadata,
            "status": "complete",
        },
    )
    print(f"[td2-kinetix] complete: {root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "heldout"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--diversity", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--soft-time-limit-seconds", type=int, default=2700)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--checkpoint-every-updates", type=int, default=10)
    parser.add_argument("--checkpoint-every-seconds", type=int, default=300)
    parser.add_argument("--eval-attempts", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        return run(build_parser().parse_args(argv))
    except (CheckpointError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
