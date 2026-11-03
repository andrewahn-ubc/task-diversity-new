"""Generate and validate deterministic Kinetix task-key banks.

Only the PRNG keys accepted by the strict no-op filter are stored.  A level is
reconstructed by passing its key to the official v0.1.0 procedural generator,
so even the largest bank is about 0.5 MiB instead of many gigabytes of physics
state.  Each training bank contains 65,536 keys; a diversity stream uses its
prefix, making diversity comparisons nested within a seed/distribution.

Generation is restart-safe.  After every candidate batch, accepted keys and
the next candidate cursor are atomically published.  Slurm may therefore stop
or requeue the job without changing the eventual bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import sys
from typing import Any, Callable, Sequence

import numpy as np

from .config import KINETIX_COMMIT, PROTOCOL, PROTOCOL_ID, SCHEMA_VERSION


class DataError(RuntimeError):
    """A task bank is absent, corrupt, or incompatible with this protocol."""


def train_bank_dir(data_root: str | Path, seed: int, distribution: int) -> Path:
    return Path(data_root).expanduser().resolve() / f"seed_{seed}" / f"distribution_{distribution}"


def heldout_bank_dir(data_root: str | Path) -> Path:
    return Path(data_root).expanduser().resolve() / "heldout"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bank_spec(kind: str, seed: int | None, distribution: int | None) -> tuple[int, tuple[int, ...]]:
    if kind == "train":
        if seed not in PROTOCOL.learner_seeds:
            raise DataError(f"invalid learner seed: {seed}")
        if distribution is None or not 0 <= distribution < PROTOCOL.num_distributions:
            raise DataError(f"invalid distribution: {distribution}")
        # Separate domain tag, learner seed, and distribution make pools
        # deterministic and disjoint in key space.
        return PROTOCOL.max_diversity, (0x54524149, int(seed), int(distribution))
    if kind == "heldout":
        if seed is not None or distribution is not None:
            raise DataError("heldout bank is global; do not pass seed/distribution")
        return PROTOCOL.heldout_tasks, (0x48454C44,)
    raise DataError(f"unknown bank kind: {kind!r}")


def candidate_keys(kind: str, seed: int | None, distribution: int | None, start: int, count: int):
    """Create deterministic JAX keys for one candidate range.

    This function imports JAX lazily so manifest and planning operations work
    in the non-Kinetix Banyan environment too.
    """

    import jax
    import jax.numpy as jnp

    _, tags = _bank_spec(kind, seed, distribution)
    key = jax.random.PRNGKey(PROTOCOL.data_root_seed)
    for tag in tags:
        key = jax.random.fold_in(key, tag)
    indexes = jnp.arange(start, start + count, dtype=jnp.uint32)
    return jax.vmap(lambda index: jax.random.fold_in(key, index))(indexes)


def make_strict_filter() -> Callable[[Any], Any]:
    """Build the JITted official generator plus strict 64-step no-op filter."""

    import jax
    import jax.numpy as jnp

    from kinetix.environment.env import make_kinetix_env_from_name
    from kinetix.environment.ued.distributions import sample_kinetix_level
    from kinetix.environment.wrappers import (
        AutoReplayWrapper,
        UnderspecifiedToGymnaxWrapper,
    )
    from kinetix.util.config import generate_params_from_config, generate_ued_params_from_config

    config = PROTOCOL.upstream_config()
    env_params, static_env_params = generate_params_from_config(config)
    ued_params = generate_ued_params_from_config(config)
    env = UnderspecifiedToGymnaxWrapper(
        AutoReplayWrapper(
            make_kinetix_env_from_name(
                config["env_name"], static_env_params=static_env_params
            )
        )
    )

    def sample(key):
        return sample_kinetix_level(
            key,
            env.physics_engine,
            env_params,
            static_env_params,
            ued_params,
            env_size_name="s",
        )

    def filter_batch(keys):
        levels = jax.vmap(sample)(keys)
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0x525354))(keys)
        _, states = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(reset_keys, levels, env_params)
        batch_size = keys.shape[0]
        action_shape = env.action_space(env_params).shape
        actions = jnp.zeros((batch_size, *action_shape), dtype=jnp.int32)

        def no_op_step(carry, step_index):
            states, first_done, first_done_reward = carry
            step_keys = jax.vmap(lambda key: jax.random.fold_in(key, step_index + 1))(keys)
            # Call the wrapper's transition directly. Gymnax's public
            # ``step`` adds a second reset through ``reset_env``, which the
            # level-conditioned Kinetix environment intentionally does not
            # implement; AutoReplayWrapper already handles terminal resets.
            _, next_states, rewards, dones, _ = jax.vmap(
                env.step_env, in_axes=(0, 0, 0, None)
            )(
                step_keys, states, actions, env_params
            )
            newly_done = (~first_done) & dones
            first_done_reward = jnp.where(newly_done, rewards, first_done_reward)
            first_done = first_done | dones
            return (next_states, first_done, first_done_reward), None

        initial = (
            states,
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.zeros((batch_size,), dtype=jnp.float32),
        )
        (_, _, first_done_reward), _ = jax.lax.scan(
            no_op_step,
            initial,
            jnp.arange(PROTOCOL.no_op_filter_steps, dtype=jnp.uint32),
        )
        # Unlike upstream's weighted sampling, exclusion is strict as stated in
        # the paper: any level solved by no-op within 64 steps is rejected.
        return first_done_reward <= 0.5

    return jax.jit(filter_batch)


def _load_progress(root: Path, *, kind: str, seed: int | None, distribution: int | None) -> tuple[np.ndarray, int]:
    partial = root / ".partial"
    progress_path = partial / "progress.json"
    keys_path = partial / "keys.npy"
    if not progress_path.exists() and not keys_path.exists():
        return np.empty((0, 2), dtype=np.uint32), 0
    if not progress_path.is_file() or not keys_path.is_file():
        raise DataError(f"incomplete partial state in {partial}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "kind": kind,
        "seed": seed,
        "distribution": distribution,
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise DataError(f"partial bank mismatch for {key}: {progress.get(key)!r} != {value!r}")
    keys = np.load(keys_path, allow_pickle=False)
    if keys.dtype != np.uint32 or keys.ndim != 2 or keys.shape[1] != 2:
        raise DataError(f"invalid partial key array: {keys_path}")
    if len(keys) != int(progress.get("accepted_count", -1)):
        raise DataError("partial accepted count does not match key array")
    return keys, int(progress["candidate_cursor"])


def _publish_progress(
    root: Path,
    keys: np.ndarray,
    cursor: int,
    *,
    kind: str,
    seed: int | None,
    distribution: int | None,
) -> None:
    partial = root / ".partial"
    _atomic_npy(partial / "keys.npy", keys)
    _atomic_json(
        partial / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "kind": kind,
            "seed": seed,
            "distribution": distribution,
            "accepted_count": len(keys),
            "candidate_cursor": cursor,
        },
    )


def generate_bank(
    data_root: str | Path,
    *,
    kind: str,
    seed: int | None = None,
    distribution: int | None = None,
    candidate_batch_size: int = 512,
    soft_limit_seconds: int = 2700,
    filter_batch: Callable[[Any], Any] | None = None,
) -> bool:
    """Advance a bank until complete or the soft deadline.

    Returns ``True`` only when the immutable final bank has been published.
    A caller should exit nonzero/requeue when it returns ``False``.
    ``filter_batch`` is injectable for focused deterministic tests.
    """

    target, _ = _bank_spec(kind, seed, distribution)
    if candidate_batch_size <= 0:
        raise DataError("candidate_batch_size must be positive")
    if soft_limit_seconds <= 0:
        raise DataError("soft_limit_seconds must be positive")
    root = (
        train_bank_dir(data_root, int(seed), int(distribution))
        if kind == "train"
        else heldout_bank_dir(data_root)
    )
    if (root / "DONE").is_file():
        validate_bank(root, expected_kind=kind, expected_seed=seed, expected_distribution=distribution)
        return True

    root.mkdir(parents=True, exist_ok=True)
    accepted, cursor = _load_progress(root, kind=kind, seed=seed, distribution=distribution)
    if len(accepted) > target:
        raise DataError(f"partial bank has {len(accepted)} keys but target is {target}")
    if filter_batch is None:
        filter_batch = make_strict_filter()

    start_time = time.monotonic()
    while len(accepted) < target:
        if time.monotonic() - start_time >= soft_limit_seconds:
            _publish_progress(
                root,
                accepted,
                cursor,
                kind=kind,
                seed=seed,
                distribution=distribution,
            )
            return False
        keys_device = candidate_keys(kind, seed, distribution, cursor, candidate_batch_size)
        keep = np.asarray(filter_batch(keys_device), dtype=np.bool_)
        keys = np.asarray(keys_device, dtype=np.uint32)
        if keep.shape != (candidate_batch_size,):
            raise DataError(f"filter returned shape {keep.shape}; expected {(candidate_batch_size,)}")
        accepted = np.concatenate((accepted, keys[keep]), axis=0)[:target]
        cursor += candidate_batch_size
        _publish_progress(
            root,
            accepted,
            cursor,
            kind=kind,
            seed=seed,
            distribution=distribution,
        )

    final_keys = root / "keys.npy"
    _atomic_npy(final_keys, accepted)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "kinetix_commit": KINETIX_COMMIT,
        "kind": kind,
        "seed": seed,
        "distribution": distribution,
        "accepted_count": len(accepted),
        "candidate_cursor": cursor,
        "filter": {"type": "strict_no_op", "steps": PROTOCOL.no_op_filter_steps},
        "keys_file": "keys.npy",
        "keys_sha256": _sha256_file(final_keys),
        "nested_prefix_sizes": list(PROTOCOL.diversity_levels) if kind == "train" else [PROTOCOL.heldout_tasks],
    }
    _atomic_json(root / "manifest.json", manifest)
    _atomic_json(root / "DONE", {"protocol_id": PROTOCOL_ID, "keys_sha256": manifest["keys_sha256"]})
    return True


def validate_bank(
    root: str | Path,
    *,
    expected_kind: str | None = None,
    expected_seed: int | None = None,
    expected_distribution: int | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    done_path = root / "DONE"
    keys_path = root / "keys.npy"
    if not manifest_path.is_file() or not done_path.is_file() or not keys_path.is_file():
        raise DataError(f"task bank is incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("protocol_id") != PROTOCOL_ID:
        raise DataError(f"task bank protocol mismatch: {root}")
    for name, expected in (
        ("kind", expected_kind),
        ("seed", expected_seed),
        ("distribution", expected_distribution),
    ):
        if expected is not None and manifest.get(name) != expected:
            raise DataError(f"task bank {name} mismatch: {manifest.get(name)!r} != {expected!r}")
    keys = np.load(keys_path, allow_pickle=False, mmap_mode="r")
    if keys.dtype != np.uint32 or keys.ndim != 2 or keys.shape[1] != 2:
        raise DataError(f"invalid task-key array: {keys_path}")
    if keys.shape[0] != int(manifest.get("accepted_count", -1)):
        raise DataError(f"task-key count mismatch: {root}")
    checksum = _sha256_file(keys_path)
    if checksum != manifest.get("keys_sha256"):
        raise DataError(f"task-key checksum mismatch: {root}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done != {"protocol_id": PROTOCOL_ID, "keys_sha256": checksum}:
        raise DataError(f"invalid completion marker: {done_path}")
    return manifest


def load_training_keys(data_root: str | Path, seed: int, distribution: int, diversity: int) -> np.ndarray:
    if diversity not in PROTOCOL.diversity_levels:
        raise DataError(f"unsupported diversity: {diversity}")
    root = train_bank_dir(data_root, seed, distribution)
    validate_bank(root, expected_kind="train", expected_seed=seed, expected_distribution=distribution)
    return np.asarray(np.load(root / "keys.npy", allow_pickle=False, mmap_mode="r")[:diversity])


def load_heldout_keys(data_root: str | Path) -> np.ndarray:
    root = heldout_bank_dir(data_root)
    validate_bank(root, expected_kind="heldout")
    return np.asarray(np.load(root / "keys.npy", allow_pickle=False, mmap_mode="r"))


def load_bank(
    data_root: str | Path,
    *,
    seed: int | None = None,
    distribution: int | None = None,
    heldout: bool = False,
) -> np.ndarray:
    """Load one complete immutable key bank as ``uint32[N,2]``.

    Training callers deliberately receive the full 65,536-key bank and select
    ``keys[:diversity]`` themselves.  This keeps one source of truth for the
    nested-prefix design.
    """

    if heldout:
        if seed is not None or distribution is not None:
            raise DataError("the heldout bank is shared; seed/distribution must be omitted")
        return load_heldout_keys(data_root)
    if seed is None or distribution is None:
        raise DataError("training banks require both seed and distribution")
    root = train_bank_dir(data_root, seed, distribution)
    validate_bank(root, expected_kind="train", expected_seed=seed, expected_distribution=distribution)
    values = np.asarray(np.load(root / "keys.npy", allow_pickle=False, mmap_mode="r"))
    if values.shape != (PROTOCOL.max_diversity, 2):
        raise DataError(f"unexpected training bank shape at {root}: {values.shape}")
    return values


def validate_all_banks(data_root: str | Path) -> None:
    """Require every one of the 21 training banks and the shared heldout bank."""

    for seed in PROTOCOL.learner_seeds:
        for distribution in range(PROTOCOL.num_distributions):
            validate_bank(
                train_bank_dir(data_root, seed, distribution),
                expected_kind="train",
                expected_seed=seed,
                expected_distribution=distribution,
            )
    validate_bank(heldout_bank_dir(data_root), expected_kind="heldout")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Compatibility mode for Slurm: one array index completely determines the
    # bank, so batch scripts do not need to duplicate the canonical mapping.
    if "--index" in raw and (not raw or raw[0] not in {"generate", "validate"}):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--data-root", required=True)
        parser.add_argument("--index", required=True, type=int)
        parser.add_argument("--candidate-batch-size", type=int, default=512)
        parser.add_argument(
            "--soft-time-limit-seconds",
            "--soft-limit-seconds",
            dest="soft_limit_seconds",
            type=int,
            default=int(os.environ.get("TD2_SOFT_LIMIT_SECONDS", "2700")),
        )
        args = parser.parse_args(raw)
        args.command = "generate-index"
        return args

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--data-root", required=True)
    generate.add_argument("--kind", choices=("train", "heldout"), required=True)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--distribution", type=int)
    generate.add_argument("--candidate-batch-size", type=int, default=512)
    generate.add_argument("--soft-limit-seconds", type=int, default=int(os.environ.get("TD2_SOFT_LIMIT_SECONDS", "2700")))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--data-root", required=True)
    return parser.parse_args(raw)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command in {"generate", "generate-index"}:
        if args.command == "generate-index":
            from .plan import data_task

            task = data_task(args.index)
            kind = task["kind"]
            seed = task["seed"]
            distribution = task["distribution"]
        else:
            kind = args.kind
            seed = args.seed
            distribution = args.distribution
        complete = generate_bank(
            args.data_root,
            kind=kind,
            seed=seed,
            distribution=distribution,
            candidate_batch_size=args.candidate_batch_size,
            soft_limit_seconds=args.soft_limit_seconds,
        )
        print("COMPLETE" if complete else "CHECKPOINTED_INCOMPLETE", flush=True)
        return 0 if complete else 75

    validate_all_banks(args.data_root)
    print(f"VALID protocol_id={PROTOCOL_ID} train_banks=21 heldout_banks=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
