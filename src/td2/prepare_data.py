"""Build the immutable task and layout bank used by the continual experiment.

The training task trees are generated one round at a time with Banyan's pinned
round-disjoint depth-six sampler at the largest requested pool width.  Smaller
diversity conditions are *prefixes* of those arrays, which makes the comparison
nested rather than independently resampled.  Each distribution receives a
deterministic 40-object leaf pool from the paper's 80-object vocabulary.  A
final logical held-out round is generated with the same sampler and the
blockers reconstructed from the training rounds.  It is padded only for the
common on-disk row stride, and the manifest selects the fixed task-layout pairs
used for adaptation.  One complete bundle is generated per learner seed, so
replicates differ in both their training banks and held-out probes.

Raw rounds are atomically retained in a specification-keyed sibling staging
directory, so an interrupted data job resumes from the first missing round.
Finalization remains transactional: it copies the raw rounds to a temporary
directory, validates every scientific invariant, hashes every payload, and
renames the directory into place only after all checks pass.  An existing
directory is accepted only when its manifest describes the same request and all
of its hashes still match.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from td2.config import TD2Config, load_config


BANYAN_COMMIT = "7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3"
SCHEMA_VERSION = 5
STAGING_PROTOCOL_VERSION = 1
NUM_DEPTHS = 6
GRID_SIZE = 6
POOL_SIZE = 40
PAPER_NUM_ITEMS = 8
PAPER_NUM_COLORS = 10
BANYAN_COLOR_STRIDE = 12
DISTRACTOR_DENSITY = 0.5
PACKED_PAD = np.uint32(5)
MAX_TOPOLOGY_ATTEMPTS = 20_000
# Banyan has 120 possible unary producer signatures. Ten training rounds may
# introduce at most 100 of them, reserving at least 20 for the disjoint held-out
# round instead of exhausting the smallest producer-signature space.
MAX_NEW_UNARY_SIGNATURES_PER_TRAIN_ROUND = 10
GENERATOR_MODULE = "banyan_grid.tasks.make_depth6_round_disjoint_dataset"
HELDOUT_GENERATOR_API = f"{GENERATOR_MODULE}._generate_round_dataset"
SPAWN_CELLS = ((0, 0), (1, 0))
_LAYOUT_SEED_SALT = 0x4C41594F  # ASCII "LAYO"
_DISTRACTOR_SEED_SALT = 0x44495354  # ASCII "DIST"
_POOL_SEED_SALT = 0x504F4F4C  # ASCII "POOL"


class DataPreparationError(RuntimeError):
    """Raised when generation or an invariant check fails."""


def paper_leaf_pool_uids(
    seed: int, *, round_index: int = 0, pool_size: int = POOL_SIZE
) -> tuple[int, ...]:
    """Select one round's 40 leaves from the paper's 8-by-10 vocabulary.

    Banyan encodes object IDs with the environment's 12-color stride, so a raw
    prefix such as ``range(40)`` is not a subset of the paper's ten-color
    experimental vocabulary.  The paper does not publish the exact 40-object
    draw; a seeded sample without replacement is the least-assumptive choice.
    """

    vocabulary = np.asarray(
        [
            item * BANYAN_COLOR_STRIDE + color
            for item in range(PAPER_NUM_ITEMS)
            for color in range(PAPER_NUM_COLORS)
        ],
        dtype=np.int32,
    )
    if not 0 < pool_size <= vocabulary.size:
        raise ValueError(f"pool_size must be in [1, {vocabulary.size}]")
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(round_index), _POOL_SEED_SALT])
    )
    selected = rng.choice(vocabulary, size=pool_size, replace=False)
    return tuple(int(value) for value in selected)


def _derived_seed(base_seed: int, data_seed: int, salt: int, index: int = 0) -> int:
    """Derive a stable unsigned 32-bit seed for one data-bundle component."""

    state = np.random.SeedSequence(
        [int(base_seed), int(data_seed), int(salt), int(index)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _canonical_json(value))


def _load_packed(path: Path) -> np.ndarray:
    with bz2.BZ2File(path, "rb") as handle:
        packed = np.load(handle, allow_pickle=False)
    if packed.dtype != np.uint32 or packed.ndim != 2:
        raise DataPreparationError(f"invalid packed ruleset array in {path}")
    return packed


def _write_packed(path: Path, packed: np.ndarray) -> None:
    with bz2.BZ2File(path, "wb", compresslevel=9) as handle:
        np.save(handle, np.asarray(packed, dtype=np.uint32), allow_pickle=False)


def _diversity_widths(config: TD2Config) -> tuple[int, ...]:
    widths: list[int] = []
    for level in config.experiment.diversity_levels:
        width = math.isqrt(int(level))
        if width * width != int(level):
            raise DataPreparationError(
                f"diversity level {level} is not a perfect square"
            )
        widths.append(width)
    if tuple(sorted(widths)) != tuple(widths) or len(set(widths)) != len(widths):
        raise DataPreparationError("diversity widths must be strictly increasing")
    return tuple(widths)


def _generation_spec(config: TD2Config, *, data_seed: int) -> dict[str, Any]:
    widths = _diversity_widths(config)
    num_rounds = config.experiment.num_distributions + 1
    leaf_pool_seed = _derived_seed(
        config.ruleset.ruleset_base_seed,
        data_seed,
        _POOL_SEED_SALT,
    )
    leaf_pool_uids_by_round = [
        list(
            paper_leaf_pool_uids(
                leaf_pool_seed, round_index=round_index
            )
        )
        for round_index in range(num_rounds)
    ]
    layout_seed = _derived_seed(
        config.experiment.adaptation.heldout_task_seed,
        data_seed,
        _LAYOUT_SEED_SALT,
    )
    distractor_seeds_by_round = [
        _derived_seed(
            config.experiment.adaptation.heldout_task_seed,
            data_seed,
            _DISTRACTOR_SEED_SALT,
            round_index,
        )
        for round_index in range(num_rounds)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "banyan_commit": BANYAN_COMMIT,
        "num_train_distributions": config.experiment.num_distributions,
        "num_rounds": num_rounds,
        "train_n_per_depth": max(
            max(widths), config.experiment.adaptation.heldout_tasks
        ),
        "heldout_n_per_depth": config.experiment.adaptation.heldout_tasks,
        "num_depths": NUM_DEPTHS,
        # The attached two-agent source YAML says depth 4; experiment.yaml
        # explicitly overrides it because Figure 6 evaluates depths 1..6.
        "attached_yaml_tree_depth": 4,
        "effective_tree_depth": config.ruleset.tree_depth,
        # Production has max(widths)=256 >= the held-out pair count. Keeping at
        # least heldout_tasks also makes the reduced smoke configuration valid.
        "max_n": max(max(widths), config.experiment.adaptation.heldout_tasks),
        "diversity_widths": list(widths),
        "diversity_levels_n2": [int(value) for value in config.experiment.diversity_levels],
        "ruleset_base_seed": config.ruleset.ruleset_base_seed,
        "data_seed": int(data_seed),
        "master_seed": int(data_seed),
        "leaf_pool_seed": leaf_pool_seed,
        "pool_size": POOL_SIZE,
        "paper_object_vocabulary": {
            "items": PAPER_NUM_ITEMS,
            "colors": PAPER_NUM_COLORS,
            "size": PAPER_NUM_ITEMS * PAPER_NUM_COLORS,
            "banyan_color_stride": BANYAN_COLOR_STRIDE,
        },
        "leaf_pool_selection": (
            "independent deterministic sample without replacement per round"
        ),
        "leaf_pool_uids_by_round": leaf_pool_uids_by_round,
        "grid_size": GRID_SIZE,
        "layout_count_per_training_distribution": max(
            max(widths), config.experiment.adaptation.heldout_tasks
        ),
        "heldout_tasks": config.experiment.adaptation.heldout_tasks,
        "heldout_task_seed": config.experiment.adaptation.heldout_task_seed,
        "layout_seed": layout_seed,
        "distractor_seeds_by_round": distractor_seeds_by_round,
        "cross_seed_semantics": (
            "each learner seed has an independently seeded training and held-out bundle"
        ),
        "distractor_density": DISTRACTOR_DENSITY,
        "ruleset_generation": {
            "training_interface": HELDOUT_GENERATOR_API,
            "training_interface_visibility": "private API at pinned commit",
            "resumable_round_staging": True,
            "staging_protocol_version": STAGING_PROTOCOL_VERSION,
            "heldout_interface": HELDOUT_GENERATOR_API,
            "heldout_interface_visibility": "private API at pinned commit",
            "storage_row_stride_n": max(
                max(widths), config.experiment.adaptation.heldout_tasks
            ),
            "packed_pad_value": int(PACKED_PAD),
            "max_topology_attempts": MAX_TOPOLOGY_ATTEMPTS,
            "max_new_unary_signatures_per_train_round": (
                MAX_NEW_UNARY_SIGNATURES_PER_TRAIN_ROUND
            ),
            "heldout_round_seed_derivation": (
                "draw round_index+1 int32 seeds from NumPy default_rng("
                "master_seed XOR (max_n * 1000003)); use the final draw"
            ),
        },
        "layout_constraints": {
            "minimum_walls": 2,
            "maximum_walls": 10,
            "minimum_open_fraction": 0.30,
            "minimum_reachable_open_fraction": 0.55,
            "spawn_cells": [list(cell) for cell in SPAWN_CELLS],
            "exact_unique_masks": True,
        },
    }


def _verify_existing(output: Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return an intact matching manifest, or fail rather than overwrite."""

    if not output.exists():
        return None
    if output.is_dir() and not any(output.iterdir()):
        # Setup scripts commonly reserve the requested path with mkdir.  An
        # empty directory contains nothing to overwrite and can still be
        # replaced transactionally once generation has completed.
        return None
    manifest_path = output / "manifest.json"
    sidecar_path = output / "manifest.sha256"
    if not output.is_dir() or not manifest_path.is_file() or not sidecar_path.is_file():
        raise DataPreparationError(
            f"refusing to overwrite existing incomplete output directory: {output}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPreparationError(f"cannot read existing manifest: {exc}") from exc
    if manifest.get("generation_spec") != spec:
        raise DataPreparationError(
            f"existing data at {output} was generated from a different specification"
        )
    sidecar_digest = sidecar_path.read_text(encoding="ascii").split()[0]
    if sidecar_digest != _sha256(manifest_path):
        raise DataPreparationError("existing manifest checksum does not match")
    hashes = manifest.get("file_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise DataPreparationError("existing manifest has no payload hashes")
    for relative, expected in hashes.items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected:
            raise DataPreparationError(f"existing payload failed checksum: {relative}")
    return manifest


def _round_paths(
    root: Path, *, storage_n: int, logical_n: int, round_index: int
) -> tuple[Path, Path]:
    run_dir = root / "rulesets" / f"n{storage_n:04d}"
    stem = f"td2_n{logical_n}_r{round_index:02d}"
    return (
        run_dir / f"{stem}.uint32.npy.bz2",
        run_dir / f"{stem}_meta.json",
    )


def _round_files(root: Path, max_n: int, rounds: int) -> list[tuple[Path, Path]]:
    run_dir = root / "rulesets" / f"n{max_n:04d}"
    result: list[tuple[Path, Path]] = []
    for round_index in range(rounds):
        stem = f"td2_n{max_n}_r{round_index:02d}"
        data = run_dir / f"{stem}.uint32.npy.bz2"
        metadata = run_dir / f"{stem}_meta.json"
        if not data.is_file() or not metadata.is_file():
            raise DataPreparationError(f"generator omitted round {round_index}")
        result.append((data, metadata))
    return result


def _atomic_write_packed(path: Path, packed: np.ndarray) -> None:
    """Publish one compressed array without exposing a partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_packed(temporary, packed)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Publish one JSON document without exposing a partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _producer_signatures(packed: np.ndarray) -> set[tuple[Any, ...]]:
    """Reconstruct producer input signatures using Banyan's public decoder."""

    from banyan_grid.environment.constants import (  # imported only for data jobs
        Colors,
        RULE_TYPE_COMBINE,
        RULE_TYPE_TERNARY_COMBINE,
        RULE_TYPE_TRANSFORM,
    )
    from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32_np

    real_colors = int(Colors.BLACK)
    rows = unpack_rules_uint32_np(packed).reshape((-1, 6))
    signatures: set[tuple[Any, ...]] = set()
    for row in rows:
        rule_type = int(row[0])
        colors = int(row[5])
        uid_a = int(row[1]) * real_colors + (colors & 0xF)
        if rule_type == int(RULE_TYPE_TRANSFORM):
            signatures.add(("U", uid_a))
        elif rule_type == int(RULE_TYPE_COMBINE):
            uid_b = int(row[2]) * real_colors + ((colors >> 4) & 0xF)
            signatures.add(("B", *sorted((uid_a, uid_b))))
        elif rule_type == int(RULE_TYPE_TERNARY_COMBINE):
            uid_b = int(row[2]) * real_colors + ((colors >> 4) & 0xF)
            item_c = int(row[4]) >> 1
            uid_c = item_c * real_colors + ((colors >> 8) & 0xF)
            signatures.add(("T", *sorted((uid_a, uid_b, uid_c))))
    return signatures


def _validate_logical_round(
    packed: np.ndarray,
    metadata: dict[str, Any],
    *,
    round_index: int,
    logical_n: int,
    num_depths: int,
    blocked_topologies: set[str],
    blocked_producers: set[tuple[Any, ...]],
) -> tuple[set[str], set[tuple[Any, ...]]]:
    """Validate one unpadded logical round against all preceding rounds."""

    expected_rows = int(logical_n) * int(num_depths)
    if packed.shape[0] != expected_rows:
        raise DataPreparationError(
            f"round {round_index} has {packed.shape[0]} rows; expected {expected_rows}"
        )
    if int(metadata.get("round_index", -1)) != int(round_index):
        raise DataPreparationError(f"round index mismatch for round {round_index}")
    if int(metadata.get("n_per_depth", -1)) != int(logical_n):
        raise DataPreparationError(f"task count mismatch in round {round_index}")

    structure = metadata.get("structure", {})
    if not isinstance(structure, dict) or any(
        int(structure.get(str(depth), structure.get(depth, -1))) != int(logical_n)
        for depth in range(1, int(num_depths) + 1)
    ):
        raise DataPreparationError(
            f"per-depth task structure mismatch in round {round_index}"
        )

    topology_sequence = metadata.get("round_depth6_topology_signatures", [])
    if not isinstance(topology_sequence, list) or len(topology_sequence) != int(
        logical_n
    ):
        raise DataPreparationError(
            f"depth-six topology count mismatch in round {round_index}"
        )
    topologies = {str(value) for value in topology_sequence}
    if len(topologies) != int(metadata["round_unique_depth6_topology_count"]):
        raise DataPreparationError(
            f"unique depth-six topology count mismatch in round {round_index}"
        )
    topology_overlap = topologies & blocked_topologies
    if topology_overlap:
        raise DataPreparationError(
            f"depth-six topology repeated across rounds at round {round_index}"
        )

    producers = _producer_signatures(packed)
    expected_producers = int(metadata["round_unique_producer_signature_count"])
    if len(producers) != expected_producers:
        raise DataPreparationError(
            f"producer signature count mismatch in round {round_index}: "
            f"decoded {len(producers)}, metadata {expected_producers}"
        )
    producer_overlap = producers & blocked_producers
    if producer_overlap:
        raise DataPreparationError(
            f"producer signature repeated across rounds at round {round_index}"
        )
    return topologies, producers


def _heldout_round_seed(*, master_seed: int, max_n: int, round_index: int) -> int:
    """Reproduce the pinned CLI's deterministic seed for one round index."""

    generator_seed = int(master_seed) ^ (int(max_n) * 1_000_003)
    rng = np.random.default_rng(generator_seed)
    round_seed = -1
    for _ in range(int(round_index) + 1):
        round_seed = int(rng.integers(0, 2**31 - 1))
    return round_seed


def _load_round_generator() -> Any:
    from banyan_grid.tasks import make_depth6_round_disjoint_dataset as generator

    return generator


def _validate_raw_round(
    packed: np.ndarray,
    metadata: dict[str, Any],
    *,
    spec: dict[str, Any],
    round_index: int,
    logical_n: int,
    blocked_topologies: set[str],
    blocked_producers: set[tuple[Any, ...]],
) -> tuple[set[str], set[tuple[Any, ...]]]:
    """Validate one durable raw round, including its deterministic inputs."""

    expected_pool = [
        int(value) for value in spec["leaf_pool_uids_by_round"][round_index]
    ]
    generator_master_seed = int(spec["master_seed"]) ^ (
        int(spec["max_n"]) * 1_000_003
    )
    expected_seed = _heldout_round_seed(
        master_seed=int(spec["master_seed"]),
        max_n=int(spec["max_n"]),
        round_index=round_index,
    )
    if [int(value) for value in metadata.get("pool_uids", [])] != expected_pool:
        raise DataPreparationError(f"round {round_index} used an unexpected leaf pool")
    if int(metadata.get("base_seed", -1)) != int(spec["ruleset_base_seed"]):
        raise DataPreparationError(f"base seed mismatch in round {round_index}")
    if int(metadata.get("max_depth", -1)) != int(spec["num_depths"]):
        raise DataPreparationError(f"tree depth mismatch in round {round_index}")
    if int(metadata.get("round_seed", -1)) != expected_seed:
        raise DataPreparationError(f"round seed mismatch in round {round_index}")
    if int(metadata.get("master_seed", -1)) != generator_master_seed:
        raise DataPreparationError(f"generator master seed mismatch in round {round_index}")
    return _validate_logical_round(
        packed,
        metadata,
        round_index=round_index,
        logical_n=logical_n,
        num_depths=int(spec["num_depths"]),
        blocked_topologies=blocked_topologies,
        blocked_producers=blocked_producers,
    )


def _generate_training_rounds(
    root: Path, spec: dict[str, Any]
) -> list[tuple[Path, Path]]:
    """Generate missing training rounds and reuse every valid completed prefix."""

    generator = _load_round_generator()
    max_n = int(spec["max_n"])
    num_rounds = int(spec["num_train_distributions"])
    generator_master_seed = int(spec["master_seed"]) ^ (max_n * 1_000_003)
    blocked_topologies: set[str] = set()
    blocked_producers: set[tuple[Any, ...]] = set()
    files: list[tuple[Path, Path]] = []

    for round_index in range(num_rounds):
        data_path, metadata_path = _round_paths(
            root,
            storage_n=max_n,
            logical_n=max_n,
            round_index=round_index,
        )
        if data_path.is_file() and metadata_path.is_file():
            try:
                packed = _load_packed(data_path)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                topologies, producers = _validate_raw_round(
                    packed,
                    metadata,
                    spec=spec,
                    round_index=round_index,
                    logical_n=max_n,
                    blocked_topologies=blocked_topologies,
                    blocked_producers=blocked_producers,
                )
            except Exception as exc:
                raise DataPreparationError(
                    f"invalid staged training round {round_index}: {exc}"
                ) from exc
            blocked_topologies.update(topologies)
            blocked_producers.update(producers)
            files.append((data_path, metadata_path))
            print(f"[td2] reusing completed data round {round_index + 1}/{num_rounds}")
            continue

        # A signal can arrive between the two atomic publishes.  Regenerating
        # this deterministic round safely repairs that one incomplete pair.
        for later_index in range(round_index + 1, num_rounds):
            later_paths = _round_paths(
                root,
                storage_n=max_n,
                logical_n=max_n,
                round_index=later_index,
            )
            if any(path.exists() for path in later_paths):
                raise DataPreparationError(
                    f"staged rounds are not a contiguous prefix at round {round_index}"
                )

        round_seed = _heldout_round_seed(
            master_seed=int(spec["master_seed"]),
            max_n=max_n,
            round_index=round_index,
        )
        print(f"[td2] generating data round {round_index + 1}/{num_rounds}")
        try:
            packed, metadata, returned_topologies, returned_producers = (
                generator._generate_round_dataset(
                    n=max_n,
                    max_depth=int(spec["num_depths"]),
                    round_index=round_index,
                    rounds_remaining=num_rounds - round_index,
                    base_pool=np.asarray(
                        spec["leaf_pool_uids_by_round"][round_index],
                        dtype=np.int32,
                    ),
                    base_seed=int(spec["ruleset_base_seed"]),
                    unary_budget=2,
                    binary_budget=2,
                    unary_root_weight=0.34,
                    binary_root_weight=0.33,
                    side_depth_cap=3,
                    deep_side_prob=0.18,
                    max_depth6_rules=32,
                    max_topology_attempts=MAX_TOPOLOGY_ATTEMPTS,
                    max_instantiation_attempts=128,
                    blocked_topology_signatures=blocked_topologies,
                    blocked_operation_signatures=blocked_producers,
                    max_new_unary_signatures_per_round=(
                        MAX_NEW_UNARY_SIGNATURES_PER_TRAIN_ROUND
                    ),
                    max_new_binary_signatures_per_round=0,
                    rng=np.random.default_rng(round_seed),
                )
            )
        except Exception as exc:
            raise DataPreparationError(
                f"Banyan training round {round_index} generation failed: {exc}"
            ) from exc

        packed = np.asarray(packed, dtype=np.uint32)
        metadata["round_seed"] = round_seed
        metadata["master_seed"] = generator_master_seed
        metadata["global_unique_depth6_topologies_after_round"] = (
            len(blocked_topologies) + len(returned_topologies)
        )
        metadata["global_unique_producer_signatures_after_round"] = (
            len(blocked_producers) + len(returned_producers)
        )
        topologies, producers = _validate_raw_round(
            packed,
            metadata,
            spec=spec,
            round_index=round_index,
            logical_n=max_n,
            blocked_topologies=blocked_topologies,
            blocked_producers=blocked_producers,
        )
        if topologies != {str(value) for value in returned_topologies}:
            raise DataPreparationError(
                f"training topology set disagrees with Banyan in round {round_index}"
            )
        if producers != {tuple(value) for value in returned_producers}:
            raise DataPreparationError(
                f"training producer set disagrees with Banyan in round {round_index}"
            )

        _atomic_write_packed(data_path, packed)
        _atomic_write_json(metadata_path, metadata)
        blocked_topologies.update(topologies)
        blocked_producers.update(producers)
        files.append((data_path, metadata_path))
        print(f"[td2] completed data round {round_index + 1}/{num_rounds}")

    return files


def _staging_root(destination: Path, spec: dict[str, Any]) -> Path:
    """Return a stable, specification-keyed root for resumable raw rounds."""

    identity = {
        "staging_protocol_version": STAGING_PROTOCOL_VERSION,
        "generation_spec": spec,
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    root = destination.parent / f".{destination.name}.prepare-{digest[:16]}"
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / "generation-spec.json"
    if sentinel.is_file():
        try:
            existing = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataPreparationError(f"cannot read data staging identity: {exc}") from exc
        if existing != identity:
            raise DataPreparationError("data staging identity does not match this run")
    else:
        unexpected = [path for path in root.iterdir() if ".partial-" not in path.name]
        if unexpected:
            raise DataPreparationError(
                f"data staging directory has no identity sentinel: {root}"
            )
        _atomic_write_json(sentinel, identity)
    return root


def _generate_heldout_round(
    root: Path,
    spec: dict[str, Any],
    training_files: Sequence[tuple[Path, Path]],
) -> tuple[Path, Path]:
    """Generate only the logical held-out tasks with the pinned private sampler.

    Banyan's public CLI can generate only one ``n`` for every round.  Its final
    n=256 round is pathologically slow even though this experiment consumes
    fewer held-out tasks.  With the repository pinned, reconstructing the
    public generator's blocker sets and calling its underlying round function
    preserves its exact disjointness contract without sampling unused tasks.
    """

    max_n = int(spec["max_n"])
    num_depths = int(spec["num_depths"])
    heldout_n = int(spec["heldout_n_per_depth"])
    round_index = int(spec["num_train_distributions"])
    if len(training_files) != round_index:
        raise DataPreparationError(
            f"expected {round_index} training rounds, found {len(training_files)}"
        )

    blocked_topologies: set[str] = set()
    blocked_producers: set[tuple[Any, ...]] = set()
    for train_index, (data_path, metadata_path) in enumerate(training_files):
        packed = _load_packed(data_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        topologies, producers = _validate_logical_round(
            packed,
            metadata,
            round_index=train_index,
            logical_n=max_n,
            num_depths=num_depths,
            blocked_topologies=blocked_topologies,
            blocked_producers=blocked_producers,
        )
        blocked_topologies.update(topologies)
        blocked_producers.update(producers)

    data_path, metadata_path = _round_paths(
        root,
        storage_n=max_n,
        logical_n=heldout_n,
        round_index=round_index,
    )
    if data_path.is_file() and metadata_path.is_file():
        try:
            packed = _load_packed(data_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            _validate_raw_round(
                packed,
                metadata,
                spec=spec,
                round_index=round_index,
                logical_n=heldout_n,
                blocked_topologies=blocked_topologies,
                blocked_producers=blocked_producers,
            )
        except Exception as exc:
            raise DataPreparationError(f"invalid staged held-out round: {exc}") from exc
        print("[td2] reusing completed held-out data round")
        return data_path, metadata_path

    try:
        generator = _load_round_generator()

        round_seed = _heldout_round_seed(
            master_seed=int(spec["master_seed"]),
            max_n=max_n,
            round_index=round_index,
        )
        packed, metadata, returned_topologies, returned_producers = (
            generator._generate_round_dataset(
                n=heldout_n,
                max_depth=num_depths,
                round_index=round_index,
                rounds_remaining=1,
                base_pool=np.asarray(
                    spec["leaf_pool_uids_by_round"][round_index], dtype=np.int32
                ),
                base_seed=int(spec["ruleset_base_seed"]),
                unary_budget=2,
                binary_budget=2,
                unary_root_weight=0.34,
                binary_root_weight=0.33,
                side_depth_cap=3,
                deep_side_prob=0.18,
                max_depth6_rules=32,
                max_topology_attempts=MAX_TOPOLOGY_ATTEMPTS,
                max_instantiation_attempts=128,
                blocked_topology_signatures=blocked_topologies,
                blocked_operation_signatures=blocked_producers,
                max_new_unary_signatures_per_round=0,
                max_new_binary_signatures_per_round=0,
                rng=np.random.default_rng(round_seed),
            )
        )
    except Exception as exc:
        raise DataPreparationError(
            f"Banyan held-out task generation failed: {exc}"
        ) from exc

    packed = np.asarray(packed, dtype=np.uint32)
    topologies, producers = _validate_logical_round(
        packed,
        metadata,
        round_index=round_index,
        logical_n=heldout_n,
        num_depths=num_depths,
        blocked_topologies=blocked_topologies,
        blocked_producers=blocked_producers,
    )
    if topologies != {str(value) for value in returned_topologies}:
        raise DataPreparationError(
            "held-out topology set disagrees with Banyan return value"
        )
    if producers != {tuple(value) for value in returned_producers}:
        raise DataPreparationError(
            "held-out producer set disagrees with Banyan return value"
        )

    generator_master_seed = int(spec["master_seed"]) ^ (max_n * 1_000_003)
    metadata["round_seed"] = round_seed
    metadata["master_seed"] = generator_master_seed
    metadata["td2_generation"] = {
        "interface": HELDOUT_GENERATOR_API,
        "visibility": "private API at pinned commit",
        "banyan_commit": BANYAN_COMMIT,
        "logical_n_per_depth": heldout_n,
        "max_topology_attempts": MAX_TOPOLOGY_ATTEMPTS,
        "blocked_depth6_topologies": len(blocked_topologies),
        "blocked_producer_signatures": len(blocked_producers),
    }
    _validate_raw_round(
        packed,
        metadata,
        spec=spec,
        round_index=round_index,
        logical_n=heldout_n,
        blocked_topologies=blocked_topologies,
        blocked_producers=blocked_producers,
    )

    _atomic_write_packed(data_path, packed)
    _atomic_write_json(metadata_path, metadata)
    print("[td2] completed held-out data round")
    return data_path, metadata_path


def _normalize_and_validate_rounds(
    files: Sequence[tuple[Path, Path]],
    *,
    max_n: int,
    num_depths: int,
    logical_n_per_depth: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Validate logical rounds, then pad them to one width and depth stride."""

    arrays = [_load_packed(data) for data, _ in files]
    logical_ns = (
        [int(max_n)] * len(files)
        if logical_n_per_depth is None
        else [int(value) for value in logical_n_per_depth]
    )
    if len(logical_ns) != len(files):
        raise DataPreparationError("logical round widths do not match round files")
    if any(value <= 0 or value > int(max_n) for value in logical_ns):
        raise DataPreparationError("logical round widths must be in [1, max_n]")
    global_rule_width = max(array.shape[1] for array in arrays)
    all_topologies: set[str] = set()
    all_producers: set[tuple[Any, ...]] = set()
    records: list[dict[str, Any]] = []

    for round_index, ((data_path, meta_path), packed, logical_n) in enumerate(
        zip(files, arrays, logical_ns)
    ):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        topologies, producers = _validate_logical_round(
            packed,
            metadata,
            round_index=round_index,
            logical_n=logical_n,
            num_depths=num_depths,
            blocked_topologies=all_topologies,
            blocked_producers=all_producers,
        )
        all_topologies.update(topologies)
        all_producers.update(producers)

        original_width = packed.shape[1]
        storage_rows = int(max_n) * int(num_depths)
        normalized = np.full(
            (storage_rows, global_rule_width), PACKED_PAD, dtype=np.uint32
        )
        for depth_index in range(int(num_depths)):
            source_start = depth_index * int(logical_n)
            source_stop = source_start + int(logical_n)
            storage_start = depth_index * int(max_n)
            storage_stop = storage_start + int(logical_n)
            normalized[storage_start:storage_stop, :original_width] = packed[
                source_start:source_stop
            ]
        _write_packed(data_path, normalized)

        metadata["original_packed_shape"] = [
            int(logical_n) * int(num_depths),
            original_width,
        ]
        metadata["td2_logical_n_per_depth"] = int(logical_n)
        metadata["td2_storage_n_per_depth"] = int(max_n)
        metadata["packed_shape"] = [storage_rows, global_rule_width]
        metadata["td2_global_rule_width"] = global_rule_width
        metadata["td2_depth_major_padding"] = {
            "packed_pad_value": int(PACKED_PAD),
            "logical_rows_per_depth": int(logical_n),
            "storage_rows_per_depth": int(max_n),
        }
        _write_json(meta_path, metadata)
        records.append(
            {
                "round_index": round_index,
                "ruleset_file": data_path.relative_to(data_path.parents[2]).as_posix(),
                "metadata_file": meta_path.relative_to(meta_path.parents[2]).as_posix(),
                "logical_n_per_depth": int(logical_n),
                "storage_n_per_depth": int(max_n),
                "unique_depth6_topologies": len(topologies),
                "unique_producer_signatures": len(producers),
            }
        )

    return records


def _reachable_one(obstacle: np.ndarray) -> np.ndarray:
    height, width = obstacle.shape
    reachable = np.zeros_like(obstacle, dtype=np.bool_)
    stack: list[tuple[int, int]] = []
    for y, x in SPAWN_CELLS:
        if 0 <= y < height and 0 <= x < width and not obstacle[y, x]:
            reachable[y, x] = True
            stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and not obstacle[ny, nx]
                and not reachable[ny, nx]
            ):
                reachable[ny, nx] = True
                stack.append((ny, nx))
    return reachable


def generate_layout_masks(
    count: int,
    *,
    seed: int,
    grid_size: int = GRID_SIZE,
    min_walls: int = 2,
    max_walls: int = 10,
    min_open_fraction: float = 0.30,
    min_reachable_fraction: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate exact-unique wall masks satisfying the paper's constraints."""

    if count <= 0:
        raise ValueError("count must be positive")
    if grid_size != GRID_SIZE:
        raise ValueError("this protocol uses a 6x6 grid")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _LAYOUT_SEED_SALT]))
    candidates = np.asarray(
        [index for index in range(grid_size * grid_size) if divmod(index, grid_size) not in SPAWN_CELLS],
        dtype=np.int32,
    )
    masks: list[np.ndarray] = []
    reachable_masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    max_attempts = max(100_000, count * 1_000)
    for _ in range(max_attempts):
        walls = int(rng.integers(min_walls, max_walls + 1))
        wall_indices = rng.choice(candidates, size=walls, replace=False)
        mask = np.zeros((grid_size, grid_size), dtype=np.bool_)
        mask.reshape(-1)[wall_indices] = True
        identity = np.packbits(mask, bitorder="little").tobytes()
        if identity in seen:
            continue
        open_count = int((~mask).sum())
        if open_count / mask.size < min_open_fraction:
            continue
        reachable = _reachable_one(mask)
        if int(reachable.sum()) / open_count < min_reachable_fraction:
            continue
        seen.add(identity)
        masks.append(mask)
        reachable_masks.append(reachable)
        if len(masks) == count:
            break
    if len(masks) != count:
        raise DataPreparationError(
            f"could generate only {len(masks)} of {count} valid unique layouts"
        )
    return np.stack(masks), np.stack(reachable_masks)


def _validate_with_banyan_layout_helper(
    obstacle_masks: np.ndarray, reachable_masks: np.ndarray
) -> None:
    from banyan_grid.utils.dataset_loader import build_reachable_masks_from_masks_np

    upstream = build_reachable_masks_from_masks_np(obstacle_masks)
    if not np.array_equal(upstream, reachable_masks):
        raise DataPreparationError("layout reachability disagrees with Banyan")


def generate_distractor_table(
    pool_uids: Sequence[int], *, seed: int, density: float = DISTRACTOR_DENSITY
) -> tuple[np.ndarray, int, int]:
    """Reproduce Banyan's dataset-global Bernoulli distractor construction."""

    from banyan_grid.environment.constants import Colors, NUM_COLORS, NUM_ITEMS
    from banyan_grid.tasks.ruleset_factory import _pair_to_output_uid_global_np

    real_colors = int(Colors.BLACK)
    pairs = [
        (int(pool_uids[left]), int(pool_uids[right]))
        for left in range(len(pool_uids))
        for right in range(left + 1, len(pool_uids))
    ]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), _DISTRACTOR_SEED_SALT])
    )
    keep = rng.random(len(pairs)) < float(density)
    table = np.zeros(
        (int(NUM_ITEMS), int(NUM_COLORS), int(NUM_ITEMS), int(NUM_COLORS), 3),
        dtype=np.int32,
    )
    for (uid_a, uid_b), selected in zip(pairs, keep):
        if not selected:
            continue
        output_uid = _pair_to_output_uid_global_np(uid_a, uid_b, base_seed=0)
        item_a, color_a = divmod(uid_a, real_colors)
        item_b, color_b = divmod(uid_b, real_colors)
        output_item, output_color = divmod(output_uid, real_colors)
        value = np.asarray([output_item, output_color, 1], dtype=np.int32)
        table[item_a, color_a, item_b, color_b] = value
        table[item_b, color_b, item_a, color_a] = value
    return table, int(keep.sum()), len(pairs)


def _distractor_path(ruleset_path: Path) -> Path:
    suffix = ".uint32.npy.bz2"
    if not ruleset_path.name.endswith(suffix):
        raise DataPreparationError(f"unexpected ruleset filename: {ruleset_path.name}")
    stem = ruleset_path.name[: -len(suffix)]
    return ruleset_path.with_name(f"{stem}.uint32_distractor_table.npy")


def _payload_hashes(root: Path) -> dict[str, str]:
    ignored = {"manifest.json", "manifest.sha256"}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in ignored
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_data(
    config_path: str | Path, output: str | Path, *, data_seed: int
) -> dict[str, Any]:
    """Generate or verify the complete immutable experiment input bundle."""

    config = load_config(config_path)
    if data_seed not in config.runtime.seeds:
        raise DataPreparationError(
            f"data seed {data_seed} is not configured: {config.runtime.seeds}"
        )
    spec = _generation_spec(config, data_seed=data_seed)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _verify_existing(destination, spec)
    if existing is not None:
        return existing

    staging = _staging_root(destination, spec)
    staged_training_files = _generate_training_rounds(staging, spec)
    _generate_heldout_round(staging, spec, staged_training_files)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        # Normalization rewrites arrays in place, so finalization uses a real
        # copy and never mutates the resumable raw rounds.
        shutil.copytree(
            staging / "rulesets",
            temporary / "rulesets",
            ignore=shutil.ignore_patterns("*.partial-*"),
        )
        training_files = _round_files(
            temporary,
            int(spec["max_n"]),
            int(spec["num_train_distributions"]),
        )
        heldout_file = _round_paths(
            temporary,
            storage_n=int(spec["max_n"]),
            logical_n=int(spec["heldout_n_per_depth"]),
            round_index=int(spec["num_train_distributions"]),
        )
        if not all(path.is_file() for path in heldout_file):
            raise DataPreparationError("staging copy omitted the held-out round")

        round_files = [*training_files, heldout_file]
        logical_ns = [int(spec["max_n"])] * int(
            spec["num_train_distributions"]
        ) + [int(spec["heldout_n_per_depth"])]
        round_records = _normalize_and_validate_rounds(
            round_files,
            max_n=int(spec["max_n"]),
            num_depths=NUM_DEPTHS,
            logical_n_per_depth=logical_ns,
        )

        total_layouts = (
            int(spec["num_train_distributions"]) * int(spec["max_n"])
            + int(spec["heldout_tasks"])
        )
        obstacle_masks, reachable_masks = generate_layout_masks(
            total_layouts, seed=int(spec["layout_seed"])
        )
        _validate_with_banyan_layout_helper(obstacle_masks, reachable_masks)
        layouts_path = temporary / "layouts.npz"
        np.savez_compressed(
            layouts_path,
            obstacle_mask=obstacle_masks,
            reachable_mask=reachable_masks,
        )

        train_records: list[dict[str, Any]] = []
        heldout_record: dict[str, Any] | None = None
        for round_index, ((ruleset_path, metadata_path), record) in enumerate(
            zip(round_files, round_records)
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pool_uids = [int(value) for value in metadata["pool_uids"]]
            expected_pool = [
                int(value)
                for value in spec["leaf_pool_uids_by_round"][round_index]
            ]
            if pool_uids != expected_pool:
                raise DataPreparationError(
                    f"round {round_index} used an unexpected leaf-object pool"
                )
            table, selected, universe = generate_distractor_table(
                pool_uids,
                seed=int(spec["distractor_seeds_by_round"][round_index]),
                density=float(spec["distractor_density"]),
            )
            table_path = _distractor_path(ruleset_path)
            np.save(table_path, table, allow_pickle=False)
            common = {
                **record,
                "distractor_file": table_path.relative_to(temporary).as_posix(),
                "distractor_pair_count": selected,
                "distractor_pair_universe": universe,
                "distractor_density": float(spec["distractor_density"]),
            }
            if round_index < int(spec["num_train_distributions"]):
                common["distribution"] = round_index + 1
                common["layout_index_start"] = round_index * int(spec["max_n"])
                common["layout_count"] = int(spec["max_n"])
                train_records.append(common)
            else:
                heldout_start = int(spec["num_train_distributions"]) * int(
                    spec["max_n"]
                )
                pairs = [
                    {
                        "pair_index": pair_index,
                        "depth": pair_index % NUM_DEPTHS + 1,
                        "local_index": pair_index,
                        "layout_index": heldout_start + pair_index,
                    }
                    for pair_index in range(int(spec["heldout_tasks"]))
                ]
                heldout_record = {
                    **common,
                    "logical_n_per_depth": int(spec["heldout_n_per_depth"]),
                    "storage_n_per_depth": int(spec["max_n"]),
                    "generation_interface": HELDOUT_GENERATOR_API,
                    "generation_interface_visibility": (
                        "private API at pinned commit"
                    ),
                    "layout_index_start": heldout_start,
                    "layout_count": int(spec["heldout_tasks"]),
                    "pairs": pairs,
                    "fixed_across_training_stages": True,
                    "fixed_across_diversity_streams": True,
                    "fixed_across_seeds": False,
                }

        if heldout_record is None:
            raise DataPreparationError("held-out round was not created")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "data_seed": int(spec["data_seed"]),
            "generation_spec": spec,
            "banyan_commit": BANYAN_COMMIT,
            "num_train_distributions": int(spec["num_train_distributions"]),
            "num_depths": NUM_DEPTHS,
            "max_n": int(spec["max_n"]),
            "diversity_widths": spec["diversity_widths"],
            "diversity_levels_n2": spec["diversity_levels_n2"],
            "ruleset_generation": spec["ruleset_generation"],
            "prefix_semantics": (
                "For width n, use local indices [0,n) independently for the task "
                "tree and layout; this yields n^2 task-layout combinations."
            ),
            "layouts_file": layouts_path.relative_to(temporary).as_posix(),
            "layout_count": total_layouts,
            "train_distributions": train_records,
            "heldout": heldout_record,
            "cross_round_validation": {
                "depth6_topologies_disjoint": True,
                "producer_signatures_disjoint": True,
                "validated_before_storage_padding": True,
            },
        }
        manifest["file_sha256"] = _payload_hashes(temporary)
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_digest = _sha256(manifest_path)
        _write_bytes(
            temporary / "manifest.sha256",
            f"{manifest_digest}  manifest.json\n".encode("ascii"),
        )
        _fsync_directory(temporary)
        if destination.exists():
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
            else:
                raise DataPreparationError(
                    f"output appeared during generation; refusing to overwrite: {destination}"
                )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = prepare_data(args.config, args.output, data_seed=args.seed)
    print(
        f"Prepared {manifest['num_train_distributions']} training distributions "
        f"and one held-out round for seed {args.seed} at {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
