"""Build the immutable task bank for the core-damage experiment.

The experiment needs a fixed 16-by-16 core and auxiliary pools as large as
256-by-256 without allowing any core task or layout into an auxiliary pool.
One fresh width-272 Banyan round is therefore generated for each learner seed.
The rows and layouts are deterministically shuffled, then partitioned into a
16-wide core and a 256-wide auxiliary bank.  Smaller auxiliary conditions are
nested prefixes of the latter bank.

This is deliberately a separate data protocol.  It neither changes nor
invalidates the schema-v5 bundles used by the original continual experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from td2.config import TD2Config, load_config
import td2.prepare_data as base_data


SCHEMA_VERSION = 1
PROTOCOL = "core_aux_damage"
CORE_N = 16
AUXILIARY_WIDTHS = (16, 32, 64, 128, 256)
AUXILIARY_MAX_N = AUXILIARY_WIDTHS[-1]
STORAGE_N = CORE_N + AUXILIARY_MAX_N
NUM_DEPTHS = base_data.NUM_DEPTHS

_POOL_SEED_SALT = 0x4341504F  # ASCII "CAPO"
_LAYOUT_SEED_SALT = 0x43414C59  # ASCII "CALY"
_DISTRACTOR_SEED_SALT = 0x43414449  # ASCII "CADI"
_PARTITION_SEED_SALT = 0x43415054  # ASCII "CAPT"


def _generation_spec(config: TD2Config, *, data_seed: int) -> dict[str, Any]:
    """Return the complete, deterministic identity of one seed bundle."""

    leaf_pool_seed = base_data._derived_seed(  # noqa: SLF001
        config.ruleset.ruleset_base_seed,
        data_seed,
        _POOL_SEED_SALT,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "banyan_commit": base_data.BANYAN_COMMIT,
        "data_seed": int(data_seed),
        "master_seed": int(data_seed),
        "ruleset_base_seed": int(config.ruleset.ruleset_base_seed),
        "num_depths": base_data.NUM_DEPTHS,
        "num_train_distributions": 1,
        "max_n": STORAGE_N,
        "core_n": CORE_N,
        "auxiliary_widths": list(AUXILIARY_WIDTHS),
        "leaf_pool_seed": leaf_pool_seed,
        "leaf_pool_uids_by_round": [
            list(base_data.paper_leaf_pool_uids(leaf_pool_seed, round_index=0))
        ],
        "layout_seed": base_data._derived_seed(  # noqa: SLF001
            config.ruleset.ruleset_base_seed,
            data_seed,
            _LAYOUT_SEED_SALT,
        ),
        "distractor_seed": base_data._derived_seed(  # noqa: SLF001
            config.ruleset.ruleset_base_seed,
            data_seed,
            _DISTRACTOR_SEED_SALT,
        ),
        "partition_seed": base_data._derived_seed(  # noqa: SLF001
            config.ruleset.ruleset_base_seed,
            data_seed,
            _PARTITION_SEED_SALT,
        ),
        "distractor_density": base_data.DISTRACTOR_DENSITY,
        "partition_semantics": (
            "Generate one width-272 round. Permute its chain-local rule indices "
            "once and use that same order at every depth; independently permute "
            "layout indices. The first 16 chains/layouts form the fixed core; "
            "the remaining 256 form the disjoint auxiliary bank, whose "
            "conditions are nested prefixes."
        ),
    }


def build_core_aux_partition(
    *, partition_seed: int, num_depths: int = NUM_DEPTHS
) -> dict[str, Any]:
    """Create the deterministic, exhaustive core/auxiliary index partition."""

    if num_depths <= 0:
        raise ValueError("num_depths must be positive")
    seed_sequence = np.random.SeedSequence(
        [int(partition_seed), _PARTITION_SEED_SALT]
    )
    rule_seed, layout_seed = seed_sequence.spawn(2)
    rule_rng = np.random.default_rng(rule_seed)
    layout_rng = np.random.default_rng(layout_seed)

    # One local index identifies the complete generated depth-six chain; rows
    # at depths 1..6 are its prefixes. Keep that chain in one bank at every
    # depth so core and auxiliary tasks are genuinely disjoint.
    rule_order = rule_rng.permutation(STORAGE_N).astype(np.int32, copy=False)
    rule_orders = np.repeat(rule_order[None, :], int(num_depths), axis=0)
    layout_order = layout_rng.permutation(STORAGE_N).astype(np.int32, copy=False)
    return {
        "core_rule_local_indices_by_depth": rule_orders[:, :CORE_N].tolist(),
        "auxiliary_rule_local_indices_by_depth": rule_orders[:, CORE_N:].tolist(),
        "core_layout_indices": layout_order[:CORE_N].tolist(),
        "auxiliary_layout_indices": layout_order[CORE_N:].tolist(),
    }


def validate_core_aux_partition(
    partition: dict[str, Any], *, num_depths: int = NUM_DEPTHS
) -> None:
    """Reject any partition that is incomplete, overlapping, or non-nested."""

    core_rules = np.asarray(
        partition["core_rule_local_indices_by_depth"], dtype=np.int32
    )
    auxiliary_rules = np.asarray(
        partition["auxiliary_rule_local_indices_by_depth"], dtype=np.int32
    )
    core_layouts = np.asarray(partition["core_layout_indices"], dtype=np.int32)
    auxiliary_layouts = np.asarray(
        partition["auxiliary_layout_indices"], dtype=np.int32
    )

    if core_rules.shape != (int(num_depths), CORE_N):
        raise base_data.DataPreparationError(
            f"core rule partition has shape {core_rules.shape}, expected "
            f"({num_depths}, {CORE_N})"
        )
    if auxiliary_rules.shape != (int(num_depths), AUXILIARY_MAX_N):
        raise base_data.DataPreparationError(
            f"auxiliary rule partition has shape {auxiliary_rules.shape}, expected "
            f"({num_depths}, {AUXILIARY_MAX_N})"
        )
    if core_layouts.shape != (CORE_N,):
        raise base_data.DataPreparationError("core layout partition has wrong shape")
    if auxiliary_layouts.shape != (AUXILIARY_MAX_N,):
        raise base_data.DataPreparationError(
            "auxiliary layout partition has wrong shape"
        )
    if not np.all(core_rules == core_rules[0]) or not np.all(
        auxiliary_rules == auxiliary_rules[0]
    ):
        raise base_data.DataPreparationError(
            "rule partitions must preserve the same chain-local indices at "
            "every depth"
        )

    expected = np.arange(STORAGE_N, dtype=np.int32)
    for depth_index in range(int(num_depths)):
        combined = np.concatenate(
            (core_rules[depth_index], auxiliary_rules[depth_index])
        )
        if not np.array_equal(np.sort(combined), expected):
            raise base_data.DataPreparationError(
                f"rule partition at depth {depth_index + 1} is not a permutation"
            )
    combined_layouts = np.concatenate((core_layouts, auxiliary_layouts))
    if not np.array_equal(np.sort(combined_layouts), expected):
        raise base_data.DataPreparationError(
            "layout partition is not a permutation"
        )


def prepare_core_aux_data(
    config_path: str | Path, output: str | Path, *, data_seed: int
) -> dict[str, Any]:
    """Generate or verify one immutable seed-specific core/auxiliary bundle."""

    config = load_config(config_path)
    if data_seed not in config.runtime.seeds:
        raise base_data.DataPreparationError(
            f"data seed {data_seed} is not configured: {config.runtime.seeds}"
        )
    spec = _generation_spec(config, data_seed=data_seed)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = base_data._verify_existing(destination, spec)  # noqa: SLF001
    if existing is not None:
        validate_core_aux_partition(
            existing["partition"], num_depths=int(existing["num_depths"])
        )
        return existing

    staging = base_data._staging_root(destination, spec)  # noqa: SLF001
    base_data._generate_training_rounds(staging, spec)  # noqa: SLF001

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.copytree(
            staging / "rulesets",
            temporary / "rulesets",
            ignore=shutil.ignore_patterns("*.partial-*"),
        )
        round_files = base_data._round_files(  # noqa: SLF001
            temporary, STORAGE_N, 1
        )
        records = base_data._normalize_and_validate_rounds(  # noqa: SLF001
            round_files,
            max_n=STORAGE_N,
            num_depths=base_data.NUM_DEPTHS,
            logical_n_per_depth=[STORAGE_N],
        )
        ruleset_path, metadata_path = round_files[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        obstacle_masks, reachable_masks = base_data.generate_layout_masks(
            STORAGE_N, seed=int(spec["layout_seed"])
        )
        base_data._validate_with_banyan_layout_helper(  # noqa: SLF001
            obstacle_masks, reachable_masks
        )
        layouts_path = temporary / "layouts.npz"
        np.savez_compressed(
            layouts_path,
            obstacle_mask=obstacle_masks,
            reachable_mask=reachable_masks,
        )

        table, selected, universe = base_data.generate_distractor_table(
            metadata["pool_uids"],
            seed=int(spec["distractor_seed"]),
            density=float(spec["distractor_density"]),
        )
        distractor_path = base_data._distractor_path(ruleset_path)  # noqa: SLF001
        np.save(distractor_path, table, allow_pickle=False)

        partition = build_core_aux_partition(
            partition_seed=int(spec["partition_seed"]),
            num_depths=base_data.NUM_DEPTHS,
        )
        validate_core_aux_partition(partition)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "data_seed": int(data_seed),
            "generation_spec": spec,
            "banyan_commit": base_data.BANYAN_COMMIT,
            "num_depths": base_data.NUM_DEPTHS,
            "num_train_distributions": 1,
            "max_n": STORAGE_N,
            "storage_n_per_depth": STORAGE_N,
            "core_n": CORE_N,
            "auxiliary_widths": list(AUXILIARY_WIDTHS),
            "ruleset_file": ruleset_path.relative_to(temporary).as_posix(),
            "ruleset_metadata_file": metadata_path.relative_to(temporary).as_posix(),
            "layouts_file": layouts_path.relative_to(temporary).as_posix(),
            "distractor_file": distractor_path.relative_to(temporary).as_posix(),
            "distractor_pair_count": selected,
            "distractor_pair_universe": universe,
            "distractor_density": float(spec["distractor_density"]),
            "round": records[0],
            "partition": partition,
            "sampling_semantics": (
                "Half of vector slots are permanently assigned to the core and "
                "half to the selected auxiliary prefix. Rules, layouts, and "
                "depths are sampled uniformly within each slot's bank."
            ),
        }
        manifest["file_sha256"] = base_data._payload_hashes(temporary)  # noqa: SLF001
        manifest_path = temporary / "manifest.json"
        base_data._write_json(manifest_path, manifest)  # noqa: SLF001
        manifest_digest = base_data._sha256(manifest_path)  # noqa: SLF001
        base_data._write_bytes(  # noqa: SLF001
            temporary / "manifest.sha256",
            f"{manifest_digest}  manifest.json\n".encode("ascii"),
        )
        base_data._fsync_directory(temporary)  # noqa: SLF001
        if destination.exists():
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
            else:
                raise base_data.DataPreparationError(
                    "output appeared during generation; refusing to overwrite: "
                    f"{destination}"
                )
        os.replace(temporary, destination)
        base_data._fsync_directory(destination.parent)  # noqa: SLF001
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-seed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = prepare_core_aux_data(
        args.config, args.output, data_seed=args.data_seed
    )
    print(
        f"Prepared {PROTOCOL} bundle for seed {args.data_seed} at "
        f"{Path(args.output).expanduser().resolve()} "
        f"(core={manifest['core_n']}^2, auxiliary up to "
        f"{max(manifest['auxiliary_widths'])}^2)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
