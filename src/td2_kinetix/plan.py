"""Canonical Slurm array mappings for the Kinetix protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import PROTOCOL, PROTOCOL_ID


DATA_TASKS = len(PROTOCOL.learner_seeds) * PROTOCOL.num_distributions + 1
STREAMS = len(PROTOCOL.learner_seeds) * len(PROTOCOL.diversity_levels)
MAIN_TASKS = STREAMS * PROTOCOL.num_distributions
HELDOUT_TASKS = STREAMS * PROTOCOL.num_distributions


def data_task(index: int) -> dict[str, Any]:
    if not 0 <= index < DATA_TASKS:
        raise IndexError(f"data index {index} outside [0,{DATA_TASKS})")
    if index == DATA_TASKS - 1:
        return {"kind": "heldout", "seed": None, "distribution": None}
    seed_index, distribution = divmod(index, PROTOCOL.num_distributions)
    return {
        "kind": "train",
        "seed": PROTOCOL.learner_seeds[seed_index],
        "distribution": distribution,
    }


def stream_task(index: int) -> dict[str, int]:
    if not 0 <= index < STREAMS:
        raise IndexError(f"stream index {index} outside [0,{STREAMS})")
    diversity_index, seed_index = divmod(index, len(PROTOCOL.learner_seeds))
    return {
        "diversity": PROTOCOL.diversity_levels[diversity_index],
        "seed": PROTOCOL.learner_seeds[seed_index],
    }


def experiment_task(index: int) -> dict[str, int]:
    if not 0 <= index < MAIN_TASKS:
        raise IndexError(f"experiment index {index} outside [0,{MAIN_TASKS})")
    distribution, stream_index = divmod(index, STREAMS)
    return {**stream_task(stream_index), "distribution": distribution, "stream_index": stream_index}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("counts")
    subparsers.add_parser("identity")
    for name in ("data-task", "stream-task", "experiment-task"):
        child = subparsers.add_parser(name)
        child.add_argument("index", type=int)
    # Shell-friendly aliases used by the Slurm launchers.  They avoid parsing
    # JSON inside failure-sensitive batch prologues.
    for name in ("data", "stream"):
        child = subparsers.add_parser(name)
        child.add_argument("--index", required=True, type=int)
    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--data-root", required=True)
    validate.add_argument("--index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "counts":
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "data_tasks": DATA_TASKS,
                    "streams": STREAMS,
                    "main_tasks": MAIN_TASKS,
                    "heldout_tasks": HELDOUT_TASKS,
                    "main_stage_array_size": STREAMS,
                    "heldout_stage_array_size": STREAMS,
                    "distributions": PROTOCOL.num_distributions,
                },
                sort_keys=True,
            )
        )
    elif args.command == "identity":
        print(PROTOCOL_ID)
    elif args.command == "stream":
        task = stream_task(args.index)
        print(f"{task['diversity']}\t{task['seed']}")
    elif args.command == "data":
        task = data_task(args.index)
        seed = "" if task["seed"] is None else task["seed"]
        distribution = "" if task["distribution"] is None else task["distribution"]
        print(f"{task['kind']}\t{seed}\t{distribution}")
    elif args.command == "validate-data":
        from .data import (
            heldout_bank_dir,
            train_bank_dir,
            validate_all_banks,
            validate_bank,
        )

        if args.index is None:
            validate_all_banks(Path(args.data_root))
        else:
            task = data_task(args.index)
            if task["kind"] == "heldout":
                validate_bank(
                    heldout_bank_dir(args.data_root), expected_kind="heldout"
                )
            else:
                validate_bank(
                    train_bank_dir(
                        args.data_root, task["seed"], task["distribution"]
                    ),
                    expected_kind="train",
                    expected_seed=task["seed"],
                    expected_distribution=task["distribution"],
                )
        print(f"VALID\t{PROTOCOL_ID}")
    else:
        function = {
            "data-task": data_task,
            "stream-task": stream_task,
            "experiment-task": experiment_task,
        }[args.command]
        print(json.dumps(function(args.index), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
