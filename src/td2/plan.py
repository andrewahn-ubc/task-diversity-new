"""Enumerate the exact production work matrix from the merged YAML config."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from td2.config import TD2Config, load_config


@dataclass(frozen=True)
class MainWorkItem:
    algorithm: str
    diversity_n2: int
    seed: int


@dataclass(frozen=True)
class AdaptationWorkItem(MainWorkItem):
    stage: int


def main_work_items(config: TD2Config) -> tuple[MainWorkItem, ...]:
    return tuple(
        MainWorkItem(algorithm, diversity_n2, seed)
        for algorithm in config.experiment.methods
        for diversity_n2 in config.experiment.diversity_levels
        for seed in config.runtime.seeds
    )


def adaptation_work_items(config: TD2Config) -> tuple[AdaptationWorkItem, ...]:
    mains = main_work_items(config)
    return tuple(
        AdaptationWorkItem(
            item.algorithm,
            item.diversity_n2,
            item.seed,
            stage,
        )
        for stage in config.experiment.adaptation.after_distributions
        for item in mains
    )


def plan_identity(config: TD2Config) -> str:
    """Hash the complete inherited config chain used to enumerate the run."""

    digest = hashlib.sha256()
    for source in config.source_files:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _indexed(items: Sequence[object], index: int, label: str):
    if not 0 <= index < len(items):
        raise SystemExit(f"{label} index must be in [0, {len(items) - 1}]")
    return items[index]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("counts")
    subparsers.add_parser("identity")
    main = subparsers.add_parser("main")
    main.add_argument("--index", required=True, type=int)
    adapt = subparsers.add_parser("adapt")
    adapt.add_argument("--index", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    mains = main_work_items(config)
    adaptations = adaptation_work_items(config)
    if args.command == "counts":
        print(f"{len(mains)}\t{len(adaptations)}")
    elif args.command == "identity":
        print(plan_identity(config))
    elif args.command == "main":
        item = _indexed(mains, args.index, "main")
        assert isinstance(item, MainWorkItem)
        print(f"{item.algorithm}\t{item.diversity_n2}\t{item.seed}")
    else:
        item = _indexed(adaptations, args.index, "adaptation")
        assert isinstance(item, AdaptationWorkItem)
        print(
            f"{item.algorithm}\t{item.diversity_n2}\t"
            f"{item.seed}\t{item.stage}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
