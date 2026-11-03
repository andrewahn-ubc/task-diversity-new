"""Validate and aggregate the two requested Kinetix continual-learning plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib.pyplot as plt

from .config import PROTOCOL


LABELS = {1: "1²", 16: "4²", 64: "8²", 256: "16²", 65536: "256²"}
COLORS = plt.cm.viridis([0.05, 0.27, 0.49, 0.71, 0.93])


def _root(output: Path, mode: str, diversity: int, seed: int, stage: int) -> Path:
    if mode == "main":
        return output / "main" / f"diversity_{diversity}" / f"seed_{seed}" / "stages" / f"stage_{stage}"
    return output / "heldout" / f"diversity_{diversity}" / f"seed_{seed}" / f"stage_{stage}"


def _rows(path: Path, *, row_type: str) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("row_type") == row_type:
                yield row


def _summary(values: list[float]) -> tuple[float, float]:
    expected = len(PROTOCOL.learner_seeds)
    if len(values) != expected:
        raise RuntimeError(
            f"unaligned aggregate point has {len(values)} seeds; expected {expected}"
        )
    centre = mean(values)
    sem = stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return centre, sem


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(output: Path) -> None:
    missing: list[str] = []
    for diversity in PROTOCOL.diversity_levels:
        for seed in PROTOCOL.learner_seeds:
            for stage in range(PROTOCOL.num_distributions):
                for mode in ("main", "heldout"):
                    root = _root(output, mode, diversity, seed, stage)
                    for name in ("DONE", "metrics.csv"):
                        if not (root / name).is_file():
                            missing.append(str(root / name))
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Kinetix results are incomplete ({len(missing)} missing files):\n{preview}")


def training_plot(output: Path, plots: Path) -> None:
    # Figure 9 is presented as a smooth learning curve.  Smooth each seed
    # independently over approximately 10M environment steps (76 PPO updates)
    # before computing the across-seed mean and SEM.
    per_stream: dict[tuple[int, int], list[tuple[int, float]]] = {}
    by_key: dict[tuple[int, int], list[float]] = {}
    for diversity in PROTOCOL.diversity_levels:
        for seed in PROTOCOL.learner_seeds:
            for stage in range(PROTOCOL.num_distributions):
                metrics = _root(output, "main", diversity, seed, stage) / "metrics.csv"
                for row in _rows(metrics, row_type="train"):
                    steps = int(row["main_environment_steps"])
                    per_stream.setdefault((diversity, seed), []).append(
                        (steps, float(row["success_rate"]))
                    )

    rolling_updates = 76
    for (diversity, _seed), observations in per_stream.items():
        observations.sort()
        values = [value for _, value in observations]
        for index, (steps, _value) in enumerate(observations):
            start = max(0, index + 1 - rolling_updates)
            smoothed = mean(values[start : index + 1])
            by_key.setdefault((diversity, steps), []).append(smoothed)

    summary: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(12, 6.6), constrained_layout=True)
    for color, diversity in zip(COLORS, PROTOCOL.diversity_levels):
        xs, ys, errors = [], [], []
        for (key_diversity, steps), values in sorted(by_key.items()):
            if key_diversity != diversity:
                continue
            centre, sem = _summary(values)
            xs.append(steps / 1_000_000)
            ys.append(centre)
            errors.append(sem)
            summary.append({"diversity": diversity, "environment_steps": steps, "mean_success_rate": centre, "sem_success_rate": sem, "num_seeds": len(values), "rolling_window_updates": rolling_updates})
        ax.plot(xs, ys, color=color, linewidth=2, label=LABELS[diversity])
        ax.fill_between(xs, [y - e for y, e in zip(ys, errors)], [y + e for y, e in zip(ys, errors)], color=color, alpha=0.16, linewidth=0)
    stage_span_millions = PROTOCOL.consumed_timesteps_per_distribution / 1_000_000
    for stage in range(1, PROTOCOL.num_distributions):
        ax.axvline(
            stage * stage_span_millions,
            color="black",
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )
    ax.set(
        title="Kinetix continual training success",
        xlabel="Continual-training environment steps (millions)",
        ylabel="Training success rate",
        xlim=(0, PROTOCOL.num_distributions * stage_span_millions),
        ylim=(0, 1.0),
    )
    ax.legend(title="Training-task diversity", ncol=5, loc="upper center")
    ax.grid(alpha=0.2)
    fig.savefig(plots / "kinetix_training_success.png", dpi=220)
    plt.close(fig)
    _write_csv(plots / "kinetix_training_success_summary.csv", summary)


def heldout_plot(output: Path, plots: Path) -> None:
    by_key: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    for diversity in PROTOCOL.diversity_levels:
        for seed in PROTOCOL.learner_seeds:
            for stage in range(PROTOCOL.num_distributions):
                metrics = _root(output, "heldout", diversity, seed, stage) / "metrics.csv"
                for row in _rows(metrics, row_type="eval"):
                    steps = int(row["adaptation_steps"])
                    episode_success = float(row["success_rate"])
                    task_success = float(row.get("task_success_rate") or episode_success)
                    by_key.setdefault((stage, diversity, steps), []).append(
                        (episode_success, task_success)
                    )

    summary: list[dict[str, object]] = []
    fig, axes = plt.subplots(4, 2, figsize=(13, 16), sharex=True, sharey=True, constrained_layout=True)
    for stage, ax in enumerate(axes.flat[: PROTOCOL.num_distributions]):
        for color, diversity in zip(COLORS, PROTOCOL.diversity_levels):
            xs, ys, errors = [], [], []
            for (key_stage, key_diversity, steps), paired_values in sorted(by_key.items()):
                if key_stage != stage or key_diversity != diversity:
                    continue
                values = [pair[0] for pair in paired_values]
                task_values = [pair[1] for pair in paired_values]
                centre, sem = _summary(values)
                task_centre, task_sem = _summary(task_values)
                xs.append(steps / 1_000_000)
                ys.append(centre)
                errors.append(sem)
                summary.append({"source_distribution": stage + 1, "diversity": diversity, "adaptation_steps": steps, "mean_success_rate": centre, "sem_success_rate": sem, "mean_task_success_rate": task_centre, "sem_task_success_rate": task_sem, "num_seeds": len(values)})
            ax.plot(xs, ys, color=color, linewidth=2, label=LABELS[diversity])
            ax.fill_between(xs, [y - e for y, e in zip(ys, errors)], [y + e for y, e in zip(ys, errors)], color=color, alpha=0.16, linewidth=0)
        ax.set_title(f"After training distribution {stage + 1}")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.0)
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Heldout success rate")
    for ax in axes[-1, :]:
        ax.set_xlabel("Heldout adaptation steps (millions)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, title="Prior training diversity", loc="center", frameon=False)
    fig.suptitle("Adaptation on the same fixed 100-task heldout set", fontsize=16)
    fig.savefig(plots / "kinetix_heldout_adaptation_success.png", dpi=220)
    plt.close(fig)
    _write_csv(plots / "kinetix_heldout_adaptation_success_summary.csv", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    validate(output)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    training_plot(output, plots)
    heldout_plot(output, plots)
    print(f"wrote two Kinetix plots and detailed summaries to {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
