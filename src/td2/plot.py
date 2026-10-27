"""Aggregate experiment CSV logs into the requested continual-learning plots.

Every run writes a recursive ``metrics.csv``.  This module deliberately reads
only files with that exact name so rerunning it cannot ingest its own summary.
Replicates are aggregated as mean +/- SEM; raw seed trajectories remain the
unit of replication for adaptation AUC and time-to-threshold statistics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from td2.config import TD2Config, load_config


REQUIRED_ID_COLUMNS = ("method", "seed", "mode", "distribution")
DEFAULT_THRESHOLD = 0.80
CORE_AUX_METHOD = "ppo_cbp"
CORE_AUX_EVAL_MODE = "core_aux_2b_eval"
CORE_AUX_WIDTHS = (16, 32, 64, 128, 256)
CORE_AUX_SEEDS = tuple(range(5))
CORE_AUX_TERMINAL_UPDATE = 7_620


class PlotDataError(RuntimeError):
    """Raised when run logs cannot be interpreted without guessing."""


@dataclass(frozen=True)
class MetricRow:
    method: str
    diversity_n: int
    diversity_n2: int
    seed: int
    mode: str
    distribution: int
    environment_steps: float
    nominal_environment_steps: float
    success_rate: float
    episodic_return: float
    adaptation_steps: float
    source_distribution: int
    global_update: int
    source: str
    ordinal: int


def _number(value: Any, *, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _integer(value: Any, *, default: int = -1) -> int:
    numeric = _number(value)
    return default if not math.isfinite(numeric) else int(numeric)


def _normal_mode(value: str) -> str:
    mode = value.strip().lower().replace("-", "_")
    aliases = {
        "training": "train",
        "evaluation": "eval",
        "evaluate": "eval",
        "adaptation": "adapt",
        "adapt_eval": "adapt",
        "adaptation_eval": "adapt",
        "heldout": "adapt",
        "held_out": "adapt",
        "heldout_eval": "adapt",
        "adapt_eval": "adapt",
    }
    return aliases.get(mode, mode)


def _parse_diversity(record: Mapping[str, str]) -> tuple[int, int]:
    n = _integer(record.get("diversity_n"))
    n2 = _integer(record.get("diversity_n2"))
    if n <= 0 and n2 > 0:
        n = math.isqrt(n2)
    if n2 <= 0 and n > 0:
        n2 = n * n
    if n <= 0 or n2 != n * n:
        raise PlotDataError(
            "every metric row needs consistent positive diversity_n and diversity_n2"
        )
    return n, n2


def read_metric_rows(output_root: str | Path) -> list[MetricRow]:
    """Read and de-duplicate all recursive ``metrics.csv`` run logs."""

    root = Path(output_root).expanduser().resolve()
    paths = sorted(
        path
        for path in root.rglob("metrics.csv")
        if not any(
            part.startswith(".") for part in path.relative_to(root).parts[:-1]
        )
    )
    if not paths:
        raise PlotDataError(f"no recursive metrics.csv files found below {root}")
    rows: list[MetricRow] = []
    ordinal = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in REQUIRED_ID_COLUMNS if name not in (reader.fieldnames or ())]
            if missing:
                raise PlotDataError(f"{path} is missing columns: {', '.join(missing)}")
            for raw in reader:
                n, n2 = _parse_diversity(raw)
                distribution = _integer(raw.get("distribution"))
                source_distribution = _integer(
                    raw.get("source_distribution"), default=distribution
                )
                rows.append(
                    MetricRow(
                        method=str(raw.get("method", "")).strip().lower(),
                        diversity_n=n,
                        diversity_n2=n2,
                        seed=_integer(raw.get("seed")),
                        mode=_normal_mode(str(raw.get("mode", ""))),
                        distribution=distribution,
                        environment_steps=_number(raw.get("environment_steps")),
                        nominal_environment_steps=_number(
                            raw.get("nominal_environment_steps")
                        ),
                        success_rate=_number(raw.get("success_rate")),
                        episodic_return=_number(raw.get("episodic_return")),
                        adaptation_steps=_number(raw.get("adaptation_steps")),
                        source_distribution=source_distribution,
                        global_update=_integer(raw.get("global_update")),
                        source=path.relative_to(root).as_posix(),
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
    if not rows:
        raise PlotDataError("metrics files contained no data rows")

    # Resume can repeat the last committed update.  Last occurrence wins, but
    # runs remain independent because seed/method/diversity are part of the key.
    deduplicated: dict[tuple[Any, ...], MetricRow] = {}
    for row in rows:
        x = row.adaptation_steps if row.mode == "adapt" else (
            row.nominal_environment_steps
            if math.isfinite(row.nominal_environment_steps)
            else row.environment_steps
        )
        key = (
            row.method,
            row.diversity_n,
            row.seed,
            row.mode,
            row.distribution,
            row.source_distribution,
            x,
        )
        deduplicated[key] = row
    return sorted(deduplicated.values(), key=lambda row: row.ordinal)


def _global_step(row: MetricRow, config: TD2Config) -> float:
    if math.isfinite(row.nominal_environment_steps):
        # The logging contract defines nominal_environment_steps as cumulative.
        return row.nominal_environment_steps
    return row.environment_steps


def _mean_sem(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return math.nan, math.nan, 0
    mean = float(finite.mean())
    sem = float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else 0.0
    return mean, sem, int(finite.size)


def _aggregate_trajectory(
    rows: Iterable[MetricRow],
    *,
    x_value,
    y_value,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    points: dict[tuple[str, int, float], list[float]] = {}
    for row in rows:
        x = float(x_value(row))
        y = float(y_value(row))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        points.setdefault((row.method, row.diversity_n, x), []).append(y)
    curves: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for (method, width, x), values in points.items():
        mean, sem, _ = _mean_sem(values)
        curves.setdefault((method, width), []).append((x, mean, sem))
    result: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, curve in curves.items():
        curve.sort(key=lambda item: item[0])
        array = np.asarray(curve, dtype=float)
        result[key] = array[:, 0], array[:, 1], array[:, 2]
    return result


def _aggregate_segmented_trajectory(
    rows: Iterable[MetricRow],
    *,
    segment_value,
    x_value,
    y_value,
) -> dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Aggregate curves without merging coincident distribution boundaries.

    The terminal evaluation of distribution ``d`` and the initial evaluation
    of ``d + 1`` intentionally share a cumulative environment-step value.
    They are different experimental measurements, so distribution remains in
    the grouping key.  Values are keyed by seed as a second guard against a
    duplicate log row being counted as an additional replicate.
    """

    points: dict[tuple[str, int, int, float], dict[int, float]] = {}
    for row in rows:
        segment = int(segment_value(row))
        x = float(x_value(row))
        y = float(y_value(row))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        points.setdefault(
            (row.method, row.diversity_n, segment, x), {}
        )[row.seed] = y

    curves: dict[tuple[str, int, int], list[tuple[float, float, float]]] = {}
    for (method, width, segment, x), seed_values in points.items():
        mean, sem, _ = _mean_sem(list(seed_values.values()))
        curves.setdefault((method, width, segment), []).append((x, mean, sem))

    result: dict[
        tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for key, curve in curves.items():
        curve.sort(key=lambda item: item[0])
        array = np.asarray(curve, dtype=float)
        result[key] = array[:, 0], array[:, 1], array[:, 2]
    return result


def _adjacent_boundary_connectors(
    curves: Mapping[
        tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
) -> list[tuple[str, int, float, float, float, float]]:
    """Return mean connectors between adjacent distribution segments."""

    by_series: dict[
        tuple[str, int],
        list[tuple[int, np.ndarray, np.ndarray]],
    ] = {}
    for (method, width, distribution), (x, mean, _sem) in curves.items():
        if x.size and mean.size:
            by_series.setdefault((method, width), []).append(
                (distribution, x, mean)
            )

    connectors: list[tuple[str, int, float, float, float, float]] = []
    for (method, width), segments in sorted(by_series.items()):
        segments.sort(key=lambda item: item[0])
        for (left_distribution, left_x, left_mean), (
            right_distribution,
            right_x,
            right_mean,
        ) in zip(segments, segments[1:]):
            if right_distribution != left_distribution + 1:
                continue
            connectors.append(
                (
                    method,
                    width,
                    float(left_x[-1]),
                    float(right_x[0]),
                    float(left_mean[-1]),
                    float(right_mean[0]),
                )
            )
    return connectors


def _style(widths: Sequence[int], methods: Sequence[str]):
    import matplotlib.pyplot as plt

    color_map = plt.get_cmap("viridis")
    denominator = max(1, len(widths) - 1)
    colors = {width: color_map(index / denominator) for index, width in enumerate(widths)}
    line_styles = {
        method: ("-" if index == 0 else "--" if index == 1 else ":")
        for index, method in enumerate(methods)
    }
    return colors, line_styles


def _label(method: str, width: int) -> str:
    method_name = {"ppo": "PPO", "ppo_cbp": "PPO + CBP"}.get(
        method, method.replace("_", " ").upper()
    )
    return f"{method_name}, {width}²"


def _boundaries(axis, config: TD2Config) -> None:
    per_distribution = float(config.ppo.total_timesteps)
    for distribution in range(1, config.experiment.num_distributions):
        axis.axvline(
            distribution * per_distribution / 1e6,
            color="0.78",
            linewidth=0.8,
            linestyle=(0, (3, 3)),
            zorder=0,
        )


def _atomic_save_figure(figure, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=220, bbox_inches="tight")
    os.replace(temporary, path)


def _line_plot(
    rows: Sequence[MetricRow],
    *,
    config: TD2Config,
    metric,
    ylabel: str,
    title: str,
    destination: Path,
    show_boundaries: bool = True,
    xlabel: str = "Continual environment steps (millions)",
) -> None:
    import matplotlib.pyplot as plt

    widths = sorted({row.diversity_n for row in rows})
    methods = sorted({row.method for row in rows})
    colors, line_styles = _style(widths, methods)
    curves = _aggregate_segmented_trajectory(
        rows,
        segment_value=lambda row: row.distribution,
        x_value=lambda row: _global_step(row, config),
        y_value=metric,
    )
    if not curves:
        raise PlotDataError(f"no finite values available for {title}")
    figure, axis = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    labelled: set[tuple[str, int]] = set()
    for (method, width, _distribution), (x, mean, sem) in sorted(curves.items()):
        x_millions = x / 1e6
        series = (method, width)
        axis.plot(
            x_millions,
            mean,
            color=colors[width],
            linestyle=line_styles[method],
            linewidth=1.8,
            label=_label(method, width) if series not in labelled else "_nolegend_",
        )
        labelled.add(series)
        axis.fill_between(
            x_millions,
            mean - sem,
            mean + sem,
            color=colors[width],
            alpha=0.16,
            linewidth=0,
        )
    for method, width, terminal_x, initial_x, terminal_mean, initial_mean in (
        _adjacent_boundary_connectors(curves)
    ):
        axis.plot(
            [terminal_x / 1e6, initial_x / 1e6],
            [terminal_mean, initial_mean],
            color=colors[width],
            linestyle=line_styles[method],
            linewidth=1.8,
            label="_nolegend_",
        )
    if show_boundaries:
        _boundaries(axis, config)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(axis="y", alpha=0.20)
    axis.legend(ncol=min(5, max(1, len(widths))), fontsize=8, frameon=False)
    if ylabel == "Evaluation success rate":
        axis.set_ylim(-0.02, 1.02)
    for suffix in (".png", ".pdf"):
        _atomic_save_figure(figure, destination.with_suffix(suffix))
    plt.close(figure)


def _seed_adaptation_summaries(
    rows: Sequence[MetricRow], *, threshold: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, int], list[MetricRow]] = {}
    for row in rows:
        if math.isfinite(row.adaptation_steps) and math.isfinite(row.success_rate):
            groups.setdefault(
                (row.method, row.diversity_n, row.seed, row.source_distribution), []
            ).append(row)
    results: list[dict[str, Any]] = []
    for (method, width, seed, source_distribution), group in sorted(groups.items()):
        by_step = {float(row.adaptation_steps): float(row.success_rate) for row in group}
        x = np.asarray(sorted(by_step), dtype=float)
        y = np.asarray([by_step[value] for value in x], dtype=float)
        if x.size == 0:
            continue
        duration = float(x[-1] - x[0])
        # Do not use ``getattr(..., np.trapz)``: Python eagerly evaluates the
        # default and NumPy 2.0+ removed that legacy attribute.
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(integrate(y, x) / duration) if duration > 0 else float(y[-1])
        crossing = _first_crossing(x, y, threshold)
        results.append(
            {
                "method": method,
                "diversity_n": width,
                "diversity_n2": width * width,
                "seed": seed,
                "source_distribution": source_distribution,
                "initial_success": float(y[0]),
                "final_success": float(y[-1]),
                "normalized_auc": auc,
                "time_to_threshold_steps": crossing,
                "threshold": threshold,
            }
        )
    return results


def _first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(y >= threshold)
    if indices.size == 0:
        return math.nan
    index = int(indices[0])
    if index == 0 or y[index] == y[index - 1]:
        return float(x[index])
    fraction = (threshold - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + np.clip(fraction, 0.0, 1.0) * (x[index] - x[index - 1]))


def _write_adaptation_summary(
    rows: Sequence[MetricRow], destination: Path, *, threshold: float
) -> None:
    seed_rows = _seed_adaptation_summaries(rows, threshold=threshold)
    if not seed_rows:
        raise PlotDataError("no finite adaptation trajectories were found")
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault(
            (row["method"], row["diversity_n"], row["source_distribution"]), []
        ).append(row)
    fields = [
        "method",
        "diversity_n",
        "diversity_n2",
        "source_distribution",
        "num_seeds",
        "initial_success_mean",
        "initial_success_sem",
        "final_success_mean",
        "final_success_sem",
        "normalized_auc_mean",
        "normalized_auc_sem",
        "threshold",
        "time_to_threshold_steps_mean",
        "time_to_threshold_steps_sem",
        "threshold_reached_fraction",
    ]
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (method, width, distribution), group in sorted(groups.items()):
            initial = _mean_sem([float(row["initial_success"]) for row in group])
            final = _mean_sem([float(row["final_success"]) for row in group])
            auc = _mean_sem([float(row["normalized_auc"]) for row in group])
            times = [float(row["time_to_threshold_steps"]) for row in group]
            time_stats = _mean_sem(times)
            reached = sum(math.isfinite(value) for value in times)
            writer.writerow(
                {
                    "method": method,
                    "diversity_n": width,
                    "diversity_n2": width * width,
                    "source_distribution": distribution,
                    "num_seeds": len(group),
                    "initial_success_mean": initial[0],
                    "initial_success_sem": initial[1],
                    "final_success_mean": final[0],
                    "final_success_sem": final[1],
                    "normalized_auc_mean": auc[0],
                    "normalized_auc_sem": auc[1],
                    "threshold": threshold,
                    "time_to_threshold_steps_mean": time_stats[0],
                    "time_to_threshold_steps_sem": time_stats[1],
                    "threshold_reached_fraction": reached / len(group),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _adaptation_plot(rows: Sequence[MetricRow], destination: Path) -> None:
    import matplotlib.pyplot as plt

    distributions = sorted({row.source_distribution for row in rows})
    if not distributions:
        raise PlotDataError("no adaptation source distributions were found")
    widths = sorted({row.diversity_n for row in rows})
    methods = sorted({row.method for row in rows})
    colors, line_styles = _style(widths, methods)
    columns = 2
    plot_rows = math.ceil(len(distributions) / columns)
    figure, axes = plt.subplots(
        plot_rows,
        columns,
        figsize=(12.0, 3.35 * plot_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    handles = []
    labels = []
    for axis, distribution in zip(axes.flat, distributions):
        subset = [row for row in rows if row.source_distribution == distribution]
        curves = _aggregate_trajectory(
            subset,
            x_value=lambda row: row.adaptation_steps,
            y_value=lambda row: row.success_rate,
        )
        for (method, width), (x, mean, sem) in sorted(curves.items()):
            (line,) = axis.plot(
                x / 1e6,
                mean,
                color=colors[width],
                linestyle=line_styles[method],
                linewidth=1.65,
                label=_label(method, width),
            )
            axis.fill_between(
                x / 1e6,
                mean - sem,
                mean + sem,
                color=colors[width],
                alpha=0.14,
                linewidth=0,
            )
            if line.get_label() not in labels:
                handles.append(line)
                labels.append(line.get_label())
        axis.axhline(DEFAULT_THRESHOLD, color="0.65", linewidth=0.8, linestyle=":")
        axis.set_title(f"After training distribution {distribution}")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.20)
    for axis in axes.flat[len(distributions) :]:
        axis.set_visible(False)
    figure.supxlabel("Held-out adaptation steps (millions)")
    figure.supylabel("Held-out success rate")
    figure.suptitle("Adaptability from frozen continual checkpoints", fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    for suffix in (".png", ".pdf"):
        _atomic_save_figure(figure, destination.with_suffix(suffix))
    plt.close(figure)


def _expected_widths(config: TD2Config) -> tuple[int, ...]:
    widths = tuple(math.isqrt(level) for level in config.experiment.diversity_levels)
    if any(width * width != level for width, level in zip(widths, config.experiment.diversity_levels)):
        raise PlotDataError("configured diversity levels are not perfect squares")
    return widths


def _validate_coverage(
    rows: Sequence[MetricRow], config: TD2Config, *, mode: str
) -> None:
    """Refuse a polished aggregate that silently omits completed streams."""

    actual = {
        (
            row.method,
            row.diversity_n,
            row.seed,
            row.source_distribution if mode == "adapt" else row.distribution,
        )
        for row in rows
    }
    if mode == "adapt":
        stages = config.experiment.adaptation.after_distributions
    elif mode in {"fixed_eval", "fixed_2b_eval", "fixed_2b_fresh_eval"}:
        stages = (1,)
    else:
        stages = tuple(range(1, config.experiment.num_distributions + 1))
    expected = {
        (method, width, seed, stage)
        for method in config.experiment.methods
        for width in _expected_widths(config)
        for seed in config.runtime.seeds
        for stage in stages
    }
    missing = sorted(expected - actual)
    if missing:
        preview = ", ".join(
            f"{method}/n={width}/seed={seed}"
            + (
                f"/stage={stage}"
                if mode == "adapt"
                else f"/distribution={stage}"
            )
            for method, width, seed, stage in missing[:8]
        )
        suffix = " ..." if len(missing) > 8 else ""
        raise PlotDataError(
            f"{mode} metrics are incomplete ({len(missing)} missing streams): "
            f"{preview}{suffix}"
        )
    unexpected = sorted(actual - expected)
    if unexpected:
        preview = ", ".join(
            f"{method}/n={width}/seed={seed}"
            + (
                f"/stage={stage}"
                if mode == "adapt"
                else f"/distribution={stage}"
            )
            for method, width, seed, stage in unexpected[:8]
        )
        suffix = " ..." if len(unexpected) > 8 else ""
        raise PlotDataError(
            f"{mode} metrics contain {len(unexpected)} unexpected streams: "
            f"{preview}{suffix}"
        )


def _validate_fixed_2b_terminal_updates(
    rows: Sequence[MetricRow], config: TD2Config
) -> None:
    """Require every fixed-2B stream to end at its exact planned update."""

    _validate_coverage(rows, config, mode="fixed_2b_eval")
    target = 2 * config.derived.total_training_updates_per_stream
    terminal_updates: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row.method, row.diversity_n, row.seed)
        terminal_updates[key] = max(terminal_updates.get(key, -1), row.global_update)

    wrong = sorted(
        (*key, update)
        for key, update in terminal_updates.items()
        if update != target
    )
    if wrong:
        preview = ", ".join(
            f"{method}/n={width}/seed={seed}: {update}"
            for method, width, seed, update in wrong[:8]
        )
        suffix = " ..." if len(wrong) > 8 else ""
        raise PlotDataError(
            "fixed_2b_eval metrics have incorrect terminal global_update "
            f"(expected {target}): {preview}{suffix}"
        )


def create_fixed_2b_plot(
    config_path: str | Path,
    source_root: str | Path,
    extension_root: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Path]:
    """Plot the original fixed 1B trajectory plus its 1B-to-2B extension."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    config = load_config(config_path)
    source_rows = read_metric_rows(source_root)
    extension_rows = read_metric_rows(extension_root)

    source_evaluation = [
        row
        for row in source_rows
        if row.mode == "eval"
        and row.distribution == 1
        and math.isfinite(row.success_rate)
    ]
    source_fixed = [
        row
        for row in source_rows
        if row.mode == "fixed_eval" and math.isfinite(row.success_rate)
    ]
    extension_fixed = [
        row
        for row in extension_rows
        if row.mode == "fixed_2b_eval" and math.isfinite(row.success_rate)
    ]
    _validate_coverage(source_fixed, config, mode="fixed_eval")
    _validate_fixed_2b_terminal_updates(extension_fixed, config)

    plot_dir = Path(extension_root).expanduser().resolve() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    destination = plot_dir / "fixed_distribution_evaluation_success_2b"
    _line_plot(
        [*source_evaluation, *source_fixed, *extension_fixed],
        config=config,
        metric=lambda row: row.success_rate,
        ylabel="Evaluation success rate",
        title=(
            "Fixed-distribution evaluation through 2B steps "
            "(LR horizon extended at 1B)"
        ),
        destination=destination,
        show_boundaries=False,
        xlabel="Fixed-distribution environment steps (millions)",
    )
    return [destination.with_suffix(".png"), destination.with_suffix(".pdf")]


def create_fixed_2b_fresh_plot(
    config_path: str | Path,
    output_root: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Path]:
    """Plot a clean fixed-distribution 2B run trained from initialization."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    config = load_config(config_path)
    rows = [
        row
        for row in read_metric_rows(output_root)
        if row.mode == "fixed_2b_fresh_eval" and math.isfinite(row.success_rate)
    ]
    _validate_fixed_2b_fresh_terminal_updates(rows, config)

    plot_dir = Path(output_root).expanduser().resolve() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    destination = plot_dir / "fixed_distribution_evaluation_success_2b_fresh"
    _line_plot(
        rows,
        config=config,
        metric=lambda row: row.success_rate,
        ylabel="Evaluation success rate",
        title="Fixed-distribution evaluation through 2B steps (clean schedule)",
        destination=destination,
        show_boundaries=False,
        xlabel="Fixed-distribution environment steps (millions)",
    )
    return [destination.with_suffix(".png"), destination.with_suffix(".pdf")]


def _validate_fixed_2b_fresh_terminal_updates(
    rows: Sequence[MetricRow], config: TD2Config
) -> int:
    """Require a complete, exactly-terminal clean-2B evaluation dataset."""

    _validate_coverage(rows, config, mode="fixed_2b_fresh_eval")
    target = 2 * config.derived.total_training_updates_per_stream
    terminal_updates: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row.method, row.diversity_n, row.seed)
        terminal_updates[key] = max(terminal_updates.get(key, -1), row.global_update)
    wrong = sorted(
        (*key, update)
        for key, update in terminal_updates.items()
        if update != target
    )
    if wrong:
        preview = ", ".join(
            f"{method}/n={width}/seed={seed}: {update}"
            for method, width, seed, update in wrong[:8]
        )
        suffix = " ..." if len(wrong) > 8 else ""
        raise PlotDataError(
            "fixed_2b_fresh_eval metrics have incorrect terminal global_update "
            f"(expected {target}): {preview}{suffix}"
        )
    return target


def _validate_fixed_1b_terminal_updates(
    rows: Sequence[MetricRow], config: TD2Config
) -> int:
    """Require a complete, exactly-terminal original fixed-1B dataset."""

    _validate_coverage(rows, config, mode="fixed_eval")
    target = config.derived.total_training_updates_per_stream
    terminal_updates: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row.method, row.diversity_n, row.seed)
        terminal_updates[key] = max(terminal_updates.get(key, -1), row.global_update)
    wrong = sorted(
        (*key, update)
        for key, update in terminal_updates.items()
        if update != target
    )
    if wrong:
        preview = ", ".join(
            f"{method}/n={width}/seed={seed}: {update}"
            for method, width, seed, update in wrong[:8]
        )
        suffix = " ..." if len(wrong) > 8 else ""
        raise PlotDataError(
            "fixed_eval metrics have incorrect terminal global_update "
            f"(expected {target}): {preview}{suffix}"
        )
    return target


def create_gru_width_terminal_plot(
    config_path: str | Path,
    run_roots: Mapping[int, str | Path],
    output_root: str | Path,
) -> list[Path]:
    """Compare exact terminal clean-2B success across recurrent widths."""

    config = load_config(config_path)
    if len(config.experiment.methods) != 1:
        raise PlotDataError(
            "GRU-width comparison requires exactly one configured method"
        )
    if not run_roots:
        raise PlotDataError("GRU-width comparison requires at least one run root")

    expected_levels = tuple(config.experiment.diversity_levels)
    expected_seeds = tuple(config.runtime.seeds)
    method = config.experiment.methods[0]
    summaries: dict[int, list[tuple[int, float, float, int]]] = {}

    for hidden_dim, run_root in sorted(run_roots.items()):
        if hidden_dim <= 0:
            raise PlotDataError(f"invalid GRU hidden width: {hidden_dim}")
        rows = [
            row
            for row in read_metric_rows(run_root)
            if row.mode == "fixed_2b_fresh_eval"
            and row.distribution == 1
            and math.isfinite(row.success_rate)
        ]
        target = _validate_fixed_2b_fresh_terminal_updates(rows, config)
        terminal: dict[tuple[int, int], float] = {}
        for row in rows:
            if row.method == method and row.global_update == target:
                terminal[(row.diversity_n2, row.seed)] = row.success_rate

        curve: list[tuple[int, float, float, int]] = []
        for level in expected_levels:
            values = [terminal[(level, seed)] for seed in expected_seeds]
            mean, sem, count = _mean_sem(values)
            curve.append((level, mean, sem, count))
        summaries[hidden_dim] = curve

    plot_dir = Path(output_root).expanduser().resolve() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    destination = plot_dir / "gru_width_terminal_evaluation_success_2b"

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, max(1, len(summaries)))
    )
    for color, (hidden_dim, curve) in zip(colors, sorted(summaries.items())):
        x = np.asarray([point[0] for point in curve], dtype=float)
        mean = np.asarray([point[1] for point in curve], dtype=float)
        sem = np.asarray([point[2] for point in curve], dtype=float)
        axis.errorbar(
            x,
            mean,
            yerr=sem,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label=f"GRU width {hidden_dim}",
        )

    axis.set_xscale("log", base=4)
    axis.set_xticks(expected_levels)
    axis.set_xticklabels(
        [f"{math.isqrt(level)}²" for level in expected_levels]
    )
    axis.set(
        xlabel="Training-task diversity (number of tasks; log₄ scale)",
        ylabel="Evaluation success rate at 2B steps",
        title="Terminal evaluation performance after 2B steps",
        ylim=(-0.02, 1.02),
    )
    axis.grid(axis="y", alpha=0.20)
    axis.legend(frameon=False)
    for suffix in (".png", ".pdf"):
        _atomic_save_figure(figure, destination.with_suffix(suffix))
    plt.close(figure)

    summary_path = destination.with_suffix(".csv")
    temporary = summary_path.with_name(
        f"{summary_path.stem}.tmp{summary_path.suffix}"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "gru_hidden_dim",
            "diversity_n",
            "diversity_n2",
            "mean_success_rate",
            "sem_success_rate",
            "num_seeds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for hidden_dim, curve in sorted(summaries.items()):
            for level, mean, sem, count in curve:
                writer.writerow(
                    {
                        "gru_hidden_dim": hidden_dim,
                        "diversity_n": math.isqrt(level),
                        "diversity_n2": level,
                        "mean_success_rate": mean,
                        "sem_success_rate": sem,
                        "num_seeds": count,
                    }
                )
    os.replace(temporary, summary_path)
    return [
        destination.with_suffix(".png"),
        destination.with_suffix(".pdf"),
        summary_path,
    ]


def create_fixed_horizon_terminal_plot(
    config_path: str | Path,
    one_b_root: str | Path,
    two_b_root: str | Path,
    output_root: str | Path,
) -> list[Path]:
    """Compare GRU-256 terminal success after independent 1B and 2B runs."""

    config = load_config(config_path)
    if len(config.experiment.methods) != 1:
        raise PlotDataError(
            "fixed-horizon comparison requires exactly one configured method"
        )
    if config.network.recurrent_hidden_dim != 256:
        raise PlotDataError(
            "fixed-horizon comparison requires a configured GRU width of 256"
        )

    expected_levels = tuple(config.experiment.diversity_levels)
    expected_seeds = tuple(config.runtime.seeds)
    method = config.experiment.methods[0]
    run_specs = (
        (1, Path(one_b_root), "fixed_eval", _validate_fixed_1b_terminal_updates),
        (
            2,
            Path(two_b_root),
            "fixed_2b_fresh_eval",
            _validate_fixed_2b_fresh_terminal_updates,
        ),
    )
    summaries: dict[int, tuple[int, list[tuple[int, float, float, int]]]] = {}

    for horizon, run_root, mode, validate in run_specs:
        rows = [
            row
            for row in read_metric_rows(run_root)
            if row.mode == mode
            and row.distribution == 1
            and math.isfinite(row.success_rate)
        ]
        target = validate(rows, config)
        terminal: dict[tuple[int, int], float] = {}
        for row in rows:
            if row.method == method and row.global_update == target:
                terminal[(row.diversity_n2, row.seed)] = row.success_rate

        curve: list[tuple[int, float, float, int]] = []
        for level in expected_levels:
            values = [terminal[(level, seed)] for seed in expected_seeds]
            mean, sem, count = _mean_sem(values)
            curve.append((level, mean, sem, count))
        summaries[horizon] = (target, curve)

    plot_dir = Path(output_root).expanduser().resolve() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    destination = plot_dir / "gru256_terminal_evaluation_success_1b_vs_2b"

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.18, 0.72, max(1, len(summaries)))
    )
    for color, (horizon, (_target, curve)) in zip(
        colors, sorted(summaries.items())
    ):
        x = np.asarray([point[0] for point in curve], dtype=float)
        mean = np.asarray([point[1] for point in curve], dtype=float)
        sem = np.asarray([point[2] for point in curve], dtype=float)
        axis.errorbar(
            x,
            mean,
            yerr=sem,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label=f"{horizon}B-step schedule",
        )

    axis.set_xscale("log", base=4)
    axis.set_xticks(expected_levels)
    axis.set_xticklabels(
        [f"{math.isqrt(level)}²" for level in expected_levels]
    )
    axis.set(
        xlabel="Training-task diversity (number of tasks; log₄ scale)",
        ylabel="Evaluation success rate at terminal checkpoint",
        title="Terminal evaluation: 1B vs 2B steps (GRU width 256)",
        ylim=(-0.02, 1.02),
    )
    axis.grid(axis="y", alpha=0.20)
    axis.legend(frameon=False)
    for suffix in (".png", ".pdf"):
        _atomic_save_figure(figure, destination.with_suffix(suffix))
    plt.close(figure)

    summary_path = destination.with_suffix(".csv")
    temporary = summary_path.with_name(
        f"{summary_path.stem}.tmp{summary_path.suffix}"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "training_steps_billions",
            "gru_hidden_dim",
            "diversity_n",
            "diversity_n2",
            "mean_success_rate",
            "sem_success_rate",
            "num_seeds",
            "terminal_global_update",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for horizon, (target, curve) in sorted(summaries.items()):
            for level, mean, sem, count in curve:
                writer.writerow(
                    {
                        "training_steps_billions": horizon,
                        "gru_hidden_dim": 256,
                        "diversity_n": math.isqrt(level),
                        "diversity_n2": level,
                        "mean_success_rate": mean,
                        "sem_success_rate": sem,
                        "num_seeds": count,
                        "terminal_global_update": target,
                    }
                )
    os.replace(temporary, summary_path)
    return [
        destination.with_suffix(".png"),
        destination.with_suffix(".pdf"),
        summary_path,
    ]


def _core_aux_expected_source(width: int, seed: int) -> str:
    return (
        "core_aux_2b/ppo_cbp/"
        f"aux_diversity_{width * width}/seed_{seed}/metrics.csv"
    )


def _validate_core_aux_2b_rows(
    rows: Sequence[MetricRow], config: TD2Config
) -> None:
    """Require all 25 exact core/aux streams at their 2B checkpoint."""

    if config.network.recurrent_hidden_dim != 256:
        raise PlotDataError("core/aux 2B plots require a configured GRU width of 256")

    expected = {
        (CORE_AUX_METHOD, width, seed)
        for width in CORE_AUX_WIDTHS
        for seed in CORE_AUX_SEEDS
    }
    actual = {(row.method, row.diversity_n, row.seed) for row in rows}
    missing = sorted(expected - actual)
    if missing:
        preview = ", ".join(
            f"{method}/aux-n={width}/seed={seed}"
            for method, width, seed in missing[:8]
        )
        suffix = " ..." if len(missing) > 8 else ""
        raise PlotDataError(
            "core_aux_2b_eval metrics are incomplete "
            f"({len(missing)} missing streams): {preview}{suffix}"
        )
    unexpected = sorted(actual - expected)
    if unexpected:
        preview = ", ".join(
            f"{method}/aux-n={width}/seed={seed}"
            for method, width, seed in unexpected[:8]
        )
        suffix = " ..." if len(unexpected) > 8 else ""
        raise PlotDataError(
            "core_aux_2b_eval metrics contain "
            f"{len(unexpected)} unexpected streams: {preview}{suffix}"
        )

    invalid_rows = [
        row
        for row in rows
        if row.mode != CORE_AUX_EVAL_MODE
        or row.distribution != 1
        or row.diversity_n2 != row.diversity_n * row.diversity_n
        or row.source != _core_aux_expected_source(row.diversity_n, row.seed)
    ]
    if invalid_rows:
        row = invalid_rows[0]
        raise PlotDataError(
            "core_aux_2b_eval row violates the expected stream layout or identity: "
            f"{row.source}"
        )

    terminal_updates: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (row.diversity_n, row.seed)
        terminal_updates[key] = max(
            terminal_updates.get(key, -1), row.global_update
        )
    wrong = sorted(
        (width, seed, update)
        for (width, seed), update in terminal_updates.items()
        if update != CORE_AUX_TERMINAL_UPDATE
    )
    if wrong:
        preview = ", ".join(
            f"aux-n={width}/seed={seed}: {update}"
            for width, seed, update in wrong[:8]
        )
        suffix = " ..." if len(wrong) > 8 else ""
        raise PlotDataError(
            "core_aux_2b_eval metrics have incorrect terminal global_update "
            f"(expected {CORE_AUX_TERMINAL_UPDATE}): {preview}{suffix}"
        )


def _core_aux_learning_summary(
    rows: Sequence[MetricRow], config: TD2Config
) -> list[dict[str, Any]]:
    by_stream: dict[tuple[int, int], dict[int, MetricRow]] = {}
    for row in rows:
        if row.global_update < 0 or not math.isfinite(row.success_rate):
            continue
        by_stream.setdefault((row.diversity_n, row.seed), {})[
            row.global_update
        ] = row

    reference_updates: tuple[int, ...] | None = None
    for key in sorted(by_stream):
        updates = tuple(sorted(by_stream[key]))
        if reference_updates is None:
            reference_updates = updates
        elif updates != reference_updates:
            raise PlotDataError(
                "core_aux_2b_eval streams do not share one evaluation-update grid; "
                f"first mismatch is aux-n={key[0]}/seed={key[1]}"
            )
    if not reference_updates:
        raise PlotDataError("core_aux_2b_eval metrics contain no finite trajectory")

    summary: list[dict[str, Any]] = []
    for width in CORE_AUX_WIDTHS:
        for update in reference_updates:
            point_rows = [
                by_stream[(width, seed)][update] for seed in CORE_AUX_SEEDS
            ]
            steps = np.asarray(
                [_global_step(row, config) for row in point_rows], dtype=float
            )
            if not np.all(np.isfinite(steps)) or not np.allclose(
                steps, steps[0], rtol=0.0, atol=0.5
            ):
                raise PlotDataError(
                    "core_aux_2b_eval streams disagree on environment steps at "
                    f"aux-n={width}, global_update={update}"
                )
            mean, sem, count = _mean_sem(
                [row.success_rate for row in point_rows]
            )
            summary.append(
                {
                    "auxiliary_diversity_n": width,
                    "auxiliary_diversity_n2": width * width,
                    "global_update": update,
                    "nominal_environment_steps": float(steps[0]),
                    "mean_core_success_rate": mean,
                    "sem_core_success_rate": sem,
                    "num_seeds": count,
                }
            )
    return summary


def _write_csv_rows(
    destination: Path,
    *,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = destination.with_name(
        f"{destination.stem}.tmp{destination.suffix}"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def create_core_aux_2b_plots(
    config_path: str | Path,
    output_root: str | Path,
) -> list[Path]:
    """Plot core retention while varying a disjoint auxiliary-task pool."""

    config = load_config(config_path)
    rows = [
        row
        for row in read_metric_rows(output_root)
        if row.mode == CORE_AUX_EVAL_MODE and math.isfinite(row.success_rate)
    ]
    _validate_core_aux_2b_rows(rows, config)
    learning = _core_aux_learning_summary(rows, config)

    plot_dir = Path(output_root).expanduser().resolve() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    learning_destination = plot_dir / "core_aux_2b_core_evaluation_success"
    terminal_destination = plot_dir / "core_aux_2b_terminal_core_success"

    import matplotlib.pyplot as plt

    colors = {
        width: color
        for width, color in zip(
            CORE_AUX_WIDTHS,
            plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(CORE_AUX_WIDTHS))),
        )
    }
    figure, axis = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    for width in CORE_AUX_WIDTHS:
        points = [
            row for row in learning if row["auxiliary_diversity_n"] == width
        ]
        x = np.asarray(
            [row["nominal_environment_steps"] for row in points], dtype=float
        ) / 1e6
        mean = np.asarray(
            [row["mean_core_success_rate"] for row in points], dtype=float
        )
        sem = np.asarray(
            [row["sem_core_success_rate"] for row in points], dtype=float
        )
        axis.plot(
            x,
            mean,
            color=colors[width],
            linewidth=1.9,
            label=f"Auxiliary {width}²",
        )
        axis.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=colors[width],
            alpha=0.16,
            linewidth=0,
        )
    axis.set(
        xlabel="Core–auxiliary training environment steps (millions)",
        ylabel="Core evaluation success rate",
        title="Core evaluation during 50/50 core–auxiliary training",
        ylim=(-0.02, 1.02),
    )
    axis.grid(axis="y", alpha=0.20)
    axis.legend(ncol=3, fontsize=8, frameon=False)
    for suffix in (".png", ".pdf"):
        _atomic_save_figure(figure, learning_destination.with_suffix(suffix))
    plt.close(figure)

    terminal = [
        row for row in learning if row["global_update"] == CORE_AUX_TERMINAL_UPDATE
    ]
    if len(terminal) != len(CORE_AUX_WIDTHS):
        raise PlotDataError("core/aux terminal summary is incomplete")
    x = np.asarray(
        [row["auxiliary_diversity_n2"] for row in terminal], dtype=float
    )
    mean = np.asarray(
        [row["mean_core_success_rate"] for row in terminal], dtype=float
    )
    sem = np.asarray(
        [row["sem_core_success_rate"] for row in terminal], dtype=float
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    axis.errorbar(
        x,
        mean,
        yerr=sem,
        color=plt.get_cmap("viridis")(0.42),
        marker="o",
        markersize=5,
        linewidth=2,
        capsize=3,
    )
    levels = tuple(width * width for width in CORE_AUX_WIDTHS)
    axis.set_xscale("log", base=4)
    axis.set_xticks(levels)
    axis.set_xticklabels([f"{width}²" for width in CORE_AUX_WIDTHS])
    axis.set(
        xlabel="Auxiliary-task diversity (number of tasks; log₄ scale)",
        ylabel="Core evaluation success rate at 2B steps",
        title="Terminal core performance versus auxiliary-task diversity",
        ylim=(-0.02, 1.02),
    )
    axis.grid(axis="y", alpha=0.20)
    for suffix in (".png", ".pdf"):
        _atomic_save_figure(figure, terminal_destination.with_suffix(suffix))
    plt.close(figure)

    learning_csv = learning_destination.with_suffix(".csv")
    _write_csv_rows(
        learning_csv,
        fields=(
            "auxiliary_diversity_n",
            "auxiliary_diversity_n2",
            "global_update",
            "nominal_environment_steps",
            "mean_core_success_rate",
            "sem_core_success_rate",
            "num_seeds",
        ),
        rows=learning,
    )
    terminal_csv = terminal_destination.with_suffix(".csv")
    _write_csv_rows(
        terminal_csv,
        fields=(
            "auxiliary_diversity_n",
            "auxiliary_diversity_n2",
            "global_update",
            "nominal_environment_steps",
            "mean_core_success_rate",
            "sem_core_success_rate",
            "num_seeds",
        ),
        rows=terminal,
    )
    return [
        learning_destination.with_suffix(".png"),
        learning_destination.with_suffix(".pdf"),
        learning_csv,
        terminal_destination.with_suffix(".png"),
        terminal_destination.with_suffix(".pdf"),
        terminal_csv,
    ]


def create_plots(
    config_path: str | Path,
    output_root: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Path]:
    """Create all requested plots and the adaptation summary table."""

    config = load_config(config_path)
    rows = read_metric_rows(output_root)
    plot_dir = Path(output_root).expanduser().resolve() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    evaluation = [
        row for row in rows if row.mode == "eval" and math.isfinite(row.success_rate)
    ]
    training = [
        row for row in rows if row.mode == "train" and math.isfinite(row.episodic_return)
    ]
    adaptation = [
        row for row in rows if row.mode == "adapt" and math.isfinite(row.success_rate)
    ]
    shared_cutoff = float(config.ppo.total_timesteps)
    fixed_continuation = [
        row
        for row in rows
        if row.mode == "fixed_eval"
        and math.isfinite(row.success_rate)
        and _global_step(row, config) > shared_cutoff
    ]
    fixed_evaluation = [
        row
        for row in evaluation
        if row.distribution == 1 and _global_step(row, config) <= shared_cutoff
    ] + fixed_continuation
    _validate_coverage(evaluation, config, mode="eval")
    _validate_coverage(training, config, mode="train")
    _validate_coverage(adaptation, config, mode="adapt")
    _validate_coverage(fixed_continuation, config, mode="fixed_eval")
    _line_plot(
        evaluation,
        config=config,
        metric=lambda row: row.success_rate,
        ylabel="Evaluation success rate",
        title="Continual evaluation (Figure 6a protocol)",
        destination=plot_dir / "figure6a_evaluation_success",
    )
    _line_plot(
        training,
        config=config,
        metric=lambda row: row.episodic_return,
        ylabel="Training episodic return",
        title="Continual training reward",
        destination=plot_dir / "training_episodic_return",
    )
    _line_plot(
        fixed_evaluation,
        config=config,
        metric=lambda row: row.success_rate,
        ylabel="Evaluation success rate",
        title="Fixed-distribution evaluation (distribution 1)",
        destination=plot_dir / "fixed_distribution_evaluation_success",
        show_boundaries=False,
        xlabel="Fixed-distribution environment steps (millions)",
    )
    _adaptation_plot(adaptation, plot_dir / "heldout_adaptation")
    summary_path = plot_dir / "adaptation_summary.csv"
    _write_adaptation_summary(adaptation, summary_path, threshold=threshold)
    return [
        plot_dir / "figure6a_evaluation_success.png",
        plot_dir / "figure6a_evaluation_success.pdf",
        plot_dir / "training_episodic_return.png",
        plot_dir / "training_episodic_return.pdf",
        plot_dir / "fixed_distribution_evaluation_success.png",
        plot_dir / "fixed_distribution_evaluation_success.pdf",
        plot_dir / "heldout_adaptation.png",
        plot_dir / "heldout_adaptation.pdf",
        summary_path,
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--source-output-root",
        type=Path,
        help="completed 1B result root (required with --fixed-2b-only)",
    )
    parser.add_argument(
        "--fixed-2b-only",
        action="store_true",
        help="write only the combined fixed-distribution 0-to-2B plot",
    )
    parser.add_argument(
        "--fixed-2b-fresh-only",
        action="store_true",
        help="write only the clean fixed-distribution 2B plot",
    )
    parser.add_argument(
        "--gru-width-comparison",
        action="store_true",
        help="compare terminal clean-2B performance across GRU widths",
    )
    parser.add_argument(
        "--gru-width-run",
        action="append",
        nargs=2,
        metavar=("WIDTH", "OUTPUT_ROOT"),
        help="GRU width and completed clean-2B result root; repeat per width",
    )
    parser.add_argument(
        "--fixed-horizon-comparison",
        action="store_true",
        help="compare terminal GRU-256 performance after independent 1B and 2B runs",
    )
    parser.add_argument(
        "--core-aux-2b-only",
        action="store_true",
        help=(
            "write only the core-evaluation learning curve and terminal "
            "core-versus-auxiliary-diversity plot"
        ),
    )
    parser.add_argument(
        "--fixed-1b-root",
        type=Path,
        help="completed original GRU-256 fixed-distribution 1B result root",
    )
    parser.add_argument(
        "--fixed-2b-root",
        type=Path,
        help="completed clean GRU-256 fixed-distribution 2B result root",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be in [0, 1]")
    modes = sum(
        (
            args.fixed_2b_only,
            args.fixed_2b_fresh_only,
            args.gru_width_comparison,
            args.fixed_horizon_comparison,
            args.core_aux_2b_only,
        )
    )
    if modes > 1:
        raise SystemExit("choose only one specialized plotting mode")
    if not args.fixed_horizon_comparison and (
        args.fixed_1b_root is not None or args.fixed_2b_root is not None
    ):
        raise SystemExit(
            "--fixed-1b-root and --fixed-2b-root require "
            "--fixed-horizon-comparison"
        )
    if args.core_aux_2b_only:
        if args.source_output_root is not None or args.gru_width_run:
            raise SystemExit(
                "core/aux plotting does not use --source-output-root "
                "or --gru-width-run"
            )
        outputs = create_core_aux_2b_plots(args.config, args.output_root)
    elif args.fixed_horizon_comparison:
        if args.source_output_root is not None or args.gru_width_run:
            raise SystemExit(
                "fixed-horizon comparison does not use --source-output-root "
                "or --gru-width-run"
            )
        if args.fixed_1b_root is None or args.fixed_2b_root is None:
            raise SystemExit(
                "--fixed-horizon-comparison requires both --fixed-1b-root "
                "and --fixed-2b-root"
            )
        outputs = create_fixed_horizon_terminal_plot(
            args.config,
            args.fixed_1b_root,
            args.fixed_2b_root,
            args.output_root,
        )
    elif args.gru_width_comparison:
        if args.source_output_root is not None:
            raise SystemExit(
                "--source-output-root is not used with --gru-width-comparison"
            )
        if not args.gru_width_run:
            raise SystemExit(
                "--gru-width-comparison requires at least one --gru-width-run"
            )
        run_roots: dict[int, Path] = {}
        for width_text, root_text in args.gru_width_run:
            try:
                width = int(width_text)
            except ValueError as error:
                raise SystemExit(f"invalid GRU width: {width_text}") from error
            if width in run_roots:
                raise SystemExit(f"duplicate GRU width: {width}")
            run_roots[width] = Path(root_text)
        outputs = create_gru_width_terminal_plot(
            args.config, run_roots, args.output_root
        )
    elif args.fixed_2b_only:
        if args.source_output_root is None:
            raise SystemExit("--fixed-2b-only requires --source-output-root")
        outputs = create_fixed_2b_plot(
            args.config,
            args.source_output_root,
            args.output_root,
            threshold=args.threshold,
        )
    elif args.fixed_2b_fresh_only:
        if args.source_output_root is not None:
            raise SystemExit(
                "--source-output-root is not used with --fixed-2b-fresh-only"
            )
        outputs = create_fixed_2b_fresh_plot(
            args.config,
            args.output_root,
            threshold=args.threshold,
        )
    else:
        if args.gru_width_run:
            raise SystemExit("--gru-width-run requires --gru-width-comparison")
        if args.source_output_root is not None:
            raise SystemExit("--source-output-root requires --fixed-2b-only")
        outputs = create_plots(args.config, args.output_root, threshold=args.threshold)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
