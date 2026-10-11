from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from td2.plot import (
    _adjacent_boundary_connectors,
    _aggregate_segmented_trajectory,
    _first_crossing,
    _seed_adaptation_summaries,
    create_plots,
    read_metric_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "smoke.yaml"
FIELDS = [
    "method",
    "diversity_n",
    "diversity_n2",
    "seed",
    "mode",
    "distribution",
    "environment_steps",
    "success_rate",
    "episodic_return",
    "adaptation_steps",
    "source_distribution",
    "nominal_environment_steps",
]


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "method": "ppo",
        "diversity_n": 1,
        "diversity_n2": 1,
        "seed": 0,
        "mode": "eval",
        "distribution": 1,
        "environment_steps": 0,
        "success_rate": 0.1,
        "episodic_return": "",
        "adaptation_steps": "",
        "source_distribution": "",
        "nominal_environment_steps": 0,
    }
    base.update(overrides)
    return base


def test_first_crossing_interpolates_and_reports_missing() -> None:
    x = np.asarray([0.0, 10.0, 20.0])
    y = np.asarray([0.2, 0.7, 0.9])
    assert _first_crossing(x, y, 0.8) == pytest.approx(15.0)
    assert math.isnan(_first_crossing(x, y, 0.95))


def test_metric_reader_deduplicates_resumed_last_row(tmp_path: Path) -> None:
    path = tmp_path / "run" / "metrics.csv"
    _write_metrics(
        path,
        [
            _row(success_rate=0.1),
            _row(success_rate=0.4),  # same committed point after resume
            _row(nominal_environment_steps=100, environment_steps=100, success_rate=0.5),
        ],
    )
    rows = read_metric_rows(tmp_path)
    assert len(rows) == 2
    assert rows[0].success_rate == pytest.approx(0.4)


def test_metric_reader_excludes_hidden_smoke_runs(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "run" / "metrics.csv", [_row(success_rate=0.4)])
    _write_metrics(
        tmp_path / ".smoke" / "job_1" / "metrics.csv",
        [_row(nominal_environment_steps=32, success_rate=0.99)],
    )
    rows = read_metric_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0].success_rate == pytest.approx(0.4)


def test_distribution_boundary_points_remain_separate_seed_aggregates(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for seed, terminal, initial in ((0, 0.8, 0.2), (1, 0.6, 0.4)):
        rows.extend(
            [
                _row(
                    seed=seed,
                    distribution=1,
                    environment_steps=100,
                    nominal_environment_steps=100,
                    success_rate=terminal,
                ),
                _row(
                    seed=seed,
                    distribution=2,
                    environment_steps=100,
                    nominal_environment_steps=100,
                    success_rate=initial,
                ),
            ]
        )
    _write_metrics(tmp_path / "metrics.csv", rows)

    curves = _aggregate_segmented_trajectory(
        read_metric_rows(tmp_path),
        segment_value=lambda row: row.distribution,
        x_value=lambda row: row.nominal_environment_steps,
        y_value=lambda row: row.success_rate,
    )

    _, terminal_mean, terminal_sem = curves[("ppo", 1, 1)]
    _, initial_mean, initial_sem = curves[("ppo", 1, 2)]
    assert terminal_mean.tolist() == pytest.approx([0.7])
    assert initial_mean.tolist() == pytest.approx([0.3])
    assert terminal_sem.tolist() == pytest.approx([0.1])
    assert initial_sem.tolist() == pytest.approx([0.1])
    connectors = _adjacent_boundary_connectors(curves)
    assert len(connectors) == 1
    assert connectors[0][:2] == ("ppo", 1)
    assert connectors[0][2:] == pytest.approx((100.0, 100.0, 0.7, 0.3))

    # Production has a small batching remainder between the terminal point
    # and the next nominal 100M boundary; that connector should be diagonal.
    offset_curves = dict(curves)
    right_x, right_mean, right_sem = offset_curves[("ppo", 1, 2)]
    offset_curves[("ppo", 1, 2)] = (right_x + 123_136, right_mean, right_sem)
    offset = _adjacent_boundary_connectors(offset_curves)
    assert offset[0][2:] == pytest.approx((100.0, 123_236.0, 0.7, 0.3))


def test_adaptation_summary_uses_seed_trajectories(tmp_path: Path) -> None:
    metric_rows = []
    for seed, values in ((0, (0.2, 0.9)), (1, (0.4, 0.7))):
        for step, value in zip((0, 100), values):
            metric_rows.append(
                _row(
                    mode="adapt_eval",
                    seed=seed,
                    distribution=2,
                    source_distribution=2,
                    adaptation_steps=step,
                    success_rate=value,
                    nominal_environment_steps="",
                )
            )
    _write_metrics(tmp_path / "metrics.csv", metric_rows)
    summaries = _seed_adaptation_summaries(read_metric_rows(tmp_path), threshold=0.8)
    assert len(summaries) == 2
    assert summaries[0]["initial_success"] == pytest.approx(0.2)
    assert summaries[0]["final_success"] == pytest.approx(0.9)
    assert summaries[0]["normalized_auc"] == pytest.approx(0.55)
    assert math.isfinite(float(summaries[0]["time_to_threshold_steps"]))
    assert math.isnan(float(summaries[1]["time_to_threshold_steps"]))


def test_create_plots_writes_all_requested_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows: list[dict[str, object]] = []
    for method in ("ppo", "ppo_cbp"):
        for distribution in (1, 2):
            global_start = (distribution - 1) * 256
            for step, success in ((0, 0.1 * distribution), (256, 0.2 * distribution)):
                rows.append(
                    _row(
                        method=method,
                        mode="eval",
                        distribution=distribution,
                        environment_steps=global_start + step,
                        nominal_environment_steps=global_start + step,
                        success_rate=success,
                    )
                )
                rows.append(
                    _row(
                        method=method,
                        mode="train",
                        distribution=distribution,
                        environment_steps=global_start + step,
                        nominal_environment_steps=global_start + step,
                        success_rate="",
                        episodic_return=success - 0.5,
                    )
                )
            for step, success in ((0, 0.1), (128, 0.85)):
                rows.append(
                    _row(
                        method=method,
                        mode="adapt_eval",
                        distribution=distribution,
                        source_distribution=distribution,
                        adaptation_steps=step,
                        nominal_environment_steps="",
                        success_rate=success,
                    )
                )
    _write_metrics(tmp_path / "runs" / "metrics.csv", rows)

    outputs = create_plots(SMOKE_CONFIG, tmp_path)
    assert len(outputs) == 7
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    summary = outputs[-1].read_text(encoding="utf-8")
    assert "normalized_auc_mean" in summary
    assert "time_to_threshold_steps_mean" in summary
