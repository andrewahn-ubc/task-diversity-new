import csv

from matplotlib.figure import Figure
import pytest

from td2_kinetix.config import PROTOCOL
from td2_kinetix.plot import heldout_plot


def test_heldout_plot_uses_episode_success_rate(tmp_path, monkeypatch):
    output = tmp_path / "results"
    for diversity in PROTOCOL.diversity_levels:
        for seed in PROTOCOL.learner_seeds:
            for stage in range(PROTOCOL.num_distributions):
                root = (
                    output
                    / "heldout"
                    / f"diversity_{diversity}"
                    / f"seed_{seed}"
                    / f"stage_{stage}"
                )
                root.mkdir(parents=True)
                with (root / "metrics.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            "row_type",
                            "adaptation_steps",
                            "success_rate",
                            "task_success_rate",
                        ),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "row_type": "eval",
                            "adaptation_steps": 0,
                            "success_rate": 0.2 + seed * 0.1,
                            "task_success_rate": 0.9,
                        }
                    )
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    plots = output / "plots"
    plots.mkdir(parents=True)
    heldout_plot(output, plots)
    with (plots / "kinetix_heldout_adaptation_success_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        first = next(csv.DictReader(handle))
    assert float(first["mean_success_rate"]) == pytest.approx(0.3)
    assert float(first["mean_task_success_rate"]) == pytest.approx(0.9)
