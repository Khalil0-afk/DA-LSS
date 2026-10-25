import csv
from pathlib import Path

import numpy as np


def generate_raw(path, seed=2026, cycles=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "battery_id", "cycle", "time_s", "voltage_v", "current_a", "soh"])
        for domain, count, shift in (("synthetic_source", 10, 0), ("synthetic_target", 8, 0.025)):
            for battery in range(count):
                rate = rng.uniform(0.0007, 0.0013)
                offset = rng.normal(0, 0.005)
                for cycle in range(1, cycles + 1):
                    health = 1 - rate * cycle - 0.025 * (cycle / cycles) ** 3
                    time = np.arange(32)
                    voltage = 3.2 + 0.15 * time / 31 + 0.3 * (1 - health) + shift + offset + rng.normal(0, 0.0015, 32)
                    current = 1.5 - 0.35 * (1 - health) - 0.1 * time / 31 + shift + rng.normal(0, 0.003, 32)
                    for t, v, c in zip(time, voltage, current):
                        writer.writerow([domain, f"cell_{battery:02d}", cycle, t, v, c, health])


def plot_run(run, features):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import load_features
    import json

    run = Path(run)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    with (run / "target_test_predictions.csv").open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    batteries = {(b.domain, b.battery_id): b for b in load_features(features)}
    keys = sorted({(r["domain"], r["battery_id"]) for r in records})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot([r["epoch"] for r in history], [r["source_val_rmse"] for r in history], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Source validation RMSE (SOH fraction)", title="Model selection: source only")
    for i, key in enumerate(keys):
        rows = [r for r in records if (r["domain"], r["battery_id"]) == key]
        cycle = [float(r["cycle"]) for r in rows]
        pred = [float(r["prediction"]) for r in rows]
        b = batteries[key]
        axes[1].plot(cycle, pred, color=f"C{i}", label=f"{key[1]} prediction")
        if b.soh is not None:
            lookup = dict(zip(b.cycles, b.soh))
            axes[1].plot(cycle, [lookup[c] for c in cycle], "--", color=f"C{i}", label=f"{key[1]} reference")
    axes[1].set(xlabel="Cycle", ylabel="SOH (fraction)", title="Held-out target batteries")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("SYNTHETIC DEMO — pipeline verification, not paper results")
    fig.savefig(run / "synthetic_demo.png", dpi=180)
    fig.savefig(run / "synthetic_demo.svg")
    plt.close(fig)
