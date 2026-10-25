import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_features
from .experiment import load_checkpoint, metrics, write_json


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def join_predictions(path, batteries):
    labels = {(b.domain, b.battery_id, float(c)): float(y)
              for b in batteries if b.soh is not None for c, y in zip(b.cycles, b.soh)}
    groups = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = r["domain"], r["battery_id"], float(r["cycle"])
            if key not in labels:
                continue
            groups.setdefault(f"{key[0]}/{key[1]}", []).append((key[2], labels[key], float(r["prediction"])))
    return {k: np.array(sorted(v)) for k, v in groups.items()}


def save_figure(fig, output, name):
    fig.savefig(Path(output) / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(Path(output) / f"{name}.svg", bbox_inches="tight")


def training_report(run, features, output):
    plt = plotting()
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    groups = join_predictions(run / "target_test_predictions.csv", load_features(features))
    if not groups:
        raise ValueError("Reference labels required for a result report")
    scores = {key: metrics(rows[:, 1], rows[:, 2]) for key, rows in groups.items()}
    worst = sorted(scores, key=lambda key: scores[key]["rmse"], reverse=True)[:3]
    write_json(output / "challenging_cases.json", {"selection": "post-hoc highest RMSE; never fed into training/tuning",
                                                   "batteries": worst, "per_battery": scores})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["task"] for r in history], label="Source MSE")
    axes[0].set(xlabel="Epoch", ylabel="MSE (SOH fraction squared)", title="Supervised training")
    axes[1].plot(epochs, [r["source_val_rmse"] for r in history], label="Source validation RMSE")
    axes[1].set(xlabel="Epoch", ylabel="RMSE (SOH fraction)", title="Source-only checkpoint selection")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    save_figure(fig, output, "training")
    plt.close(fig)
    fig, axes = plt.subplots(len(worst), 2, figsize=(10, 3 * len(worst)), squeeze=False, constrained_layout=True)
    for row, key in zip(axes, worst):
        data = groups[key]
        row[0].plot(data[:, 0], data[:, 1], label="Reference")
        row[0].plot(data[:, 0], data[:, 2], label="Prediction")
        row[0].set(title=key, xlabel="Cycle", ylabel="SOH fraction")
        row[0].legend()
        row[1].plot(data[:, 0], data[:, 2] - data[:, 1])
        row[1].axhline(0, color="black", linewidth=0.6)
        row[1].set(xlabel="Cycle", ylabel="Prediction - reference", title=f"RMSE = {scores[key]['rmse']:.5f}")
    save_figure(fig, output, "challenging_cases")
    plt.close(fig)
    all_rows = np.concatenate(list(groups.values()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].scatter(all_rows[:, 1], all_rows[:, 2], s=6, alpha=0.5)
    low, high = float(all_rows[:, 1:].min()), float(all_rows[:, 1:].max())
    axes[0].plot([low, high], [low, high], "k--", linewidth=0.8)
    axes[0].set(xlabel="Reference SOH", ylabel="Predicted SOH", title="Held-out target predictions")
    axes[1].hist(all_rows[:, 2] - all_rows[:, 1], bins=30)
    axes[1].set(xlabel="Prediction - reference", ylabel="Window count", title="Residual distribution")
    save_figure(fig, output, "residuals")
    plt.close(fig)
    return scores


@torch.no_grad()
def hidden_report(checkpoint, features, output, battery_key, device="cpu"):
    plt = plotting()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, scaler, cfg = load_checkpoint(checkpoint, device)
    cells = {b.key: b for b in load_features(features)}
    if battery_key not in cells or len(cells[battery_key].cycles) < cfg.sequence_length:
        raise ValueError("Select a battery with one complete window")
    b = cells[battery_key]
    x = scaler.transform(b.features[-cfg.sequence_length:])
    hidden, _, _ = model.encode(torch.as_tensor(x[None], device=device), cfg.use_gate and not cfg.source_only)
    vectors = hidden[0].cpu().numpy().astype(float)
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.maximum(norms, 1e-12)
    corr = np.clip(unit @ unit.T, -1, 1)
    np.save(output / "hidden.npy", vectors)
    np.save(output / "temporal_correlation.npy", corr)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
    ax.set(xlabel="Window position", ylabel="Window position", title=f"{model.cfg.backbone}: {battery_key}")
    fig.colorbar(im, ax=ax, label="Pearson correlation across state channels")
    save_figure(fig, output, "temporal_correlation")
    plt.close(fig)
    write_json(output / "metadata.json", {"battery": battery_key, "backbone": model.cfg.backbone,
                                           "window": "last complete input window", "reference_soh_used": False,
                                           "zero_variance_rows": int((norms < 1e-12).sum()),
                                           "interpretation": "descriptive correlation, not proof of long-range causal dependence"})


def group_score(predictions, features, grouping, output):
    grouping = json.loads(Path(grouping).read_text(encoding="utf-8"))
    names = {key: name for domain in grouping["domains"].values() for key, name in domain["groups"].items()}
    joined = join_predictions(predictions, load_features(features))
    groups = {}
    for key, data in joined.items():
        if key not in names:
            raise ValueError(f"No retrospective grouping for {key}")
        groups.setdefault(names[key], []).append(data)
    result = {}
    for group, parts in groups.items():
        rows = np.concatenate(parts)
        result[group] = {"battery_count": len(parts), "window_count": len(rows),
                         "pooled_windows": metrics(rows[:, 1], rows[:, 2]),
                         "macro_battery": {key: float(np.mean([metrics(p[:, 1], p[:, 2])[key] for p in parts]))
                                           for key in ("rmse", "mae", "mape_percent")}}
    write_json(output, {"scope": "post-hoc groups only", "groups": result})
    return result


def sensitivity_report(suite, features, output):
    plt = plotting()
    root, output = Path(suite), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    case = json.loads((root / "preregistered_case.json").read_text())["selected_battery"]
    cells = load_features(features)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for column, variants in enumerate((("reference", "weight_0.25", "weight_1.25"),
                                      ("reference", "epsilon_0.001", "epsilon_0.025"))):
        for name in variants:
            path = root / name / "target_test_predictions.csv"
            data = join_predictions(path, cells)[case]
            axes[0, column].plot(data[:, 0], data[:, 2], label=name)
            axes[1, column].scatter(data[:, 1], data[:, 2] - data[:, 1], s=8, label=name)
        axes[0, column].plot(data[:, 0], data[:, 1], "k--", label="Reference SOH")
        axes[0, column].set(xlabel="Cycle", ylabel="SOH fraction")
        axes[1, column].set(xlabel="Reference SOH", ylabel="Prediction - reference")
        for ax in axes[:, column]:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle(f"Prespecified target case: {case}")
    save_figure(fig, output, "tmr_sensitivity")
    plt.close(fig)


def group_crossval(runs, features, grouping, output):
    root, output = Path(runs), Path(output)
    paths = sorted(root.rglob("target_test_predictions.csv"))
    if not paths:
        raise ValueError("No finalized cross-validation prediction files found")
    output.mkdir(parents=True, exist_ok=False)
    collected, seen = {}, set()
    for path in paths:
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        direction = f"{manifest['source_domain']}->{manifest['target_domain']}"
        split = direction, manifest["source_fold"], manifest["target_fold"]
        if split in seen:
            raise ValueError("Duplicate split: provide one method's cross-validation root, not a mixed-method suite")
        seen.add(split)
        destination = output / path.parent.relative_to(root) / "group_metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        scores = group_score(path, features, grouping, destination)
        for group, values in scores.items():
            collected.setdefault(direction, {}).setdefault(group, []).append(values)
    summary = {}
    for direction, groups in collected.items():
        summary[direction] = {}
        for group, results in groups.items():
            summary[direction][group] = {"contributing_combinations": len(results)}
            for aggregation in ("pooled_windows", "macro_battery"):
                values = np.array([r[aggregation]["rmse"] for r in results])
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else None
                summary[direction][group][aggregation] = {
                    "rmse_mean": mean, "rmse_std_ddof1": sd,
                    "rmse_x100_mean": mean * 100,
                    "rmse_x100_std_ddof1": None if sd is None else sd * 100}
    write_json(output / "summary.json", {"results": summary,
                                          "note": "Groups absent from a partition are omitted, never scored as zero; SD describes overlapping partitions"})
    return summary
