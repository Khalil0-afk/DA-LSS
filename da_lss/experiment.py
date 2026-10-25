import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Standardizer, WindowPool, file_sha256, load_features, split_batteries
from .model import DALSS, ModelConfig, temporal_monotonicity


@dataclass
class TrainConfig:
    sequence_length: int = 128
    stride: int = 1
    batch_per_domain: int = 128
    epochs: int = 80
    steps_per_epoch: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_domain: float = 1.774
    lambda_phy: float = 0.703
    lambda_aux: float = 1.0
    epsilon: float = 0.005
    patience: int = 15
    seed: int = 2026
    use_gate: bool = True
    source_only: bool = False

    def __post_init__(self):
        if min(self.sequence_length, self.stride, self.batch_per_domain,
               self.epochs, self.steps_per_epoch, self.patience) < 1:
            raise ValueError("Training counts must be positive")
        if self.sequence_length < 2 or self.learning_rate <= 0:
            raise ValueError("Require sequence_length >= 2 and positive learning rate")
        if min(self.weight_decay, self.lambda_domain, self.lambda_phy,
               self.lambda_aux, self.epsilon) < 0:
            raise ValueError("Loss coefficients and tolerances must be nonnegative")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def metrics(reference, prediction):
    y, p = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y <= 0):
        raise ValueError("Metrics require matching, finite predictions and positive reference SOH")
    error = p - y
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mape_percent": float(np.mean(np.abs(error / y)) * 100)}


@torch.no_grad()
def calibrate_gate(model, source, target, device, batch_size=128):
    model.eval()
    if model.cfg.backbone != "lss":
        return
    collected = []
    for pool, labeled in ((source, True), (target, False)):
        k, h = model.cfg.health_points, model.cfg.hidden
        mass = torch.zeros(k, device=device, dtype=torch.float64)
        mass2 = torch.zeros_like(mass)
        first = torch.zeros(k, h, device=device, dtype=torch.float64)
        second = torch.zeros_like(first)
        for x, keys in pool.batches(batch_size):
            _, u, aux = model.candidate(torch.as_tensor(x, device=device))
            if labeled:
                soh = torch.tensor([pool.batteries[i].soh[end] for i, end in keys], device=device)
            else:
                soh = aux
            q = model.gate.weights(soh).double()
            mass += q.sum(0)
            mass2 += q.square().sum(0)
            first += q.T @ u.double().mean(1)
            second += q.T @ u.double().square().mean(1)
        collected.append((mass, mass2, first, second))
    model.gate.fit_moments(*collected)


@torch.no_grad()
def predict(model, pool, device, use_gate=True, batch_size=128):
    model.eval()
    records = []
    for x, keys in pool.batches(batch_size):
        prediction = model.predict_soh(torch.as_tensor(x, device=device), use_gate=use_gate).cpu().numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite predictions")
        for (i, end), value in zip(keys, prediction):
            b = pool.batteries[i]
            records.append({"domain": b.domain, "battery_id": b.battery_id,
                            "cycle": float(b.cycles[end]), "prediction": float(value)})
    return records


def score(records, batteries):
    lookup = {(b.domain, b.battery_id, float(c)): float(y)
              for b in batteries if b.soh is not None for c, y in zip(b.cycles, b.soh)}
    groups, truth, predicted = {}, [], []
    for r in records:
        key = r["domain"], r["battery_id"], r["cycle"]
        if key not in lookup:
            continue
        y = lookup[key]
        truth.append(y)
        predicted.append(r["prediction"])
        groups.setdefault(f"{key[0]}/{key[1]}", ([], []))[0].append(y)
        groups[f"{key[0]}/{key[1]}"][1].append(r["prediction"])
    if not truth:
        return {"labeled_windows": 0, "total_windows": len(records)}
    per_battery = {key: metrics(*pair) for key, pair in groups.items()}
    return {"labeled_windows": len(truth), "total_windows": len(records),
            "pooled_windows": metrics(truth, predicted), "per_battery": per_battery,
            "macro_battery": {k: float(np.mean([v[k] for v in per_battery.values()]))
                              for k in ("rmse", "mae", "mape_percent")}}


def save_predictions(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "battery_id", "cycle", "prediction"])
        writer.writeheader()
        writer.writerows(records)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DALSS(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scaler = Standardizer(np.array(checkpoint["scaler"]["mean"], dtype=np.float32),
                          np.array(checkpoint["scaler"]["scale"], dtype=np.float32))
    return model, scaler, TrainConfig(**checkpoint["train_config"])


def run_experiment(features, output, source, target, train_cfg=None, model_cfg=None,
                   source_fold=0, target_fold=0, device="cpu", resume=False,
                   evaluate_tests=True):
    cfg, mcfg = train_cfg or TrainConfig(), model_cfg or ModelConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(cfg.seed)
    all_batteries = load_features(features)

    eligible = [b for b in all_batteries if len(b.cycles) >= cfg.sequence_length]
    roles = split_batteries(eligible, source, target, source_fold, target_fold, cfg.seed)
    for role in ("source_train", "source_val"):
        if any(b.soh is None for b in roles[role]):
            raise ValueError(f"{role} requires source labels")
    scaler = Standardizer.fit(roles["source_train"])
    pools = {key: WindowPool(value, scaler, cfg.sequence_length, cfg.stride) for key, value in roles.items()}
    manifest = {"roles": {r: [b.key for b in bs] for r, bs in roles.items()},
                "window_counts": {r: len(p) for r, p in pools.items()},
                "battery_window_counts": {r: {b.key: len(e) for b, e in zip(p.batteries, p.ends)}
                                          for r, p in pools.items()},
                "excluded_too_short": [b.key for b in all_batteries if len(b.cycles) < cfg.sequence_length],
                "source_fold": source_fold, "target_fold": target_fold,
                "source_domain": source, "target_domain": target,
                "feature_sha256": file_sha256(features),
                "train_config": asdict(cfg), "model_config": asdict(mcfg),
                "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                "device": device, "aggregation": "pooled windows plus macro battery",
                "evaluate_tests": evaluate_tests,
                "target_labels_used_for_training": False}
    manifest["adaptation_window_fraction"] = len(pools["target_adapt"]) / (len(pools["target_adapt"]) + len(pools["target_test"]))
    if resume:
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume manifest differs: data, configuration, device or split changed")
    else:
        write_json(output / "manifest.json", manifest)
    model = DALSS(mcfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, stale, history, start_epoch = float("inf"), 0, [], 0
    if resume:
        last = torch.load(output / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        rng.bit_generator.state = last["numpy_rng"]
        random.setstate(last["python_rng"])
        torch.set_rng_state(last["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in last["cuda_rng"]])
        best, stale, history, start_epoch = last["best"], last["stale"], last["history"], last["epoch"]
    start = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        if stale >= cfg.patience:
            break
        model.train()
        totals = {"task": 0.0, "domain": 0.0, "tmr": 0.0, "auxiliary": 0.0}
        for _ in range(cfg.steps_per_epoch):
            sx, sn, sm, sy = pools["source_train"].sample(cfg.batch_per_domain, rng)
            if cfg.source_only:
                x, following, mask = sx, sn, sm
            else:
                tx, tn, tm, _ = pools["target_adapt"].sample(cfg.batch_per_domain, rng)
                x, following, mask = np.concatenate((sx, tx)), np.concatenate((sn, tn)), np.concatenate((sm, tm))
            x = torch.as_tensor(x, device=device)
            use_gate = cfg.use_gate and not cfg.source_only
            result = model(x, reversal=cfg.lambda_domain, use_gate=use_gate)
            n = len(sy)
            labels = torch.as_tensor(sy, device=device)
            task = F.mse_loss(result["prediction"][:n], labels)
            auxiliary = F.mse_loss(result["auxiliary"][:n], labels)
            domain = task * 0
            if not cfg.source_only:
                domain_labels = torch.cat((torch.zeros(n, device=device), torch.ones(n, device=device)))
                domain = F.binary_cross_entropy_with_logits(result["domain"], domain_labels)
            tmr = task * 0
            if cfg.lambda_phy:
                next_prediction = model.predict_soh(torch.as_tensor(following, device=device), use_gate=use_gate)
                tmr = temporal_monotonicity(result["prediction"], next_prediction,
                                           torch.as_tensor(mask, device=device), cfg.epsilon)
            loss = task + domain + cfg.lambda_phy * tmr + cfg.lambda_aux * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in (("task", task), ("domain", domain), ("tmr", tmr), ("auxiliary", auxiliary)):
                totals[key] += float(value.detach()) / cfg.steps_per_epoch


        if cfg.use_gate and not cfg.source_only:
            calibrate_gate(model, pools["source_train"], pools["target_adapt"], device)
        records = predict(model, pools["source_val"], device, cfg.use_gate and not cfg.source_only)
        val = score(records, roles["source_val"])["pooled_windows"]["rmse"]
        entry = {"epoch": epoch + 1, **totals, "source_val_rmse": val,
                 "supported_health_points": int(model.gate.supported.sum())}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if val < best:
            best, stale = val, 0
            torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                        "train_config": asdict(cfg), "scaler": {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()},
                        "epoch": epoch + 1, "source_val_rmse": val}, output / "best.pt")
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
                    "numpy_rng": rng.bit_generator.state, "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else []}, output / "last.pt.tmp")
        (output / "last.pt.tmp").replace(output / "last.pt")
        write_json(output / "history.json", history)
        if stale >= cfg.patience:
            break
    write_json(output / "history.json", history)
    model, _, _ = load_checkpoint(output / "best.pt", device)
    result = {"elapsed_seconds": time.perf_counter() - start, "best_source_val_rmse": best,
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "inference_parameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("domain."))}
    if not evaluate_tests:
        result["test_evaluation"] = "withheld: source-validation-only tuning trial"
        write_json(output / "metrics.json", result)
        return result

    predictions = {}
    for role in ("source_test", "target_test"):
        predictions[role] = predict(model, pools[role], device, cfg.use_gate and not cfg.source_only)
        save_predictions(output / f"{role}_predictions.csv", predictions[role])
    for role in predictions:
        result[role] = score(predictions[role], roles[role])
    write_json(output / "metrics.json", result)
    return result
