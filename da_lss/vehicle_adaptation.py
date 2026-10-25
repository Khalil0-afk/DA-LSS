import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import WindowPool, file_sha256, load_features
from .experiment import calibrate_gate, load_checkpoint, predict, score, write_json
from .model import temporal_monotonicity


@dataclass
class VehicleAdaptationConfig:
    adaptation_vehicle_count: int = 4
    test_vehicle_count: int = 20
    windows_per_vehicle: int = 32
    stride: int = 32
    epochs: int = 10
    updates_per_epoch: int = 4
    batch_per_domain: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 2026

    def __post_init__(self):
        if min(self.adaptation_vehicle_count, self.test_vehicle_count, self.windows_per_vehicle,
               self.stride, self.epochs, self.updates_per_epoch, self.batch_per_domain) < 1:
            raise ValueError("Vehicle adaptation counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid adaptation optimizer settings")


def vehicle_identities(path):
    identities = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vehicle = row.get("vehicle_id", "")
            if not vehicle:
                raise ValueError("A vehicle_id is required on every adaptation row")
            key = f"{row['domain']}/{row['battery_id']}"
            if key in identities and identities[key] != vehicle:
                raise ValueError("One battery key cannot represent multiple vehicles")
            identities[key] = vehicle
    return identities


def adapt_vehicle(checkpoint, source_features, adaptation_features, test_vehicle_ids, output,
                  config=None, device="cpu", source_manifest=None):
    cfg = config or VehicleAdaptationConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(cfg.seed)
    parent = json.loads(Path(source_manifest or Path(checkpoint).parent / "manifest.json").read_text(encoding="utf-8"))
    if parent["feature_sha256"] != file_sha256(source_features):
        raise ValueError("Source data must match the pretrained split manifest")
    if len(set(test_vehicle_ids)) != len(test_vehicle_ids) or len(test_vehicle_ids) != cfg.test_vehicle_count:
        raise ValueError("Expected a unique, preregistered held-out test vehicle list")
    identities = vehicle_identities(adaptation_features)
    vehicles = sorted(set(identities.values()))
    if len(vehicles) != cfg.adaptation_vehicle_count or set(vehicles) & set(test_vehicle_ids):
        raise ValueError("Adaptation and test identities must be disjoint at vehicle level")
    model, scaler, training = load_checkpoint(checkpoint, device)
    if model.cfg.backbone != "lss":
        raise ValueError("The reviewer-response vehicle protocol uses a pretrained DA-LSS backbone")
    public = {b.key: b for b in load_features(source_features)}
    roles = {role: [public[key] for key in parent["roles"][role]] for role in ("source_train", "source_val")}
    if any(b.soh is None for cells in roles.values() for b in cells):
        raise ValueError("Source supervision and source validation labels are required")
    source = WindowPool(roles["source_train"], scaler, training.sequence_length, training.stride)
    validation = WindowPool(roles["source_val"], scaler, training.sequence_length, training.stride)
    target_cells = [b.unlabeled() for b in load_features(adaptation_features)]
    candidates = {vehicle: [] for vehicle in vehicles}
    for i, b in enumerate(target_cells):
        for end in range(training.sequence_length - 1, len(b.cycles), cfg.stride):
            candidates[identities[b.key]].append((i, end))
    selected = []
    for vehicle in vehicles:
        windows = candidates[vehicle]
        if len(windows) < cfg.windows_per_vehicle:
            raise ValueError(f"Insufficient distinct adaptation windows for vehicle {vehicle}")
        selected.extend(windows[i] for i in sorted(rng.choice(len(windows), cfg.windows_per_vehicle, replace=False)))
    used = sorted({i for i, _ in selected})
    target = WindowPool([target_cells[i] for i in used], scaler, training.sequence_length, cfg.stride)
    for new_index, old_index in enumerate(used):
        target.ends[new_index] = np.array(sorted(end for i, end in selected if i == old_index))
    manifest = {"protocol": "pre-deployment unsupervised vehicle adaptation", "config": asdict(cfg),
                "initial_checkpoint_sha256": file_sha256(checkpoint), "adaptation_vehicles": vehicles,
                "held_out_test_vehicles": list(test_vehicle_ids), "target_soh_used": False,
                "normalization": "frozen source-pretraining scaler", "tmr_scope": "source pairs only",
                "selection": "uniform distinct windows per vehicle, then vehicle-uniform replacement sampling",
                "windows": [{"battery": target_cells[i].key, "end_cycle": float(target_cells[i].cycles[end])}
                            for i, end in selected], "source_manifest": parent,
                "adaptation_features_sha256": file_sha256(adaptation_features)}
    write_json(output / "manifest.json", manifest)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, history = float("inf"), []
    by_vehicle = {vehicle: [] for vehicle in vehicles}
    for i, b in enumerate(target.batteries):
        by_vehicle[identities[b.key]].extend((i, int(end)) for end in target.ends[i])
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for _ in range(cfg.updates_per_epoch):
            sx, sn, valid, sy = source.sample(cfg.batch_per_domain, rng)
            draws = []
            for _ in range(cfg.batch_per_domain):
                vehicle = vehicles[int(rng.integers(len(vehicles)))]
                group = by_vehicle[vehicle]
                i, end = group[int(rng.integers(len(group)))]
                draws.append(target.window(i, end))
            x = torch.as_tensor(np.concatenate((sx, np.stack(draws))), device=device)
            result = model(x, reversal=training.lambda_domain, use_gate=training.use_gate)
            n = len(sy)
            labels = torch.as_tensor(sy, device=device)
            task = F.mse_loss(result["prediction"][:n], labels)
            auxiliary = F.mse_loss(result["auxiliary"][:n], labels)
            domain = F.binary_cross_entropy_with_logits(result["domain"], torch.cat(
                (torch.zeros(n, device=device), torch.ones(n, device=device))))
            following = model.predict_soh(torch.as_tensor(sn, device=device), training.use_gate)
            tmr = temporal_monotonicity(result["prediction"][:n], following,
                                       torch.as_tensor(valid, device=device), training.epsilon)
            loss = task + domain + training.lambda_aux * auxiliary + training.lambda_phy * tmr
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite vehicle adaptation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if training.use_gate:
            calibrate_gate(model, source, target, device)
        records = predict(model, validation, device, training.use_gate)
        value = score(records, roles["source_val"])["pooled_windows"]["rmse"]
        history.append({"epoch": epoch + 1, "updates": (epoch + 1) * cfg.updates_per_epoch,
                        "source_val_rmse": value, "mean_optimization_loss": float(np.mean(losses))})
        if value < best:
            best = value
            torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg),
                        "train_config": asdict(training), "scaler": {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()},
                        "vehicle_adaptation": asdict(cfg), "epoch": epoch + 1, "source_val_rmse": value}, output / "best.pt")
    write_json(output / "history.json", history)
    result = {"updates_completed": cfg.epochs * cfg.updates_per_epoch, "best_source_val_rmse": best,
              "adaptation_window_count": len(selected), "held_out_ev_test_records_loaded": False, "target_soh_used": False}
    write_json(output / "metrics.json", result)
    return result
