import csv
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import FEATURES
from .experiment import load_checkpoint, metrics, write_json


class OnlineEstimator:
    def __init__(self, model, scaler, length=128, stride=32, device="cpu", use_gate=True):
        if length < 2 or not 1 <= stride <= length:
            raise ValueError("Require length >= 2 and 1 <= stride <= length")
        self.model = model.eval()
        self.scaler = scaler
        self.length, self.stride, self.device = length, stride, device
        self.use_gate = use_gate
        self.cells = {}
        self.invalid_vectors = 0

    @torch.no_grad()
    def push(self, domain, cell_id, timestamp, features):
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (4,) or not np.isfinite(vector).all() or not np.isfinite(timestamp):
            self.invalid_vectors += 1
            return None
        key = domain, cell_id
        state = self.cells.setdefault(key, {"buffer": deque(maxlen=self.length),
                                           "valid_count": 0, "last_timestamp": -float("inf")})
        if timestamp <= state["last_timestamp"]:
            raise ValueError(f"Timestamps must increase strictly within cell {key}")
        state["last_timestamp"] = timestamp
        state["buffer"].append(vector.copy())
        state["valid_count"] += 1
        count = state["valid_count"]
        if count < self.length or (count - self.length) % self.stride:
            return None
        raw = np.stack(state["buffer"])
        x = torch.as_tensor(self.scaler.transform(raw)[None], device=self.device)
        y = self.model.predict_soh(x, use_gate=self.use_gate).item()
        if not np.isfinite(y):
            raise FloatingPointError("Nonfinite online prediction")
        return {"domain": domain, "battery_id": cell_id, "timestamp": float(timestamp),
                "prediction": y, "valid_count": count}


def replay(checkpoint, input_csv, output, device="cpu", stride=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, scaler, cfg = load_checkpoint(checkpoint, device)
    estimator = OnlineEstimator(model, scaler, cfg.sequence_length, stride, device,
                                cfg.use_gate and not cfg.source_only)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    references, predictions = {}, []
    vehicles, protocols = {}, {}


    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"domain", "battery_id", "timestamp", *FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Replay CSV requires {sorted(required)}")
        for row in reader:
            domain, cell = row["domain"], row["battery_id"]
            vehicle = row.get("vehicle_id") or f"{domain}/{cell}"
            timestamp = float(row["timestamp"])
            prediction = estimator.push(domain, json.dumps([vehicle, cell]), timestamp, [float(row[c]) for c in FEATURES])
            if prediction is None:
                continue
            prediction["battery_id"] = cell
            prediction["vehicle_id"] = vehicle
            prediction["charging_protocol"] = row.get("charging_protocol", "unspecified") or "unspecified"
            predictions.append(prediction)
            key = domain, vehicle, cell, timestamp

            if row.get("soh", ""):
                y = float(row["soh"])
                if not np.isfinite(y) or not 0 < y <= 1.5:
                    raise ValueError("Replay SOH must be a finite positive fraction")
                references[key] = y
            vehicles[key] = vehicle
            protocols[key] = row.get("charging_protocol", "unspecified") or "unspecified"
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "battery_id", "vehicle_id", "charging_protocol", "timestamp", "prediction", "valid_count"])
        writer.writeheader()
        writer.writerows(predictions)
    per_vehicle, per_protocol, per_vehicle_protocol = {}, {}, {}
    for r in predictions:
        key = r["domain"], r["vehicle_id"], r["battery_id"], r["timestamp"]
        if key in references:
            pair = per_vehicle.setdefault(vehicles[key], ([], []))
            pair[0].append(references[key])
            pair[1].append(r["prediction"])
            for grouping, name in ((per_protocol, protocols[key]),
                                   (per_vehicle_protocol, f"{vehicles[key]}/{protocols[key]}")):
                values = grouping.setdefault(name, ([], []))
                values[0].append(references[key])
                values[1].append(r["prediction"])
    by_vehicle = {v: metrics(*pair) for v, pair in per_vehicle.items()}
    frozen = all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    if not frozen:
        raise RuntimeError("Model state mutated during replay")
    result = {"prediction_count": len(predictions), "invalid_vectors_skipped": estimator.invalid_vectors,
              "length": cfg.sequence_length, "stride": stride, "batch_size": 1,
              "cold_start_padding": False, "state_frozen": frozen, "per_vehicle": by_vehicle,
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
    result["per_charging_protocol"] = {name: metrics(*pair) for name, pair in per_protocol.items()}
    result["per_vehicle_and_protocol"] = {name: metrics(*pair) for name, pair in per_vehicle_protocol.items()}
    result["protocol_assignment"] = "metadata of the prediction endpoint; mixed-history windows are not relabeled"
    if by_vehicle:
        result["vehicle_summary"] = {metric: {"mean": float(np.mean([v[metric] for v in by_vehicle.values()])),
                                              "std_ddof1": float(np.std([v[metric] for v in by_vehicle.values()], ddof=1))
                                              if len(by_vehicle) > 1 else None}
                                     for metric in ("rmse", "mae", "mape_percent")}
    write_json(output / "metrics.json", result)
    return result


class InferenceModule(nn.Module):
    def __init__(self, model, scaler, use_gate):
        super().__init__()
        self.model = model
        self.use_gate = use_gate
        self.register_buffer("mean", torch.as_tensor(scaler.mean))
        self.register_buffer("scale", torch.as_tensor(scaler.scale))

    def forward(self, raw_features):
        x = (raw_features - self.mean) / self.scale
        return self.model.predict_soh(x, use_gate=self.use_gate)


@torch.no_grad()
def export_torchscript(checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, scaler, cfg = load_checkpoint(checkpoint, "cpu")

    model.domain = nn.Identity()
    module = InferenceModule(model, scaler, cfg.use_gate and not cfg.source_only).eval()
    generator = torch.Generator().manual_seed(2026)
    example = torch.randn(1, cfg.sequence_length, 4, generator=generator)
    traced = torch.jit.trace(module, example, check_trace=True)
    frozen = torch.jit.freeze(traced)
    frozen.save(str(output / "model.ts"))
    reloaded = torch.jit.load(str(output / "model.ts"))
    errors = []
    for batch in (1, 2):
        x = torch.randn(batch, cfg.sequence_length, 4, generator=generator)
        expected, actual = module(x), reloaded(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float((actual - expected).abs().max()))
    metadata = {"format": "frozen TorchScript", "input": ["batch", cfg.sequence_length, 4],
                "input_units": "unstandardized features in documented order", "features": list(FEATURES),
                "output": "SOH fraction", "normalization_embedded": True,
                "domain_classifier_inactive": True, "sequence_length_is_fixed": True,
                "max_reload_parity_error": max(errors),
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "torch_version": str(torch.__version__)}
    write_json(output / "metadata.json", metadata)
    return metadata
