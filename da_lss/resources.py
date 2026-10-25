import csv
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import file_sha256, segment_features
from .experiment import load_checkpoint, write_json


@torch.no_grad()
def count_operations(model, sequence_length=128, use_gate=True):
    if model.cfg.backbone != "lss":
        raise ValueError("The explicit scalar operation convention currently covers DA-LSS only")
    counts = {"macs": 0, "bias_additions": 0}
    handles = []

    def linear(module, inputs, output):
        counts["macs"] += output.numel() * module.in_features
        counts["bias_additions"] += output.numel() if module.bias is not None else 0

    def convolution(module, inputs, output):
        counts["macs"] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        counts["bias_additions"] += output.numel() if module.bias is not None else 0

    for name, module in model.named_modules():
        if name.startswith("domain"):
            continue
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear))
        elif isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(convolution))
    device = next(model.parameters()).device
    try:
        model.eval().predict_soh(torch.zeros(1, sequence_length, 4, device=device), use_gate)
    finally:
        for handle in handles:
            handle.remove()
    h, k, n = model.cfg.hidden, model.cfg.health_points, sequence_length * model.cfg.hidden
    if use_gate:
        counts["macs"] += k * h
    other = counts["bias_additions"] + 12 * n + h
    if use_gate:
        other += 4 * k + 4 * k * h + 2 * h
    return {"macs_per_window": counts["macs"], "other_accounted_flops": other,
            "accounted_mflops_per_window": (2 * counts["macs"] + other) / 1e6,
            "macs_per_new_valid_point_stride32": counts["macs"] / 32,
            "accounted_mflops_per_new_valid_point_stride32": (2 * counts["macs"] + other) / 32e6,
            "convention": "one multiply-add is two FLOPs; dense/Conv1d hooks plus explicit gate matrix product",
            "other_included": ["bias additions", "step offset", "ZOH scalar arithmetic", "six recurrence operations per state point",
                               "two temporal means", "negative A", "health-gate scalar arithmetic"],
            "unaccounted_operators": ["InstanceNorm implementation", "softplus", "exp/expm1", "softmax implementation"],
            "is_complete_hardware_instruction_count": False, "domain_classifier_included": False}


def integrate_power(input_csv, completed_windows):
    if completed_windows < 1:
        raise ValueError("A positive completed-window count is required")
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["timestamp_s"]) for r in rows])
    power = np.array([float(r["power_w"]) for r in rows])
    if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(power).all() or np.any(np.diff(t) <= 0) or np.any(power < 0):
        raise ValueError("Power samples must be finite, nonnegative and strictly chronological")
    energy_j = float(np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(t)))
    duration_s = float(t[-1] - t[0])
    return {"duration_s": duration_s, "completed_windows": completed_windows,
            "platform_energy_j": energy_j, "mean_platform_power_w": energy_j / duration_s,
            "energy_per_window_mj": 1000 * energy_j / completed_windows,
            "idle_power_subtracted": False, "integration": "trapezoidal, actual sample timestamps",
            "scope": "user-supplied post-warmup platform-level power trace"}


@torch.no_grad()
def profile_resources(checkpoint, output, device="cpu", repetitions=100, warmup=10,
                      exported_model=None, raw_segments=None):
    import psutil

    if repetitions < 1 or warmup < 0:
        raise ValueError("Invalid repetition count")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, scaler, cfg = load_checkpoint(checkpoint, device)
    use_gate = cfg.use_gate and not cfg.source_only
    counts = count_operations(model, cfg.sequence_length, use_gate)
    if raw_segments is None:
        raw_features = np.zeros((cfg.sequence_length, 4), dtype=np.float32)
    else:
        if len(raw_segments) != cfg.sequence_length:
            raise ValueError("One completed raw segment per sequence point is required")
        raw_features = np.array([segment_features(v, c) for v, c in raw_segments], dtype=np.float32)
    x = torch.as_tensor(scaler.transform(raw_features)[None], device=device)

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model.predict_soh(x, use_gate)
    synchronize()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def memory_sampler():
        while not stop.wait(0.001):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=memory_sampler, daemon=True)
    sampler.start()
    inference, end_to_end = [], []
    try:
        for _ in range(repetitions):
            synchronize()
            start = time.perf_counter()
            model.predict_soh(x, use_gate).item()
            synchronize()
            inference.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            features = raw_features if raw_segments is None else np.array(
                [segment_features(v, c) for v, c in raw_segments], dtype=np.float32)
            window = np.stack([row for row in features])
            inputs = torch.as_tensor(scaler.transform(window)[None], device=device)
            model.predict_soh(inputs, use_gate).item()
            synchronize()
            end_to_end.append((time.perf_counter() - start) * 1000)
    finally:
        stop.set()
        sampler.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    result = {**counts, "inference_parameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("domain.")),
              "checkpoint_sha256": file_sha256(checkpoint),
              "exported_model_sha256": file_sha256(exported_model) if exported_model else None,
              "torch_version": str(torch.__version__),
              "parameter_millions": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("domain.")) / 1e6,
              "exported_model_mb": Path(exported_model).stat().st_size / 1e6 if exported_model else None,
              "inference_latency_mean_ms": float(np.mean(inference)),
              "end_to_end_latency_mean_ms": float(np.mean(end_to_end)),
              "inference_latency_p95_ms": float(np.percentile(inference, 95)),
              "peak_process_rss_mb_sampled": peak[0] / 1e6, "rss_sampling_interval_ms": 1,
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6 if str(device).startswith("cuda") else None,
              "device": device, "precision": "float32", "batch_size": 1, "sequence_length": cfg.sequence_length,
              "warmup": warmup, "repetitions": repetitions,
              "end_to_end_includes": ["feature extraction" if raw_segments is not None else "precomputed features",
                                      "window organization", "source normalization", "tensor transfer", "inference", "scalar output"],
              "end_to_end_excludes": ["waiting to acquire a full window", "CAN acquisition/decoding and asynchronous signal matching"],
              "memory_scope": "sampled process RSS including runtime; CUDA allocator peak reported separately",
              "power_w": None, "energy_per_window_mj": None}
    write_json(output / "resources.json", result)
    return result
