import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ("mean_voltage", "mean_current", "voltage_entropy", "voltage_std")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Battery:
    domain: str
    battery_id: str
    cycles: np.ndarray
    features: np.ndarray
    soh: object = None

    @property
    def key(self):
        return f"{self.domain}/{self.battery_id}"

    def unlabeled(self):
        return Battery(self.domain, self.battery_id, self.cycles, self.features, None)


def load_features(path):
    groups = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"domain", "battery_id", "cycle", *FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            key = row["domain"], row["battery_id"]
            if not all(key):
                raise ValueError("Domain and battery_id must not be empty")
            groups.setdefault(key, []).append(row)
    batteries = []
    for (domain, battery_id), rows in sorted(groups.items()):
        rows.sort(key=lambda r: float(r["cycle"]))
        cycles = np.array([float(r["cycle"]) for r in rows])
        x = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=np.float32)
        y = np.array([float(r["soh"]) if r.get("soh", "") else np.nan for r in rows], dtype=np.float32)
        if not np.isfinite(cycles).all() or not np.isfinite(x).all():
            raise ValueError(f"Nonfinite input: {domain}/{battery_id}")
        if np.any(np.diff(cycles) <= 0):
            raise ValueError(f"Duplicate cycles: {domain}/{battery_id}")
        if not np.isnan(y).all() and (not np.isfinite(y).all() or np.any((y <= 0) | (y > 1.5))):
            raise ValueError("SOH must be complete positive fractions (not percentages), or entirely absent")
        batteries.append(Battery(domain, battery_id, cycles, x, None if np.isnan(y).all() else y))
    if not batteries:
        raise ValueError("Empty feature table")
    return batteries


def split_batteries(batteries, source, target, source_fold=0, target_fold=0, seed=2026):
    if source == target or not 0 <= source_fold < 5 or not 0 <= target_fold < 4:
        raise ValueError("Require distinct domains, source fold 0..4, target fold 0..3")
    folds = {}

    for domain, count in ((source, 5), (target, 4)):
        ids = sorted(b.key for b in batteries if b.domain == domain)
        if len(ids) < count:
            raise ValueError(f"Need at least {count} eligible batteries in {domain}")
        rng = np.random.default_rng(seed)
        folds[domain] = [set(x.tolist()) for x in np.array_split(rng.permutation(ids), count)]
    stest = folds[source][source_fold]
    sval = folds[source][(source_fold + 1) % 5]
    ttest = folds[target][target_fold]
    roles = {name: [] for name in ("source_train", "source_val", "source_test", "target_adapt", "target_test")}
    for b in batteries:
        if b.domain == source:
            role = "source_test" if b.key in stest else "source_val" if b.key in sval else "source_train"
        elif b.domain == target:
            role = "target_test" if b.key in ttest else "target_adapt"
        else:
            continue
        roles[role].append(b.unlabeled() if role == "target_adapt" else b)
    return roles


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, source_train):
        x = np.concatenate([b.features for b in source_train])
        return cls(x.mean(0), np.maximum(x.std(0), 1e-6))

    def transform(self, x):
        return ((x - self.mean) / self.scale).astype(np.float32)


class WindowPool:
    def __init__(self, batteries, scaler, length=128, stride=1):
        if length < 2 or stride < 1:
            raise ValueError("length >= 2 and stride >= 1 required")
        self.batteries = batteries
        self.length = length
        self.x = [scaler.transform(b.features) for b in batteries]
        self.ends = [np.arange(length - 1, len(b.cycles), stride) for b in batteries]
        if not batteries or any(len(e) == 0 for e in self.ends):
            raise ValueError("Every pool battery must contain a full window")

    def __len__(self):
        return sum(len(e) for e in self.ends)

    def window(self, battery, end):
        return self.x[battery][end - self.length + 1:end + 1]

    def sample(self, size, rng):
        indices = rng.integers(len(self.batteries), size=size)
        current, following, valid, labels = [], [], [], []
        for i in indices:
            b = self.batteries[i]
            end = int(rng.choice(self.ends[i]))
            adjacent = end + 1 < len(b.cycles) and np.isclose(b.cycles[end + 1] - b.cycles[end], 1)
            current.append(self.window(i, end))
            following.append(self.window(i, end + 1 if adjacent else end))
            valid.append(adjacent)
            labels.append(np.nan if b.soh is None else b.soh[end])
        return np.stack(current), np.stack(following), np.array(valid), np.array(labels, dtype=np.float32)

    def batches(self, batch_size=128):
        batch, keys = [], []
        for i, ends in enumerate(self.ends):
            for end in ends:
                batch.append(self.window(i, end))
                keys.append((i, int(end)))
                if len(batch) == batch_size:
                    yield np.stack(batch), keys
                    batch, keys = [], []
        if batch:
            yield np.stack(batch), keys


def segment_features(voltage, current, smoothing=15, bins=32, voltage_range=(2.0, 4.5)):
    v, c = np.asarray(voltage, dtype=float), np.asarray(current, dtype=float)
    if v.ndim != 1 or v.shape != c.shape or smoothing < 1 or bins < 2:
        raise ValueError("Expected paired 1D signals and positive smoothing/bins")
    if voltage_range[0] >= voltage_range[1]:
        raise ValueError("Invalid voltage range")
    finite = np.isfinite(v) & np.isfinite(c)
    v, c = v[finite], c[finite]
    if len(v) < smoothing:
        raise ValueError("Segment too short after nonfinite removal")
    keep = (np.abs(v - v.mean()) <= 3 * v.std()) & (np.abs(c - c.mean()) <= 3 * c.std())
    v, c = v[keep], c[keep]
    if len(v) < smoothing:
        raise ValueError("Segment too short after 3-sigma rejection")
    kernel = np.ones(smoothing) / smoothing
    v, c = np.convolve(v, kernel, "valid"), np.convolve(c, kernel, "valid")
    if np.any((v < voltage_range[0]) | (v > voltage_range[1])):
        raise ValueError("Voltage outside configured entropy range; choose a suitable cell range")
    counts, _ = np.histogram(v, bins=bins, range=voltage_range)
    p = counts[counts > 0] / counts.sum()
    return [v.mean(), c.mean(), -(p * np.log(p)).sum(), v.std()]


def prepare_raw(input_path, output_path, smoothing=15, bins=32, voltage_range=(2.0, 4.5)):
    groups = {}
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"domain", "battery_id", "cycle", "time_s", "voltage_v", "current_a"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Raw CSV requires {sorted(required)}")
        for row in reader:
            cycle = float(row["cycle"])
            if not np.isfinite(cycle):
                raise ValueError("Raw cycle numbers must be finite")
            groups.setdefault((row["domain"], row["battery_id"], cycle), []).append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "battery_id", "cycle", *FEATURES, "soh"])
        for key, rows in sorted(groups.items()):
            rows.sort(key=lambda r: float(r["time_s"]))
            times = np.array([float(r["time_s"]) for r in rows])
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Raw timestamps must be finite and unique: {key}")
            labels = {r["soh"] for r in rows if r.get("soh", "")}
            if len(labels) > 1:
                raise ValueError(f"Inconsistent segment labels: {key}")
            features = segment_features([float(r["voltage_v"]) for r in rows],
                                        [float(r["current_a"]) for r in rows], smoothing, bins, voltage_range)
            writer.writerow([*key, *features, next(iter(labels), "")])
    provenance = {"input_sha256": file_sha256(input_path), "output_sha256": file_sha256(output_path),
                  "smoothing": smoothing, "entropy_bins": bins, "entropy_range_v": list(voltage_range),
                  "entropy_log_base": "e", "outlier_sigma": 3,
                  "alignment": "joint voltage/current rejection mask", "boundary": "valid trailing average",
                  "selection": "upstream completed segments; one group per domain/battery/cycle",
                  "segment_count": len(groups)}
    output_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
