import csv
from pathlib import Path

import numpy as np

from .data import FEATURES, file_sha256, load_features, segment_features
from .experiment import write_json


def audit_features(path, length=128):
    cells = load_features(path)
    domains = {}
    for b in cells:
        info = domains.setdefault(b.domain, {"batteries": 0, "eligible_batteries": 0,
                                             "rows": 0, "windows_stride1": 0, "missing_labels": 0,
                                             "nonunit_cycle_gaps": 0})
        info["batteries"] += 1
        info["eligible_batteries"] += int(len(b.cycles) >= length)
        info["rows"] += len(b.cycles)
        info["windows_stride1"] += max(0, len(b.cycles) - length + 1)
        info["missing_labels"] += len(b.cycles) if b.soh is None else 0
        info["nonunit_cycle_gaps"] += int((~np.isclose(np.diff(b.cycles), 1)).sum())
    return {"sha256": file_sha256(path), "sequence_length": length,
            "domains": domains, "features": list(FEATURES), "input_valid": True,
            "scope": "format and counts only; not a test-label-dependent quality filter"}


def import_mit_mat(path, output, batch_id, nominal_capacity_ah, smoothing=15,
                   current_threshold=0.01, bins=32):
    import h5py

    if nominal_capacity_ah <= 0 or current_threshold < 0 or not batch_id:
        raise ValueError("Explicit positive nominal capacity, batch ID and valid threshold required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"format": "MIT v7.3 batch/summary/cycles", "nominal_capacity_ah": nominal_capacity_ah,
              "smoothing": smoothing, "entropy_bins": bins, "entropy_range_v": [2.0, 4.5],
              "charging_current_threshold_a": current_threshold,
              "selection": "longest contiguous positive-current run; no SOH-dependent selection",
              "written_cycles": 0, "skipped_cycles": [], "batch_id": batch_id,
              "identity_warning": "Resolve repeated physical cells/continuations across batches before splitting",
              "source_sha256": file_sha256(path)}

    with h5py.File(path, "r") as f, output.open("x", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["domain", "battery_id", "cycle", *FEATURES, "soh"])
        batch = f["batch"]
        summaries = np.asarray(batch["summary"]).reshape(-1)
        cycle_groups = np.asarray(batch["cycles"]).reshape(-1)
        if len(summaries) != len(cycle_groups):
            raise ValueError("MIT summary/cycle group cell counts differ")
        for cell, (summary_ref, cycles_ref) in enumerate(zip(summaries, cycle_groups)):
            summary, cycles = f[summary_ref], f[cycles_ref]
            cycle_numbers = np.asarray(summary["cycle"]).reshape(-1)
            capacities = np.asarray(summary["QDischarge"]).reshape(-1)
            voltage_refs = np.asarray(cycles["V"]).reshape(-1)
            current_refs = np.asarray(cycles["I"]).reshape(-1)

            if not len(cycle_numbers) == len(capacities) == len(voltage_refs) == len(current_refs):
                raise ValueError(f"Ambiguous cycle/summary alignment for {batch_id}_c{cell}; inspect the archive")
            for index, (cycle, capacity) in enumerate(zip(cycle_numbers, capacities)):
                key = {"battery_id": f"{batch_id}_c{cell}", "cycle": float(cycle)}
                v = np.asarray(f[voltage_refs[index]]).reshape(-1)
                current = np.asarray(f[current_refs[index]]).reshape(-1)
                if v.shape != current.shape:
                    raise ValueError(f"Voltage/current lengths differ: {key}")
                active = np.isfinite(v) & np.isfinite(current) & (current > current_threshold)
                padded = np.concatenate(([False], active, [False])).astype(int)
                starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
                if not len(starts):
                    report["skipped_cycles"].append({**key, "reason": "no positive-current segment"})
                    continue
                longest = int(np.argmax(ends - starts))
                start, end = starts[longest], ends[longest]
                try:
                    values = segment_features(v[start:end], current[start:end], smoothing, bins)
                except ValueError as error:
                    report["skipped_cycles"].append({**key, "reason": str(error)})
                    continue


                soh = float(capacity) / nominal_capacity_ah
                if not np.isfinite(soh) or not 0 < soh <= 1.5:
                    raise ValueError(f"Invalid SOH reference: {key}; verify units/denominator")
                writer.writerow(["MIT", key["battery_id"], cycle, *values, soh])
                report["written_cycles"] += 1
    report["feature_sha256"] = file_sha256(output)
    write_json(output.with_suffix(".provenance.json"), report)
    return report
