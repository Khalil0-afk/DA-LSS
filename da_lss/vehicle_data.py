import csv
import json
from itertools import groupby
from pathlib import Path

import numpy as np

from .experiment import write_json


def decode_can(input_csv, mapping_json, output_csv):
    mapping = json.loads(Path(mapping_json).read_text(encoding="utf-8"))
    messages = {int(m["frame_id"]): m["signals"] for m in mapping["messages"]}
    records = []
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = row["frame_id"]
            frame_id = int(text, 16) if text.lower().startswith("0x") else int(text)
            if frame_id not in messages:
                continue
            payload = bytes.fromhex(row["data_hex"])
            for signal in messages[frame_id]:
                offset, width = int(signal["byte_offset"]), int(signal["byte_length"])
                order = signal["byte_order"]
                if offset < 0 or width < 1 or offset + width > len(payload) or order not in ("little", "big"):
                    raise ValueError("Invalid byte-aligned CAN signal mapping")
                raw = int.from_bytes(payload[offset:offset + width], order, signed=signal["signed"])
                value = raw * float(signal["scale"]) + float(signal["offset"])
                timestamp = float(row["timestamp"])
                if not np.isfinite(value) or not np.isfinite(timestamp):
                    raise ValueError("CAN timestamps and decoded values must be finite")
                records.append({"vehicle_id": row["vehicle_id"], "battery_id": row.get("battery_id", ""),
                                "segment_id": row.get("segment_id", ""), "timestamp": timestamp,
                                "signal": signal["name"], "scope": signal["scope"], "value": value})
    with open(output_csv, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["vehicle_id", "battery_id", "segment_id", "timestamp", "signal", "scope", "value"])
        writer.writeheader()
        writer.writerows(records)
    return {"decoded_signal_count": len(records), "mapping_origin": mapping.get("origin", "user supplied")}


def match_signals(input_csv, output_csv, max_age_s):
    if max_age_s < 0:
        raise ValueError("max_age_s must be nonnegative")
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        observations = list(csv.DictReader(f))
    observations.sort(key=lambda r: float(r["timestamp"]))
    states, records, skipped = {}, [], 0
    for timestamp, rows in groupby(observations, key=lambda r: float(r["timestamp"])):
        rows = list(rows)
        if not np.isfinite(timestamp):
            raise ValueError("Nonfinite observation timestamp")
        for row in rows:
            vehicle, cell, name = row["vehicle_id"], row["battery_id"], row["signal"]
            value = float(row["value"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite observation value")
            key = (vehicle, "" if row["scope"] == "pack" else cell, name)
            states[key] = timestamp, value
        for row in rows:
            if row["signal"] != "cell_voltage_v" or row["scope"] != "cell":
                continue
            vehicle, cell = row["vehicle_id"], row["battery_id"]
            current = states.get((vehicle, "", "pack_current_a"))
            if current is None or timestamp - current[0] > max_age_s:
                skipped += 1
                continue
            records.append({"vehicle_id": vehicle, "battery_id": cell, "segment_id": row["segment_id"],
                            "timestamp": timestamp, "cell_voltage_v": float(row["value"]),
                            "pack_current_a": current[1], "current_timestamp": current[0]})
    with open(output_csv, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["vehicle_id", "battery_id", "segment_id", "timestamp",
                                              "cell_voltage_v", "pack_current_a", "current_timestamp"])
        writer.writeheader()
        writer.writerows(records)
    return {"matched_samples": len(records), "unmatched_voltage_samples": skipped,
            "matching": "same vehicle; past-or-equal pack current only", "max_age_s": max_age_s}


def capacity_soh(times, current_a, soc, nominal_capacity_ah, max_gap_s=60, min_soc_change=0.05):
    t, current, soc = (np.asarray(x, dtype=float) for x in (times, current_a, soc))
    if t.ndim != 1 or t.shape != current.shape or t.shape != soc.shape or len(t) < 2:
        raise ValueError("Expected paired time, current and cell-SOC series")
    if not all(np.isfinite(x).all() for x in (t, current, soc)):
        raise ValueError("Capacity inputs must be finite")
    gaps = np.diff(t)
    if np.any(gaps <= 0) or np.any(gaps > max_gap_s):
        raise ValueError("Non-increasing time or an unsupported integration gap")
    if nominal_capacity_ah <= 0 or min_soc_change <= 0 or np.any((soc < 0) | (soc > 1)):
        raise ValueError("Invalid nominal capacity, SOC threshold or SOC scale")
    if np.any(current > 0) and np.any(current < 0):
        raise ValueError("A capacity segment must not mix charging and discharging")
    delta_soc = float(soc[-1] - soc[0])
    if abs(delta_soc) < min_soc_change:
        raise ValueError("Insufficient SOC excursion")
    delta_q = float(np.sum(0.5 * (current[1:] + current[:-1]) * gaps) / 3600)
    capacity = delta_q / delta_soc
    if capacity <= 0:
        raise ValueError("Current polarity and endpoint SOC change disagree")
    return {"delta_charge_ah": delta_q, "delta_soc": delta_soc, "usable_capacity_ah": capacity,
            "soh_fraction": capacity / nominal_capacity_ah,
            "assumptions": ["reliable endpoint cell SOC", "unit coulombic efficiency",
                             "series current; negligible balancing and no parallel diversion"],
            "reference_type": "integration-derived estimate; not the BMS-reported evaluation reference"}


def capacity_table(input_csv, output_json, max_gap_s=60, min_soc_change=0.05):
    groups = {}
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["scope"] not in ("cell", "pack"):
                raise ValueError("Capacity scope must be cell or pack")
            key = row["vehicle_id"], row["battery_id"], row["segment_id"], row["scope"]
            groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: float(r["timestamp"]))
        nominal = {float(r["nominal_capacity_ah"]) for r in rows}
        if len(nominal) != 1:
            raise ValueError("Inconsistent nominal capacity within a segment")
        values = capacity_soh([float(r["timestamp"]) for r in rows], [float(r["current_a"]) for r in rows],
                              [float(r["soc"]) for r in rows], nominal.pop(), max_gap_s, min_soc_change)
        result.append(dict(zip(("vehicle_id", "battery_id", "segment_id", "scope"), key), **values))
    write_json(output_json, {"segments": result})
    return result


def matched_to_segments(input_csv, output_csv):
    with open(input_csv, newline="", encoding="utf-8-sig") as source, open(output_csv, "x", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)
        writer.writerow(["domain", "battery_id", "cycle", "time_s", "voltage_v", "current_a", "soh"])
        for row in csv.DictReader(source):
            if not row["segment_id"]:
                raise ValueError("Segment identity must be supplied by the acquisition protocol")
            writer.writerow(["EV", f"{row['vehicle_id']}:{row['battery_id']}", float(row["segment_id"]),
                             row["timestamp"], row["cell_voltage_v"], row["pack_current_a"], ""])


def match_references(predictions_csv, references_csv, output_csv, max_age_s):
    if max_age_s < 0:
        raise ValueError("Reference maximum age must be nonnegative")
    references = {}
    with open(references_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "cell":
                raise ValueError("Cell evaluation requires cell-level BMS references")
            y, t = float(row["bms_soh"]), float(row["timestamp"])
            if not np.isfinite(t) or not np.isfinite(y) or not 0 < y <= 1.5:
                raise ValueError("Invalid BMS reference timestamp or SOH fraction")
            key = row["vehicle_id"], row["battery_id"]
            references.setdefault(key, []).append((t, y))
    for key, rows in references.items():
        rows.sort()
        if len({t for t, _ in rows}) != len(rows):
            raise ValueError("Duplicate cell reference timestamp")
    with open(predictions_csv, newline="", encoding="utf-8-sig") as f, open(output_csv, "x", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=[*reader.fieldnames, "reference_timestamp", "bms_soh"])
        writer.writeheader()
        for row in reader:
            history = references.get((row["vehicle_id"], row["battery_id"]), [])
            t = float(row["timestamp"])
            index = int(np.searchsorted([v[0] for v in history], t, side="right")) - 1
            valid = index >= 0 and t - history[index][0] <= max_age_s
            writer.writerow({**row, "reference_timestamp": history[index][0] if valid else "",
                             "bms_soh": history[index][1] if valid else ""})
