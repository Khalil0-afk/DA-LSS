import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = []

    def execute(name, arguments):
        print(f"Running {name} ...", flush=True)
        with (output / "logs" / f"{name}.md").open("w", encoding="utf-8") as f:
            f.write(f"# {name.replace('_', ' ').title()}\n\n````text\n")
            f.flush()
            result = subprocess.run([sys.executable, "-m", "da_lss", *map(str, arguments)], cwd=root, env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
            f.write("\n````\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {output / 'logs' / (name + '.md')}")
        completed.append(name)

    with (output / "can.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_id", "battery_id", "segment_id", "timestamp", "frame_id", "data_hex"])
        for t in range(32):
            payload = (3200 + int(200 * t / 31)).to_bytes(2, "little") + (150).to_bytes(2, "little", signed=True)
            writer.writerow(["synthetic_vehicle", "cell_01", 1, t, "0x100", payload.hex()])
    execute("decode", ["decode-can", "--input", output / "can.csv", "--mapping", root / "configs/can_mapping.synthetic.json",
                       "--output", output / "signals.csv"])
    execute("matching", ["match-signals", "--input", output / "signals.csv", "--max-age-s", 0,
                         "--output", output / "matched.csv"])
    execute("raw_segments", ["matched-segments", "--input", output / "matched.csv", "--output", output / "segments.csv"])
    execute("features", ["prepare", "--input", output / "segments.csv", "--config", root / "configs/preprocessing.json",
                         "--output", output / "features.csv"])
    (output / "capacity.csv").write_text(
        "vehicle_id,battery_id,segment_id,scope,timestamp,current_a,soc,nominal_capacity_ah\n"
        "synthetic_vehicle,cell_01,1,cell,0,1,0.2,2\n"
        "synthetic_vehicle,cell_01,1,cell,1800,1,0.45,2\n"
        "synthetic_vehicle,cell_01,1,cell,3600,1,0.7,2\n", encoding="utf-8")
    execute("capacity", ["capacity-soh", "--input", output / "capacity.csv", "--max-gap-s", 1800,
                         "--output", output / "capacity.json"])
    (output / "power.csv").write_text("timestamp_s,power_w\n0,10\n1,20\n3,20\n", encoding="utf-8")
    execute("power_arithmetic", ["power-energy", "--input", output / "power.csv", "--completed-windows", 5,
                                 "--output", output / "synthetic_energy.json"])
    voltage = np.tile(np.linspace(3.2, 3.4, 32), (128, 1))
    np.savez(output / "raw_window.npz", voltage=voltage, current=np.full_like(voltage, 1.5))
    execute("demo", ["demo", "--output", output / "synthetic", "--device", args.device])
    checkpoint = output / "synthetic/run/best.pt"
    execute("export", ["export", "--checkpoint", checkpoint, "--output", output / "export"])
    execute("resources", ["resources", "--checkpoint", checkpoint,
                           "--raw-segments-npz", output / "raw_window.npz", "--exported-model", output / "export/model.ts",
                           "--output", output / "resources", "--device", args.device])
    with (output / "adaptation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "battery_id", "vehicle_id", "cycle", "mean_voltage", "mean_current",
                         "voltage_entropy", "voltage_std", "soh"])
        for vehicle in range(4):
            for cycle in range(1, 1121):
                writer.writerow(["EV", f"vehicle_{vehicle}_cell", f"adapt_{vehicle}", cycle,
                                 3.28 + 0.00001 * cycle + vehicle * 0.002, 1.45 - 0.00001 * cycle,
                                 0.5 + 0.02 * np.sin(cycle / 40), 0.025, ""])
    reserved = [f"reserved_test_{i:02d}" for i in range(20)]
    (output / "test_vehicles.json").write_text(json.dumps(reserved), encoding="utf-8")
    execute("vehicle_adaptation", ["adapt-vehicle", "--checkpoint", checkpoint,
                                    "--source-features", output / "synthetic/synthetic_features.csv",
                                    "--adaptation-features", output / "adaptation.csv", "--test-vehicles", output / "test_vehicles.json",
                                    "--config", root / "configs/vehicle_adaptation.json", "--output", output / "adapted",
                                    "--device", args.device])
    result = json.loads((output / "adapted/metrics.json").read_text())
    assert result["updates_completed"] == 40 and result["adaptation_window_count"] == 128
    manifest = json.loads((output / "adapted/manifest.json").read_text())
    assert len(manifest["adaptation_vehicles"]) == 4 and len(manifest["held_out_test_vehicles"]) == 20
    (output / "verification.json").write_text(json.dumps({"synthetic_only": True,
        "completed": completed, "adaptation": result, "real_vehicle_accuracy_reproduced": False,
        "synthetic_power_trace_is_not_a_hardware_measurement": True}, indent=2), encoding="utf-8")
    print(f"Completed {len(completed)} response-specific stages: {output}", flush=True)


if __name__ == "__main__":
    main()
