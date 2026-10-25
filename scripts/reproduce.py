import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/reproduction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extended", action="store_true", help="Also run 40 combinations, two tuning trials and all reports")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "tmp").mkdir()
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", TEMP=str(output / "tmp"), TMP=str(output / "tmp"),
               LOKY_MAX_CPU_COUNT="4")
    completed = []

    def execute(name, arguments, module=True):
        cmd = [sys.executable] + (["-m", "da_lss"] if module else []) + list(map(str, arguments))
        print(f"Running {name} ...", flush=True)
        with (output / "logs" / f"{name}.md").open("w", encoding="utf-8") as log:
            log.write(f"# {name.replace('_', ' ').title()}\n\n````text\n")
            log.flush()
            result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.write("\n````\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed. Read {output / 'logs' / (name + '.md')}")
        completed.append(name)
        (output / "verification.json").write_text(json.dumps(
            {"synthetic_only": True, "paper_accuracy_reproduced": False,
             "python": sys.executable, "device": args.device, "completed": completed}, indent=2), encoding="utf-8")

    execute("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], module=False)
    execute("demo", ["demo", "--output", output / "demo", "--device", args.device, "--plot"])
    features, run = output / "demo/synthetic_features.csv", output / "demo/run"
    checkpoint = run / "best.pt"
    execute("audit", ["audit", "--features", features, "--output", output / "audit.json"])
    execute("predict", ["predict", "--checkpoint", checkpoint, "--features", features,
                         "--output", output / "frozen", "--device", args.device])
    execute("export", ["export", "--checkpoint", checkpoint, "--output", output / "deployment"])
    replay_csv = output / "replay.csv"
    with features.open(newline="", encoding="utf-8") as f, replay_csv.open("w", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        names = ["timestamp" if x == "cycle" else x for x in reader.fieldnames]
        writer = csv.DictWriter(out, fieldnames=names)
        writer.writeheader()
        for row in reader:
            row["timestamp"] = row.pop("cycle")
            writer.writerow(row)
    execute("replay", ["replay", "--checkpoint", checkpoint, "--input", replay_csv,
                        "--output", output / "online", "--device", args.device])
    execute("report", ["report", "--run", run, "--features", features, "--output", output / "report"])
    execute("benchmark", ["benchmark", "--checkpoint", checkpoint, "--output", output / "benchmark",
                           "--device", args.device])
    common = ["--features", features, "--source", "synthetic_source", "--target", "synthetic_target",
              "--config", root / "configs/smoke.json", "--device", args.device]
    if args.extended:
        execute("crossval", ["crossval", *common, "--both-directions", "--output", output / "crossval"])
        execute("ablation", ["suite", *common, "--suite", "ablation", "--output", output / "ablation"])
        execute("sensitivity", ["suite", *common, "--suite", "sensitivity", "--output", output / "sensitivity"])
        execute("sensitivity_plot", ["sensitivity-report", "--suite", output / "sensitivity", "--features", features,
                                      "--output", output / "sensitivity_report"])
        execute("tuning", ["tune", *common, "--trials", "2", "--output", output / "tuning"])
        execute("clustering", ["cluster", "--features", features, "--output", output / "groups.json"])
        execute("group_score", ["group-score", "--features", features, "--grouping", output / "groups.json",
                                 "--predictions", run / "target_test_predictions.csv", "--output", output / "group_metrics.json"])
        execute("group_crossval", ["group-crossval", "--runs", output / "crossval", "--features", features,
                                    "--grouping", output / "groups.json", "--output", output / "group_crossval"])
        execute("heatmap", ["heatmap", "--checkpoint", checkpoint, "--features", features,
                             "--battery", "synthetic_target/cell_00", "--output", output / "heatmap"])
    print(f"Completed {len(completed)} stages: {output}", flush=True)


if __name__ == "__main__":
    main()
