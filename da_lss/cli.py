import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .data import WindowPool, load_features, prepare_raw
from .experiment import (TrainConfig, load_checkpoint, predict, run_experiment,
                         save_predictions, score, write_json)
from .model import ModelConfig


def config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    unknown = set(raw) - {"training", "model"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {unknown}")
    return TrainConfig(**raw.get("training", {})), ModelConfig(**raw.get("model", {}))


def main():
    parser = argparse.ArgumentParser(description="Independent DA-LSS technical reproduction")
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert canonical raw charging CSV to features")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--smoothing", type=int, default=15)
    prepare.add_argument("--bins", type=int, default=32)
    prepare.add_argument("--voltage-range", nargs=2, type=float, default=(2, 4.5))
    prepare.add_argument("--config")
    demo = commands.add_parser("demo", help="Synthetic pipeline verification; not a paper benchmark")
    demo.add_argument("--output", required=True)
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--plot", action="store_true")
    for name in ("train", "crossval", "suite", "tune"):
        p = commands.add_parser(name)
        p.add_argument("--features", required=True)
        p.add_argument("--source", required=True)
        p.add_argument("--target", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--config")
        p.add_argument("--device", default="cpu")
        if name != "crossval":
            p.add_argument("--source-fold", type=int, default=0)
            p.add_argument("--target-fold", type=int, default=0)
        if name == "crossval":
            p.add_argument("--both-directions", action="store_true")
        if name in ("train", "crossval"):
            p.add_argument("--resume", action="store_true")
        if name == "suite":
            p.add_argument("--suite", choices=["ablation", "sensitivity"], default="ablation")
            p.add_argument("--all-folds", action="store_true")
            p.add_argument("--both-directions", action="store_true")
        if name == "tune":
            p.add_argument("--trials", type=int, default=10)
    for name in ("predict", "benchmark"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cpu")
        if name == "predict":
            p.add_argument("--features", required=True)
        else:
            p.add_argument("--repetitions", type=int, default=100)
    cluster = commands.add_parser("cluster", help="Retrospective only: use AFTER predictions are final")
    cluster.add_argument("--features", required=True)
    cluster.add_argument("--output", required=True)
    for name in ("replay", "export", "heatmap"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", required=True)
        if name != "export":
            p.add_argument("--device", default="cpu")
        if name == "replay":
            p.add_argument("--input", required=True)
            p.add_argument("--stride", type=int, default=32)
        if name == "heatmap":
            p.add_argument("--features", required=True)
            p.add_argument("--battery", required=True, help="domain/battery_id")
    for name in ("audit", "report", "group-score", "group-crossval", "sensitivity-report"):
        p = commands.add_parser(name)
        p.add_argument("--features", required=True)
        p.add_argument("--output", required=True)
        if name == "audit":
            p.add_argument("--length", type=int, default=128)
        if name == "report":
            p.add_argument("--run", required=True)
        if name in ("group-score", "group-crossval"):
            p.add_argument("--grouping", required=True)
        if name == "group-score":
            p.add_argument("--predictions", required=True)
        if name == "group-crossval":
            p.add_argument("--runs", required=True)
        if name == "sensitivity-report":
            p.add_argument("--suite", required=True)
    native = commands.add_parser("import-mit", help="Documented MATLAB v7.3 batch layout")
    native.add_argument("--input", required=True)
    native.add_argument("--output", required=True)
    native.add_argument("--batch-id", required=True)
    native.add_argument("--nominal-capacity-ah", required=True, type=float)
    for name in ("decode-can", "match-signals", "match-references", "matched-segments", "capacity-soh", "power-energy"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        if name == "decode-can":
            p.add_argument("--mapping", required=True)
        if name in ("match-signals", "match-references"):
            p.add_argument("--max-age-s", required=True, type=float)
        if name == "match-references":
            p.add_argument("--references", required=True)
        if name == "capacity-soh":
            p.add_argument("--max-gap-s", type=float, default=60)
            p.add_argument("--min-soc-change", type=float, default=0.05)
        if name == "power-energy":
            p.add_argument("--completed-windows", required=True, type=int)
    adaptation = commands.add_parser("adapt-vehicle")
    adaptation.add_argument("--checkpoint", required=True)
    adaptation.add_argument("--source-features", required=True)
    adaptation.add_argument("--adaptation-features", required=True)
    adaptation.add_argument("--test-vehicles", required=True)
    adaptation.add_argument("--output", required=True)
    adaptation.add_argument("--config")
    adaptation.add_argument("--source-manifest")
    adaptation.add_argument("--device", default="cpu")
    resources = commands.add_parser("resources")
    resources.add_argument("--checkpoint", required=True)
    resources.add_argument("--output", required=True)
    resources.add_argument("--device", default="cpu")
    resources.add_argument("--exported-model")
    resources.add_argument("--raw-segments-npz")
    resources.add_argument("--repetitions", type=int, default=100)
    resources.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        if args.config:
            settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
            fixed = {"outlier_sigma": 3, "smoothing_direction": "trailing", "boundary": "valid_only_no_padding",
                     "nonfinite_samples": "remove_joint_voltage_current_rows", "entropy_log_base": "e",
                     "insufficient_segment": "error"}
            for key, expected in fixed.items():
                if key in settings and settings[key] != expected:
                    raise ValueError(f"Unsupported preprocessing rule: {key}")
            args.smoothing = settings.get("smoothing", args.smoothing)
            args.bins = settings.get("entropy_bins", args.bins)
            args.voltage_range = settings.get("entropy_voltage_range_v", args.voltage_range)
        prepare_raw(args.input, args.output, args.smoothing, args.bins, args.voltage_range)
    elif args.command == "demo":
        from .demo import generate_raw, plot_run
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=False)
        generate_raw(root / "synthetic_raw.csv")
        prepare_raw(root / "synthetic_raw.csv", root / "synthetic_features.csv")
        cfg = TrainConfig(epochs=3, steps_per_epoch=6, batch_per_domain=8, patience=3)
        result = run_experiment(root / "synthetic_features.csv", root / "run",
                                "synthetic_source", "synthetic_target", cfg, device=args.device)
        write_json(root / "NOTICE.json", {"synthetic": True, "paper_accuracy_reproduced": False,
                                         "purpose": "End-to-end software verification only"})
        if args.plot:
            plot_run(root / "run", root / "synthetic_features.csv")
        print(json.dumps(result, indent=2))
    elif args.command in ("train", "crossval", "suite", "tune"):
        cfg, mcfg = config(args.config)
        if args.command == "train":
            run_experiment(args.features, args.output, args.source, args.target, cfg, mcfg,
                           args.source_fold, args.target_fold, args.device, resume=args.resume)
        elif args.command == "crossval":
            from .workflows import run_crossval
            run_crossval(args.features, args.output, args.source, args.target, cfg, mcfg,
                         args.device, args.both_directions, args.resume)
        elif args.command == "suite":
            from .workflows import run_suite
            run_suite(args.features, args.output, args.source, args.target, cfg, mcfg, args.suite,
                      args.device, args.source_fold, args.target_fold, args.all_folds, args.both_directions)
        else:
            from .workflows import tune
            tune(args.features, args.output, args.source, args.target, cfg, mcfg, args.trials,
                 args.device, args.source_fold, args.target_fold)
    elif args.command in ("predict", "benchmark"):
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        model, scaler, cfg = load_checkpoint(args.checkpoint, args.device)
        use_gate = cfg.use_gate and not cfg.source_only
        if args.command == "predict":
            batteries = load_features(args.features)
            pool = WindowPool([b.unlabeled() for b in batteries], scaler, cfg.sequence_length, cfg.stride)
            records = predict(model, pool, args.device, use_gate)
            save_predictions(output / "predictions.csv", records)
            write_json(output / "metrics.json", score(records, batteries))
        else:
            from .analysis import benchmark
            write_json(output / "benchmark.json", benchmark(model, cfg.sequence_length, args.device,
                                                             args.repetitions, use_gate=use_gate))
    elif args.command == "cluster":
        from .analysis import retrospective_groups
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, retrospective_groups(load_features(args.features)))
    elif args.command == "replay":
        from .deployment import replay
        replay(args.checkpoint, args.input, args.output, args.device, args.stride)
    elif args.command == "export":
        from .deployment import export_torchscript
        export_torchscript(args.checkpoint, args.output)
    elif args.command == "heatmap":
        from .reporting import hidden_report
        hidden_report(args.checkpoint, args.features, args.output, args.battery, args.device)
    elif args.command == "audit":
        from .importers import audit_features
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, audit_features(args.features, args.length))
    elif args.command == "report":
        from .reporting import training_report
        training_report(args.run, args.features, args.output)
    elif args.command == "group-score":
        from .reporting import group_score
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        group_score(args.predictions, args.features, args.grouping, args.output)
    elif args.command == "sensitivity-report":
        from .reporting import sensitivity_report
        sensitivity_report(args.suite, args.features, args.output)
    elif args.command == "group-crossval":
        from .reporting import group_crossval
        group_crossval(args.runs, args.features, args.grouping, args.output)
    elif args.command == "import-mit":
        from .importers import import_mit_mat
        import_mit_mat(args.input, args.output, args.batch_id, args.nominal_capacity_ah)
    elif args.command in ("decode-can", "match-signals", "match-references", "matched-segments", "capacity-soh", "power-energy"):
        from .vehicle_data import capacity_table, decode_can, match_signals, match_references, matched_to_segments
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "decode-can":
            result = decode_can(args.input, args.mapping, destination)
        elif args.command == "match-signals":
            result = match_signals(args.input, destination, args.max_age_s)
        elif args.command == "matched-segments":
            matched_to_segments(args.input, destination)
            result = None
        elif args.command == "match-references":
            match_references(args.input, args.references, destination, args.max_age_s)
            result = None
        elif args.command == "capacity-soh":
            result = capacity_table(args.input, destination, args.max_gap_s, args.min_soc_change)
        else:
            from .resources import integrate_power
            result = integrate_power(args.input, args.completed_windows)
            write_json(destination, result)
        if result is not None and args.command in ("decode-can", "match-signals"):
            write_json(destination.with_suffix(".provenance.json"), result)
    elif args.command == "adapt-vehicle":
        from .vehicle_adaptation import VehicleAdaptationConfig, adapt_vehicle
        raw = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        reserved = json.loads(Path(args.test_vehicles).read_text(encoding="utf-8"))
        adapt_vehicle(args.checkpoint, args.source_features, args.adaptation_features, reserved, args.output,
                      VehicleAdaptationConfig(**raw), args.device, args.source_manifest)
    elif args.command == "resources":
        from .resources import profile_resources
        segments = None
        if args.raw_segments_npz:
            import numpy as np
            with np.load(args.raw_segments_npz, allow_pickle=False) as arrays:
                if arrays["voltage"].shape != arrays["current"].shape:
                    raise ValueError("Raw voltage/current segment arrays must match")
                segments = list(zip(arrays["voltage"], arrays["current"]))
        profile_resources(args.checkpoint, args.output, args.device, args.repetitions, args.warmup,
                          args.exported_model, segments)


if __name__ == "__main__":
    main()
