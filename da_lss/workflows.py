from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .data import load_features, split_batteries
from .experiment import run_experiment, write_json


def run_crossval(features, output, source, target, cfg, mcfg, device="cpu",
                 both_directions=False, resume=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=resume)
    directions = [(source, target)] + ([(target, source)] if both_directions else [])
    summary = {}
    for index, (src, tgt) in enumerate(directions):
        results = []
        for sf in range(5):
            for tf in range(4):
                run = root / f"direction_{index}" / f"source{sf}_target{tf}"


                r = run_experiment(features, run, src, tgt, cfg, mcfg, sf, tf, device,
                                   resume=resume and run.exists())
                results.append(r["target_test"])
        if any(r["labeled_windows"] != r["total_windows"] for r in results):
            raise ValueError("Complete target test labels required for cross-validation summaries")
        summary[f"{src}->{tgt}"] = {
            aggregation: {metric: {"mean": float(np.mean([r[aggregation][metric] for r in results])),
                                   "std_ddof1": float(np.std([r[aggregation][metric] for r in results], ddof=1))}
                          for metric in ("rmse", "mae", "mape_percent")}
            for aggregation in ("pooled_windows", "macro_battery")}
        write_json(root / "summary.json", {"partition_combinations_per_direction": 20,
                                             "note": "Overlapping partitions; SD is not a confidence interval",
                                             "results": summary})
    return summary


def experiment_variants(suite, cfg, mcfg):
    if suite == "ablation":
        return {
            "proposed": (cfg, mcfg),
            "DA-LSS-1_no_alignment": (replace(cfg, lambda_domain=0), mcfg),
            "DA-LSS-2_no_TMR": (replace(cfg, lambda_phy=0), mcfg),
            "DA-LSS-3_LSTM": (replace(cfg, lambda_aux=0, use_gate=False), replace(mcfg, backbone="lstm")),
            "source_only": (replace(cfg, source_only=True), mcfg),
            "no_health_gate": (replace(cfg, use_gate=False), mcfg),
        }
    if suite == "sensitivity":
        return {name: (replace(cfg, lambda_phy=weight, epsilon=epsilon), mcfg)
                for name, weight, epsilon in (("reference", 0.703, 0.005),
                                              ("weight_0.25", 0.25, 0.005),
                                              ("weight_1.25", 1.25, 0.005),
                                              ("epsilon_0.001", 0.703, 0.001),
                                              ("epsilon_0.025", 0.703, 0.025))}
    raise ValueError("Unknown suite")


def run_suite(features, output, source, target, cfg, mcfg, suite="ablation", device="cpu",
              source_fold=0, target_fold=0, all_folds=False, both_directions=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    batteries = [b for b in load_features(features) if len(b.cycles) >= cfg.sequence_length]
    selected = split_batteries(batteries, source, target, source_fold, target_fold, cfg.seed)["target_test"]
    selected = sorted(selected, key=lambda b: b.key)
    selected_key = selected[int(np.random.default_rng(cfg.seed).integers(len(selected)))].key

    write_json(root / "preregistered_case.json", {"selected_battery": selected_key, "seed": cfg.seed,
                                                  "source_fold": source_fold, "target_fold": target_fold,
                                                  "selection": "uniform target-test battery before predictions"})
    results = {}
    for name, (training, model) in experiment_variants(suite, cfg, mcfg).items():
        destination = root / name
        if all_folds:
            result = run_crossval(features, destination, source, target, training, model, device, both_directions)
        else:
            result = run_experiment(features, destination, source, target, training, model,
                                    source_fold, target_fold, device)
        results[name] = result
        write_json(root / "results.json", {"suite": suite, "all_folds": all_folds,
                                            "case_selected_before_predictions": selected_key, "results": results})
    return results


def tune(features, output, source, target, cfg, mcfg, trials=10, device="cpu",
         source_fold=0, target_fold=0):
    import optuna

    if trials < 1:
        raise ValueError("trials must be positive")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    search = {"learning_rate": [0.0001, 0.003, "log"], "lambda_domain": [0.1, 2.5],
              "lambda_phy": [0.0, 1.5], "epsilon": [0.001, 0.005, 0.025]}
    write_json(root / "search_protocol.json", {"search_space": search, "seed": cfg.seed,
                                               "n_trials": trials, "early_stopping_patience": cfg.patience,
                                               "objective": "source-validation pooled RMSE only",
                                               "test_predictions_generated": False,
                                               "origin": "declared independent search space; paper does not specify it"})
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))

    def objective(trial):
        trial_cfg = replace(cfg, learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                            lambda_domain=trial.suggest_float("lambda_domain", 0.1, 2.5),
                            lambda_phy=trial.suggest_float("lambda_phy", 0, 1.5),
                            epsilon=trial.suggest_categorical("epsilon", [0.001, 0.005, 0.025]))
        result = run_experiment(features, root / f"trial_{trial.number:03d}", source, target, trial_cfg, mcfg,
                                source_fold, target_fold, device, evaluate_tests=False)
        write_json(root / f"trial_{trial.number:03d}" / "config.json",
                   {"training": asdict(trial_cfg), "model": asdict(mcfg)})
        return result["best_source_val_rmse"]

    study.optimize(objective, n_trials=trials)
    best = replace(cfg, **study.best_params)
    write_json(root / "best_config.json", {"training": asdict(best), "model": asdict(mcfg)})
    write_json(root / "study.json", {"best_trial": study.best_trial.number, "best_source_val_rmse": study.best_value,
                                     "trials": [{"number": t.number, "value": t.value, "parameters": t.params}
                                                for t in study.trials]})
    return study.best_value
