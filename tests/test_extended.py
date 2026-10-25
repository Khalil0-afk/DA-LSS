import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from da_lss.data import FEATURES, Standardizer, WindowPool, load_features, prepare_raw
from da_lss.deployment import OnlineEstimator, export_torchscript, replay
from da_lss.experiment import TrainConfig, load_checkpoint, predict, run_experiment
from da_lss.importers import audit_features, import_mit_mat
from da_lss.model import DALSS, ModelConfig
from da_lss.reporting import group_crossval
from da_lss.workflows import experiment_variants, tune
from test_core import batteries, save_features


def checkpoint(path, model, length=4):
    torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg),
                "train_config": asdict(TrainConfig(sequence_length=length)),
                "scaler": {"mean": [0.0] * 4, "scale": [1.0] * 4}}, path)


class DeploymentTests(unittest.TestCase):
    def test_inference_never_calls_discriminator(self):
        model = DALSS(ModelConfig(hidden=8)).eval()
        def forbidden(*args):
            raise AssertionError("Discriminator called during inference")
        hook = model.domain.register_forward_hook(forbidden)
        model.predict_soh(torch.randn(2, 8, 4))
        hook.remove()

    def test_all_backbones_forward_backward_and_inference(self):
        for backbone in ("lss", "lstm", "bigru", "transformer"):
            with self.subTest(backbone=backbone):
                model = DALSS(ModelConfig(hidden=8, backbone=backbone))
                x = torch.randn(2, 10, 4)
                out = model(x)
                self.assertEqual(tuple(out["hidden"].shape), (2, 10, 8))
                out["prediction"].square().sum().backward()
                self.assertTrue(torch.isfinite(model.regression.weight.grad).all())
                model.eval()
                torch.testing.assert_close(model.predict_soh(x), model(x)["prediction"])

    def test_fifo_128_32_cold_start_and_cell_isolation(self):
        model = DALSS(ModelConfig(hidden=8)).eval()
        scaler = Standardizer(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        online = OnlineEstimator(model, scaler)
        rows = np.random.default_rng(0).normal(size=(160, 4)).astype(np.float32)
        outputs = []
        for t, features in enumerate(rows):
            for cell in ("a", "b"):
                value = online.push("EV", cell, t, features)
                if value:
                    outputs.append(value)
        self.assertEqual([r["valid_count"] for r in outputs], [128, 128, 160, 160])
        self.assertEqual([r["timestamp"] for r in outputs], [127, 127, 159, 159])
        with torch.no_grad():
            expected = model.predict_soh(torch.from_numpy(rows[-128:][None])).item()
        self.assertAlmostEqual(outputs[-1]["prediction"], expected, places=6)

    def test_invalid_vectors_do_not_advance_stride_and_time_reversal_rejected(self):
        model = DALSS(ModelConfig(hidden=8)).eval()
        scaler = Standardizer(np.zeros(4), np.ones(4))
        online = OnlineEstimator(model, scaler, length=4, stride=2)
        for t in range(3):
            self.assertIsNone(online.push("EV", "a", t, [1, 2, 3, 4]))
        self.assertIsNone(online.push("EV", "a", 3, [np.nan] * 4))
        self.assertIsNotNone(online.push("EV", "a", 4, [1, 2, 3, 4]))
        self.assertEqual(online.invalid_vectors, 1)
        with self.assertRaises(ValueError):
            online.push("EV", "a", 4, [1, 2, 3, 4])

    def test_torchscript_reload_prediction_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for backbone in ("lss", "lstm"):
                path = root / f"{backbone}.pt"
                checkpoint(path, DALSS(ModelConfig(hidden=8, backbone=backbone)))
                result = export_torchscript(path, root / backbone)
                self.assertLess(result["max_reload_parity_error"], 1e-5)
                self.assertTrue(result["domain_classifier_inactive"])
                self.assertTrue((root / backbone / "model.ts").exists())

    def test_replay_reference_changes_leave_predictions_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(root / "model.pt", DALSS(ModelConfig(hidden=8)))
            for repeat in range(2):
                path = root / f"replay{repeat}.csv"
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["domain", "battery_id", "timestamp", *FEATURES, "soh", "vehicle_id"])
                    for t in range(8):
                        writer.writerow(["EV", "a", t, 3.3, 1.5, 0.2, 0.01, 0.95 - repeat * 0.2, "vehicle1"])
                result = replay(root / "model.pt", path, root / f"run{repeat}", stride=2)
                self.assertTrue(result["state_frozen"])
                self.assertEqual(result["prediction_count"], 3)
            self.assertEqual((root / "run0/predictions.csv").read_bytes(),
                             (root / "run1/predictions.csv").read_bytes())


class WorkflowTests(unittest.TestCase):
    def test_group_summary_units_and_absent_group_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = batteries()
            save_features(root / "features.csv", cells)
            grouping = {"domains": {"T": {"groups": {b.key: "FAS" if b.battery_id == "0" else "TAS"
                                                        for b in cells if b.domain == "T"}}}}
            (root / "groups.json").write_text(json.dumps(grouping))
            for sf, error in enumerate((0.01, 0.03)):
                run = root / "runs" / str(sf)
                run.mkdir(parents=True)
                (run / "manifest.json").write_text(json.dumps({"source_domain": "S", "target_domain": "T",
                                                                "source_fold": sf, "target_fold": 0}))
                b = next(b for b in cells if b.key == "T/0")
                with (run / "target_test_predictions.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["domain", "battery_id", "cycle", "prediction"])
                    writer.writerow(["T", "0", b.cycles[-1], float(b.soh[-1]) + error])
            result = group_crossval(root / "runs", root / "features.csv", root / "groups.json", root / "out")
            self.assertNotIn("TAS", result["S->T"])
            self.assertEqual(result["S->T"]["FAS"]["contributing_combinations"], 2)
            self.assertAlmostEqual(result["S->T"]["FAS"]["pooled_windows"]["rmse_x100_mean"], 2.0)

    def test_prepare_emits_numeric_cycle_order_for_online_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.csv"
            with raw.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["domain", "battery_id", "cycle", "time_s", "voltage_v", "current_a", "soh"])
                for cycle in (10, 2, 1):
                    for t in range(16):
                        writer.writerow(["S", "a", cycle, t, 3.3, 1.5, 0.9])
            prepare_raw(raw, root / "features.csv")
            with (root / "features.csv").open(newline="", encoding="utf-8") as f:
                cycles = [float(row["cycle"]) for row in csv.DictReader(f)]
            self.assertEqual(cycles, [1, 2, 10])

    def test_resume_after_interruption_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "features.csv"
            save_features(path, batteries())
            cfg = TrainConfig(sequence_length=4, epochs=3, steps_per_epoch=1, batch_per_domain=2)
            mcfg = ModelConfig(hidden=8, min_support=2)
            run_experiment(path, root / "continuous", "S", "T", cfg, mcfg)
            original = DALSS.forward
            calls = 0
            def interrupt(model, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return original(model, *args, **kwargs)
            with patch.object(DALSS, "forward", interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_experiment(path, root / "resumed", "S", "T", cfg, mcfg)
            run_experiment(path, root / "resumed", "S", "T", cfg, mcfg, resume=True)
            first, _, _ = load_checkpoint(root / "continuous/best.pt")
            second, _, _ = load_checkpoint(root / "resumed/best.pt")
            for key, tensor in first.state_dict().items():
                torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)
            self.assertEqual((root / "continuous/history.json").read_bytes(),
                             (root / "resumed/history.json").read_bytes())

    def test_tuning_trials_never_generate_test_predictions(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("Optional optuna not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "features.csv"
            save_features(path, batteries())
            tune(path, root / "study", "S", "T",
                 TrainConfig(sequence_length=4, epochs=1, steps_per_epoch=1, batch_per_domain=2),
                 ModelConfig(hidden=8), trials=2)
            self.assertEqual(len(list((root / "study").glob("trial_*/best.pt"))), 2)
            self.assertEqual(list((root / "study").glob("trial_*/*predictions.csv")), [])
            self.assertTrue((root / "study/best_config.json").exists())

    def test_sensitivity_constants_match_revised_paper(self):
        variants = experiment_variants("sensitivity", TrainConfig(), ModelConfig())
        settings = {(cfg.lambda_phy, cfg.epsilon) for cfg, _ in variants.values()}
        self.assertEqual(settings, {(0.703, 0.005), (0.25, 0.005), (1.25, 0.005),
                                    (0.703, 0.001), (0.703, 0.025)})

    def test_audit_is_label_independent_in_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            cells = batteries()
            save_features(path, cells)
            first = audit_features(path, 4)
            for b in cells:
                b.soh[:] = 0.5
            save_features(path, cells)
            second = audit_features(path, 4)
            self.assertEqual(first["domains"], second["domains"])

    def test_native_mit_hdf5_layout_and_nominal_soh(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("Optional h5py not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "fixture.mat"
            with h5py.File(path, "w") as f:
                batch, summary, cycles = f.create_group("batch"), f.create_group("s"), f.create_group("c")
                summary["cycle"] = [[1.0, 2.0]]
                summary["QDischarge"] = [[1.1, 1.045]]
                for field in ("summary", "cycles"):
                    d = batch.create_dataset(field, (1, 1), dtype=h5py.ref_dtype)
                    d[0, 0] = summary.ref if field == "summary" else cycles.ref
                for field in ("V", "I"):
                    refs = cycles.create_dataset(field, (2, 1), dtype=h5py.ref_dtype)
                    for cycle in range(2):
                        values = np.linspace(3.2, 3.4, 32) if field == "V" else np.full(32, 1.5)
                        dataset = f.create_dataset(f"{field}_{cycle}", data=values)
                        refs[cycle, 0] = dataset.ref
            result = import_mit_mat(path, root / "features.csv", "fixture", 1.1)
            self.assertEqual(result["written_cycles"], 2)
            cell = load_features(root / "features.csv")[0]
            np.testing.assert_allclose(cell.soh, [1.0, 0.95])
            self.assertTrue((root / "features.provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
