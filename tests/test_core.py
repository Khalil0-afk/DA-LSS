import csv
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from da_lss.data import (Battery, FEATURES, Standardizer, WindowPool, load_features,
                         segment_features, split_batteries)
from da_lss.experiment import (TrainConfig, load_checkpoint, metrics, predict,
                               run_experiment)
from da_lss.model import (DALSS, GradientReversal, HealthGate, ModelConfig,
                          diagonal_zoh, temporal_monotonicity)

torch.set_num_threads(2)


def batteries():
    rng = np.random.default_rng(8)
    return [Battery(domain, str(i), np.arange(1, 10, dtype=float),
                    rng.normal(size=(9, 4)).astype(np.float32), np.linspace(1, 0.8, 9).astype(np.float32))
            for domain, count in (("S", 10), ("T", 8)) for i in range(count)]


def save_features(path, cells):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "battery_id", "cycle", *FEATURES, "soh"])
        for b in cells:
            for j, cycle in enumerate(b.cycles):
                writer.writerow([b.domain, b.battery_id, cycle, *b.features[j], b.soh[j] if b.soh is not None else ""])


class MathTests(unittest.TestCase):
    def test_zoh_matches_augmented_matrix_exponential(self):
        a = torch.tensor([-0.4, -2.0], dtype=torch.float64)
        dt = torch.tensor([0.3, 0.3], dtype=torch.float64)
        bx = torch.tensor([0.8, -0.2], dtype=torch.float64)
        transition, write = diagonal_zoh(a, dt, bx)
        block = torch.zeros(3, 3, dtype=torch.float64)
        block[:2, :2] = torch.diag(a)
        block[:2, 2] = bx
        expected = torch.matrix_exp(block * 0.3)
        torch.testing.assert_close(torch.diag(transition), expected[:2, :2])
        torch.testing.assert_close(write, expected[:2, 2])

    def test_zoh_small_step_is_finite_and_first_order(self):
        a = torch.tensor([-1e-8], dtype=torch.float64)
        _, write = diagonal_zoh(a, torch.tensor([0.2]), torch.tensor([3.0]))
        torch.testing.assert_close(write, torch.tensor([0.6], dtype=torch.float64))

    def test_gradient_reversal_sign(self):
        x = torch.tensor([2.0], requires_grad=True)
        (GradientReversal.apply(x, 1.774) * 3).sum().backward()
        self.assertAlmostEqual(x.grad.item(), -5.322, places=5)

    def test_gate_bounds_and_unsupported_fallback(self):
        gate = HealthGate(ModelConfig(hidden=8))
        y = torch.tensor([0.8, 0.95])
        torch.testing.assert_close(gate(y), torch.ones(2, 8))
        gate.discrepancy.fill_(100)
        gate.supported.fill_(True)
        self.assertTrue(((gate(y) >= 0.5) & (gate(y) <= 1)).all())
        low = gate(y)
        gate.relevance.fill_(100)
        self.assertTrue((gate(y) > low).all())

    def test_identical_domain_moments_disable_modulation(self):
        gate = HealthGate(ModelConfig(hidden=4, health_points=3, min_support=2))
        stats = (torch.ones(3) * 10, torch.ones(3) * 5,
                 torch.ones(3, 4) * 10, torch.ones(3, 4) * 20)
        gate.fit_moments(stats, stats)
        self.assertTrue(gate.supported.all())
        torch.testing.assert_close(gate(torch.tensor([0.9])), torch.ones(1, 4))

    def test_tmr_ignores_invalid_pairs_and_tolerance(self):
        current = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        next_y = torch.tensor([0.904, 0.82, 10.0])
        loss = temporal_monotonicity(current, next_y, torch.tensor([True, True, False]))
        self.assertAlmostEqual(loss.item(), 0.0075, places=6)
        loss.backward()
        self.assertEqual(current.grad[-1].item(), 0)

    def test_no_valid_tmr_has_zero_differentiable_loss(self):
        x = torch.tensor([0.8], requires_grad=True)
        temporal_monotonicity(x, x, torch.tensor([False])).backward()
        self.assertEqual(x.grad.item(), 0)

    def test_forward_backprop_and_batch_independence(self):
        torch.manual_seed(4)
        model = DALSS(ModelConfig(hidden=8))
        x = torch.randn(3, 12, 4)
        out = model(x)
        self.assertEqual(out["hidden"].shape, (3, 12, 8))
        (out["prediction"].square().mean() + out["auxiliary"].square().mean()).backward()
        self.assertTrue(torch.isfinite(model.log_decay.grad).all())
        self.assertGreater(model.input_projection.weight.grad.abs().sum().item(), 0)
        model.eval()
        torch.testing.assert_close(model(x)["prediction"][:1], model(x[:1])["prediction"])

    def test_metrics_fraction_and_percent_units(self):
        r = metrics([1, 0.5], [0.9, 0.6])
        self.assertAlmostEqual(r["rmse"], 0.1)
        self.assertAlmostEqual(r["mae"], 0.1)
        self.assertAlmostEqual(r["mape_percent"], 15)


class DataTests(unittest.TestCase):
    def test_all_twenty_partitions_disjoint_and_cover_target(self):
        cells = batteries()
        seen = {b.key: 0 for b in cells if b.domain == "T"}
        for sf in range(5):
            for tf in range(4):
                roles = split_batteries(cells, "S", "T", sf, tf)
                keys = [b.key for bs in roles.values() for b in bs]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(len(keys), 18)
                self.assertTrue(all(b.soh is None for b in roles["target_adapt"]))
                for b in roles["target_test"]:
                    seen[b.key] += 1
        self.assertTrue(all(count == 5 for count in seen.values()))

    def test_scaler_only_fits_source_training(self):
        roles = split_batteries(batteries(), "S", "T")
        scaler = Standardizer.fit(roles["source_train"])
        expected = np.concatenate([b.features for b in roles["source_train"]]).mean(0)
        np.testing.assert_allclose(scaler.mean, expected)
        roles["target_adapt"][0].features[:] = 10000
        np.testing.assert_allclose(Standardizer.fit(roles["source_train"]).mean, expected)

    def test_windows_and_successors_stay_within_battery(self):
        cells = batteries()
        for i, b in enumerate(cells):
            b.features[:] = i
        pool = WindowPool(cells, Standardizer(np.zeros(4), np.ones(4)), length=4)
        x, following, mask, _ = pool.sample(100, np.random.default_rng(6))
        self.assertTrue(np.all(x == x[:, :1, :1]))
        np.testing.assert_array_equal(x, following)
        self.assertTrue(mask.any())
        self.assertTrue((~mask).any())

    def test_gap_is_not_a_consecutive_tmr_pair(self):
        b = batteries()[0]
        b.cycles = np.arange(9) * 2
        pool = WindowPool([b], Standardizer.fit([b]), length=4)
        self.assertFalse(pool.sample(50, np.random.default_rng(0))[2].any())

    def test_constant_segment_and_outlier_alignment(self):
        features = segment_features(np.full(32, 3.3), np.full(32, 1.5))
        np.testing.assert_allclose(features, [3.3, 1.5, 0, 0], atol=1e-14)
        v, c = np.full(40, 3.3), np.full(40, 1.5)
        v[20], c[20] = 100, 100
        np.testing.assert_allclose(segment_features(v, c), features, atol=1e-14)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            segment_features([3.2], [1.5])
        with self.assertRaises(ValueError):
            ModelConfig(kappa=1)
        with self.assertRaises(ValueError):
            metrics([0], [1])

    def test_csv_roundtrip_and_duplicate_cycle_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            cells = batteries()
            save_features(path, cells)
            loaded = load_features(path)
            np.testing.assert_array_equal(loaded[0].features, cells[0].features)
            cells[0].cycles[1] = cells[0].cycles[0]
            save_features(path, cells)
            with self.assertRaises(ValueError):
                load_features(path)


class IntegrationTests(unittest.TestCase):
    def test_source_only_ignores_target_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = batteries()
            cfg = TrainConfig(sequence_length=4, epochs=1, steps_per_epoch=1,
                              batch_per_domain=2, source_only=True)
            states = []
            for index in range(2):
                if index:
                    for b in cells:
                        if b.domain == "T":
                            b.features[:] *= 10
                path = root / f"data{index}.csv"
                save_features(path, cells)
                run_experiment(path, root / f"run{index}", "S", "T", cfg, ModelConfig(hidden=8))
                model, _, _ = load_checkpoint(root / f"run{index}" / "best.pt")
                states.append(model.state_dict())
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_unlabeled_target_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = batteries()
            for b in cells:
                if b.domain == "T":
                    b.soh = None
            path = root / "unlabeled.csv"
            save_features(path, cells)
            result = run_experiment(path, root / "run", "S", "T",
                                    TrainConfig(sequence_length=4, epochs=1, steps_per_epoch=1,
                                                batch_per_domain=2, lambda_phy=0, use_gate=False),
                                    ModelConfig(hidden=8))
            self.assertEqual(result["target_test"]["labeled_windows"], 0)
            self.assertGreater(result["target_test"]["total_windows"], 0)

    def test_target_label_changes_cannot_change_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = batteries()
            cfg = TrainConfig(sequence_length=4, epochs=2, steps_per_epoch=2, batch_per_domain=3, patience=2)
            model_cfg = ModelConfig(hidden=8, min_support=2)
            for index in range(2):
                if index:
                    for b in cells:
                        if b.domain == "T":
                            b.soh[:] = 0.5
                path = root / f"data{index}.csv"
                save_features(path, cells)
                run_experiment(path, root / f"run{index}", "S", "T", cfg, model_cfg)
            first, scaler, _ = load_checkpoint(root / "run0" / "best.pt")
            second, _, _ = load_checkpoint(root / "run1" / "best.pt")
            for key, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
            before = {k: v.clone() for k, v in first.state_dict().items()}
            pool = WindowPool([b.unlabeled() for b in cells], scaler, length=4)
            records = predict(first, pool, "cpu")
            self.assertEqual(len(records), 18 * 6)
            for key, value in first.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertEqual((root / "run0" / "target_test_predictions.csv").read_bytes(),
                             (root / "run1" / "target_test_predictions.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
