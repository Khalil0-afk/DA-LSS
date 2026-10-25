import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from da_lss.data import Battery, FEATURES
from da_lss.deployment import replay
from da_lss.experiment import TrainConfig, load_checkpoint, run_experiment
from da_lss.model import DALSS, ModelConfig
from da_lss.resources import count_operations, integrate_power, profile_resources
from da_lss.vehicle_adaptation import VehicleAdaptationConfig, adapt_vehicle
from da_lss.vehicle_data import capacity_soh, decode_can, match_references, match_signals
from test_core import batteries, save_features
from test_extended import checkpoint


class ResponseCommitmentTests(unittest.TestCase):
    def test_partial_charge_and_discharge_capacity_formulas(self):
        charged = capacity_soh([0, 1800, 3600], [1, 1, 1], [0.2, 0.45, 0.7], 2, max_gap_s=1800)
        discharged = capacity_soh([0, 1800, 3600], [-1, -1, -1], [0.7, 0.45, 0.2], 2, max_gap_s=1800)
        self.assertAlmostEqual(charged["usable_capacity_ah"], 2)
        self.assertAlmostEqual(discharged["soh_fraction"], 1)
        with self.assertRaises(ValueError):
            capacity_soh([0, 100], [1, 1], [0.8, 0.5], 2, max_gap_s=100)
        with self.assertRaises(ValueError):
            capacity_soh([0, 100], [1, 1], [0.8, 0.81], 2, max_gap_s=100)

    def test_can_mapping_and_past_only_current_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {"messages": [{"frame_id": 256, "signals": [
                {"name": "cell_voltage_v", "scope": "cell", "byte_offset": 0, "byte_length": 2,
                 "byte_order": "little", "signed": False, "scale": 0.001, "offset": 0},
                {"name": "pack_current_a", "scope": "pack", "byte_offset": 2, "byte_length": 2,
                 "byte_order": "little", "signed": True, "scale": 0.01, "offset": 0}]}]}
            (root / "map.json").write_text(json.dumps(mapping))
            with (root / "can.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["vehicle_id", "battery_id", "segment_id", "timestamp", "frame_id", "data_hex"])
                writer.writerow(["v1", "c1", 1, 1, "0x100", (3300).to_bytes(2, "little").hex() + (-150).to_bytes(2, "little", signed=True).hex()])
            decode_can(root / "can.csv", root / "map.json", root / "decoded.csv")
            result = match_signals(root / "decoded.csv", root / "matched.csv", 0)
            self.assertEqual(result["matched_samples"], 1)
            with (root / "matched.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertAlmostEqual(float(row["cell_voltage_v"]), 3.3)
            self.assertAlmostEqual(float(row["pack_current_a"]), -1.5)
            with (root / "observations.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["vehicle_id", "battery_id", "segment_id", "timestamp", "signal", "scope", "value"])
                writer.writerow(["v1", "c1", 1, 1, "cell_voltage_v", "cell", 3.3])
                writer.writerow(["v1", "", 1, 2, "pack_current_a", "pack", 1.5])
            self.assertEqual(match_signals(root / "observations.csv", root / "past.csv", 10)["matched_samples"], 0)

    def test_bms_matching_does_not_broadcast_pack_or_future_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "predictions.csv").write_text("vehicle_id,battery_id,timestamp,prediction\nv1,c1,1,0.9\nv1,c2,1,0.9\n")
            (root / "references.csv").write_text("vehicle_id,battery_id,timestamp,bms_soh,scope\nv1,c1,2,0.95,cell\n")
            match_references(root / "predictions.csv", root / "references.csv", root / "matched.csv", 10)
            with (root / "matched.csv").open() as f:
                self.assertTrue(all(row["bms_soh"] == "" for row in csv.DictReader(f)))
            (root / "pack.csv").write_text("vehicle_id,battery_id,timestamp,bms_soh,scope\nv1,c1,0,0.95,pack\n")
            with self.assertRaises(ValueError):
                match_references(root / "predictions.csv", root / "pack.csv", root / "forbidden.csv", 10)

    def test_nonuniform_power_integration_matches_energy_per_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power.csv"
            path.write_text("timestamp_s,power_w\n0,10\n1,20\n3,20\n")
            result = integrate_power(path, 5)
            self.assertAlmostEqual(result["platform_energy_j"], 55)
            self.assertAlmostEqual(result["mean_platform_power_w"], 55 / 3)
            self.assertAlmostEqual(result["energy_per_window_mj"], 11000)

    def test_operation_scope_and_stride_amortization(self):
        model = DALSS(ModelConfig(hidden=8))
        counts = count_operations(model, 4)
        self.assertGreater(counts["macs_per_window"], 0)
        self.assertEqual(counts["macs_per_window"] / 32, counts["macs_per_new_valid_point_stride32"])
        self.assertFalse(counts["domain_classifier_included"])
        expected = (2 * counts["macs_per_window"] + counts["other_accounted_flops"]) / 1e6
        self.assertEqual(counts["accounted_mflops_per_window"], expected)

    def test_resource_report_leaves_unmeasured_power_unset(self):
        try:
            import psutil
        except ImportError:
            self.skipTest("Optional psutil not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(root / "model.pt", DALSS(ModelConfig(hidden=8)))
            raw = [(np.linspace(3.2, 3.4, 32), np.ones(32))] * 4
            result = profile_resources(root / "model.pt", root / "out", repetitions=2, warmup=1, raw_segments=raw)
            self.assertIsNone(result["power_w"])
            self.assertGreater(result["peak_process_rss_mb_sampled"], 0)
            self.assertIn("feature extraction", result["end_to_end_includes"])

    def test_shared_cell_ids_remain_isolated_by_vehicle_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(root / "model.pt", DALSS(ModelConfig(hidden=8)))
            with (root / "replay.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["domain", "battery_id", "vehicle_id", "charging_protocol", "timestamp", *FEATURES, "soh"])
                for t in range(6):
                    for vehicle, protocol in (("v1", "AC_slow"), ("v2", "DC_fast")):
                        writer.writerow(["EV", "cell_01", vehicle, protocol, t, 3.3, 1.5, 0.2, 0.01, 0.95])
            result = replay(root / "model.pt", root / "replay.csv", root / "out", stride=2)
            self.assertEqual(result["prediction_count"], 4)
            self.assertEqual(set(result["per_vehicle"]), {"v1", "v2"})
            self.assertEqual(set(result["per_charging_protocol"]), {"AC_slow", "DC_fast"})

    def test_vehicle_adaptation_is_fixed_budget_label_blind_and_vehicle_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            save_features(source, batteries())
            run_experiment(source, root / "pretrained", "S", "T",
                           TrainConfig(sequence_length=4, epochs=1, steps_per_epoch=1, batch_per_domain=2),
                           ModelConfig(hidden=8))
            cells = [Battery("EV", f"cell_{i}", np.arange(8),
                             np.random.default_rng(i).normal(size=(8, 4)).astype(np.float32), np.full(8, 0.95))
                     for i in range(4)]
            reserved = [f"test_{i}" for i in range(20)]
            cfg = VehicleAdaptationConfig(windows_per_vehicle=2, stride=2, batch_per_domain=2)
            for repeat in range(2):
                path = root / f"adapt{repeat}.csv"
                with path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["domain", "battery_id", "vehicle_id", "cycle", *FEATURES, "soh"])
                    for i, b in enumerate(cells):
                        for cycle, features in zip(b.cycles, b.features):
                            writer.writerow([b.domain, b.battery_id, f"adapt_{i}", cycle, *features, 0.95 - repeat * 0.2])
                result = adapt_vehicle(root / "pretrained/best.pt", source, path, reserved, root / f"out{repeat}", cfg)
                self.assertEqual(result["updates_completed"], 40)
                self.assertEqual(result["adaptation_window_count"], 8)
                self.assertFalse(result["held_out_ev_test_records_loaded"])
            first, _, _ = load_checkpoint(root / "out0/best.pt")
            second, _, _ = load_checkpoint(root / "out1/best.pt")
            for name, tensor in first.state_dict().items():
                torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                adapt_vehicle(root / "pretrained/best.pt", source, root / "adapt0.csv",
                              ["adapt_0", *reserved[1:]], root / "overlap", cfg)


if __name__ == "__main__":
    unittest.main()
