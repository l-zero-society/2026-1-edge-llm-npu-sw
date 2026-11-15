import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gelu_standalone_calibration as gelu


class GeLUStandaloneCalibrationTests(unittest.TestCase):
    def test_signed_int10_addressing(self):
        np.testing.assert_array_equal(gelu.address([-512, 0, 511]), [0, 512, 1023])
        self.assertEqual(len(np.unique(gelu.address(np.arange(-512, 512)))), 1024)

    def test_deterministic_gelu_tanh(self):
        x = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_array_equal(gelu.gelu(x), gelu.gelu(x))

    def test_deterministic_lut_and_format(self):
        a = gelu.make_lut(0.1, 0.05)
        b = gelu.make_lut(0.1, 0.05)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1024,))
        self.assertEqual(a.dtype, np.int8)
        self.assertEqual(len(a.tobytes()), 1024)
        self.assertGreaterEqual(int(a.min()), -127)
        self.assertLessEqual(int(a.max()), 127)

    def test_rne(self):
        np.testing.assert_array_equal(gelu.rne([0.5, 1.5, 2.5, -0.5, -1.5]), [0, 2, 2, 0, -2])

    def test_percentile_candidates(self):
        values = np.arange(1, 101, dtype=float)
        first = gelu.percentile_candidates(values)
        self.assertEqual(first, gelu.percentile_candidates(values))
        self.assertEqual(first[-1][0], 100.0)
        self.assertAlmostEqual(first[-1][1], 100.0 / 127.0)
        self.assertEqual(len({x[1] for x in first}), len(first))

    def test_saturation_accounting(self):
        result, _ = gelu.group_metrics(np.array([[50.0]]), 0.1, 0.001, gelu.make_lut(0.1, 0.001))
        self.assertGreater(result["saturation_rate"], 0)

    def test_observed_distribution_metric(self):
        result, _ = gelu.group_metrics(np.array([[0.0, 0.1, 0.1, 1.0]]), 0.1, 0.05, gelu.make_lut(0.1, 0.05))
        self.assertIn("total_nmse", result)
        self.assertGreaterEqual(result["output_code_utilization"], 0)

    def test_balanced_prefill_decode(self):
        self.assertEqual(gelu.balanced(1.0, 3.0), 2.0)
        self.assertEqual(gelu.balanced(None, 3.0), 3.0)

    def test_local_selection_and_absmax_retained(self):
        candidates = [
            {"score": 2.0, "table_nmse": 1.0, "worst_total_nmse": 2.0, "clipping_fraction": 0.0, "saturation_rate": 0.0, "s_act": 2.0, "retained_percentile": 100.0},
            {"score": 1.0, "table_nmse": 1.0, "worst_total_nmse": 1.0, "clipping_fraction": 0.01, "saturation_rate": 0.01, "s_act": 1.0, "retained_percentile": 99.0},
        ]
        self.assertEqual(gelu.select_candidate(candidates)["retained_percentile"], 99.0)
        with self.assertRaises(RuntimeError):
            gelu.select_candidate(candidates[1:])

    def test_scale_pair_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            value = {"clustering_performed": False, "pairs": [{"layer": 0, "s10": 0.1, "S_act": 0.2}]}
            path.write_text(json.dumps(value))
            self.assertEqual(json.loads(path.read_text()), value)

    def test_gate(self):
        old = {"kl": 1.0, "nmse": 1.0, "top1": 0.5}
        self.assertTrue(gelu.gate(old, {"kl": 1.05, "nmse": 1.05, "top1": 0.49})["passed"])
        self.assertFalse(gelu.gate(old, {"kl": 1.051, "nmse": 1.0, "top1": 0.5})["passed"])

    def test_hash_fingerprint_changes(self):
        self.assertNotEqual(gelu.digest({"a": 1}), gelu.digest({"a": 2}))


if __name__ == "__main__":
    unittest.main()
