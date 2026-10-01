import copy
import unittest
import numpy as np

from scripts import pilot_absmax_calibration as p


class PilotAbsmaxTests(unittest.TestCase):
    def test_positions(self):
        self.assertEqual(p.CAL_POSITIONS, [0, 5, 10, 15, 20, 25])
        self.assertEqual(p.VAL_POSITIONS, [0, 4, 8, 12, 16, 20])

    def test_selector_identity(self):
        self.assertIs(p.select_absmax, p.compare.select_absmax)

    def test_fixed_scope(self):
        self.assertEqual(len(p.CAL_POSITIONS), 6)
        self.assertEqual(p.RESERVOIR_CAPACITY, 32)
        self.assertEqual(p.CAL_RESPONSE_LIMIT, 16)
        self.assertEqual(len(p.VAL_POSITIONS) * p.VAL_TARGETS, 144)

    def test_candidates_and_no_expansion(self):
        self.assertEqual(p.SX_JS, [-2, -1, 0, 1, 2])
        self.assertEqual(p.S10_IS, [-4, -3, -2, -1, 0, 1, 2, 3, 4])
        self.assertEqual((min(p.SX_JS), max(p.SX_JS)), (-2, 2))
        self.assertEqual((min(p.S10_IS), max(p.S10_IS)), (-4, 4))

    def test_effective_shift(self):
        x = np.array([[1e-8], [1e8]])
        _, _, info = p.select_absmax(x, 1.0, np.array([3, 28]))
        self.assertGreaterEqual(info["effective_shift_min"], 0)
        self.assertLessEqual(info["effective_shift_max"], 31)

    def test_independent_caches(self):
        a, b, c = object(), object(), object()
        p.independent([a, b, c])
        with self.assertRaises(AssertionError):
            p.independent([a, b, a])

    def test_classification(self):
        old = dict(kl=.1, nmse=.2, top1=.9)
        self.assertEqual(p.classify(old, dict(kl=.095, nmse=.2, top1=.9), 0)["label"], "CLEAR_BENEFIT")
        self.assertEqual(p.classify(old, dict(kl=.099, nmse=.199, top1=.9), 0)["label"], "NEUTRAL_BUT_SAFE")
        self.assertEqual(p.classify(old, dict(kl=.106, nmse=.2, top1=.9), .2)["label"], "REGRESSION")
        self.assertEqual(p.classify(old, dict(kl=.1, nmse=.2, top1=.889), .2)["label"], "REGRESSION")

    def test_frozen_fingerprint(self):
        spec = {"x": dict(sx=.1, s10=.2, wq=np.array([[1]], np.int8),
                          params=dict(multiplier=np.array([3]), shift=np.array([5])))}
        before = p.compare.fingerprint(spec)
        self.assertEqual(before, p.compare.fingerprint(copy.deepcopy(spec)))
        spec["x"]["params"]["shift"][0] += 1
        self.assertNotEqual(before, p.compare.fingerprint(spec))


if __name__ == "__main__":
    unittest.main()
