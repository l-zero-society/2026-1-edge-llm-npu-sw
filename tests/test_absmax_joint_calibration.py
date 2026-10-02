"""Tiny numerical/mock tests; no model or artifact loading required."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import absmax_joint_calibration as joint
from compare_k_selectors_e2e import select_absmax


class JointCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 2.], [3., 4.]])
        self.labels = np.array(["prefill", "decode"])
        self.ref = np.ones((2, 1))
        self.wq = np.ones((1, 2), np.int8)
        self.sw = np.ones(1)

    def evaluate(self, sx=1., s10=1., cache=None):
        return joint.evaluate_joint_candidate(
            self.x, self.labels, self.ref, self.wq, self.sw, sx, s10,
            sx_j=0, s10_i=0, acc_cache={} if cache is None else cache)

    def search(self, js, indices):
        return joint.search_joint_pair(self.x, self.labels, self.ref,
                                      self.wq, self.sw, 1., 1.,
                                      sx_js=js, s10_is=indices)

    @staticmethod
    def fake_candidate(x, labels, ref, wq, sw, sx, s10, *, sx_j, s10_i, acc_cache):
        score = 1. + abs(sx_j) + abs(s10_i)
        return dict(sx=sx, s10=s10, sx_j=sx_j, s10_i=s10_i,
                    score=score, worst=score, clip_rate=0., feasible=True)

    def test_cartesian_product(self):
        with patch.object(joint, "evaluate_joint_candidate", side_effect=self.fake_candidate) as evaluate:
            result = self.search([-1, 0, 1], [-1, 0, 1])
        self.assertEqual(evaluate.call_count, 9)
        self.assertEqual({(c["sx_j"], c["s10_i"]) for c in result["candidates"]},
                         {(j, i) for j in [-1, 0, 1] for i in [-1, 0, 1]})

    def test_old_pair_added_without_zero(self):
        with patch.object(joint, "evaluate_joint_candidate", side_effect=self.fake_candidate):
            result = self.search([1, 2], [1, 2])
        self.assertEqual(len(result["candidates"]), 5)
        self.assertTrue(result["old"]["is_old_pair"])

    def test_worse_candidates_keep_old(self):
        with patch.object(joint, "evaluate_joint_candidate", side_effect=self.fake_candidate):
            result = self.search([-1, 1], [-1, 1])
        self.assertIs(result["selected"], result["old"])

    def test_no_regression(self):
        result = self.search([-1, 0, 1], [-1, 0, 1])
        self.assertLessEqual(result["selected"]["score"],
                             result["old"]["score"] + joint.LOCAL_REGRESSION_TOL)

    def test_regression_guard_is_runtime_error(self):
        with patch.object(joint, "evaluate_joint_candidate", side_effect=self.fake_candidate), \
                patch.object(joint, "candidate_key", side_effect=lambda c, *args: -c["score"]):
            with self.assertRaisesRegex(RuntimeError, "locally regressive"):
                self.search([0, 1], [0, 1])

    def test_candidate_specific_shifts(self):
        first = dict(multiplier=np.array([1]), shift=np.array([4]), status=np.array(["ok"]))
        second = dict(multiplier=np.array([1]), shift=np.array([5]), status=np.array(["ok"]))
        with patch.object(joint.Profile, "approximate", side_effect=[first, second]), \
                patch.object(joint, "select_absmax", wraps=select_absmax) as selector:
            self.evaluate(s10=1.)
            self.evaluate(s10=2.)
        np.testing.assert_array_equal(selector.call_args_list[0].args[2], [4])
        np.testing.assert_array_equal(selector.call_args_list[1].args[2], [5])

    def test_effective_shift_rejection(self):
        with patch.object(joint, "select_absmax", return_value=(
                np.ones_like(self.x, dtype=np.int8), np.array([100, 100]), {})), \
                patch.object(joint.base, "integer_dot") as dot:
            candidate = self.evaluate()
        self.assertFalse(candidate["feasible"])
        self.assertIn("effective shift", candidate["rejection_reason"])
        dot.assert_not_called()

    def test_invalid_profile_rejection(self):
        params = dict(multiplier=np.array([0]), shift=np.array([0]), status=np.array(["inaccurate"]))
        with patch.object(joint.Profile, "approximate", return_value=params), \
                patch.object(joint, "select_absmax") as selector:
            self.assertFalse(self.evaluate()["feasible"])
        selector.assert_not_called()

    def test_identical_codes_reuse_acc_even_with_different_k(self):
        cache = {}
        q = np.ones_like(self.x, dtype=np.int8)
        with patch.object(joint, "select_absmax", side_effect=[
                (q, np.zeros(2, dtype=int), {}), (q, -np.ones(2, dtype=int), {})]), \
                patch.object(joint.base, "integer_dot", wraps=joint.base.integer_dot) as dot:
            self.evaluate(cache=cache)
            self.evaluate(cache=cache)
        self.assertEqual(dot.call_count, 1)
        self.assertEqual(len(cache), 1)

    def test_different_codes_separate_acc(self):
        cache = {}
        q = np.ones_like(self.x, dtype=np.int8)
        with patch.object(joint, "select_absmax", side_effect=[
                (q, np.zeros(2, dtype=int), {}), (q * 2, np.zeros(2, dtype=int), {})]), \
                patch.object(joint.base, "integer_dot", wraps=joint.base.integer_dot) as dot:
            self.evaluate(cache=cache)
            self.evaluate(cache=cache)
        self.assertEqual(dot.call_count, 2)

    def test_tie_prefers_old_parameters(self):
        old = dict(score=1., worst=2., clip_rate=0., sx=1., s10=1., sx_j=0, s10_i=0)
        far = dict(old, sx=2., sx_j=4)
        self.assertLess(joint.candidate_key(old, 1., 1.), joint.candidate_key(far, 1., 1.))

    def test_balanced_pooled_group_score(self):
        ref = np.array([[1.], [2.], [1.]])
        out = np.array([[2.], [4.], [3.]])
        scores = joint.output_scores(out, ref, np.array(["prefill", "prefill", "decode"]))
        self.assertEqual(scores["prefill_nmse"], 1.)
        self.assertEqual(scores["decode_nmse"], 4.)
        self.assertEqual(scores["score"], 2.5)

    def test_worst_group_score(self):
        scores = joint.output_scores(np.array([[2.], [4.]]), self.ref, self.labels)
        self.assertEqual(scores["worst"], 9.)

    def test_wq_invariant(self):
        weight = np.ones((1, 2))
        bad = dict(wq=np.zeros((1, 2), np.int8), sx=1., s10=1.)
        with self.assertRaises(AssertionError):
            joint.calibrate_layer_joint(weight, bad, {"prefill": self.x}, sx_js=[0], s10_is=[0])

    def test_proposed_selected_params_and_unchanged_weight(self):
        old = dict(wq=self.wq, sx=1., s10=1., params={})
        selected = self.evaluate(sx=2., s10=2.)
        proposed = joint.proposed_spec(old, selected)
        self.assertIs(proposed["params"], selected["params"])
        self.assertIs(proposed["wq"], old["wq"])
        self.assertEqual(old["sx"], 1.)
        self.assertEqual(proposed["sx"], 2.)

    def test_gate_kl_boundary(self):
        old = dict(kl=1., nmse=1., top1=.5)
        self.assertTrue(joint.e2e_gate(old, dict(old, kl=1.05))["pass"])
        self.assertFalse(joint.e2e_gate(old, dict(old, kl=1.050001))["kl_pass"])

    def test_gate_nmse_boundary(self):
        old = dict(kl=1., nmse=1., top1=.5)
        self.assertTrue(joint.e2e_gate(old, dict(old, nmse=1.05))["pass"])
        self.assertFalse(joint.e2e_gate(old, dict(old, nmse=1.050001))["nmse_pass"])

    def test_gate_one_sided_top1_boundary(self):
        old = dict(kl=1., nmse=1., top1=.5)
        self.assertTrue(joint.e2e_gate(old, dict(old, top1=.49))["pass"])
        self.assertTrue(joint.e2e_gate(old, dict(old, top1=.8))["pass"])
        self.assertFalse(joint.e2e_gate(old, dict(old, top1=.489999))["top1_pass"])

    def test_selector_identity(self):
        self.assertIs(joint.select_absmax, select_absmax)

    def test_zero_energy_and_missing_group(self):
        scores = joint.output_scores(np.zeros((2, 1)), np.zeros((2, 1)),
                                     np.array(["prefill", "prefill"]))
        self.assertEqual(scores["score"], 0.)
        self.assertIsNone(scores["decode_nmse"])


if __name__ == "__main__":
    unittest.main()
