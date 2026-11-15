import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_joint_mlp_consumer_pilot as joint


class JointMLPConsumerTests(unittest.TestCase):
    def candidate(self, **updates):
        value = dict(post_residual_nmse_balanced=1., post_residual_worst_group=1., down_nmse_balanced=1.,
                     down_nmse_worst=1., post_gpalu_nmse=1., gpalu_clip_rate=0., down_saturation_rate=0.,
                     gelu_nmse=1., movement=0., s10=1., s_act=1., s_h_target=1., s8_down=1.)
        value.update(updates); return value

    def test_external_interfaces_remain_int8(self):
        self.assertEqual((-128, 127), joint.mixed.output_range("ORDINARY_INT8"))

    def test_fullscale_effective_identity(self):
        self.assertAlmostEqual(.5 * 2**10 / 256, joint.prior.effective_scale(.5, 256, 10))

    def test_sact_not_prepruned(self):
        self.assertEqual(joint.PERCENTILES, (99.,99.5,99.9,99.95,99.99,99.995,100.))

    def test_down_scale_changes_ms(self):
        sw=np.array([.1,.2]); a=joint.Profile().approximate(.3*sw/.4); b=joint.Profile().approximate(.3*sw/.8)
        self.assertFalse(np.array_equal(a["shift"],b["shift"]) and np.array_equal(a["multiplier"],b["multiplier"]))

    def test_direct_down_no_second_quantizer(self):
        self.assertNotIn("select_absmax", joint.direct_down_scale.__code__.co_names)

    def test_residual_objective(self):
        labels=np.array(["prefill","decode"]); r=np.ones((2,2)); d=np.zeros((2,2)); ref=np.ones((2,2))
        self.assertEqual(0., joint.balanced(joint.residual_objective(r,d,r,ref*0,labels).values()))

    def test_balanced_groups(self): self.assertEqual(2., joint.balanced([1.,3.]))

    def test_guarded_rejects_severe_local_error(self):
        good=self.candidate(post_residual_nmse_balanced=2.)
        bad=self.candidate(post_residual_nmse_balanced=1.,gelu_nmse=3.)
        _,guarded,_,_=joint.guarded_select([good,bad]); self.assertIs(guarded,good)

    def test_selection_deterministic(self):
        a=self.candidate(s10=1.); b=self.candidate(s10=2.)
        self.assertIs(min([b,a],key=joint.residual_key),a)

    def test_e2e_not_selection_input(self):
        self.assertNotIn("e2e", joint.residual_key.__code__.co_names)

    def test_historical_anchor_inclusion(self):
        rows=joint.s10_candidates(1.,2.,3.); values=[r[0] for r in rows]
        self.assertIn(2.,values); self.assertIn(3.,values)

    def test_no_dynamic_gpalu_scale(self):
        self.assertNotIn("row", joint.prior.fullscale_codes.__code__.co_names)

    def test_down_wq_sw_not_arguments_to_search(self):
        spec={"sw":np.array([.1]),"wq":np.array([[1]],np.int8),"sout":.2}
        original_wq=spec["wq"].copy(); original_sw=spec["sw"].copy()
        joint.direct_down_scale(np.array([[1]],np.int8),.3,spec,.4)
        np.testing.assert_array_equal(spec["wq"],original_wq); np.testing.assert_array_equal(spec["sw"],original_sw)


if __name__ == "__main__": unittest.main()
