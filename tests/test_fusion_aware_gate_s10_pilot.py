import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_fusion_aware_gate_s10_pilot as fusion


class FusionAwareGateTests(unittest.TestCase):
    def test_grid(self):
        grid=fusion.s10_grid(.25)
        self.assertEqual(len(grid),9);self.assertEqual(grid[4],(0,.25))

    def test_s10_recomputes_profile(self):
        sw=np.array([.1,.2]);sx=.1
        a=fusion.Profile().approximate(sx*sw/.1);b=fusion.Profile().approximate(sx*sw/.2)
        self.assertFalse(np.array_equal(a["multiplier"],b["multiplier"]) and np.array_equal(a["shift"],b["shift"]))

    def test_feasible_bounds_candidate_specific(self):
        self.assertNotEqual(fusion.mixed.effective_bounds(np.array([5])),fusion.mixed.effective_bounds(np.array([20])))

    def test_output_contracts(self):
        self.assertEqual(fusion.mixed.output_range("GATE_PREACT_INT10"),(-512,511))
        self.assertEqual(fusion.mixed.output_range("ORDINARY_INT8"),(-128,127))

    def test_lut_address_and_size(self):
        np.testing.assert_array_equal(fusion.gelu_base.address([-512,0,511]),[0,512,1023])
        lut=fusion.gelu_base.make_lut(.1,.1);self.assertEqual(len(lut),1024);self.assertEqual(len(lut.tobytes()),1024)

    def test_inner_standalone_ignores_fusion(self):
        g=np.array([[0.,1.],[-1.,2.]]);q=np.array([[0,10],[-10,20]],np.int16);labels=np.array(["prefill","decode"])
        selected,_=fusion.inner_sact(g,q,labels,.1)
        self.assertIn("total_nmse_balanced",selected);self.assertNotIn("fusion_full_nmse",selected)

    def test_outer_uses_fusion_and_incumbent(self):
        def c(j,score):
            return dict(candidate_j=j,s10_candidate=1*2**(j/8),fusion_full_nmse_balanced=score,fusion_full_worst_group=score,
                        fusion_gate_only_nmse_balanced=10,gelu_nmse_balanced=10,gate_nmse_balanced=10,gate_output_saturation_rate=0)
        selected,old=fusion.select_outer([c(0,2),c(1,1)],1.)
        self.assertEqual(selected["candidate_j"],1);self.assertEqual(old["candidate_j"],0)

    def test_incumbent_wins_worse_candidate(self):
        def c(j,score):
            return dict(candidate_j=j,s10_candidate=1*2**(j/8),fusion_full_nmse_balanced=score,fusion_full_worst_group=score,
                        fusion_gate_only_nmse_balanced=score,gelu_nmse_balanced=score,gate_nmse_balanced=score,gate_output_saturation_rate=0)
        self.assertEqual(fusion.select_outer([c(0,1),c(1,2)],1.)[0]["candidate_j"],0)

    def test_balanced_and_fusion_metrics(self):
        self.assertEqual(fusion.balanced([1.,3.]),2.)
        a=np.array([[1.,2.]]);u=np.array([[2.,3.]])
        self.assertEqual(fusion.mixed.nmse(a*u,a*u),0.)

    def test_tie_break_deterministic(self):
        base=dict(fusion_full_nmse_balanced=1,fusion_full_worst_group=1,fusion_gate_only_nmse_balanced=1,
                  gelu_nmse_balanced=1,gate_nmse_balanced=1,gate_output_saturation_rate=0)
        a=dict(base,s10_candidate=1.);b=dict(base,s10_candidate=2.)
        self.assertLess(fusion.outer_key(a,1.),fusion.outer_key(b,1.))

    def test_qparam_binary(self):
        p=fusion.Profile();m=np.array([1,2]);s=np.array([3,4]);data=p.pack(m,s).astype("<u4").tobytes()
        self.assertEqual(len(data),8)

    def test_baseline_hash_immutability_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x";p.write_text("a");before=fusion.sha(p);p.write_text("b");self.assertNotEqual(before,fusion.sha(p))


if __name__=="__main__":unittest.main()
