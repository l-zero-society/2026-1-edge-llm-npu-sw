from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_gpalu_fullscale_downaware_pilot as full


class FullscaleDownawareTests(unittest.TestCase):
    def test_int8_product_fits_int16(self):
        values=np.array([-128*127,127*127]);self.assertTrue(np.all((values>=-32768)&(values<=32767)))
    def test_ms_bounds_and_determinism(self):
        a=full.approximate_scalar(.123);b=full.approximate_scalar(.123);self.assertEqual(a,b);self.assertTrue(0<a["M_G"]<=65535 and 0<=a["S_G"]<=31)
    def test_signed_rne_positive_negative_half_even(self):
        ideal,hw,_=full.fullscale_codes(np.array([1,3,5,7,-1,-3,-5,-7]),.5,1,1);np.testing.assert_array_equal(hw,[0,2,2,4,0,-2,-2,-4]);np.testing.assert_array_equal(ideal,hw)
    def test_int8_saturation(self):
        _,hw,info=full.fullscale_codes(np.array([1000,-1000]),1.,1,0);np.testing.assert_array_equal(hw,[127,-128]);self.assertEqual(info["hw_clip_count"],2)
    def test_effective_scale_identity(self):self.assertEqual(full.effective_scale(.25,2,3),1.)
    def test_target_ratio(self):self.assertEqual(.25/.5,.5)
    def test_actual_reconstruction_uses_effective(self):
        rep=full.approximate_scalar(.3);effective=full.effective_scale(.2,rep["M_G"],rep["S_G"]);self.assertAlmostEqual(effective,.2/rep["alpha_hw"])
    def test_direct_down_has_no_input_quantizer(self):
        self.assertNotIn("select_absmax",full.direct_down.__code__.co_names)
    def test_down_ratio_and_changed_ms(self):
        sw=np.array([.1,.2]);a=full.Profile().approximate(.2*sw/.3);b=full.Profile().approximate(.4*sw/.3);self.assertFalse(np.array_equal(a["multiplier"],b["multiplier"]) and np.array_equal(a["shift"],b["shift"]))
    def test_down_wq_and_s8_invariant_shape(self):
        spec={"wq":np.array([[1]],np.int8),"sw":np.array([.1]),"sout":.2};before=spec["wq"].copy();self.assertEqual(spec["sout"],.2);np.testing.assert_array_equal(before,spec["wq"])
    def test_consumer_selector_uses_down_nmse(self):
        base={"down_nmse_worst":1.,"local_hw_nmse_balanced":1.,"hw_clip_rate":0.,"alpha_relative_error":0.,"down_output_saturation_rate":0.,"anchor_log2_offset":0.,"s_h_target":1.}
        a=dict(base,down_nmse_balanced=1.);b=dict(base,down_nmse_balanced=.5);self.assertLess(full.down_key(b),full.down_key(a))
    def test_error_cancellation_warning(self):
        self.assertTrue(full.cancellation_warning({"local_hw_nmse_balanced":2.1},{"local_hw_nmse_balanced":1.}))
    def test_e2e_not_in_selection_key(self):self.assertNotIn("e2e",full.down_key.__code__.co_names)
    def test_scale_anchor_deterministic(self):
        a=full.scale_anchors(np.arange(10.),np.arange(10.),.1,.2);b=full.scale_anchors(np.arange(10.),np.arange(10.),.1,.2);self.assertEqual(a,b)
    def test_baseline_hash_immutability(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x";p.write_text("a");before=full.sha(p);p.write_text("b");self.assertNotEqual(before,full.sha(p))


if __name__=="__main__":unittest.main()
