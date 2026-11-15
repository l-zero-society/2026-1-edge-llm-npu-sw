import json
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpalu_width_sweep_pilot as sweep


class GPALUWidthSweepTests(unittest.TestCase):
    def test_signed_ranges(self):
        self.assertEqual((-128,127),sweep.signed_bounds(8));self.assertEqual((-512,511),sweep.signed_bounds(10));self.assertEqual((-2048,2047),sweep.signed_bounds(12))
    def test_int8_product_fits_int16(self): self.assertLessEqual(max(abs(-128*-128),abs(-128*127),abs(127*127)),32767)
    def test_ms_constraints_and_determinism(self):
        a=sweep.full.approximate_scalar(.123);b=sweep.full.approximate_scalar(.123);self.assertEqual(a,b);self.assertTrue(0<a['M_G']<=65535);self.assertTrue(0<=a['S_G']<=31)
    def test_signed_rne(self):
        _,q,_=sweep.width_codes(np.array([-3,-1,1,3]),.5,1,1,8);np.testing.assert_array_equal(q,[ -2,0,0,2])
    def test_effective_scale(self): self.assertEqual(2.,sweep.effective_scale(.5,256,10))
    def test_width_clipping(self):
        p=np.array([-10000,10000]);
        for bits in sweep.WIDTHS:
            _,q,_=sweep.width_codes(p,1.,1,0,bits);lo,hi=sweep.signed_bounds(bits);self.assertGreaterEqual(q.min(),lo);self.assertLessEqual(q.max(),hi)
    def test_same_upstream_width_independent(self):
        self.assertNotIn('bits',sweep.prepare_layer.__code__.co_varnames)
    def test_e2e_not_calibration_selection(self): self.assertNotIn('e2e',sweep.select_candidate.__code__.co_names)
    def test_fp_down_isolation(self):
        self.assertNotIn('direct_down',sweep.patch_width.__code__.co_names)
        self.assertNotIn('down_spec',sweep.patch_width.__code__.co_names)
    def test_norm_rope_not_patched(self):
        names=sweep.patch_width.__code__.co_names;self.assertNotIn('norm',names);self.assertNotIn('rotary',names)
    def test_all_width_artifact_contract(self): self.assertEqual((8,10,12),sweep.WIDTHS)


if __name__=='__main__':unittest.main()
