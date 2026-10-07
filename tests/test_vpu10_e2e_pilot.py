import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_vpu10_e2e_pilot as vpu


class VPU10Tests(unittest.TestCase):
    def test_int10_range(self): self.assertEqual((-512,511),vpu.signed_bounds())
    def test_accumulator_width(self): self.assertEqual(35,vpu.accumulator_bits(10,4096))
    def test_normalized_index(self):
        self.assertEqual((0,0),vpu.normalized_index(0));idx,exp=vpu.normalized_index(1);self.assertEqual(512,idx);self.assertEqual(0,exp)
    def test_delta_index(self):
        self.assertEqual(1023,vpu.delta_to_index(-512,511));self.assertEqual(7,vpu.delta_to_index(2,-5))
    def test_norm_tables(self):
        a,b=vpu.norm_tables(14),vpu.norm_tables(8);self.assertEqual(1024,len(a["rsqrt"]));self.assertLessEqual(b["rsqrt"].max(),1023)
    def test_exponent_corrections(self):
        self.assertEqual(256,vpu.adjust_rsqrt(512,2,8));self.assertEqual(128,vpu.adjust_recip(512,2));self.assertEqual(1024,vpu.adjust_recip(512,-1))
    def test_rms_arithmetic(self):
        x=np.array([[1.,-1.,2.,-2.]])
        y,d=vpu.rms_hw_numpy(x,.01,.01,np.zeros(4),1e-6,vpu.norm_tables(14));self.assertEqual(x.shape,y.shape);self.assertTrue(np.isfinite(y).all());self.assertIn("scale_lut_mse",d)
    def test_softmax_arithmetic(self):
        x=np.array([[1.,2.,3.]])
        y,d=vpu.softmax_hw_numpy(x,.01,vpu.norm_tables(14));self.assertAlmostEqual(1.,float(y.sum()),places=6);self.assertIn("exp_lut_mse",d)
    def test_rope_tables(self):
        c,s=vpu.rope_tables();self.assertEqual((1024,),c.shape);self.assertEqual(16384,int(c[0]));self.assertEqual(16384,int(s[256]))
    def test_q016_angle(self): self.assertEqual(256,vpu.q016_angle_index(1,.25));self.assertEqual(512,vpu.q016_angle_index(2,.25))
    def test_rope_saturation(self):
        c,s=vpu.rope_tables();x=np.full((1,4),1e6);cos=np.ones((1,4));sin=np.zeros((1,4));y,d=vpu.rope_hw_numpy(x,cos,sin,1.,c,s,True);self.assertLessEqual(y.max(),511);self.assertGreater(d["input_clipping_rate"],0)
    def test_gpalu_baseline_reuse(self):
        p=ROOT/"diagnostics/gpalu_width_sweep_pilot/int10/gpalu_parameters.json";d=json.loads(p.read_text());self.assertEqual(10,d["output_bits"]);self.assertEqual(18,len(d["layers"]))
    def test_e2e_not_selection(self): self.assertNotIn("e2e",vpu.collect_calibration.__code__.co_names)
    def test_done_order(self):
        text=(ROOT/"scripts/run_vpu10_e2e_pilot.py").read_text();self.assertLess(text.index("subprocess.run([\"git\",\"push\""),text.index("(OUT/\"DONE\").write_text"))


if __name__=="__main__":unittest.main()
