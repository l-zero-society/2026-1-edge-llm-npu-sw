import sys
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_activation_lut_calibration as lut

class ActivationLUTTests(unittest.TestCase):
    def test_addresses(self):
        np.testing.assert_array_equal(lut.address([-512,0,511]),[0,512,1023])
    def test_unique_addresses(self):
        self.assertEqual(len(np.unique(lut.address(np.arange(-512,512)))),1024)
    def test_gelu_deterministic(self):
        x=np.array([[-2.,0.,2.]])
        np.testing.assert_array_equal(lut.gelu(x),lut.gelu(x))
    def test_table_deterministic(self):
        np.testing.assert_array_equal(lut.table(.1,.1),lut.table(.1,.1))
    def test_table_format(self):
        t=lut.table(.1,.1)
        self.assertEqual(t.shape,(1024,));self.assertEqual(t.dtype,np.int8)
        self.assertEqual(len(t.tobytes()),1024)
        self.assertGreaterEqual(t.min(),-127);self.assertLessEqual(t.max(),127)
    def test_rne(self):
        np.testing.assert_array_equal(lut.qinput([.5,1.5,2.5,-.5,-1.5],1),[0,2,2,0,-2])
    def test_input_clipping(self):
        np.testing.assert_array_equal(lut.qinput([-999,999],1),[-512,511])
    def test_output_saturation(self):
        np.testing.assert_array_equal(lut.qoutput([-999,999],1),[-127,127])
    def test_sl_candidates(self):
        v=np.arange(100.)
        np.testing.assert_array_equal(lut.scales(v,511),np.percentile(v,lut.PERCENTILES)/511)
    def test_sa_candidates(self):
        self.assertEqual(lut.scales(np.ones(10),127),[1/127])
    def test_no_duplicates(self):
        self.assertEqual(len(lut.scales(np.zeros(10),511)),1)
    def test_decomposition(self):
        g=np.array([[.14,.26,1.17]]);q=lut.qinput(g,.1);a=lut.gelu(g);aq=lut.gelu(q*.1)
        al=lut.table(.1,.2)[lut.address(q)]*.2
        for x,y in [(aq,a),(al,aq),(al,a)]:
            self.assertAlmostEqual(lut.metric(x,y)['nmse'],float(np.sum((x-y)**2)/np.sum(y*y)))
    def test_product_metric(self):
        self.assertAlmostEqual(lut.metric(np.array([[2.,4.]]),np.array([[1.,2.]]))['nmse'],1)
    def test_balanced(self):
        rows=[dict(layer=0,x=1),dict(layer=0,x=3),dict(layer=1,x=6)]
        self.assertEqual(lut.balanced(rows,'x'),4)
    def test_gate(self):
        old=dict(kl=1.,nmse=1.,top1=.5)
        self.assertTrue(lut.gate(old,dict(kl=1.05,nmse=1.05,top1=.49))['passed'])
        self.assertFalse(lut.gate(old,dict(kl=1.06,nmse=1,top1=.5))['passed'])
    def test_frozen_hash_check(self):
        lut.verify_hashes({'a':'x'},{'a':'x'})
        with self.assertRaises(RuntimeError):lut.verify_hashes({'a':'x'},{'a':'y'})
    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';p.write_text(json.dumps(dict(identity='a',value=2)))
            self.assertEqual(lut.checkpoint(p,'a'),2)
            with self.assertRaises(RuntimeError):lut.checkpoint(p,'b')

if __name__=='__main__':unittest.main()
