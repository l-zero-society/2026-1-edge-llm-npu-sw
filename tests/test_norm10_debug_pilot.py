"""Small arithmetic tests; no model or GGUF loading."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_norm10_debug_pilot as n

class Norm10DebugTests(unittest.TestCase):
    def test_bounds(self):
        q,_=n.old.quantize10(np.array([-1000.,1000.]),1.);np.testing.assert_array_equal(q,[-512,511])
    def test_accumulator(self): self.assertEqual(n.old.accumulator_bits(10,4096),35)
    def test_inv_n(self):self.assertEqual(n.inv_n(2048),512);self.assertEqual(n.inv_n(3),349525)
    def test_mean_multiply_shift(self):self.assertEqual(n.mean_sq([1,1,2]),1)
    def test_epsilon(self):self.assertEqual(n.epsilon_q(.0001,.01),1);self.assertEqual(n.epsilon_q(.00025,.01),2)
    def test_index(self):
        for value,expected in [(0,(0,0)),(1,(512,0)),(2,(512,1)),(3,(768,1)),(1023,(1023,9)),(1024,(512,10))]:
            self.assertEqual(n.old.normalized_index(value),expected)
    def test_table(self):self.assertEqual(len(n.rsqrt_table()),1024);self.assertEqual(n.rsqrt_table()[512],16384)
    def test_odd(self):self.assertEqual(n.corrected_scale(16384,1),11585)
    def test_even(self):self.assertEqual(n.corrected_scale(16384,2),8192)
    def test_phase2(self):
        q,_,_=n.phase2([-3,-1,1,3],8192);np.testing.assert_array_equal(q,[-2,-1,0,1])
    def test_saturation(self):
        q,_,_=n.phase2([-512,511],32768);np.testing.assert_array_equal(q,[-512,511])
    def test_golden_vector(self):
        spec=dict(input_scale=1.,output_scale=1.,eps=0.)
        y,d=n.exact(np.array([[2.,-2.,2.,-2.]]),spec,np.zeros(4),trace=True)
        np.testing.assert_array_equal(y,[[1,-1,1,-1]])
        tr=d['traces'][0];self.assertEqual(tr['sq_sum'],16);self.assertEqual(tr['meanSq'],4);self.assertEqual(tr['adjusted_scale_integer'],8192)
    def test_old_mismatch(self):
        x=np.array([[2.,-2.]])
        spec=dict(input_scale=1.,output_scale=.01,eps=0.)
        new,_=n.exact(x,spec,np.zeros(2))
        old,_=n.old.rms_hw_numpy(x,1.,.01,np.zeros(2),0.,n.old.norm_tables(14))
        self.assertTrue(n.mismatch(old,new));self.assertFalse(n.mismatch(old,old))
    def test_first_divergence(self):
        initial=dict(nmse=0.,top1=1.,hidden_nmse=0.,input_clip=0.,output_clip=0.,newly_enabled_module='',number_of_int10_norms=0)
        current=dict(initial,nmse=.1,newly_enabled_module='n0',number_of_int10_norms=1)
        self.assertEqual(n.first_jump([initial,current])['module'],'n0')
        self.assertFalse(n.first_jump([initial])['found'])
    def test_completion_guard(self):
        for tests,report,push in [(False,True,True),(True,False,True),(True,True,False)]:self.assertFalse(n.done_allowed(tests,report,push))
        self.assertTrue(n.done_allowed(True,True,True))
    def test_rtl_core_scale_explicit(self):
        x=np.array([[1.,-1.]])
        a,_=n.exact(x,dict(input_scale=1.,output_scale=.25,eps=0),np.zeros(2))
        np.testing.assert_array_equal(a,[[.25,-.25]])

if __name__=='__main__':unittest.main()
