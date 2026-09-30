import unittest
from unittest.mock import patch
import numpy as np
from scripts import compare_k_selectors_e2e as c

class SelectorTests(unittest.TestCase):
    def test_zero(self):
        q,k,_=c.select_absmax(np.zeros((2,4)),1.,np.array([16,20]))
        np.testing.assert_array_equal(k,0);np.testing.assert_array_equal(q,0)
    def test_inside(self):
        _,k,_=c.select_absmax(np.array([[508.,0.]]),1.,np.array([16,20]))
        self.assertEqual(k[0],2)
    def test_low(self):
        _,k,d=c.select_absmax(np.array([[1e-8]]),1.,np.array([31]))
        self.assertEqual(k[0],0);self.assertEqual(d['clamped'],1)
    def test_high(self):
        _,k,d=c.select_absmax(np.array([[1e8]]),1.,np.array([2,20]))
        self.assertEqual(k[0],2);self.assertEqual(d['clamped'],1)
    def test_shifts(self):
        _,k,d=c.select_absmax(np.array([[1e-8],[1e8]]),1.,np.array([3,28]))
        self.assertGreaterEqual(d['effective_shift_min'],0);self.assertLessEqual(d['effective_shift_max'],31)
    def test_manual_output(self):
        x=np.array([[[508.,-200.]]]); wq=np.array([[2,3]],np.int8)
        params={'multiplier':np.array([7]),'shift':np.array([5])}
        with patch.object(c.row,'select_rows',c.select_absmax):
            y,_=c.row.apply_rows(x,wq,1.,.1,params)
        q=c.row.quantize_input(x[0],4.)
        acc=c.row.integer_dot(q,wq,'fp64-exact')
        expected=c.Profile().apply(acc,params['multiplier'],params['shift']-2)*.1
        np.testing.assert_allclose(y[0],expected,rtol=1e-6)
    def test_no_mse_in_hw_selector(self):
        with patch.object(c,'MSE3',side_effect=AssertionError):
            c.select_absmax(np.ones((1,3)),1.,np.array([16]))

if __name__=='__main__':unittest.main()
