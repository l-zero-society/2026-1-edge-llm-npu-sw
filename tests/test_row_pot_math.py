import sys
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from test_e2e_row_pot import select_rows,apply_rows,check_baseline
from test_e2e_linear_quant import quantized_linear
from static_quant.hardware import Profile
from test_factorized_row_requant import row_path


class RowPoTTests(unittest.TestCase):
    def fixture(self):
        return dict(wq=np.array([[2,-3],[4,5]],np.int8),sx=1.,s10=.05,k=0,
                    params=dict(multiplier=np.array([13,23]),shift=np.array([12,15])))

    def test_k_zero_is_existing_path(self):
        x=np.array([[[127.,17.],[127.,-30.]]]);s=self.fixture()
        _,ks,_=select_rows(x[0],s['sx'],s['params']['shift']);np.testing.assert_array_equal(ks,0)
        a,_=apply_rows(x,**s);b,_=quantized_linear(x,**s);np.testing.assert_array_equal(a,b)

    def test_input_scale_and_shift_only_no_channel_regeneration(self):
        x=np.array([[[31.75,2.25],[508.,-200.]]]);s=self.fixture()
        before={k:v.copy() for k,v in s['params'].items()}
        with patch.object(Profile,'approximate',side_effect=AssertionError('regenerated')):
            y,_=apply_rows(x,**s);_,ks,_=select_rows(x[0],s['sx'],s['params']['shift'])
        for row,k in enumerate(ks):
            _,raw=row_path(x[0,row:row+1],s['wq'],s['sx'],s['params'],2.**int(k),int(k))
            np.testing.assert_array_equal(y[0,row],(np.clip(raw,-512,511)*s['s10']).astype(np.float32)[0])
        for key in before:np.testing.assert_array_equal(before[key],s['params'][key])
        self.assertEqual(list(ks),[-2,2])

    def test_selection_is_activation_only_and_minimum_mse(self):
        rng=np.random.default_rng(43);x=rng.normal(size=(11,13))*20;shift=np.array([15,24])
        q,k,_=select_rows(x,.2,shift)
        for i,row in enumerate(x):
            k0=int(np.rint(np.log2(np.max(np.abs(row))/127/.2)))
            candidates=[v for v in [k0,k0-1,k0+1] if -8<=v<=7 and (shift-v>=0).all() and (shift-v<=31).all()]
            errors=[np.square(np.clip(np.rint(row/(.2*2.**v)),-127,127)*(.2*2.**v)-row).mean() for v in candidates]
            self.assertEqual(k[i],candidates[int(np.argmin(errors))])

    def test_feasibility_filters_and_empty_set_errors(self):
        x=np.array([[31.75,0.]])
        _,k,stats=select_rows(x,1.,np.array([15,30]))
        self.assertEqual(int(k[0]),-1);self.assertEqual(stats['bos_shift_restricted_rows'],1)
        self.assertGreaterEqual(stats['effective_shift_min'],0);self.assertLessEqual(stats['effective_shift_max'],31)
        with self.assertRaises(ValueError):select_rows(np.array([[1.,0.]]),1.,np.array([15,30]))

    def test_zero_rows_and_finite_validation(self):
        q,k,_=select_rows(np.zeros((2,7)),1.,np.array([10,31]));np.testing.assert_array_equal(q,0);np.testing.assert_array_equal(k,0)
        for bad in [np.nan,np.inf]:
            with self.assertRaises(ValueError):select_rows(np.array([[bad]]),1.,np.array([12]))

    def test_baseline_reproduction_checks_all_statistics(self):
        obj=dict(layers=[{'all':{'se':1.}}],normalized_hidden={'se':2.},logits={'fp_nll_sum':3.,'nll_sum':4.})
        check_baseline(obj,obj)
        changed=dict(obj,logits={'fp_nll_sum':3.,'nll_sum':4.1})
        with self.assertRaises(AssertionError):check_baseline(changed,obj)


if __name__=='__main__':unittest.main()
