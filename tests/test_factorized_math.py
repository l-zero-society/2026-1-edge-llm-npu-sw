import copy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from test_factorized_row_requant import row_requant, row_path, calibrate, evaluate, choose_bos
from static_quant.core import quantize_input, quantize_weight, exact_accumulator
from static_quant.hardware import Profile


class FactorizedTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(81)
        x=rng.normal(size=(12,7));x[0,0]=15
        pos=np.tile(np.arange(6),2);w=rng.normal(size=(4,7));wq,sw=quantize_weight(w)
        baseline=dict(s_X=.12,s_10=.2)
        return x,pos,w,wq,sw,baseline

    def test_c_one_is_existing_path_exactly(self):
        acc=np.array([[-2**31,-1,1,2**31-1]],np.int64)
        params=dict(multiplier=np.array([65535,1,3,65535]),shift=np.array([0,1,1,31]))
        expected=Profile().apply(acc,params['multiplier'],params['shift'],saturate=False)
        np.testing.assert_array_equal(row_requant(acc,params),expected)
        np.testing.assert_array_equal(row_requant(acc,params,1.,0),expected)

    def test_arbitrary_uses_same_scalar_on_input_and_before_rounding(self):
        x,pos,w,wq,sw,base=self.fixture();c=2.3
        p=Profile().approximate(base['s_X']*sw/base['s_10'])
        q,actual=row_path(x,wq,base['s_X'],p,c)
        np.testing.assert_array_equal(q,quantize_input(x,base['s_X']*c))
        acc=exact_accumulator(q,wq)
        expected=np.array([[round(Fraction(int(a)*int(m),2**int(s))*Fraction.from_float(c))
                            for a,m,s in zip(row,p['multiplier'],p['shift'])] for row in acc])
        np.testing.assert_array_equal(actual,expected)

    def test_pot_exact_extended_shifts_and_signed_ties(self):
        acc=np.array([[-7,-3,-1,1,3,7],[2**31-1,-2**31,7,-7,5,-5]],np.int64)
        p=dict(multiplier=np.array([1,3,7,65535,2,1]),shift=np.array([0,1,2,31,4,1]))
        for k in range(-8,9):
            expected=np.array([[round(Fraction(int(a)*int(m),2**int(s))*Fraction(2)**k)
                                for a,m,s in zip(row,p['multiplier'],p['shift'])] for row in acc],np.int64)
            np.testing.assert_array_equal(row_requant(acc,p,2.**k,k),expected)
        # Applying the row factor after INT10 would give a different answer.
        p=dict(multiplier=np.array([1]),shift=np.array([0]))
        self.assertEqual(int(row_requant(np.array([[1024]]),p,.5,-1)[0,0]),512)

    def test_calibration_only_and_frozen_common_scale(self):
        x,pos,w,wq,sw,base=self.fixture()
        selected=calibrate(x,pos,w,sw,base,rows=16);frozen=copy.deepcopy(selected)
        for validation in [x,x*10]:
            result=evaluate(validation,pos,w,sw,base,selected)
            self.assertTrue(all(r['s_10']==base['s_10'] for r in result))
            for key in result[1]:
                if key.startswith('non_bos_'):self.assertEqual(result[1][key],result[2][key])
        self.assertEqual(selected,frozen)
        self.assertEqual(selected['selection_split'],'calibration')
        self.assertEqual(selected['normal_search']['sampling']['bos_rows'],0)
        self.assertEqual(selected['arbitrary']['mse'],min(r['mse'] for r in selected['arbitrary_candidates']))
        self.assertEqual(selected['pot']['mse'],min(r['mse'] for r in selected['pot_candidates']))

    def test_bos_never_generates_another_channel_parameter_array(self):
        x,pos,w,wq,sw,base=self.fixture();p=Profile().approximate(base['s_X']*sw/base['s_10'])
        before=copy.deepcopy(p)
        with patch.object(Profile,'approximate',side_effect=AssertionError('BOS qparam regeneration')):
            chosen,table=choose_bos(x[pos==0],w,wq,base['s_X'],base['s_10'],p,[(1.,None),(3.,None),(4.,2)],'int64')
        for key in p:np.testing.assert_array_equal(p[key],before[key])
        self.assertFalse(any('multiplier' in row or 'shift' in row for row in table))

    def test_invalid_correction_rejected(self):
        p=dict(multiplier=np.array([1]),shift=np.array([1]));a=np.array([[1]])
        for bad in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):row_requant(a,p,bad)
        with self.assertRaises(ValueError):row_requant(a,p,3.,1)


if __name__=='__main__':unittest.main()
