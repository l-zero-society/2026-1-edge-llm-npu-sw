"""Focused numerical oracles for the diagnostic; no model/download required."""
from pathlib import Path
import sys
import shutil
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from quant_diagnostic_math import ErrorMetric, activation_stats, paths, integer_dot, scale_from_threshold, configure_blas, dot
from diagnose_linear_quant_error import evaluate
from static_quant.core import quantize_weight
from static_quant.hardware import Profile


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'optional native reducer requires C compiler')
    def test_native_metric_reduction_all_paths_and_sweeps(self):
        from quant_diagnostic_reduce import NativeMetrics, NAMES
        native = NativeMetrics()
        x,w,sx,sw,s10,p = self.fixture()
        candidates = [dict(policy='clipped',threshold=1.27,s_X=.01)]
        expected = evaluate(x,w,sx,sw,s10,Profile(),p,'int64',2,2,candidates,[.01,.1])
        actual = evaluate(x,w,sx,sw,s10,Profile(),p,'int64',2,2,candidates,[.01,.1],native)
        for erows, arows in zip([[expected[0]],expected[1],expected[2]], [[actual[0]],actual[1],actual[2]]):
            for e,a in zip(erows,arows):
                self.assertEqual(e.keys(),a.keys())
                for key in e:
                    if isinstance(e[key],float):
                        np.testing.assert_allclose(a[key],e[key],rtol=1e-12,atol=1e-14)
                    else:self.assertEqual(a[key],e[key])
        metrics = {k:ErrorMetric() for k in NAMES}
        invalid = np.array([[float('nan')]])
        with self.assertRaises(ValueError):native.add(metrics,*[invalid]*6)

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS Accelerate only')
    def test_accelerate_fp64_and_integer_oracles(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(11,37)); w = rng.normal(size=(7,37))
        expected = x @ w.T
        configure_blas('accelerate')
        try:
            np.testing.assert_allclose(dot(x,w),expected,rtol=1e-13,atol=1e-13)
            self.test_acceleration_exact_full_k_and_signs()
            self.test_streaming_metrics_and_reference_denominators()
        finally:
            configure_blas('numpy')

    def fixture(self):
        x=np.array([[.05,.15,-.25,5.],[-.05,.35,.75,-2.],[0.,0.,0.,0.]])
        w=np.array([[.01,.17,-.02,1.3],[2.,-.1,.7,-.21],[0.,0.,0.,0.]])
        _,sw=quantize_weight(w);sx=.2;s10=.03
        p=Profile().approximate(sx*sw/s10)
        return x,w,sx,sw,s10,p

    def test_all_paths_dense_oracle_and_nonadditivity(self):
        x,w,sx,sw,s10,p=self.fixture()
        got=paths(x,w,sx,sw,s10,Profile(),p['multiplier'],p['shift'])
        xq=np.clip(np.rint(x/sx),-127,127).astype(np.int64)
        wq=np.clip(np.rint(w/sw[:,None]),-127,127).astype(np.int64)
        acc=np.array([[sum(int(a)*int(b) for a,b in zip(xx,ww)) for ww in wq] for xx in xq])
        np.testing.assert_allclose(got['input_only'],(xq*sx)@w.T)
        np.testing.assert_allclose(got['weight_only'],x@(wq*sw[:,None]).T)
        np.testing.assert_array_equal(got['int8_pair'],acc*(sx*sw))
        np.testing.assert_array_equal(got['ideal_int10'],np.clip(np.rint(acc*(sx*sw/s10)),-512,511)*s10)
        hw=np.empty_like(acc)
        for i in range(len(x)):
            for j in range(len(w)):
                a=int(acc[i,j])*int(p['multiplier'][j]);den=2**int(p['shift'][j]);q,rem=divmod(a,den)
                q+=int(2*rem>den or (2*rem==den and q%2==1));hw[i,j]=min(511,max(-512,q))
        np.testing.assert_array_equal(got['hw'],hw*s10)
        errors={k:v-got['reference'] for k,v in got.items() if k in ['input_only','weight_only','int8_pair']}
        self.assertFalse(np.isclose(np.square(errors['int8_pair']).mean(),np.square(errors['input_only']).mean()+np.square(errors['weight_only']).mean()))

    def test_streaming_metrics_and_reference_denominators(self):
        x,w,sx,sw,s10,p=self.fixture()
        dense=paths(x,w,sx,sw,s10,Profile(),p['multiplier'],p['shift'])
        base,sweep,output=evaluate(x,w,sx,sw,s10,Profile(),p,'int64',1,2,
             [dict(policy='same',threshold=127*sx,s_X=sx)],[s10])
        for name in ['input_only','weight_only','int8_pair','ideal_int10']:
            error=dense[name]-dense['reference']
            self.assertAlmostEqual(base[name+'_mse'],np.square(error).mean())
            self.assertAlmostEqual(base[name+'_mae'],np.abs(error).mean())
            self.assertAlmostEqual(base[name+'_nmse'],np.square(error).sum()/np.square(dense['reference']).sum())
            self.assertAlmostEqual(base[name+'_max_absolute_error'],np.abs(error).max())
        for key in ['input_only_nmse','int8_pair_nmse','final_local_total_nmse']:
            self.assertAlmostEqual(base[key],sweep[0][key])
        self.assertAlmostEqual(base['final_local_total_nmse'],output[0]['final_local_total_nmse'])
        self.assertAlmostEqual(base['actual_requant_mse'],np.square(dense['hw']-dense['ideal_int10']).mean())

    def test_acceleration_exact_full_k_and_signs(self):
        rng=np.random.default_rng(19)
        for k in [31,257,2048,16384]:
            x=rng.integers(-127,128,(3,k),dtype=np.int8);w=rng.integers(-127,128,(4,k),dtype=np.int8)
            np.testing.assert_array_equal(integer_dot(x,w,'fp64-exact'),integer_dot(x,w))
        x=np.full((1,16384),127,dtype=np.int8)
        np.testing.assert_array_equal(integer_dot(x,-x,'fp64-exact'),[[-16384*127**2]])
        with self.assertRaises(ValueError):integer_dot(x.astype(float),x,'fp64-exact')
        with self.assertRaises(OverflowError):integer_dot(np.zeros((1,140000),np.int8),np.zeros((1,140000),np.int8),'fp64-exact')

    def test_rne_zero_boundary_saturation_and_exact_percentile(self):
        x=np.array([0.,.49,.5,.51,-.5,1.,2.,127.,128.],dtype=np.float32)
        stats=activation_stats(x,1.)
        self.assertEqual(stats['zero_rate'],4/9)  # half ties round to even zero
        self.assertEqual(stats['below_half_scale_rate'],2/9)  # strict inequality
        self.assertEqual(stats['saturation_rate'],2/9)
        self.assertEqual(stats['clipping_rate'],1/9)
        self.assertAlmostEqual(stats['p99.9'],np.percentile(np.abs(x),99.9))
        self.assertAlmostEqual(stats['std'],x.astype(float).std())

    def test_zero_reference_and_zero_scale_fallback(self):
        metric=ErrorMetric();metric.add(np.ones(3),np.zeros(3))
        self.assertIsNone(metric.result()['nmse'])
        metric=ErrorMetric();metric.add(np.zeros(3),np.zeros(3))
        self.assertEqual(metric.result()['nmse'],0)
        self.assertEqual(scale_from_threshold(0,0),1)
        self.assertEqual(scale_from_threshold(0,127),1)
        with self.assertRaises(ValueError):activation_stats(np.array([np.nan]),1)


if __name__=='__main__':unittest.main()
