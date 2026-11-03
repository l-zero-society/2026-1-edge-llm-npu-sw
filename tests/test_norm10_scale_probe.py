"""Pure arithmetic and publication isolation tests; no Gemma/GGUF loading."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_norm10_scale_probe as probe


class Norm10ScaleProbeTests(unittest.TestCase):
    def test_original_ppl_overflow_reproduces(self):
        score=probe.response.Scores()
        score.metric.add(np.ones((1,8)),np.ones((1,8)))
        score.n=1;score.nll=1000.
        with self.assertRaises(OverflowError):
            probe.response.merged([score.raw()])

    def test_large_finite_nll_survives_summary(self):
        score=probe.response.Scores()
        score.metric.add(np.ones((1,8)),np.ones((1,8)))
        score.n=1;score.nll=1000.
        result=probe.merge_scores([score.raw()])
        self.assertEqual(result['nll'],1000.)
        self.assertEqual(result['log_ppl'],1000.)
        self.assertIsNone(result['ppl'])
        self.assertEqual(result['ppl_status'],'exceeds_float64_range')

    def test_representable_ppl_unchanged(self):
        score=probe.response.Scores()
        score.metric.add(np.ones((1,8)),np.ones((1,8)))
        score.n=1;score.nll=5.
        a=probe.merge_scores([score.raw()]);b=probe.response.merged([score.raw()])
        for key,value in b.items():self.assertEqual(a[key],value)

    def test_input_quantization_saturates_before_integer_cast(self):
        q,clip=probe.quantize_input10(np.array([1e30,-1e30]),.01)
        np.testing.assert_array_equal(q,[511,-512]);self.assertEqual(clip,1.)

    def test_original_fixed_width_conversion_exceeds_int64(self):
        with np.errstate(invalid='ignore'):
            intermediate=probe.old.round_even(np.array([1e30])/.01)
        self.assertNotEqual(float(intermediate[0]),float(np.rint(1e30/.01)))
        fixed,_=probe.quantize_input10(np.array([1e30]),.01)
        self.assertEqual(int(fixed[0]),511)

    def test_invalid_output_scale(self):
        for scale in (0.,-1.,np.inf,np.nan):
            with self.assertRaises(ValueError):probe.static_gain(np.ones(4),scale)

    def test_genuine_statistic_width_violation_is_retained(self):
        result=probe.statistic([0]*256,1<<35)
        self.assertTrue(result['overflow'])
        self.assertFalse(result['intermediate_checks']['epsilon_uint35'])
        self.assertEqual(result['normalized_index'],0)

    def test_nonfinite_input_rejected(self):
        for invalid in (np.inf,-np.inf,np.nan):
            with self.assertRaises(FloatingPointError):
                probe.quantize_input10(np.array([invalid]),1.)

    def test_int10_bounds(self):
        np.testing.assert_array_equal(probe.sat10(np.array([-1000, 1000])), [-512, 511])

    def test_worst_case_square_sum(self):
        result=probe.statistic([-512]*4096)
        self.assertEqual(result['square_sum'], 1 << 30)
        self.assertEqual(result['required_square_sum_bits'], 31)
        self.assertFalse(result['overflow'])

    def test_accumulator_35(self):
        self.assertEqual(probe.statistic([511]*2048)['accumulator_bits'],35)

    def test_inverse_vector_length(self):
        self.assertEqual(probe.debug.inv_n(4096),256)
        self.assertEqual(probe.debug.inv_n(3),349525)

    def test_mean_square_truncation(self):
        s=probe.statistic([1,1,2]);self.assertEqual(s['mean_square'],1)
        self.assertEqual(s['exact_integer_mean'],2)
        self.assertTrue(s['intermediate_checks']['mean_product_uint59'])

    def test_epsilon_rne(self):
        self.assertEqual(probe.debug.epsilon_q(.00025,.01),2)
        self.assertEqual(probe.statistic([0,0],4)['statistic'],4)

    def test_normalized_index(self):
        s=probe.statistic([3,3]);self.assertEqual(s['normalized_index'],576);self.assertEqual(s['exponent'],3)

    def test_lut_q214(self):
        self.assertEqual(probe.debug.rsqrt_table()[512],16384)
        self.assertEqual(len(probe.debug.rsqrt_table()),1024)

    def test_exponent_correction(self):
        self.assertEqual(probe.debug.corrected_scale(16384,1),11585)
        self.assertEqual(probe.debug.corrected_scale(16384,2),8192)

    def test_signed_ties(self):
        self.assertEqual([probe.rounded_shift(i,1) for i in (-7,-5,-3,-1,1,3,5,7)],[-4,-2,-2,0,0,2,2,4])
        self.assertEqual(probe.rounded_shift(-3,0),-3)
        self.assertEqual(probe.rounded_shift(3,-2),12)

    def test_gain_format_and_sign(self):
        gamma=np.array([-1.,0.,1.,2.]);p=probe.static_gain(gamma,.01)
        np.testing.assert_array_equal(p['sign'],[-1,0,1,1])
        self.assertTrue(np.all(p['multiplier']<=65535));self.assertTrue(np.all(p['shift']<=31))
        np.testing.assert_allclose(p['multiplier']/np.exp2(p['shift']),[100,0,100,200])

    def test_unrepresentable_gain_recorded(self):
        p=probe.static_gain(np.array([1e8]),1.)
        self.assertFalse(p['representable'][0]);self.assertTrue(p['inaccurate'][0])

    def test_fused_gamma_output_gain(self):
        x=np.array([1.,-1.,1.,-1.]);gamma=np.array([1.,2.,-1.,0.])
        spec=dict(input_scale=1.,output_scale=.01,eps=0.)
        y,d=probe.mode_row(x,spec,gamma,'FUSED_REQUANT_ORACLE')
        np.testing.assert_allclose(y,[1,-2,-1,0]);self.assertFalse(d['overflow'])

    def test_fused_saturation(self):
        x=np.ones(4);spec=dict(input_scale=1.,output_scale=.001,eps=0.)
        y,d=probe.mode_row(x,spec,np.ones(4),'FUSED_REQUANT_ORACLE')
        np.testing.assert_allclose(y,.511);self.assertEqual(d['output_clip'],1.)

    def test_early_truncation_loss(self):
        x=np.array([1.,2.,-1.,-2.]);spec=dict(input_scale=1.,output_scale=.01,eps=0.)
        ref,_=probe.mode_row(x,spec,np.ones(4),'INPUT_QUANT_ORACLE')
        post,_=probe.mode_row(x,spec,np.ones(4),'POST_GAIN_DIAGNOSTIC')
        fused,_=probe.mode_row(x,spec,np.ones(4),'FUSED_REQUANT_ORACLE')
        self.assertLess(probe.metrics(fused,ref)['nmse'],probe.metrics(post,ref)['nmse'])

    def test_literal_adapter_explicit(self):
        x=np.array([2.,-2.]);spec=dict(input_scale=1.,output_scale=.01,eps=0.)
        literal,_=probe.mode_row(x,spec,np.ones(2),'RTL_LITERAL')
        np.testing.assert_allclose(literal,[.01,-.01])

    def test_integer_fused_matches_ideal_when_gain_exact(self):
        x=np.array([1.,2.,-1.,-2.]);spec=dict(input_scale=.1,output_scale=.01,eps=1e-6)
        gamma=np.array([1.,.5,-1.,2.])
        a,_=probe.mode_row(x,spec,gamma,'FUSED_IDEAL_COEFFICIENT')
        b,_=probe.mode_row(x,spec,gamma,'FUSED_REQUANT_ORACLE')
        np.testing.assert_array_equal(a,b)

    def test_arbitrary_precision_signed_rounding(self):
        self.assertEqual(probe.rounded_shift(-(1<<90),70),-(1<<20))

    def test_done_does_not_require_push(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory)/'report';f.write_text('report')
            self.assertTrue(probe.scientific_done([f],True))
            self.assertFalse(probe.scientific_done([f],False))
            self.assertFalse(probe.scientific_done([f.with_name('missing')],True))

    def test_publication_failure_keeps_done(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'DONE').write_text('EXPERIMENT_COMPLETE')
            with patch.object(probe,'OUT',out), patch.object(probe.subprocess,'check_output',side_effect=RuntimeError('offline')):
                probe.publish('head','index')
            self.assertTrue((out/'DONE').is_file());self.assertTrue((out/'PUSH_FAILED').is_file())
            self.assertFalse((out/'FAILED').exists())

    def test_publication_only_whitelist(self):
        self.assertNotIn('diagnostics/norm10_scale_probe/status.json',probe.PUBLISH_FILES)
        self.assertFalse(any('/work/' in p for p in probe.PUBLISH_FILES))


if __name__=='__main__':unittest.main()
