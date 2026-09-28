import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_response_quant as e
from static_quant.core import quantize_weight


class ResponseEvalTests(unittest.TestCase):
    def test_mask_excludes_headers_suffix_padding_includes_first(self):
        offsets=[(0,3),(3,8),(8,10),(10,14),(14,19),(19,25),(0,0)]
        targets=e.response_indices(offsets,10,19)
        self.assertEqual(targets,[3,4]);self.assertEqual([t-1 for t in targets],[2,3])

    def test_ambiguous_boundary_rejected(self):
        with self.assertRaises(ValueError):e.response_indices([(0,5),(5,12),(12,15)],10,15)

    def test_empty_or_unpredictable_content_rejected(self):
        for offsets,start,end in [([(0,2)],0,2),([(0,2)],3,4)]:
            with self.assertRaises(ValueError):e.response_indices(offsets,start,end)

    def test_scores_same_explicit_target_positions(self):
        q=torch.tensor([[1.,3.,2.,0.,-1.],[4.,0.,2.,1.,-2.]])
        s=e.Scores();s.add(torch,q,q,[0,2]);r=e.merged([s.raw()])
        expected=float(-torch.log_softmax(q.double(),-1)[[0,1],[0,2]].mean())
        self.assertAlmostEqual(r['nll'],expected);self.assertEqual(r['tokens'],2)
        self.assertEqual(r['kl'],0);self.assertEqual(r['top1'],1);self.assertEqual(r['in5'],1)

    def fixture(self):
        rng=np.random.default_rng(3);wq,sw=quantize_weight(rng.normal(size=(5,7)))
        sx=.1;s10=.03
        return rng.normal(size=(1,4,7)),dict(wq=wq,sx=sx,s10=s10,params=e.base.Profile().approximate(sx*sw/s10),k=1)

    def test_bos_path_and_absolute_decode_position(self):
        x,s=self.fixture();a=e.quant_call(x,s,'bos_only',0,[0],np.zeros(16,np.int64))
        b=e.base.quantized_linear(x,**s)[0];np.testing.assert_array_equal(a,b)
        a=e.quant_call(x[:,:1],s,'bos_only',8,[0],np.zeros(16,np.int64))
        b=e.base.quantized_linear(x[:,:1],**dict(s,k=0))[0];np.testing.assert_array_equal(a,b)

    def test_general_path_reuses_params_and_response_histogram(self):
        x,s=self.fixture();hist=np.zeros(16,np.int64)
        with patch.object(e.base.Profile,'approximate',side_effect=AssertionError('no recalibration')):
            a=e.quant_call(x,s,'general_pot',0,[1,3],hist)
            b=e.row.apply_rows(x,**s)[0]
        np.testing.assert_array_equal(a,b);self.assertEqual(int(hist.sum()),2)

    def test_buckets_cover_each_response_once(self):
        for t in range(100):self.assertEqual(sum(lo<=t<hi for lo,hi,_ in e.BUCKETS),1)

    def test_generation_prefix_and_eos_censoring(self):
        r=e.generation_comparison([3,4,1],[3,5,1],[1])
        self.assertEqual(r['shared_prefix'],1);self.assertEqual(r['first_divergence'],1)
        self.assertEqual(r['eos_step_difference'],0)
        self.assertIsNone(e.generation_comparison([3,4],[3,4],[1])['eos_step_difference'])


if __name__=='__main__':unittest.main()
