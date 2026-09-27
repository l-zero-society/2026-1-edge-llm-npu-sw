import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'), str(Path(__file__).resolve().parents[1]/'src')]
from test_e2e_linear_quant import quantized_linear, patch_linears, Metric, logit_metrics
from test_factorized_row_requant import row_path
from static_quant.hardware import Profile


class E2ELinearTests(unittest.TestCase):
    def spec(self):
        return dict(wq=np.array([[2,-3],[4,5]],np.int8), sx=.25, s10=.05, k=2,
                    params=Profile().approximate(np.array([.02,.03])))

    def test_prefill_matches_existing_path_and_position_zero_only(self):
        s=self.spec(); x=np.array([[[.3,1.2],[.3,1.2],[-2.,.1]]])
        result,stats=quantized_linear(x,**s,chunk=1)
        for i,row in enumerate(x[0]):
            k=s['k'] if i==0 else 0
            _,raw=row_path(row[None],s['wq'],s['sx'],s['params'],2.**k,k)
            np.testing.assert_array_equal(result[0,i],(np.clip(raw,-512,511)*s['s10']).astype(np.float32)[0])
        self.assertEqual(stats['bos_outputs'],2)
        self.assertEqual(stats['non_bos_outputs'],4)

    def test_invalid_shift_is_rejected(self):
        s=self.spec();s['k']=40
        with self.assertRaises(ValueError):quantized_linear(np.ones((1,3,2)),**s)

    def test_outputs_propagate_and_only_requested_modules_change(self):
        first,second=torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,2,bias=False)
        source=SimpleNamespace(torch=torch,modules={'x.down_proj':first,'x.up_proj':second})
        original_first,original_second=first.forward,second.forward
        x=torch.ones((1,3,2));s=self.spec();calls=[]
        def wrapped(x,**spec):
            calls.append(np.array(x,copy=True));return np.asarray(x)*2,{}
        with patch('test_e2e_linear_quant.quantized_linear',side_effect=wrapped):
            with patch_linears(source,{'x.down_proj':s,'x.up_proj':s},'all_linear',{}) as names:
                y=second(torch.sigmoid(first(x)))
                np.testing.assert_allclose(calls[1],torch.sigmoid(x*2).numpy())
                np.testing.assert_allclose(y.numpy(),2*torch.sigmoid(x*2).numpy())
                self.assertEqual(len(names),2)
            with patch_linears(source,{'x.down_proj':s,'x.up_proj':s},'down_only',{}) as names:
                self.assertEqual(names,['x.down_proj']);self.assertEqual(second.forward,original_second)
        self.assertEqual(first.forward,original_first);self.assertEqual(second.forward,original_second)

    def test_identity_metrics_and_next_token_targets(self):
        x=torch.tensor([[1.,2.,3.,4.,5.,6.],[6.,5.,4.,3.,2.,1.]])
        r=logit_metrics(torch,x,x,[1,2])
        self.assertEqual(r['kl_sum'],0)
        self.assertEqual(r['top1_count'],2)
        self.assertEqual(r['top5_exact_count'],2)
        self.assertEqual(r['targets'],1)
        self.assertAlmostEqual(r['nll_sum'],float(-torch.log_softmax(x[0].double(),-1)[2]))
        m=Metric();m.add(x.numpy(),x.numpy())
        self.assertEqual(m.result()['nmse'],0);self.assertAlmostEqual(m.result()['cosine'],1)


if __name__=='__main__':unittest.main()
