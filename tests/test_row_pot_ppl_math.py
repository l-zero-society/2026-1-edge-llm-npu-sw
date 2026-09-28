import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import diagnose_row_pot_ppl as d
from static_quant.core import quantize_weight
from test_factorized_row_requant import row_path


class PPLDiagnosisTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(45);w=rng.normal(size=(5,7));wq,sw=quantize_weight(w)
        sx=.1;s10=.04;p=d.Profile().approximate(sx*sw/s10)
        return rng.normal(size=(4,7))*10,w,sw,dict(wq=wq,sx=sx,s10=s10,k=0,params=p)

    def test_existing_paths_reproduce(self):
        x,w,sw,s=self.fixture()
        for mode,fn in [('bos_only',d.base.quantized_linear),('general_pot',d.row.apply_rows)]:
            y,k,_,_=d.apply_mode(x,s,w,sw,mode);old,_=fn(x[None],**s)
            np.testing.assert_array_equal(y,old[0])

    def test_caps_only_restrict_non_bos(self):
        x=np.array([[508.,-8.],[127.,17.]])
        ks,valid,_,_=d.candidates(x,1.,np.array([15,20]),cap=0,bos_k=2)
        self.assertTrue(np.all(ks[:,0][valid[:,0]]==2))
        self.assertTrue(np.all(ks[:,1][valid[:,1]]<=0))
        with self.assertRaises(ValueError):d.candidates(np.array([[127.,0.],[508.,0.]]),1.,np.array([15,20]),cap=0,bos_k=0)

    def test_shift_feasibility(self):
        k,v,_,_=d.candidates(np.array([[63.5,0.]]),1.,np.array([2,31]))
        self.assertTrue(np.all(k[v]>=0));self.assertTrue(np.all(k[v]<=2))

    def test_d3_matches_actual_path_brute_force(self):
        x,w,sw,s=self.fixture();y,k,_,_=d.select_oracle(x,s,w,sw,'final')
        ks,valid,_,_=d.candidates(x,s['sx'],s['params']['shift'])
        for r in range(len(x)):
            scores=[];outputs=[];values=[]
            for i in range(3):
                if not valid[i,r]:continue
                kk=int(ks[i,r]);_,raw=row_path(x[r:r+1],s['wq'],s['sx'],s['params'],2.**kk,kk)
                out=np.clip(raw,-512,511)*s['s10'];scores.append(np.square(out-d.dot(x[r:r+1],w)).sum());outputs.append(out);values.append(kk)
            best=int(np.argmin(scores));self.assertEqual(k[r],values[best]);np.testing.assert_array_equal(y[r],outputs[best].astype(np.float32)[0])

    def test_oracles_are_row_local_without_targets(self):
        x,w,sw,s=self.fixture();changed=x.copy();changed[-1]*=2
        for objective in ['pair','final']:
            a=d.select_oracle(x,s,w,sw,objective);b=d.select_oracle(changed,s,w,sw,objective)
            np.testing.assert_array_equal(a[1][:-1],b[1][:-1]);np.testing.assert_array_equal(a[0][:-1],b[0][:-1])

    def test_d1_has_no_weight_or_label_input(self):
        import inspect
        self.assertEqual(list(inspect.signature(d.candidates).parameters),['x','sx','shift','cap','bos_k'])

    def test_token_decomposition_alignment(self):
        fp=torch.tensor([[1.,2.,3.],[2.,1.,0.],[0.,1.,2.]])
        q=fp+5.;r=d.token_decomposition(torch,fp,q,[0,2,1])
        np.testing.assert_allclose(r['target_logit_fp'],[3.,1.])
        np.testing.assert_allclose(r['nll_fp'],r['nll_quant'],atol=1e-14)
        np.testing.assert_allclose(r['nll_quant']-r['nll_fp'],-(r['target_logit_quant']-r['target_logit_fp'])+(r['lse_quant']-r['lse_fp']),atol=1e-14)

    def test_recenter_selection_is_calibration_only(self):
        rows=[dict(selection_split='calibration',feasible=True,nmse=.2,j=0),dict(selection_split='calibration',feasible=True,nmse=.1,j=1)]
        chosen=d.choose_base(rows);self.assertEqual(chosen['j'],1)
        for val in [0.,1e9]:
            with self.assertRaises(ValueError):d.choose_base(rows+[dict(selection_split='validation',feasible=True,nmse=val,j=-1)])
        self.assertEqual(chosen,d.choose_base(rows))


if __name__=='__main__':unittest.main()
