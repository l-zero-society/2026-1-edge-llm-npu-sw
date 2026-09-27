import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from test_s10_calibration import scales,parameters,select,score_candidates,evaluate
from test_factorized_row_requant import evaluate as previous_evaluate
from static_quant.core import quantize_weight


class S10Tests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(31);x=rng.normal(size=(14,7));x[0,0]=30
        pos=np.tile(np.arange(7),2);w=rng.normal(size=(5,7));_,sw=quantize_weight(w)
        groups={g:(x[m],np.full(m.sum(),1/m.sum())) for g,m in [('bos',pos==0),('non_bos',pos!=0)]}
        return x,pos,w,sw,groups

    def test_baseline_matches_previous_pot(self):
        x,pos,w,sw,groups=self.fixture();sx=.04;k=2;s10=.07
        prior=previous_evaluate(x,pos,w,sw,dict(s_X=sx,s_10=s10),
            dict(s_X_normal=sx,s_10=s10,arbitrary={'C':4.},pot={'C':4.,'k':k}))[-1]
        new=evaluate(x,pos,w,sw,{'frozen':dict(s_X_normal=sx,k=k,s_10=s10,old_s_10=s10,ratio=1.)})[0]
        for group in ['bos','non_bos']:
            for metric in ['mse','mae','nmse','int10_clip_rate']:
                self.assertEqual(prior[group+'_'+metric],new[group+'_'+metric])

    def test_candidate_grid_recomputes_channel_params_but_freezes_input(self):
        x,pos,w,sw,groups=self.fixture();sx=.04;k=2;old=.07
        grid=scales(old);self.assertEqual(len(grid),33);self.assertIn(old,grid)
        self.assertEqual((grid[0],grid[-1]),(old/4,old*4))
        first,_=parameters(sx,sw,grid[0],k);last,_=parameters(sx,sw,grid[-1],k)
        self.assertFalse(np.array_equal(first['multiplier'],last['multiplier']) and np.array_equal(first['shift'],last['shift']))
        report=score_candidates(groups,w,sw,[(sx,k,s) for s in [old/2,old,old*2]],old)
        self.assertTrue(all((r['s_X_normal'],r['k'])==(sx,k) for r in report['candidates']))
        for r in report['candidates']:
            self.assertEqual(r['balanced_score'],.5*(r['bos_nmse']+r['non_bos_nmse']))

    def test_validation_independent_and_common_scale(self):
        x,pos,w,sw,groups=self.fixture();old=.07
        selection=score_candidates(groups,w,sw,[(.04,2,s) for s in [old/2,old,old*2]],old)
        snapshot=copy.deepcopy(selection);r=selection['selected']
        policies={'selected':{k:r[k] for k in ['s_X_normal','k','s_10','old_s_10','ratio']}}
        for xx in [x,x*100]:
            result=evaluate(xx,pos,w,sw,policies)[0]
            self.assertEqual(result['s_10'],r['s_10'])
            self.assertNotIn('bos_s_10',result);self.assertNotIn('non_bos_s_10',result)
        self.assertEqual(selection,snapshot)

    def test_effective_shift_feasibility_and_feasible_selection(self):
        sw=np.array([.01,.02]);_,good=parameters(.1,sw,.1,2)
        self.assertTrue(good['feasible'])
        for k in [-40,40]:
            _,bad=parameters(.1,sw,.1,k);self.assertFalse(bad['feasible'])
        base=dict(balanced_score=.1,worst_group_score=.15,balanced_clip_rate=.01,ratio=1.,s_10=.1)
        rows=[dict(base,feasible=False),dict(base,balanced_score=.2,feasible=True)]
        self.assertEqual(select(rows)['balanced_score'],.2)
        self.assertEqual(select(rows,False)['balanced_score'],.1)

    def test_deterministic_tie_breaks(self):
        base=dict(balanced_score=.1,worst_group_score=.2,balanced_clip_rate=.01,ratio=1.,s_10=.1,feasible=True)
        self.assertEqual(select([base,dict(base,worst_group_score=.19,ratio=2.)])['ratio'],2.)
        self.assertEqual(select([base,dict(base,balanced_clip_rate=0.,ratio=2.)])['ratio'],2.)
        self.assertEqual(select([dict(base,ratio=2.),base])['ratio'],1.)

    def test_production_source_and_lut_contract_unchanged(self):
        identity=json.loads((ROOT/'diagnostics/bos_aware/provenance.json').read_text())
        for path,digest in identity['source_hashes'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)
        # Pure experiment functions expose no production or LUT write operations.
        import subprocess
        tracked=['src/LUT.py']
        tracked+=subprocess.check_output(['git','ls-files','configs/gemma-linear-scale-policy.json','luts/*.bin'],cwd=ROOT,text=True).splitlines()
        for path in tracked:
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT),path)


if __name__=='__main__':unittest.main()
