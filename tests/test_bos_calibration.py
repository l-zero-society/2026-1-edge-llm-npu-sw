import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from static_quant.sampling import RepresentativeRowSampler, valid_positions
from static_quant.core import InputStats, Options, LinearCalibration, Reservoir, quantize_weight, quantize_input, exact_accumulator
from static_quant.search import activation_candidates, joint_scale_search
from static_quant.hardware import Profile
from static_quant.cli import main


class SamplingTests(unittest.TestCase):
    def test_uniform_control_matches_historical_reservoir(self):
        x=np.arange(3000).reshape(1000,3);p=np.tile(np.arange(100),10)
        old=Reservoir(64,42);new=RepresentativeRowSampler(64,42,representative=False)
        for start in range(0,len(x),77):
            old.add(x[start:start+77]);new.add(x[start:start+77],p[start:start+77])
        np.testing.assert_array_equal(old.values,new.values)

    def test_bos_guarantee_determinism_and_streaming(self):
        positions=np.tile(np.arange(100),8);x=np.arange(len(positions)*2).reshape(-1,2)
        a=RepresentativeRowSampler(64,42);b=RepresentativeRowSampler(64,42)
        a.add(x,positions)
        for i in range(0,len(x),13):b.add(x[i:i+13],positions[i:i+13])
        for u,v in zip(a.selected(),b.selected()):np.testing.assert_array_equal(u,v)
        meta=a.metadata();self.assertEqual(meta['retained'],64);self.assertEqual(meta['bos_rows'],8)
        self.assertEqual(meta['early_rows'],16);self.assertEqual(meta['general_rows'],40)
        self.assertAlmostEqual(a.selected()[2].sum(),1)
        for seed in range(20):
            s=RepresentativeRowSampler(3,seed);s.add(x,positions)
            self.assertGreaterEqual(s.metadata()['bos_rows'],1)
            self.assertEqual(s.metadata()['retained'],3)

    def test_short_strata_redistribution_and_explicit_disable(self):
        p=np.array([0,1,2]+list(range(9,100)))
        s=RepresentativeRowSampler(64,7);s.add(p[:,None],p)
        m=s.metadata();self.assertEqual((m['bos_rows'],m['early_rows'],m['general_rows']),(1,2,61))
        s=RepresentativeRowSampler(64);s.add(np.ones((2,3)),np.array([0,1]))
        self.assertEqual(s.metadata()['retained'],2)
        s=RepresentativeRowSampler(1,bos_rows=0,early_rows=0);s.add(p[:,None],p)
        self.assertEqual(len(s.values),1)
        with self.assertRaises(ValueError):
            s=RepresentativeRowSampler(2);s.add(p[:,None],p);s.selected()

    def test_population_weights_and_padding(self):
        p=np.concatenate([np.zeros(2,int),np.ones(6,int),np.full(92,20)])
        s=RepresentativeRowSampler(12,4,bos_rows=4,early_rows=4);s.add(p[:,None],p)
        x,pos,w,_=s.selected()
        self.assertAlmostEqual(w[pos==0].sum(),.02)
        self.assertAlmostEqual(w[pos==1].sum(),.06)
        self.assertAlmostEqual(w[pos==20].sum(),.92)
        np.testing.assert_array_equal(valid_positions([[0,1,1,0],[1,1,1,0]]),[0,1,0,1,2])


class JointSearchTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'darwin', 'real-run accelerated backend is macOS only')
    def test_real_runner_group_metrics_match_dense(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from run_bos_aware_calibration import metrics_for_policies
        rng=np.random.default_rng(92);x=rng.normal(size=(13,9));w=rng.normal(size=(5,9))
        pos=np.array([0,1,2,9,10,0,1,9,10,11,12,13,14]);wq,sw=quantize_weight(w)
        policies=[dict(policy='a',s_X=.03,s_10=.05),dict(policy='b',s_X=.03,s_10=.1),dict(policy='c',s_X=.06,s_10=.08)]
        actual=metrics_for_policies(x,pos,w,sw,policies,Profile(),m_chunk=4,n_chunk=3)
        for policy,row in zip(policies,actual):
            acc=exact_accumulator(quantize_input(x,policy['s_X']),wq)
            pars=Profile().approximate(policy['s_X']*sw/policy['s_10'])
            hw=Profile().apply(acc,pars['multiplier'],pars['shift'])*policy['s_10'];ref=x@w.T
            for name,mask in [('bos',pos==0),('non_bos',pos!=0),('all',np.ones(len(pos),bool))]:
                mse=np.square(hw[mask]-ref[mask]).mean();energy=np.square(ref[mask]).mean()
                self.assertAlmostEqual(row[name+'_final_local_total_mse'],mse)
                self.assertAlmostEqual(row[name+'_final_local_total_nmse'],mse/energy)

    def fixture(self,fixed=False):
        rng=np.random.default_rng(4);x=rng.normal(size=(45,11));x[0,0]=40
        w=rng.normal(size=(7,11));_,sw=quantize_weight(w)
        s=RepresentativeRowSampler(32,2,bos_rows=4,early_rows=6);s.add(x,np.tile(np.arange(15),3))
        candidates=activation_candidates(float(np.abs(x).max()),np.abs(x).ravel())
        o=Options(n_chunk=3,k_chunk=4,mode='fixed' if fixed else 'mse',s10=.1 if fixed else None,
                  fixed_lut_scale=.1 if fixed else None)
        return x,w,sw,s,candidates,o

    def test_lowest_output_mse_and_independent_dense_oracle(self):
        x,w,sw,s,c,o=self.fixture();p=Profile()
        sx,s10,params,report=joint_scale_search(lambda sl:w[sl],w.shape,sw,s,c,.3,p,o)
        self.assertEqual(report['selected']['mse'],min(r['mse'] for r in report['candidates']))
        rows,pos,weights,_=s.selected();wq,_=quantize_weight(w)
        for candidate in report['candidates']:
            q=quantize_input(rows,candidate['s_X']);acc=exact_accumulator(q,wq)
            pars=p.approximate(candidate['s_X']*sw/candidate['s10'])
            y=p.apply(acc,pars['multiplier'],pars['shift'])*candidate['s10']
            expected=weights @ np.square(y-rows@w.T).mean(axis=1)
            self.assertAlmostEqual(expected,candidate['mse'],places=12)
        old=p.approximate(sx*sw/s10)
        np.testing.assert_array_equal(params['multiplier'],old['multiplier'])
        np.testing.assert_array_equal(params['shift'],old['shift'])

    def test_fixed_lut_and_zero_finite(self):
        x,w,sw,s,c,o=self.fixture(True)
        _,s10,_,r=joint_scale_search(lambda sl:w[sl],w.shape,sw,s,c,.3,Profile(),o)
        self.assertEqual(s10,.1);self.assertTrue(all(v['s10']==.1 for v in r['candidates']))
        x=np.zeros((3,5));w=np.zeros((2,5));_,sw=quantize_weight(w)
        s=RepresentativeRowSampler(3);s.add(x,np.array([0,1,10]))
        c=activation_candidates(0,np.zeros(10))
        sx,s10,_,r=joint_scale_search(lambda sl:w[sl],w.shape,sw,s,c,1.,Profile(),Options(mode='mse'))
        self.assertEqual((sx,s10),(1.,1.));self.assertEqual(r['selected']['mse'],0)
        json.dumps(r,allow_nan=False)
        for bad in [np.nan,np.inf,-1]:
            with self.assertRaises(ValueError):activation_candidates(bad,np.ones(5))
        with self.assertRaises(ValueError):s.add(np.full((1,5),np.nan),np.array([0]))

    def test_validation_cannot_change_selection(self):
        x,w,sw,s,c,o=self.fixture()
        def replay(callback):callback(x,np.tile(np.arange(15),3))
        def run(value):
            op=LinearCalibration('test',w.shape,lambda sl:w[sl],np.abs(x).max()/127,Profile(),o,c)
            report,_=op.run(replay,lambda cb:cb(np.full((2,11),value),np.array([0,1])))
            return op,report
        a,ar=run(1);b,br=run(100)
        self.assertEqual(ar['selection'],br['selection'])
        self.assertEqual((a.sx,a.s10),(b.sx,b.s10))
        np.testing.assert_array_equal(a.params['multiplier'],b.params['multiplier'])
        self.assertNotEqual(ar['validation']['local_total']['mse'],br['validation']['local_total']['mse'])
        q=quantize_input(x,a.sx);wq,_=quantize_weight(w);acc=exact_accumulator(q,wq)
        hw=Profile().apply(acc,a.params['multiplier'],a.params['shift'])*a.s10;ref=x@w.T
        positions=np.tile(np.arange(15),3)
        for group,mask in [('bos',positions==0),('non_bos',positions!=0),('all',np.ones(len(x),bool))]:
            expected=np.square(hw[mask]-ref[mask]).mean()
            self.assertAlmostEqual(ar['calibration']['position_metrics'][group]['mse'],expected)

    def test_cli_actually_passes_per_operation_constraints(self):
        names=['model.layers.0.self_attn.q_proj','model.layers.0.self_attn.k_proj','model.layers.0.mlp.gate_proj']
        w=np.array([[1.,-2.],[.1,.3]])
        class Source:
            def __init__(self,args):
                self.names=names;self.modules={n:types.SimpleNamespace(weight=w) for n in names};self.metadata={}
            def read_weight(self,name):return lambda sl:w[sl]
            def replay(self,name,split='calibration',with_positions=False):
                return lambda cb:cb(np.array([[1.,2.],[.2,.3],[.4,-.5]]),np.array([0,1,10]))
        with tempfile.TemporaryDirectory() as tmp,patch('static_quant.cli.GemmaSource',Source),contextlib.redirect_stdout(io.StringIO()):
            manifest=main(['--calibration-data','unused','--output-dir',tmp,'--scale-mode','mse','--activation-search','output-mse'])
            self.assertEqual([e['s_10'] for e in manifest['modules']],[.2,.2,.1])
            search=json.loads((Path(tmp)/'scale_search.json').read_text())
            self.assertTrue(all(search[n]['sampling']['bos_rows']>=1 for n in names))
            self.assertTrue(all(c['s10']==.1 for c in search[names[-1]]['candidates']))


if __name__=='__main__':unittest.main()
