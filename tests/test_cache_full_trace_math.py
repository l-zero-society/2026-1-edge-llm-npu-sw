import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_cache_full_divergence as d


class CacheFullTraceTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(83)
        x=rng.normal(size=(1,4,7)).astype(np.float32)
        wq,sw=d.base.quantize_weight(rng.normal(size=(5,7)))
        sx=.03;s10=.015
        spec=dict(wq=wq,sx=sx,s10=s10,k=1,params=d.base.Profile().approximate(sx*sw/s10))
        name='model.layers.0.mlp.down_proj';module=SimpleNamespace(forward=lambda x:x)
        source=SimpleNamespace(torch=torch,modules={name:module})
        return x,spec,source,name

    def traced(self,x,spec,source,name,mode='general_pot',control=None,reference=None,offset=0):
        trace={}
        with d.instrument(source,{name:spec},mode,offset,trace,control,reference):
            y=source.modules[name].forward(torch.from_numpy(x)).numpy()
        return y,trace[0]

    def test_identical_history_and_absolute_position(self):
        ex=dict(prompt_ids=[2,3,4],targets=[5,6],predictor_positions=[2,3])
        self.assertEqual(d.histories(ex,1,[2,3,4,5]),[2,3,4,5])
        with self.assertRaises(AssertionError):d.histories(ex,1,[2,3,4,6])
        ex['predictor_positions'][1]=4
        with self.assertRaises(AssertionError):d.histories(ex,1,[2,3,4,5])

    def test_general_is_existing_path_exactly(self):
        x,s,source,name=self.fixture()
        with patch.object(d.base.Profile,'approximate',side_effect=AssertionError('no parameter regeneration')):
            y,tr=self.traced(x,s,source,name)
            expected=d.row.apply_rows(x,**s)[0]
        np.testing.assert_array_equal(y,expected)
        q,k,_=d.row.select_rows(x[0],s['sx'],s['params']['shift'])
        np.testing.assert_array_equal(tr['xq'],q[-1]);self.assertEqual(tr['k'],k[-1])
        self.assertGreaterEqual(tr['shift_min'],0);self.assertLessEqual(tr['shift_max'],31)

    def test_bos_absolute_offset(self):
        x,s,source,name=self.fixture();x=x[:,:1]
        y,tr=self.traced(x,s,source,name,'bos_only',offset=15)
        np.testing.assert_array_equal(y,d.base.quantized_linear(x,**dict(s,k=0))[0]);self.assertEqual(tr['k'],0)
        y,tr=self.traced(x,s,source,name,'bos_only',offset=0)
        np.testing.assert_array_equal(y,d.base.quantized_linear(x,**s)[0]);self.assertEqual(tr['k'],s['k'])

    def test_force_k_changes_only_selected_row_scale(self):
        x,s,source,name=self.fixture();_,ref=self.traced(x,s,source,name)
        ref=dict(ref,k=0);untouched=x.copy()
        y,tr=self.traced(x,s,source,name,control='force_full_k',reference={0:ref})
        np.testing.assert_array_equal(x,untouched);np.testing.assert_array_equal(tr['x'],x[0,-1])
        np.testing.assert_array_equal(tr['xq'],d.row.quantize_input(x[0,-1:],s['sx'])[0])
        np.testing.assert_array_equal(y[0,:-1],d.row.apply_rows(x,**s)[0][0,:-1])

    def test_force_xq_injects_only_codes_and_matching_k(self):
        x,s,source,name=self.fixture();_,ref=self.traced(x,s,source,name)
        altered=x.copy();altered[0,-1]*=1.1
        y,tr=self.traced(altered,s,source,name,control='force_full_Xq',reference={0:ref})
        np.testing.assert_array_equal(tr['x'],altered[0,-1]);self.assertEqual(tr['k'],ref['k'])
        for key in ['xq','acc','raw','q10','y']:np.testing.assert_array_equal(tr[key],ref[key])

    def test_force_q10_changes_output_only(self):
        x,s,source,name=self.fixture();_,ref=self.traced(x,s,source,name)
        altered=x.copy();altered[0,-1]*=1.1
        _,native=self.traced(altered,s,source,name)
        _,tr=self.traced(altered,s,source,name,control='force_full_q10',reference={0:ref})
        for key in ['x','k','xq','acc','raw']:np.testing.assert_array_equal(tr[key],native[key])
        np.testing.assert_array_equal(tr['q10'],ref['q10']);np.testing.assert_array_equal(tr['y'],ref['y'])

    def test_boundary_distance(self):
        np.testing.assert_allclose(d.boundaries(np.array([-1.5,-1.,-.5,0.,.5,1.,1.5]),1),[0,.5,0,.5,0,.5,0])

    def test_measured_fp_parity_and_complete_modes(self):
        p=d.ROOT/'diagnostics/cache_full_divergence_summary.json'
        if not p.exists():self.skipTest('real-model run not completed')
        s=json.loads(p.read_text());self.assertEqual(s['completed_modes'],d.TAGS)
        f=s['controls'][0];self.assertEqual(f['top1_agreement'],1)
        self.assertLess(f['max_abs_nll_delta'],.001);self.assertLess(f['logits_nmse'],1e-8)
        self.assertEqual(s['kv_rows'],4*16*18)
        self.assertEqual(s['trace']['force_full_k']['k_mismatch_fraction'],0)
        self.assertEqual(s['trace']['force_full_Xq']['xq_any_fraction'],0)
        self.assertEqual(s['trace']['force_full_Xq']['q10_any_fraction'],0)
        self.assertEqual(s['trace']['force_full_q10']['q10_any_fraction'],0)
        for m in d.TAGS[1:]:
            self.assertGreaterEqual(s['trace'][m]['effective_shift_min'],0)
            self.assertLessEqual(s['trace'][m]['effective_shift_max'],31)

    def test_no_tracked_production_changes(self):
        changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=d.ROOT,text=True).splitlines()
        self.assertFalse(any(p.startswith(('src/','configs/','rtl/','calibration_outputs/')) for p in changed))


if __name__=='__main__':unittest.main()
