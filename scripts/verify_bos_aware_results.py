#!/usr/bin/env python3
"""Validate real experiment coverage, provenance, frozen selection and contracts."""
import csv
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/bos_aware'


def main():
    records=[json.loads(p.read_text()) for p in (OUT/'results').glob('*.json')]
    down=[r for r in records if r['module']=='down_proj']
    assert {r['layer'] for r in down}==set(range(18))
    provenance=json.loads((OUT/'provenance.json').read_text())
    for path,digest in provenance['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    historical={(r['layer'],r['module'],r['split']):r for r in csv.DictReader((ROOT/'diagnostics/quant_error_attribution.csv').open())}
    max_replay=0.;max_ratio=0.
    for d in records:
        for tag,s in d['searches'].items():
            m=s['sampling'];count=int(tag.split('_')[-1])
            assert m['retained']<=count and sum(m[k] for k in ['bos_rows','early_rows','general_rows','unknown_rows'])==m['retained']
            assert math.isclose(sum(m['row_objective_weights']),1,rel_tol=1e-12)
            assert s['selection_split']=='calibration'
            if tag.startswith('stratified'):assert m['bos_rows']>=1
            eligible=[r for r in s['candidates'] if r['bad_ratio_channels']==0]
            assert s['selected']==min(eligible,key=lambda r:r['mse'])
            for row in s['candidates']:
                assert row['s_X']>0 and row['s10']>0 and math.isfinite(row['mse'])
                if d['module']=='gate_proj':assert row['s10']==.1
                if d['module'] in ['q_proj','k_proj']:assert row['s10']==.2
            frozen=[r for r in d['evaluation'] if r['policy']==tag]
            assert len(frozen)==2
            for r in frozen:
                assert (r['s_X'],r['s_10'])==(s['selected']['s_X'],s['selected']['s10'])
        for r in d['evaluation']:
            assert r['bad_ratio_channels']==0
            max_ratio=max(max_ratio,r['ratio_max_relative_error'])
            if r['policy']=='historical':
                old=historical[str(d['layer']),d['module'],r['split']]
                expected=float(old['final_local_total_mse'])
                max_replay=max(max_replay,abs(r['all_final_local_total_mse']/expected-1))
                assert math.isclose(r['s_X'],float(old['s_X']),rel_tol=1e-14)
                assert math.isclose(r['s_10'],float(old['s_10']),rel_tol=1e-14)
            for key,value in r.items():
                if key.endswith(('_nmse','_mse','_rate')) and value is not None:
                    assert math.isfinite(value) and value>=0,(key,value)
        name=f'model.layers.{d["layer"]}.'+('mlp.' if d['module'] in ['gate_proj','up_proj','down_proj'] else 'self_attn.')+d['module']
        selected=OUT/'selected'/(name+'.json');complete=OUT/'results'/(name+'.json')
        assert selected.stat().st_mtime<=complete.stat().st_mtime
    assert max_replay<2e-5
    protected=['src/static_quant/hardware.py','src/LUT.py']
    unchanged={}
    import subprocess
    protected += subprocess.check_output(['git','ls-files','luts/*.bin'],cwd=ROOT,text=True).splitlines()
    for file in protected:
        original=subprocess.check_output(['git','show','HEAD:'+file],cwd=ROOT)
        unchanged[file]=original==(ROOT/file).read_bytes();assert unchanged[file]
    import ast
    before=ast.parse(subprocess.check_output(['git','show','HEAD:src/static_quant/core.py'],cwd=ROOT,text=True))
    after=ast.parse((ROOT/'src/static_quant/core.py').read_text())
    for name in ['quantize_input','quantize_weight','exact_accumulator','check_int32']:
        original=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        current=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(original)==ast.dump(current),name
        unchanged['core.'+name]=True
    result=dict(passed=True,down_proj_layers=18,total_operations=len(records),historical_mse_max_relative_difference=max_replay,
                maximum_ratio_relative_error=max_ratio,protected_files_unchanged=unchanged,
                scope='offline calibration output consistency; not RTL or end-to-end accuracy')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
