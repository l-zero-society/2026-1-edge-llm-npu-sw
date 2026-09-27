#!/usr/bin/env python3
"""Cached down_proj diagnostic: balanced s10 search with frozen PoT input scales."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from test_factorized_row_requant import (Profile, quantize_input, quantize_weight, row_requant,
    integer_dot, dot, configure_blas, ErrorMetric, token_positions, sha256_file, write_json, write_csv)


def scales(old):
    if not np.isfinite(old) or old<=0:raise ValueError('positive baseline scale required')
    return sorted(set([float(old)]+[float(old*2.**(i/8)) for i in range(-16,17)]))


def parameters(sx,sw,s10,k,bounds=(0,31)):
    p=Profile().approximate(sx*sw/s10);s=p['shift'];effective=s-k
    info=dict(base_shift_min=int(s.min()),base_shift_max=int(s.max()),
              bos_effective_shift_min=int(effective.min()),bos_effective_shift_max=int(effective.max()),
              effective_shift_min=int(min(s.min(),effective.min())),
              effective_shift_max=int(max(s.max(),effective.max())),
              bad_ratio_channels=int((p['status']!='ok').sum()))
    info['feasible']=bool(info['bad_ratio_channels']==0 and info['effective_shift_min']>=bounds[0] and info['effective_shift_max']<=bounds[1])
    return p,info


def select(rows,feasible=True):
    eligible=[r for r in rows if r['feasible'] or not feasible]
    if not eligible:raise ValueError('no hardware-feasible candidate')
    best=min(r['balanced_score'] for r in eligible)
    tied=[r for r in eligible if np.isclose(r['balanced_score'],best,rtol=1e-6,atol=1e-12)]
    return min(tied,key=lambda r:(r['worst_group_score'],r['balanced_clip_rate'],abs(np.log2(r['ratio'])),r['s_10']))


def calibration_groups(x,pos,previous):
    sampling=previous['normal_search']['sampling'];normal=np.flatnonzero(pos!=0)
    ids=np.asarray(sampling['selected_row_ids'],np.int64)
    weights=np.asarray(sampling['row_objective_weights'],np.float64)
    assert sampling['observed']==len(normal) and sampling['bos_rows']==0 and np.isclose(weights.sum(),1)
    bos=np.asarray(x[pos==0],np.float64)
    return {'bos':(bos,np.full(len(bos),1/len(bos))),
            'non_bos':(np.asarray(x[normal[ids]],np.float64),weights)}


def score_candidates(groups,w,sw,specs,old,bounds=(0,31),backend='int64'):
    """Calibration-only inputs; shared ACC/ref across all output scales."""
    wq,check=quantize_weight(w);np.testing.assert_array_equal(check,sw)
    refs={g:dot(x,w) for g,(x,weights) in groups.items()};accs={};rows=[]
    for sx,k,s10 in specs:
        p,info=parameters(sx,sw,s10,k,bounds)
        row=dict(s_X_normal=float(sx),k=int(k),old_s_10=float(old),s_10=float(s10),ratio=float(s10/old),**info)
        for group,(x,weights) in groups.items():
            kg=k if group=='bos' else 0;key=(group,sx,kg)
            if key not in accs:accs[key]=integer_dot(quantize_input(x,sx*2.**kg),wq,backend)
            raw=row_requant(accs[key],p,2.**kg,kg);ref=refs[group]
            error=np.clip(raw,-512,511)*s10-ref
            mse=float(weights@np.square(error).mean(axis=1));energy=float(weights@np.square(ref).mean(axis=1))
            if energy==0:raise ValueError('balanced NMSE undefined for zero-energy calibration group')
            row.update({group+'_mse':mse,group+'_nmse':mse/energy,
                        group+'_mae':float(weights@np.abs(error).mean(axis=1)),
                        group+'_int10_clip_rate':float(weights@((raw < -512)|(raw > 511)).mean(axis=1))})
        row['balanced_score']=(row['bos_nmse']+row['non_bos_nmse'])/2
        row['worst_group_score']=max(row['bos_nmse'],row['non_bos_nmse'])
        row['balanced_clip_rate']=(row['bos_int10_clip_rate']+row['non_bos_int10_clip_rate'])/2
        rows.append(row)
    return dict(selected=select(rows),unconstrained_best=select(rows,False),candidates=rows,
                selection_split='calibration',objective='0.5 BOS NMSE + 0.5 non-BOS NMSE',
                score_tie_rtol=1e-6,score_tie_atol=1e-12,shift_bounds=list(bounds))


def evaluate(x,pos,w,sw,policies,bounds=(0,31),backend='int64'):
    wq,check=quantize_weight(w);np.testing.assert_array_equal(check,sw)
    params={tag:parameters(p['s_X_normal'],sw,p['s_10'],p['k'],bounds) for tag,p in policies.items()}
    stats={tag:{g:dict(error=ErrorMetric(),grid=ErrorMetric(),pair=ErrorMetric(),clips=0,count=0) for g in ['bos','non_bos']} for tag in policies}
    for start in range(0,len(x),512):
        block=np.asarray(x[start:start+512],np.float64);pb=pos[start:start+512]
        for group,mask in [('bos',pb==0),('non_bos',pb!=0)]:
            if not mask.any():continue
            xx=block[mask];ref=dot(xx,w);accs={}
            for tag,p in policies.items():
                k=p['k'] if group=='bos' else 0;sx=p['s_X_normal'];s10=p['s_10'];key=(sx,k)
                if key not in accs:accs[key]=integer_dot(quantize_input(xx,sx*2.**k),wq,backend)
                acc=accs[key];raw=row_requant(acc,params[tag][0],2.**k,k);s=stats[tag][group]
                s['error'].add(np.clip(raw,-512,511)*s10,ref)
                s['grid'].add(np.clip(np.rint(ref/s10),-512,511)*s10,ref)
                s['pair'].add(acc*(sx*2.**k*sw),ref)
                s['clips']+=int(((raw < -512)|(raw > 511)).sum());s['count']+=raw.size
    result=[]
    for tag,p in policies.items():
        row=dict(scheme=tag,split='validation',**p,**params[tag][1])
        for g,s in stats[tag].items():
            row.update({g+'_'+key:value for key,value in s['error'].result().items()})
            row[g+'_int10_clip_rate']=s['clips']/s['count']
            row[g+'_grid_lower_bound_nmse']=s['grid'].result()['nmse']
            row[g+'_int8_pair_nmse']=s['pair'].result()['nmse']
        row['balanced_score']=(row['bos_nmse']+row['non_bos_nmse'])/2
        row['worst_group_score']=max(row['bos_nmse'],row['non_bos_nmse'])
        result.append(row)
    return result


def materialize(out):
    records=sorted([json.loads(p.read_text()) for p in out.glob('layer_*.json')],key=lambda r:r['layer'])
    rows=[dict(layer=d['layer'],**r) for d in records for r in d['evaluation']]
    joint=[dict(layer=d['layer'],**r) for p in out.glob('joint_*.json') for d in [json.loads(p.read_text())] for r in d['evaluation'] if r['scheme']=='joint']
    write_csv(out.parent/'s10_calibration_3layer.csv',[r for r in rows+joint if r['layer'] in (7,8,17)])
    if len(records)==18:write_csv(out.parent/'s10_calibration_all_down.csv',rows)
    candidates=[dict(layer=d['layer'],**r) for d in records for r in d['search']['candidates']]
    write_csv(out/'candidates.csv',candidates)
    def table(headers,rr):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
    report=['# Common s_10 calibration with frozen factorized PoT scales',
        'Diagnostic only. Previous PoT s_X_normal, k, s_W, W_q, calibration/validation observations are frozen in the primary pass. '
        'Existing activation caches are reused; no model forward, activation capture, download, LUT, ABI, QB, RTL or production changes. '
        'One common s_10 is used by both groups. Every candidate regenerates ONLY the normal channel multiplier/shift array; BOS applies the existing scalar 2^k before RNE and saturation.',
        'Candidates: old s_10 * 2^(i/8), i=-16..16, exact old baseline included. Selection uses 0.5*NMSE_BOS + 0.5*NMSE_nonBOS on calibration only. '
        'The same 256 prior non-BOS row IDs and within-non-BOS weights are reused; all 128 calibration BOS rows are used independently. '
        'BOS is never weighted by corpus frequency. Balanced-score ties use rtol=1e-6, atol=1e-12, then lower worst-group NMSE, lower equal-group clipping, and closer log-distance to baseline.',
        'Practical effective-shift support is explicitly 0..31, the current unsigned five-bit shift domain, for BOTH groups. '
        'The previous observed BOS range 11..28 was descriptive, not a hardware limit; previous normal shifts already reach 31. '
        'No negative or extended shift is accepted for the main result. Unconstrained and feasible optima, base/BOS/combined shift ranges and ratio status are recorded for every operation.',
        table(['Layer','Scheme','sX normal','k','s10','Ratio','BOS NMSE %','Normal NMSE %','BOS clip %','Normal clip %','Effective shifts'],
              [[r['layer'],r['scheme'],f"{r['s_X_normal']:.8g}",r['k'],f"{r['s_10']:.8g}",f"{r['ratio']:.6g}",f"{100*r['bos_nmse']:.6f}",f"{100*r['non_bos_nmse']:.6f}",f"{100*r['bos_int10_clip_rate']:.7g}",f"{100*r['non_bos_int10_clip_rate']:.7g}",f"{r['effective_shift_min']}..{r['effective_shift_max']}"] for r in rows+joint]),
        'NMSEs above use full independent validation. Candidate tables contain calibration estimates. Output-grid lower bounds use the exact FP reference rounded directly onto each candidate INT10 grid. '
        'INT8-pair errors are independent counterfactuals; no subtraction of MSEs is used as additive attribution.']
    for tag in ['frozen','calibrated']:
        rr=[r for r in rows if r['scheme']==tag]
        if rr:report.append(f'{tag}: BOS median/max NMSE {100*np.median([r["bos_nmse"] for r in rr]):.6f}% / {100*max(r["bos_nmse"] for r in rr):.6f}%; non-BOS median/max {100*np.median([r["non_bos_nmse"] for r in rr]):.6f}% / {100*max(r["non_bos_nmse"] for r in rr):.6f}%.')
    if joint:report.append('Optional joint pass is restricted to layers 7/8/17: 3 input-scale factors {2^-0.25,1,2^0.25}, k±1, and 3 output-scale factors {2^-0.125,1,2^0.125} around the s10-only selected point (27 candidates). Same calibration rows/objective/shift bounds; no validation selection.')
    if (out/'decision.json').exists():report.append(json.loads((out/'decision.json').read_text())['explanation'])
    if (out/'answers.md').exists():report.append((out/'answers.md').read_text())
    (out.parent/'s10_calibration_report.md').write_text('\n\n'.join(report)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--layers',default='7,8,17');ap.add_argument('--joint',action='store_true')
    ap.add_argument('--model',default='/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf')
    args=ap.parse_args();layers=list(range(18)) if args.layers=='all' else [int(i) for i in args.layers.split(',')]
    out=ROOT/'diagnostics/s10_calibration';out.mkdir(exist_ok=True);prior=ROOT/'diagnostics/factorized_row_requant'
    if set(layers)-{7,8,17} or args.joint:
        assert json.loads((out/'decision.json').read_text())['proceed'],'positive first-three decision required'
    if args.joint:assert set(layers)<={7,8,17},'joint experiment is three-layer only'
    previous_identity=json.loads((prior/'provenance.json').read_text())
    assert sha256_file(ROOT/'scripts/test_factorized_row_requant.py')==previous_identity['script_sha256']
    old_identity=json.loads((ROOT/'diagnostics/bos_aware/provenance.json').read_text())
    for p,h in old_identity['source_hashes'].items():assert sha256_file(ROOT/p)==h,p
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1';manifest=json.loads((art/'manifest.json').read_text())
    identity=dict(script_sha256=sha256_file(__file__),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        previous_provenance_sha256=sha256_file(prior/'provenance.json'),model_sha256=sha256_file(args.model),
        effective_shift_bounds=[0,31],ratio_exponents=[-16,16,8],profile=Profile().metadata())
    assert identity['model_sha256']==previous_identity['model_sha256'] and identity['profile']==previous_identity['profile']
    if (out/'provenance.json').exists():assert json.loads((out/'provenance.json').read_text())==identity
    write_json(out/'provenance.json',identity)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,time=time.time()))+'\n')
    import gguf
    reader=gguf.GGUFReader(args.model);tensors={t.name:t for t in reader.tensors}
    positions=token_positions(reader,manifest,ROOT/'calibration_outputs/data-oasst1-v1');configure_blas('accelerate')
    entries={e['module_name']:e for e in manifest['modules']}
    for layer in layers:
        filename=f'joint_{layer}.json' if args.joint else f'layer_{layer}.json'
        if (out/filename).exists():continue
        started=time.monotonic();past=json.loads((prior/f'layer_{layer}.json').read_text());frozen=next(r for r in past['evaluation'] if r['scheme']=='pot')
        selected=json.loads((prior/f'selected_{layer}.json').read_text());sx=selected['s_X_normal'];k=selected['pot']['k'];old=selected['s_10']
        assert (sx,k,old)==(frozen['s_X_normal'],frozen['k'],frozen['s_10'])
        for c in past['cache_files'].values():
            p=Path(c['path']);assert p.stat().st_size==c['bytes'] and p.stat().st_mtime_ns==c['mtime_ns']
        t=tensors[f'blk.{layer}.ffn_down.weight'];w=gguf.quants.dequantize(t.data,t.tensor_type).reshape(tuple(reversed(t.shape)))
        wq,sw=quantize_weight(w);entry=entries[f'model.layers.{layer}.mlp.down_proj']
        np.testing.assert_array_equal(wq,np.load(art/entry['int8_weight_file'],mmap_mode='r'))
        with np.load(art/Path(entry['module_manifest']).parent/'scales.npz') as z:np.testing.assert_array_equal(sw,z['op0000.s_W'])
        xc=np.load(past['cache_files']['calibration']['path'],mmap_mode='r');groups=calibration_groups(xc,positions['calibration'],selected)
        specs=[(sx,k,s) for s in scales(old)]
        if args.joint:
            center=json.loads((out/f'layer_{layer}.json').read_text())['search']['selected']
            specs=[(sx*2.**dx,k+dk,center['s_10']*2.**ds) for dx in [-.25,0,.25] for dk in [-1,0,1] for ds in [-.125,0,.125]]
        for g,(x,_) in groups.items():
            q=quantize_input(x[:2],sx*2.**(k if g=='bos' else 0));qw=wq[:2]
            np.testing.assert_array_equal(integer_dot(q,qw,'fp64-exact'),integer_dot(q,qw,'int64'))
        search=score_candidates(groups,w,sw,specs,old,backend='fp64-exact');choice=search['selected']
        write_json(out/('selected_'+filename),search)
        print('SELECTED',layer,'joint' if args.joint else 's10',choice['s_10'],choice['ratio'],choice['balanced_score'],flush=True)
        policy=lambda sx,k,s:dict(s_X_normal=sx,k=k,old_s_10=old,s_10=s,ratio=s/old)
        policies=dict(frozen=policy(sx,k,old))
        if args.joint:
            policies['calibrated']=policy(sx,k,center['s_10']);policies['joint']=policy(choice['s_X_normal'],choice['k'],choice['s_10'])
        else:policies['calibrated']=policy(sx,k,choice['s_10'])
        xv=np.load(past['cache_files']['validation']['path'],mmap_mode='r')
        evaluation=evaluate(xv,positions['validation'],w,sw,policies,backend='fp64-exact')
        for g in ['bos','non_bos']:
            for metric in ['mse','mae','nmse','int10_clip_rate']:
                assert np.isclose(evaluation[0][g+'_'+metric],frozen[g+'_'+metric],rtol=2e-12,atol=1e-15),(layer,g,metric)
        write_json(out/filename,dict(layer=layer,search=search,evaluation=evaluation,seconds=time.monotonic()-started,
            previous_result_sha256=sha256_file(prior/f'layer_{layer}.json'),previous_selection_sha256=sha256_file(prior/f'selected_{layer}.json')))
        materialize(out);print('DONE',layer,[(r['scheme'],r['bos_nmse'],r['non_bos_nmse']) for r in evaluation],flush=True)


if __name__=='__main__':main()
