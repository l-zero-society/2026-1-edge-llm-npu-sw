#!/usr/bin/env python3
"""Frozen Final policy, HW-selector-only inference; never rerun FP/MSE3."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import finalize_downproj_ptq as final
row = final.cached.row
base = final.base
Profile = final.Profile
MSE3 = row.select_rows
OUT = ROOT / 'diagnostics/k_selector_compare'


def select_absmax(x, sx, shift):
    x = row.finite(np.asarray(x, np.float64), 'activation')
    row.positive_scale(sx, 'base scale')
    if x.ndim != 2: raise ValueError('expected rows by channels')
    maximum = np.abs(x).max(axis=1)
    k0 = np.rint(np.log2(np.where(maximum > 0, maximum / (127*sx), 1.))).astype(np.int64)
    shifts = Profile().effective(shift)
    lo, hi = max(-8, int(shifts.max())-31), min(7, int(shifts.min()))
    if lo > hi: raise ValueError('empty feasible range')
    k = np.clip(k0, lo, hi)
    q = np.empty(x.shape, np.int8)
    for kk in np.unique(k):
        mask = k == kk
        q[mask] = row.quantize_input(x[mask], sx*2.**int(kk))
    assert 0 <= shifts.min()-k.max() <= shifts.max()-k.min() <= 31
    return q, k, dict(clamped=int((k != k0).sum()),
        effective_shift_min=int(shifts.min()-k.max()), effective_shift_max=int(shifts.max()-k.min()))


def empty_stats():
    return dict(rows=0, inputs=0, agreement=0, abs_delta_sum=0, max_abs_delta=0,
        clamped=0, hw_clips=0, mse3_clips=0, hw_zeros=0, mse3_zeros=0,
        effective_shift_min=31, effective_shift_max=0, delta={}, hw_hist={}, mse3_hist={})


def add_stats(s, x, sx, q, k, qm, km, info):
    delta = k-km
    s['rows'] += len(k); s['inputs'] += x.size
    s['agreement'] += int((delta == 0).sum())
    s['abs_delta_sum'] += int(np.abs(delta).sum())
    s['max_abs_delta'] = max(s['max_abs_delta'], int(np.abs(delta).max()))
    s['clamped'] += info['clamped']
    for tag, codes, ks in [('hw', q, k), ('mse3', qm, km)]:
        s[tag+'_clips'] += int((np.abs(x) > 127*sx*np.exp2(ks)[:,None]).sum())
        s[tag+'_zeros'] += int((codes == 0).sum())
    for name, values in [('delta',delta), ('hw_hist',k), ('mse3_hist',km)]:
        for value,count in zip(*np.unique(values,return_counts=True)):
            key=str(int(value)); s[name][key]=s[name].get(key,0)+int(count)
    s['effective_shift_min']=min(s['effective_shift_min'],info['effective_shift_min'])
    s['effective_shift_max']=max(s['effective_shift_max'],info['effective_shift_max'])


def fingerprint(specs):
    h=hashlib.sha256()
    for name,s in sorted(specs.items()):
        h.update(name.encode()); h.update(repr((s['sx'],s['s10'])).encode())
        h.update(s['wq'].tobytes())
        for key in ['multiplier','shift']: h.update(s['params'][key].tobytes())
    return h.hexdigest()


def aggregate(records, stored, provenance):
    assert len(records)==len(stored)==24
    assert all(r['targets']==32 for r in records)
    assert all(a['index']==b['index'] for a,b in zip(records,stored))
    modes=[]
    for label,key in [('FP','FP'),('Final MSE3','calibrated')]:
        modes.append(dict(mode=label,**final.response.merged([r['scores'][key]['all'] for r in stored])))
    assert all(r['tokens']==768 for r in modes)
    n=sum(r['targets'] for r in records); assert n==768
    nll=sum(r['nll_sum'] for r in records)/n
    modes.append(dict(mode='HW absmax',tokens=n,nll=nll,ppl=math.exp(nll),
        kl=None,nmse=None,cosine=None,top1=None,in5=None,overlap=None))
    conv=[]
    for hw,old in zip(records,stored):
        a=final.response.merged([old['scores']['calibrated']['all']])['nll']
        b=hw['nll_sum']/hw['targets']; d=b-a
        conv.append(dict(index=hw['index'],targets=hw['targets'],mse3_nll=a,hw_nll=b,delta_nll=d,
            outcome='equal' if abs(d)<1e-3 else ('better' if d<0 else 'worse')))
    stats=[]; hist=[]
    for layer in list(range(18))+['all']:
        s=empty_stats()
        for r in records:
            for l in (range(18) if layer=='all' else [layer]):
                a=r['stats'][str(l)]
                for k,v in a.items():
                    if isinstance(v,dict):
                        for kk,vv in v.items(): s[k][kk]=s[k].get(kk,0)+vv
                    elif k=='effective_shift_min': s[k]=min(s[k],v)
                    elif k in ('effective_shift_max','max_abs_delta'): s[k]=max(s[k],v)
                    else: s[k]+=v
        stats.append(dict(layer=layer,rows=s['rows'],agreement=s['agreement']/s['rows'],
            mean_abs_delta=s['abs_delta_sum']/s['rows'],max_abs_delta=s['max_abs_delta'],
            clamp_rate=s['clamped']/s['rows'],hw_clip_rate=s['hw_clips']/s['inputs'],
            mse3_clip_rate=s['mse3_clips']/s['inputs'],hw_zero_rate=s['hw_zeros']/s['inputs'],
            mse3_zero_rate=s['mse3_zeros']/s['inputs'],effective_shift_min=s['effective_shift_min'],
            effective_shift_max=s['effective_shift_max']))
        for kind in ('delta','hw_hist','mse3_hist'):
            for k,count in sorted(s[kind].items(),key=lambda z:int(z[0])):
                hist.append(dict(layer=layer,kind=kind,k=int(k),count=count,fraction=count/s['rows']))
    summary=dict(provenance=provenance,modes=modes,stats=stats,
        relative_ppl_change=modes[2]['ppl']/modes[1]['ppl']-1,
        relative_kl_change=None,relative_nmse_change=None,top1_pp_change=None,
        conversation_outcomes={k:sum(r['outcome']==k for r in conv) for k in ['better','worse','equal']},
        missing_reference='Stored short JSONs contain aggregate metrics only, no aligned FP logits or hidden vectors. New HW-vs-FP KL/NMSE/cosine/top-k and hidden NMSE cannot be recovered without forbidden FP inference. These metrics are unmeasured, not zero.')
    summary['top_disagreement_layers']=sorted(stats[:-1],key=lambda r:r['agreement'])[:3]
    summary['delta_histogram']={str(r['k']):r['count'] for r in hist
                                if r['layer']=='all' and r['kind']=='delta'}
    summary['interpretation']=(
        'Absmax is a promising HW candidate on this 768-target likelihood test: '
        f"PPL changes by {100*summary['relative_ppl_change']:.4f}% and k agrees on "
        f"{100*stats[-1]['agreement']:.4f}% of identical-input rows. "
        'This does not establish FP-logit fidelity or hardware adoption; the required '
        'aligned FP reference vectors are absent and FP inference was explicitly forbidden.')
    def check_finite(value):
        if isinstance(value,dict):
            for v in value.values(): check_finite(v)
        elif isinstance(value,list):
            for v in value: check_finite(v)
        elif isinstance(value,float): assert math.isfinite(value)
    check_finite(summary)
    assert all(0<=r['effective_shift_min']<=r['effective_shift_max']<=31 for r in stats)
    base.write_json(OUT/'summary.json',summary)
    for name,rows in [('per_conversation',conv),('k_stats',stats),('k_delta_histogram',hist)]:base.write_csv(OUT/(name+'.csv'),rows)
    lines=['# Frozen k-selector comparison','| Mode | NLL | PPL | KL | Logits NMSE | Cosine | Top1 | FP-top1-in-Q-top5 | Top5 overlap |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in modes:
        vals=[r.get(k) for k in ['nll','ppl','kl','nmse','cosine','top1','in5','overlap']]
        lines.append('| '+r['mode']+' | '+' | '.join('not measured' if v is None else f'{v:.8g}' for v in vals)+' |')
    lines += ['',summary['missing_reference'],'',f"PPL relative change: {100*summary['relative_ppl_change']:.5f}%.",
        f"Conversation NLL outcomes: {summary['conversation_outcomes']}.",
        'Selector rates include all prompt prefill rows and 31 cached decode rows per chat; both selectors see the identical HW trajectory activation.',
        f'All-layer selector statistics: {stats[-1]}',
        f"Delta k histogram (HW minus MSE3): {summary['delta_histogram']}",
        'Clipping/zero-rate differences compare both selectors on identical HW-trajectory inputs, not the independently propagated MSE3 trajectory.',
        '| Layer | k agreement | Mean absolute delta k | HW clipping | MSE3 clipping | HW zero | MSE3 zero | Shift min | Shift max |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append('| '+str(r['layer'])+' | '+' | '.join(f'{r[k]:.7g}' for k in
            ['agreement','mean_abs_delta','hw_clip_rate','mse3_clip_rate','hw_zero_rate',
             'mse3_zero_rate','effective_shift_min','effective_shift_max'])+' |')
    lines += [
        'Hidden comparison omitted: aligned FP hidden vectors were not saved in the baseline artifacts.',
        summary['interpretation'],
        'No recalibration, FP/MSE3 inference, long validation or generation. Only 18 down_proj patched. Frozen parameter and weight fingerprints verified before/after every conversation.']
    (OUT/'report.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen_path=final.OUT/'decisions.json'
    frozen_bytes=frozen_path.read_bytes()
    head=subprocess.check_output(['git','show','HEAD:diagnostics/final_downproj_ptq/decisions.json'],cwd=ROOT)
    assert head==frozen_bytes, 'working parameters differ from HEAD'
    stored=[json.loads((final.PROGRESS/f'short_{i:03d}.json').read_text()) for i in range(24)]
    for key in ['FP','calibrated']: assert sum(r['scores'][key]['all']['n'] for r in stored)==768
    source,_,_,previous,_,_=final.load_source_and_policies()
    specs,_=final.restore_final(source,previous)
    assert set(specs)=={f'model.layers.{i}.mlp.down_proj' for i in range(18)}
    examples,_=final.validation_examples(source)
    assert len(examples)==24 and sum(min(32,len(e['targets'])) for e in examples)==768
    digest=fingerprint(specs)
    provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_sha256=hashlib.sha256(frozen_bytes).hexdigest(),parameter_sha256=digest,
        script_sha256=base.sha256_file(__file__),baseline_sha256=[base.sha256_file(final.PROGRESS/f'short_{i:03d}.json') for i in range(24)],
        command=sys.argv,fp_inference_calls=0,mse3_inference_calls=0)
    records=[]
    for i,e in enumerate(examples):
        assert e['index']==stored[i]['index']
        path=OUT/f'conversation_{i:03d}.json'
        if path.exists():
            record=json.loads(path.read_text()); assert record['parameter_sha256']==digest
        else:
            stats={str(l):empty_stats() for l in range(18)}; calls=[0]
            def selector(x,sx,shift):
                layer=calls[0]%18; calls[0]+=1
                spec=specs[f'model.layers.{layer}.mlp.down_proj']
                assert sx==spec['sx']; np.testing.assert_array_equal(shift,spec['params']['shift'])
                q,k,info=select_absmax(x,sx,shift)
                qm,km,counts=MSE3(x,sx,shift)
                add_stats(stats[str(layer)],x,sx,q,k,qm,km,info)
                counts.update(info)
                return q,k,counts
            cache=None; total=0.
            with patch.object(row,'select_rows',selector):
                for t,target in enumerate(e['targets'][:32]):
                    ids=e['prompt_ids'] if t==0 else [e['targets'][t-1]]
                    logits,_,cache=final.cached.run_mode(source,specs,ids,cache,layers=False)
                    z=logits.double(); nll=(source.torch.logsumexp(z,dim=-1)-z[:,target]).item()
                    assert math.isfinite(nll); total+=nll
            assert calls[0]==18*32
            record=dict(index=e['index'],targets=32,nll_sum=total,stats=stats,parameter_sha256=digest)
            assert fingerprint(specs)==digest and frozen_path.read_bytes()==frozen_bytes
            base.write_json(path,record)
        records.append(record)
        print('HW',i+1,24,'NLL',record['nll_sum']/32,flush=True)
    aggregate(records,stored,provenance)

if __name__=='__main__':main()
