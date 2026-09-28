#!/usr/bin/env python3
"""Matched-prefix cache/full trace. Frozen down_proj only; no policy search."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_response_quant as e
row=e.row; base=e.base; ROOT=e.ROOT
OUT=ROOT/'diagnostics/cache_full_divergence'
TAGS=['FP','bos_only','general_pot','force_full_k','force_full_Xq','force_full_q10']


def histories(example,t,cached_history):
    full=example['prompt_ids']+example['targets'][:t]
    assert full==cached_history, 'token history mismatch'
    assert len(full)-1==example['predictor_positions'][t], 'absolute position mismatch'
    return full


def override_codes(x,q,ks,sx,reference,control,index=-1):
    """Only selected row; input X is never modified."""
    q=q.copy();ks=ks.copy()
    if control in ('force_full_k','force_full_Xq'):
        ks[index]=reference['k']
        q[index]=row.quantize_input(x[index][None],sx*2.**int(ks[index]))[0]
        if control=='force_full_Xq':q[index]=reference['xq']
    return q,ks


def numeric(cache,full):
    a=np.asarray(cache,np.float64);b=np.asarray(full,np.float64);delta=a-b
    re=float(np.square(b).sum());qe=float(np.square(a).sum());se=float(np.square(delta).sum())
    return dict(nmse=se/re if re else (0. if not se else None),cosine=float((a*b).sum()/np.sqrt(re*qe)) if re*qe else float(not se),max_abs=float(np.abs(delta).max()),mean_abs=float(np.abs(delta).mean()))


def codes(cache,full):
    delta=np.asarray(cache,np.int64)-np.asarray(full,np.int64)
    return dict(mismatch_fraction=float(np.mean(delta!=0)),mean_abs_code_delta=float(np.abs(delta).mean()),max_abs_code_delta=int(np.abs(delta).max()))


def boundaries(x,scale):
    d=np.asarray(x,np.float64)/scale
    return np.abs(np.abs(d-np.floor(d))-.5)


def take_kv(cache):
    result=[]
    for layer in range(18):
        item=cache.layers[layer]
        result.append((item.keys[0,:,-1,:].detach().numpy().copy(),item.values[0,:,-1,:].detach().numpy().copy()))
    return result


@contextmanager
def instrument(source,specs,mode,offset,traces,control=None,reference=None):
    originals={}
    try:
        for name,spec in specs.items():
            layer=int(name.split('.')[2]);module=source.modules[name];original=module.forward; originals[name]=original
            def forward(x,spec=spec,layer=layer,original=original):
                xx=x.detach().numpy();trace={'x':xx[0,-1].copy()};traces[layer]=trace
                if mode=='FP':return original(x)
                collected=[];selection=row.select_rows;dot=base.integer_dot
                def select(arr,sx,shift):
                    q,ks,counts=selection(arr,sx,shift)
                    if control:q,ks=override_codes(arr,q,ks,sx,reference[layer],control)
                    shifts=base.Profile().effective(shift)
                    assert np.all(shifts.min()-ks>=0) and np.all(shifts.max()-ks<=31)
                    trace['k']=int(ks[-1]);return q,ks,counts
                def accumulate(q,w,backend):
                    acc=dot(q,w,backend)
                    trace['xq']=q[-1].copy();collected.append(acc[-1].copy());return acc
                if mode=='bos_only':
                    trace['k']=spec['k'] if offset+xx.shape[1]-1==0 else 0
                    with patch.object(base,'integer_dot',accumulate):
                        out=base.quantized_linear(xx,**dict(spec,k=spec['k'] if offset==0 else 0))[0]
                else:
                    with patch.object(row,'select_rows',select),patch.object(row,'integer_dot',accumulate):
                        out=row.apply_rows(xx,**spec)[0]
                trace['acc']=np.concatenate(collected)
                shift=spec['params']['shift']-trace['k']; assert np.all((shift>=0)&(shift<=31))
                trace['shift_min']=int(shift.min());trace['shift_max']=int(shift.max())
                trace['raw']=base.Profile().apply(trace['acc'][None],spec['params']['multiplier'],shift,saturate=False)[0]
                trace['q10']=np.clip(trace['raw'],-512,511)
                expected=(trace['q10']*spec['s10']).astype(np.float32)
                np.testing.assert_array_equal(expected,out[0,-1])
                if control=='force_full_q10':
                    trace['q10']=reference[layer]['q10'].copy()
                    out[0,-1]=(trace['q10']*spec['s10']).astype(np.float32)
                trace['y']=out[0,-1].copy()
                return source.torch.from_numpy(out).to(x.dtype)
            module.forward=forward
        yield
    finally:
        for name,fn in originals.items():source.modules[name].forward=fn


def run(source,specs,ids,mode,cache=None,control=None,reference=None,kv=False):
    offset=cache.get_seq_length() if cache is not None else 0
    traces={};context={}
    with instrument(source,specs,mode,offset,traces,control,reference):
        h,states,newcache=e.forward_hidden(source,ids,[len(ids)-1],context,cache,True,True)
    for l in range(18):traces[l]['hidden']=states[l][0].numpy().copy()
    with source.torch.inference_mode():logits=source.model.lm_head(h).detach()
    return dict(trace=traces,logits=logits,cache=newcache,absolute=offset+len(ids)-1,kv=take_kv(newcache) if kv else None)


def compare(full,cached,specs,tag,index,t,trace_rows,boundary_rows,kv_rows):
    assert full['absolute']==cached['absolute']
    for l in range(18):
        a=cached['trace'][l];b=full['trace'][l];r=dict(mode=tag,prompt=index,token=t,layer=l,absolute=full['absolute'])
        for field in ['x','hidden']:
            r.update({field+'_'+k:v for k,v in numeric(a[field],b[field]).items()})
        if tag!='FP':
            r.update(k_full=b['k'],k_cache=a['k'],k_match=a['k']==b['k'],effective_shift_min=a['shift_min'],effective_shift_max=a['shift_max'])
            for field in ['xq','acc','raw','q10']:r.update({field+'_'+k:v for k,v in codes(a[field],b[field]).items()})
            r['acc_nmse']=numeric(a['acc'],b['acc'])['nmse']
            r.update({'y_'+k:v for k,v in numeric(a['y'],b['y']).items()})
            sx=specs[f'model.layers.{l}.mlp.down_proj']['sx']
            if a['k']!=b['k']:
                for side,tr in [('full',b),('cache',a)]:
                    err=[]
                    for k in [b['k'],a['k']]:
                        scale=sx*2.**k;qq=row.quantize_input(tr['x'][None],scale)
                        err.append(float(np.square(qq[0].astype(np.float64)*scale-tr['x'].astype(np.float64)).mean()))
                    r[side+'_mse_full_k']=err[0];r[side+'_mse_cache_k']=err[1];r[side+'_mse_full_minus_cache_k']=err[0]-err[1]
            mismatch=a['xq']!=b['xq']
            if mismatch.any():
                for side,tr in [('full',b),('cache',a)]:
                    distance=boundaries(tr['x'],sx*2.**tr['k'])
                    for label,mask in [('mismatched',mismatch),('matched',~mismatch)]:
                        if mask.any():
                            pp=np.quantile(distance[mask],[.5,.9,.99])
                            boundary_rows.append(dict(mode=tag,prompt=index,token=t,layer=l,side=side,elements=label,count=int(mask.sum()),median=float(pp[0]),p90=float(pp[1]),p99=float(pp[2]),k_match=a['k']==b['k']))
        trace_rows.append(r)
        if tag=='general_pot':
            kvr=dict(prompt=index,token=t,layer=l,absolute=full['absolute'])
            for j,label in enumerate(['K','V']):kvr.update({label+'_'+k:v for k,v in numeric(cached['kv'][l][j],full['kv'][l][j]).items()})
            kv_rows.append(kvr)


def logit_comparison(torch,full,cache,target,tag,index,t):
    q=cache.double();f=full.double();lq=torch.log_softmax(q,-1);lf=torch.log_softmax(f,-1)
    metric=base.Metric();metric.add(q.numpy(),f.numpy())
    delta=float(lf[0,target]-lq[0,target])
    return dict(mode=tag,prompt=index,token=t,metric=vars(metric),kl=float((lf.exp()*(lf-lq)).sum()),top1=int(q.argmax(-1)==f.argmax(-1)),nll_delta=delta,abs_nll_delta=abs(delta))


def save(trace,boundary,kv,outputs):
    # Uniform union schema, without vector dumps.
    def csv(path,rows):
        if rows:
            keys=list(dict.fromkeys(k for r in rows for k in r));base.write_csv(path,[{k:r.get(k) for k in keys} for r in rows])
    csv(OUT.parent/'cache_full_divergence_trace.csv',trace)
    csv(OUT.parent/'cache_full_divergence_boundary.csv',boundary)
    csv(OUT.parent/'cache_full_divergence_kv.csv',kv)
    base.write_json(OUT/'logit_statistics.json',outputs)
    first=[];aggregates={};controls=[]
    for tag in TAGS:
        rr=[r for r in trace if r['mode']==tag]
        if not rr:continue
        groups=sorted({(r['prompt'],r['token']) for r in rr})
        for index,t in groups:
            ll=[r for r in rr if r['prompt']==index and r['token']==t]
            conditions={'k_mismatch':lambda r:r.get('k_match',True)==False,'xq_mismatch':lambda r:r.get('xq_mismatch_fraction',0)>0,'q10_mismatch':lambda r:r.get('q10_mismatch_fraction',0)>0}
            conditions.update({f'hidden_nmse_gt_{v:g}':lambda r,v=v:r['hidden_nmse']>v for v in [1e-6,1e-4,1e-2]})
            first.append(dict(mode=tag,prompt=index,token=t,**{k:next((r['layer'] for r in ll if fn(r)),None) for k,fn in conditions.items()}))
        agg=dict(rows=len(rr),x_nmse_max=max(r['x_nmse'] for r in rr),x_max_abs_max=max(r['x_max_abs'] for r in rr),hidden_nmse_max=max(r['hidden_nmse'] for r in rr))
        if tag!='FP':
            agg.update(k_mismatch_fraction=float(np.mean([not r['k_match'] for r in rr])),xq_any_fraction=float(np.mean([r['xq_mismatch_fraction']>0 for r in rr])),q10_any_fraction=float(np.mean([r['q10_mismatch_fraction']>0 for r in rr])),xq_element_mismatch_fraction=float(np.mean([r['xq_mismatch_fraction'] for r in rr])),q10_element_mismatch_fraction=float(np.mean([r['q10_mismatch_fraction'] for r in rr])),effective_shift_min=min(r['effective_shift_min'] for r in rr),effective_shift_max=max(r['effective_shift_max'] for r in rr))
        aggregates[tag]=agg
        oo=[r for r in outputs if r['mode']==tag];mm=base.merge_metrics([r['metric'] for r in oo])
        controls.append(dict(mode=tag,targets=len(oo),top1_agreement=float(np.mean([r['top1'] for r in oo])),kl=float(np.mean([r['kl'] for r in oo])),logits_nmse=mm['nmse'],mean_abs_nll_delta=float(np.mean([r['abs_nll_delta'] for r in oo])),max_abs_nll_delta=max(r['abs_nll_delta'] for r in oo),mean_nll_delta=float(np.mean([r['nll_delta'] for r in oo]))))
    csv(OUT.parent/'cache_full_divergence_first.csv',first);csv(OUT.parent/'cache_full_divergence_controls.csv',controls)
    summary=dict(prompts=4,targets_per_mode=64,trace=aggregates,controls=controls,first=first,completed_modes=list(aggregates),kv_rows=len(kv),controls_scope='Only current matched predictor row injected; prompt KV and other rows retain their own cached path. Full references use exactly the same prefix length.',boundary_scope='Exact per-row element quantiles; only rows with Xq mismatches; no pooled-percentile approximation.')
    base.write_json(OUT.parent/'cache_full_divergence_summary.json',summary)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report-only',action='store_true');ap.add_argument('--resume-controls',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    if args.report_only:
        import csv
        summary=json.loads((OUT.parent/'cache_full_divergence_summary.json').read_text())
        lines=['# Full-prefix versus cached down_proj trace','4 prompts × 16 targets; 18 down_proj only. Frozen global w_BOS=0.25. All other operations remain FP. Every full forward uses prompt + exactly the same GT response prefix as the cached history. Target t is predicted at absolute position prompt_length+t-1. No future tokens enter either path.','| mode | top1 % | KL | logits NMSE | mean abs NLL delta | max abs NLL delta |','|---|---:|---:|---:|---:|---:|']
        for r in summary['controls']:lines.append(f'| {r["mode"]} | {100*r["top1_agreement"]:.4f} | {r["kl"]:.9g} | {r["logits_nmse"]:.9g} | {r["mean_abs_nll_delta"]:.9g} | {r["max_abs_nll_delta"]:.9g} |')
        if (OUT/'answers.md').exists():lines.append((OUT/'answers.md').read_text())
        (OUT.parent/'cache_full_divergence_report.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n');return
    previous=ROOT/'diagnostics/response_quant_eval';past=json.loads((previous/'provenance.json').read_text())
    assert base.sha256_file(ROOT/'scripts/evaluate_response_quant.py')==past['script_sha256']
    examples=json.loads((previous/'examples.json').read_text())[:4]
    assert [x['index'] for x in examples]==[0,1,2,3] and all(x['response_tokens']>=16 for x in examples)
    base.write_json(OUT/'examples.json',[{k:x[k] for k in ['index','anchor_id','response_id','prompt_tokens','response_tokens']} for x in examples])
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1';manifest=json.loads((art/'manifest.json').read_text())
    source=base.RecordedGGUFSource(e.MODEL,e.DATA,manifest);base.configure_blas('accelerate')
    source.tokenizer.chat_template=e.native_template(e.MODEL)
    assert source.tokenizer.chat_template==json.loads((previous/'protocol.json').read_text())['template']
    frozen_path=ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json';assert base.sha256_file(frozen_path)==past['frozen_selection_sha256']
    filtered=dict(manifest,modules=[m for m in manifest['modules'] if m['module_name'].endswith('.down_proj')])
    specs,missing,hashes=base.load_artifacts(source,art,filtered,json.loads(frozen_path.read_text()));assert not missing and len(specs)==18
    base.write_json(OUT/'provenance.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=base.sha256_file(__file__),previous_provenance_sha256=base.sha256_file(previous/'provenance.json'),examples_sha256=base.sha256_file(previous/'examples.json'),frozen_selection_sha256=base.sha256_file(frozen_path),artifact_hashes=hashes,profile=base.Profile().metadata(),model_sha256=past['model_sha256'],command=sys.argv))
    trace=[];boundary=[];kv=[];outputs=[];references={};seeds={}
    tags=TAGS
    if args.resume_controls:
        import csv
        def load_csv(path):
            def value(v):
                if v=='':return None
                if v=='True':return True
                if v=='False':return False
                try:return float(v) if any(c in v for c in '.eE') else int(v)
                except ValueError:return v
            return [{k:value(v) for k,v in r.items() if v!=''} for r in csv.DictReader(open(path))]
        trace=load_csv(OUT.parent/'cache_full_divergence_trace.csv')
        boundary=load_csv(OUT.parent/'cache_full_divergence_boundary.csv')
        kv=load_csv(OUT.parent/'cache_full_divergence_kv.csv')
        outputs=json.loads((OUT/'logit_statistics.json').read_text())
        assert all(sum(r['mode']==m for r in trace)==64*18 for m in TAGS[:3])
        assert all(r['mode'] in TAGS[:3] for r in trace)
        # A terminated process cannot retain the unsaved reference vectors. Rebuild
        # General FULL references only; keep all completed baseline statistics.
        for ex in examples:
            for t in range(16):
                ids=histories(ex,t,ex['prompt_ids']+ex['targets'][:t])
                full=run(source,specs,ids,'general_pot',kv=True);full.pop('cache')
                references[ex['index'],t]=full
                if t==0:
                    cached=run(source,specs,ex['prompt_ids'],'general_pot',kv=True)
                    c=cached.pop('cache');seeds[ex['index']]=dict(cache=c,record=cached)
                    for l in range(18):np.testing.assert_array_equal(cached['trace'][l]['q10'],full['trace'][l]['q10'])
                print('REBUILD FULL',ex['index'],t,flush=True)
        tags=TAGS[3:]
    for tag in tags:
        mode=tag if tag in e.MODES else 'general_pot';control=None if tag in e.MODES else tag
        for ex in examples:
            index=ex['index'];cache=None;history=ex['prompt_ids'].copy()
            if control:
                cache=copy.deepcopy(seeds[index]['cache'])
            for t in range(16):
                if t:history.append(ex['targets'][t-1])
                ids=histories(ex,t,history)
                if not control:
                    full=run(source,specs,ids,mode,kv=tag=='general_pot')
                    # Full temporary KV is never used by cached execution.
                    full.pop('cache')
                    if tag=='general_pot':references[index,t]=full
                else:full=references[index,t]
                if control and t==0:
                    cached=seeds[index]['record']  # Identical t0 injections; no duplicate prefill.
                else:
                    feed=ex['prompt_ids'] if t==0 else [ex['targets'][t-1]]
                    cached=run(source,specs,feed,mode,cache,control,full['trace'] if control else None,kv=tag=='general_pot')
                    cache=cached.pop('cache')
                assert cached['absolute']==len(history)-1==full['absolute']
                if tag=='general_pot' and t==0:seeds[index]=dict(cache=copy.deepcopy(cache),record=cached)
                compare(full,cached,specs,tag,index,t,trace,boundary,kv)
                outputs.append(logit_comparison(source.torch,full['logits'],cached['logits'],ex['targets'][t],tag,index,t))
                print('TRACE',tag,index,t,flush=True)
            save(trace,boundary,kv,outputs)
        print('MODE COMPLETE',tag,flush=True)
        if tag=='FP':
            oo=[r for r in outputs if r['mode']=='FP'];assert all(r['top1']==1 and r['abs_nll_delta']<.001 for r in oo),'FP parity failed'
    save(trace,boundary,kv,outputs)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
