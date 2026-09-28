#!/usr/bin/env python3
"""Sequential, resumable down_proj-only likelihood diagnosis. No production writes."""
import argparse
from contextlib import contextmanager
import csv
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import test_e2e_row_pot as row
base=row.base
from quant_diagnostic_math import dot,integer_dot
from static_quant.core import quantize_input
from static_quant.hardware import Profile,finite
OUT=ROOT/'diagnostics/row_pot_ppl'
MODES=['bos_only','general_pot','cap0','cap1','cap2','pair_oracle','int10_oracle','recentered','arbitrary_absmax']


def candidates(x,sx,shift,cap=None,bos_k=None):
    x=finite(np.asarray(x,np.float64),'activation')
    maximum=np.abs(x).max(axis=1)
    k0=np.rint(np.log2(np.where(maximum>0,maximum/(127*sx),1.))).astype(np.int64)
    ks=k0[None,:]+np.array([0,-1,1])[:,None]
    valid=(ks>=-8)&(ks<=7)&(ks>=int(shift.max())-31)&(ks<=int(shift.min()))
    if cap is not None:valid[:,1:] &= ks[:,1:]<=cap
    if bos_k is not None:valid[:,0] &= ks[:,0]==bos_k
    missing=np.flatnonzero(~valid.any(axis=0))
    if len(missing):raise ValueError(f'empty strict candidate set: row(s) {missing[:16].tolist()}, k0 {k0[missing[:16]].tolist()}, cap={cap}, shift-k bounds=[{int(shift.max())-31},{int(shift.min())}]')
    qs=[];errors=np.full(ks.shape,np.inf)
    for i in range(3):
        q=np.zeros(x.shape,np.int8)
        for k in np.unique(ks[i,valid[i]]):
            ids=np.flatnonzero(valid[i]&(ks[i]==k));scale=sx*2.**int(k)
            q[ids]=quantize_input(x[ids],scale)
            errors[i,ids]=np.square(q[ids].astype(np.float64)*scale-x[ids]).mean(axis=1)
        qs.append(q)
    return ks,valid,qs,errors


def quant_rows(x,spec,ks):
    q=np.empty(x.shape,np.int8)
    for k in np.unique(ks):
        mask=ks==k;q[mask]=quantize_input(x[mask],spec['sx']*2.**int(k))
    return q


def hw_output(q,ks,spec):
    out=np.empty((len(q),len(spec['wq'])),np.float32);clips=0
    for start in range(0,len(spec['wq']),512):
        sl=slice(start,start+512);acc=integer_dot(q,spec['wq'][sl],'fp64-exact')
        raw=Profile().apply(acc,spec['params']['multiplier'][sl],spec['params']['shift'][sl][None,:]-ks[:,None],saturate=False)
        clips+=int(((raw < -512)|(raw > 511)).sum())
        out[:,sl]=(np.clip(raw,-512,511)*spec['s10']).astype(np.float32)
    return out,clips


def select_oracle(x,spec,w,sw,objective):
    """Current-row FP Linear output only. No labels or reference trajectory inputs."""
    ks,valid,qs,activation=candidates(x,spec['sx'],spec['params']['shift'])
    pair=np.zeros(ks.shape);final=np.zeros(ks.shape)
    ys=np.empty((3,len(x),len(spec['wq'])),np.float32)
    clip_rows=np.zeros(ks.shape,np.int64)
    for start in range(0,len(spec['wq']),512):
        sl=slice(start,start+512);ref=dot(x,w[sl])
        for i in range(3):
            safe=np.where(valid[i],ks[i],0)
            acc=integer_dot(qs[i],spec['wq'][sl],'fp64-exact')
            value=acc*(spec['sx']*np.exp2(safe))[:,None]*sw[sl]
            pair[i]+=np.square(value-ref).sum(axis=1)
            raw=Profile().apply(acc,spec['params']['multiplier'][sl],spec['params']['shift'][sl][None,:]-safe[:,None],saturate=False)
            real=np.clip(raw,-512,511)*spec['s10']
            final[i]+=np.square(real-ref).sum(axis=1)
            ys[i,:,sl]=real.astype(np.float32)
            clip_rows[i]+=((raw < -512)|(raw > 511)).sum(axis=1)
    pair[~valid]=np.inf;final[~valid]=np.inf
    ids=np.arange(len(x))
    chosen={key:ks[score.argmin(axis=0),ids] for key,score in [('activation',activation),('pair',pair),('final',final)]}
    score=pair if objective=='pair' else final;index=score.argmin(axis=0)
    return ys[index,ids],chosen[objective],chosen,int(clip_rows[index,ids].sum())


def apply_mode(x,spec,w,sw,mode):
    x=np.asarray(x,np.float64);comparison=None
    if mode=='arbitrary_absmax':
        scales=np.abs(x).max(axis=1)/127;scales=np.where(scales>0,scales,spec['sx'])
        # Quantize through the existing helper with normalized X and unit scale.
        q=quantize_input(x/scales[:,None],1.)
        y=np.empty((len(x),len(spec['wq'])),np.float32);clips=0
        for start in range(0,len(spec['wq']),512):
            sl=slice(start,start+512);acc=integer_dot(q,spec['wq'][sl],'fp64-exact')
            raw=np.rint(acc*(scales[:,None]*sw[sl]/spec['s10']))
            clips+=int(((raw < -512)|(raw > 511)).sum())
            y[:,sl]=(np.clip(raw,-512,511)*spec['s10']).astype(np.float32)
        ks=None
    else:
        if mode=='bos_only':
            ks=np.zeros(len(x),np.int64);ks[0]=spec['k'];q=quant_rows(x,spec,ks);y,clips=hw_output(q,ks,spec)
        elif mode in ['pair_oracle','int10_oracle']:
            y,ks,comparison,clips=select_oracle(x,spec,w,sw,'pair' if mode=='pair_oracle' else 'final')
            q=quant_rows(x,spec,ks)
        else:
            cap=int(mode[-1]) if mode.startswith('cap') else None
            kk,valid,qq,score=candidates(x,spec['sx'],spec['params']['shift'],cap,spec['k'] if cap is not None else None)
            ix=score.argmin(axis=0);ids=np.arange(len(x));ks=kk[ix,ids]
            q=np.stack(qq)[ix,ids];y,clips=hw_output(q,ks,spec)
        scales=spec['sx']*np.exp2(ks)
    stats=dict(inputs=x.size,outputs=y.size,zeros=int((q==0).sum()),activation_clips=int((np.abs(x)>127*scales[:,None]).sum()),int10_clips=clips)
    if ks is not None:
        stats.update(effective_shift_min=int(spec['params']['shift'].min()-ks.max()),effective_shift_max=int(spec['params']['shift'].max()-ks.min()))
        assert 0<=stats['effective_shift_min']<=stats['effective_shift_max']<=31
    return y,ks,stats,comparison


@contextmanager
def patched(source,specs,weights,scales,mode,trace,stats,confusion):
    originals={}
    try:
        for name,spec in specs.items():
            module=source.modules[name];originals[name]=module.forward
            def forward(x,name=name,spec=spec):
                try:y,k,counts,cmp=apply_mode(x.detach().numpy()[0],spec,weights[name],scales[name],mode)
                except ValueError as exc:raise ValueError(name+': '+str(exc)) from exc
                stats[name]=counts
                if k is not None:trace[int(name.split('.')[2])]=k.astype(np.int8)
                if cmp is not None:
                    for a,b in [('activation','pair'),('activation','final'),('pair','final')]:
                        key=a+'_vs_'+b
                        matrix=confusion.setdefault(key,np.zeros((16,16),np.int64))
                        np.add.at(matrix,(cmp[a]+8,cmp[b]+8),1)
                return source.torch.from_numpy(y[None]).to(x.dtype)
            module.forward=forward
        yield
    finally:
        for name,fn in originals.items():source.modules[name].forward=fn


def token_decomposition(torch,ref,quant,ids):
    arrays={k:[] for k in ['nll_fp','nll_quant','target_logit_fp','target_logit_quant','lse_fp','lse_quant']}
    for start in range(0,len(ids)-1,32):
        end=min(start+32,len(ids)-1);a=ref[start:end].double();b=quant[start:end].double()
        targets=torch.tensor(ids[start+1:end+1]);rr=torch.arange(end-start)
        af=a[rr,targets];bq=b[rr,targets];al=torch.logsumexp(a,-1);bl=torch.logsumexp(b,-1)
        for k,v in zip(arrays,[al-af,bl-bq,af,bq,al,bl]):arrays[k].append(v.numpy())
    return {k:np.concatenate(v) for k,v in arrays.items()}


def aggregate(mode):
    records=[json.loads(p.read_text()) for p in sorted((OUT/mode).glob('sequence_*.json'))]
    if len(records)!=32:return None,[]
    layers=[]
    for layer in range(18):
        g={k:base.merge_metrics([r['metrics']['layers'][layer][k] for r in records]) for k in ['all','bos','non_bos']}
        layers.append(dict(mode=mode,layer=layer,**g['all'],bos_nmse=g['bos']['nmse'],non_bos_nmse=g['non_bos']['nmse']))
    lm=[r['metrics']['logits'] for r in records];tokens=sum(r['tokens'] for r in lm);targets=sum(r['targets'] for r in lm)
    logit=base.merge_metrics([r['metric'] for r in lm]);norm=base.merge_metrics([r['metrics']['normalized_hidden'] for r in records])
    stats=[v for r in records for v in r['stats'].values()]
    result=dict(mode=mode,sequences=32,tokens=tokens,hidden_nmse=layers[-1]['nmse'],norm_hidden_nmse=norm['nmse'],
        logits_nmse=logit['nmse'],logits_cosine=logit['cosine'],kl=sum(r['kl_sum'] for r in lm)/tokens,
        top1=sum(r['top1_count'] for r in lm)/tokens,top5_overlap=sum(r['top5_overlap_sum'] for r in lm)/tokens,
        nll=sum(r['nll_sum'] for r in lm)/targets,fp_nll=sum(r['fp_nll_sum'] for r in lm)/targets,
        first_layer_above_1percent=next((r['layer'] for r in layers if r['nmse']>.01),None),
        first_layer_above_5percent=next((r['layer'] for r in layers if r['nmse']>.05),None),
        worst_layer=max(layers,key=lambda r:r['nmse'])['layer'],
        activation_zero_rate=sum(r['zeros'] for r in stats)/sum(r['inputs'] for r in stats),
        activation_clip_rate=sum(r['activation_clips'] for r in stats)/sum(r['inputs'] for r in stats),
        int10_clip_rate=sum(r['int10_clips'] for r in stats)/sum(r['outputs'] for r in stats))
    result['ppl']=math.exp(result['nll'])
    if mode!='arbitrary_absmax':
        result.update(effective_shift_min=min(r['effective_shift_min'] for r in stats),effective_shift_max=max(r['effective_shift_max'] for r in stats))
        hist=np.zeros(16,np.int64)
        for p in sorted((OUT/mode).glob('sequence_*.npz')):
            with np.load(p) as z:hist+=np.bincount(z['k'].ravel().astype(np.int64)+8,minlength=16)
        result['k_histogram']={str(i-8):int(v) for i,v in enumerate(hist)}
    return result,layers


def failure(mode,sequence,exc):
    base.write_json(OUT/(mode+'_failure.json'),dict(mode=mode,sequence=sequence,error=str(exc),traceback=traceback.format_exc()))
    print('FAILED',mode,sequence,str(exc),flush=True)


def run_modes(source,specs,weights,scales,modes,decompose=False):
    for mode in modes:(OUT/mode).mkdir(exist_ok=True)
    for i,ids in enumerate(source.datasets['validation']):
        pending=[m for m in modes if not (OUT/(m+'_failure.json')).exists() and not (OUT/m/f'sequence_{i:03d}.json').exists()]
        if not pending:continue
        ref,ref_states,ref_norm=base.forward_prefill(source,ids)
        for mode in pending:
            started=time.monotonic();trace={};stats={};confusion={}
            try:
                with patched(source,specs,weights,scales,mode,trace,stats,confusion):logits,states,norm=base.forward_prefill(source,ids)
                assert len(stats)==18
                metrics=row.measure(source.torch,logits,states,norm,ref,ref_states,ref_norm,ids)
                arrays={}
                if trace:arrays['k']=np.stack([trace[l] for l in range(18)])
                arrays.update(confusion)
                if decompose:
                    old=json.loads((ROOT/f'diagnostics/e2e_row_pot/sequence_{i:03d}.json').read_text())
                    row.check_baseline(metrics,old['modes'][mode])
                    arrays.update(token_decomposition(source.torch,ref,logits,ids))
                    arrays['targets']=np.array(ids[1:],np.int32)
                    np.testing.assert_allclose(arrays['nll_quant'].sum(),metrics['logits']['nll_sum'],rtol=1e-12,atol=1e-10)
                np.savez_compressed(OUT/mode/f'sequence_{i:03d}.npz',**arrays)
                base.write_json(OUT/mode/f'sequence_{i:03d}.json',dict(sequence=i,metrics=metrics,stats=stats,seconds=time.monotonic()-started,propagated_boundaries=17))
                print('DONE',mode,i,'NLL',metrics['logits']['nll_sum']/metrics['logits']['targets'],'seconds',time.monotonic()-started,flush=True)
                del logits,states,norm
            except Exception as exc:
                if decompose:raise  # baseline mismatch is a hard blocker
                failure(mode,i,exc)
        del ref,ref_states,ref_norm


def nll_analysis():
    rows=[];summary={};features=[]
    for mode in ['bos_only','general_pot']:
        for i in range(32):
            with np.load(OUT/mode/f'sequence_{i:03d}.npz') as z:
                kk=z['k'][:,:-1]
                for t in range(len(z['targets'])):
                    r=dict(mode=mode,sequence=i,predictor_position=t,target_position=t+1,target_token_id=int(z['targets'][t]))
                    r.update({k:float(z[k][t]) for k in ['nll_fp','nll_quant','target_logit_fp','target_logit_quant','lse_fp','lse_quant']})
                    r.update(delta_nll=r['nll_quant']-r['nll_fp'],delta_target_logit=r['target_logit_quant']-r['target_logit_fp'],delta_lse=r['lse_quant']-r['lse_fp'])
                    k=kk[:,t];r.update(positive_layers=int((k>0).sum()),negative_layers=int((k<0).sum()),sum_k=int(k.sum()),max_k=int(k.max()),mean_k=float(k.mean()),max_positive_k=int(max(0,k.max())))
                    assert math.isclose(r['delta_nll'],-r['delta_target_logit']+r['delta_lse'],abs_tol=1e-10)
                    rows.append(r)
        rr=[r for r in rows if r['mode']==mode];v=np.array([r['delta_nll'] for r in rr]);order=np.sort(v)[::-1]
        summary[mode]=dict(mean=float(v.mean()),median=float(np.median(v)),**{f'p{p}':float(np.percentile(v,p)) for p in [90,95,99]},max=float(v.max()),fraction_positive=float((v>0).mean()),fraction_negative=float((v<0).mean()),total_excess_nll=float(v.sum()),
            mean_delta_target_logit=float(np.mean([r['delta_target_logit'] for r in rr])),mean_delta_lse=float(np.mean([r['delta_lse'] for r in rr])))
        summary[mode]['tails']={str(p):dict(tokens=math.ceil(len(v)*p/100),excess_nll=float(order[:math.ceil(len(v)*p/100)].sum()),fraction_net_excess=float(order[:math.ceil(len(v)*p/100)].sum()/v.sum()),fraction_positive_burden=float(np.maximum(order[:math.ceil(len(v)*p/100)],0).sum()/np.maximum(v,0).sum())) for p in [1,5,10]}
    with gzip.open(ROOT/'diagnostics/row_pot_ppl_token_nll.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    base.write_json(OUT/'nll_decomposition.json',summary)
    rr=[r for r in rows if r['mode']=='general_pot']
    for population in ['all','non_bos']:
        subset=[r for r in rr if population=='all' or r['predictor_position']>0]
        groups={'no_positive':lambda r:r['positive_layers']==0,'positive_1_3':lambda r:1<=r['positive_layers']<=3,'positive_4_8':lambda r:4<=r['positive_layers']<=8,'positive_gt8':lambda r:r['positive_layers']>8,
                'max0':lambda r:r['max_positive_k']==0,'max1':lambda r:r['max_positive_k']==1,'max2':lambda r:r['max_positive_k']==2,'max_ge3':lambda r:r['max_positive_k']>=3}
        for group,pred in groups.items():
            selected=[r for r in subset if pred(r)]
            features.append(dict(population=population,kind='group',feature=group,count=len(selected),mean_delta_nll=float(np.mean([r['delta_nll'] for r in selected])) if selected else None,pearson=None))
        for key in ['positive_layers','negative_layers','sum_k','max_k','mean_k']:
            x=np.array([r[key] for r in subset]);y=np.array([r['delta_nll'] for r in subset])
            features.append(dict(population=population,kind='correlation',feature=key,count=len(subset),mean_delta_nll=None,pearson=float(np.corrcoef(x,y)[0,1]) if np.std(x) else None))
    base.write_csv(ROOT/'diagnostics/row_pot_ppl_k_analysis.csv',features)
    print('A/B',json.dumps(summary),flush=True)


def choose_base(rows):
    if any(r['selection_split']!='calibration' for r in rows):raise ValueError('calibration candidates only')
    valid=[r for r in rows if r.get('feasible')]
    if not valid:raise ValueError('no feasible recenter candidate')
    return min(valid,key=lambda r:(r['nmse'],abs(r['j']),r['j']))


def recenter(source,specs,weights,scales):
    dest=OUT/'recenter_selection.json'
    if dest.exists():return json.loads(dest.read_text())
    positions=np.concatenate([np.arange(len(ids)) for ids in source.datasets['calibration']])
    bos=np.flatnonzero(positions==0);normal=np.flatnonzero(positions!=0)
    sample=np.sort(np.random.default_rng(42).choice(normal,size=min(256,len(normal)),replace=False))
    ids=np.concatenate([bos,sample]);wts=np.r_[np.full(len(bos),1/len(positions)),np.full(len(sample),len(normal)/len(sample)/len(positions))]
    selection={};rows=[];cache_info={}
    for name,spec in specs.items():
        layer=int(name.split('.')[2]);directory=Path('/tmp/bos-aware-capture')/str(layer)/'calibration'
        marker=json.loads((directory/'identity.json').read_text())
        assert marker==dict(model=source.manifest['gguf']['sha256'],data=source.manifest['datasets']['calibration']['file_sha256'],source=base.sha256_file(ROOT/'scripts/quant_diagnostic_source.py'),tokens=len(positions))
        path=directory/(name+'.input.npy');cache=np.load(path,mmap_mode='r');x=np.asarray(cache[ids],np.float64)
        cache_info[name]=dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,selected_data_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest())
        ref=dot(x,weights[name]);energy=float(wts@np.square(ref).mean(axis=1));local=[]
        for j in range(-4,5):
            sx=spec['sx']*2.**(j/4);r=dict(layer=layer,j=j,s_X=sx,s_10=spec['s10'],selection_split='calibration')
            try:
                params=Profile().approximate(sx*scales[name]/spec['s10']);assert np.all(params['status']=='ok')
                candidate=dict(spec,sx=sx,params=params)
                y,_,_,_=apply_mode(x,candidate,weights[name],scales[name],'recentered')
                mse=float(wts@np.square(y.astype(np.float64)-ref).mean(axis=1))
                r.update(feasible=True,mse=mse,nmse=mse/energy)
            except Exception as exc:r.update(feasible=False,error=str(exc))
            local.append(r);rows.append(r)
        selection[name]=choose_base(local)
        print('RECENTER',layer,selection[name],flush=True)
    result=dict(selection_split='calibration',objective='estimated population-pooled final Linear output NMSE',
        sampling='all 128 BOS plus 256 uniform non-BOS rows without replacement, seed 42; population weights',
        row_ids=ids.tolist(),row_weights=wts.tolist(),calibration_data=source.manifest['datasets']['calibration'],cache_info=cache_info,selected=selection,candidates=rows)
    base.write_json(dest,result);base.write_csv(ROOT/'diagnostics/row_pot_ppl_sx_recenter.csv',rows)
    return result


def report():
    modes=[];layers=[];failures={}
    for mode in MODES:
        r,ll=aggregate(mode)
        if r:modes.append(r);layers+=ll
        f=OUT/(mode+'_failure.json')
        if f.exists():failures[mode]=json.loads(f.read_text())
    if not modes:return
    fp=dict(mode='FP',sequences=32,tokens=10141,hidden_nmse=0.,norm_hidden_nmse=0.,logits_nmse=0.,logits_cosine=1.,kl=0.,top1=1.,top5_overlap=1.,nll=modes[0]['fp_nll'],ppl=math.exp(modes[0]['fp_nll']))
    comparisons={}
    for mode in ['pair_oracle','int10_oracle']:
        files=sorted((OUT/mode).glob('sequence_*.npz'))
        if len(files)!=32:continue
        sums={k:np.zeros((16,16),np.int64) for k in ['activation_vs_pair','activation_vs_final','pair_vs_final']};actual=np.zeros((16,16),np.int64)
        for f in files:
            with np.load(f) as z, np.load(OUT/'general_pot'/f.name) as d1:
                for k in sums:sums[k]+=z[k]
                np.add.at(actual,(d1['k'].ravel().astype(int)+8,z['k'].ravel().astype(int)+8),1)
        comparisons[mode]={k:dict(agreement=float(np.trace(v)/v.sum()),confusion=v.tolist()) for k,v in sums.items()}
        comparisons[mode]['actual_trajectory_vs_D1']=dict(agreement=float(np.trace(actual)/actual.sum()),confusion=actual.tolist())
    summary=dict(modes=[fp]+modes,failures=failures,selector_agreement=comparisons,
        nll_decomposition=json.loads((OUT/'nll_decomposition.json').read_text()) if (OUT/'nll_decomposition.json').exists() else None)
    base.write_json(ROOT/'diagnostics/row_pot_ppl_diagnosis_summary.json',summary)
    base.write_csv(ROOT/'diagnostics/row_pot_ppl_modes.csv',[{k:v for k,v in r.items() if k!='k_histogram'} for r in [fp]+modes])
    base.write_csv(ROOT/'diagnostics/row_pot_ppl_layers.csv',layers)
    lines=['# Row PoT likelihood diagnosis','Full held-out validation: 32 sequences, 10,141 positions, 10,109 next-token targets. Only 18 down_proj operations are quantized; all other operations remain FP. Static production/calibration artifacts are unchanged.',
        '| mode | hidden NMSE % | normalized hidden NMSE % | logits NMSE % | KL | top1 % | PPL |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in [fp]+modes:lines.append(f'| {r["mode"]} | {100*r["hidden_nmse"]:.6f} | {100*r["norm_hidden_nmse"]:.6f} | {100*r["logits_nmse"]:.6f} | {r["kl"]:.7f} | {100*r["top1"]:.4f} | {r["ppl"]:.6f} |')
    for mode,f in failures.items():lines.append(f'{mode}: not a complete E2E result. Sequence {f["sequence"]}: {f["error"]}')
    lines+=['D2/D3 compare current propagated X against X @ recovered FP W.T, never cached reference-model inputs or labels. Both oracles are diagnostic and are not implementable runtime selector proposals. Same-input k confusion is measured on each oracle trajectory; actual-trajectory agreement versus D1 is reported separately.',
        'E uses cached calibration activations only: all 128 BOS rows and a deterministic 256-row uniform non-BOS sample, with population weights to estimate pooled output NMSE. Selection is frozen before its validation forward. This is a bounded calibration estimate, not full-token exhaustive calibration. Existing s10 is fixed. Validation does not select parameters.',
        'F is an arbitrary absmax-row reference with exact FP64 ratio and RNE, bypassing M/S approximation. Absmax is not an optimized scalar threshold and does not guarantee the minimum possible output error; compare it as a diagnostic reference, not a proven upper bound.',
        'Token NLL uses predictor position p and target token p+1. k features come from predictor position p, not the future target row. Tail shares use signed net excess NLL; positive-burden shares are also provided and prevent interpreting >100% net contributions incorrectly. Correlation does not establish causality.',
        'Per-mode localization, non-BOS hidden NMSE, clipping and zero rates, histograms, and k confusion are in the CSV/JSON outputs. No large reference logits or activation caches are written; FP forward is shared within a stage when new logits comparisons require it. C0/C4/D1 reuse already measured modes.']
    if (OUT/'answers.md').exists():lines.append((OUT/'answers.md').read_text())
    (ROOT/'diagnostics/row_pot_ppl_diagnosis_report.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['all','ab','c','d','e','f','report'],default='all');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=='report':report();return
    old=json.loads((ROOT/'diagnostics/e2e_row_pot/provenance.json').read_text())
    assert base.sha256_file(ROOT/'scripts/test_e2e_row_pot.py')==old['script_sha256']
    assert base.sha256_file(ROOT/'scripts/test_e2e_linear_quant.py')==old['base_script_sha256']
    for p,h in {**old['artifact_hashes'],**old['source_hashes']}.items():assert base.sha256_file(ROOT/p)==h
    prior=json.loads((ROOT/'diagnostics/e2e_row_pot_summary.json').read_text())
    assert abs(prior['fp_ppl']-186.6371)<.0001 and abs(prior['modes']['bos_only']['ppl']-220.7119)<.0001 and abs(prior['modes']['general_pot']['ppl']-351.7771)<.0001
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1';manifest=json.loads((art/'manifest.json').read_text())
    frozen=json.loads((ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json').read_text())
    source=base.RecordedGGUFSource('/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf',ROOT/'calibration_outputs/data-oasst1-v1',manifest)
    base.configure_blas('accelerate')
    filtered=dict(manifest,modules=[e for e in manifest['modules'] if e['module_name'].endswith('.down_proj')])
    specs,missing,hashes=base.load_artifacts(source,art,filtered,frozen);assert not missing and len(specs)==18
    weights={n:source.weight(n) for n in specs};scales={}
    for e in filtered['modules']:
        with np.load(art/Path(e['module_manifest']).parent/'scales.npz') as z:scales[e['module_name']]=z['op0000.s_W'].copy()
    identity=dict(script_sha256=base.sha256_file(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        previous_provenance_sha256=base.sha256_file(ROOT/'diagnostics/e2e_row_pot/provenance.json'),
        source_hashes={**old['source_hashes'],'scripts/test_e2e_row_pot.py':old['script_sha256'],'scripts/test_e2e_linear_quant.py':old['base_script_sha256']},
        artifact_hashes=hashes,model_sha256=old['model_sha256'],frozen_selection_sha256=base.sha256_file(ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json'),validation=manifest['datasets']['validation'],profile=Profile().metadata())
    if (OUT/'provenance.json').exists():assert json.loads((OUT/'provenance.json').read_text())==identity
    else:base.write_json(OUT/'provenance.json',identity)
    with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,time=time.time()))+'\n')
    stages=['ab','c','d','e','f'] if args.stage=='all' else [args.stage]
    for stage in stages:
        print('STAGE',stage,flush=True)
        if stage=='ab':
            run_modes(source,specs,weights,scales,['bos_only','general_pot'],decompose=True)
            nll_analysis()
            actual={m:aggregate(m)[0]['ppl'] for m in ['bos_only','general_pot']}
            for m,ppl in actual.items():np.testing.assert_allclose(ppl,prior['modes'][m]['ppl'],rtol=1e-12)
            base.write_json(OUT/'baseline_reproduction.json',dict(ppl=actual,fp_ppl=math.exp(aggregate('bos_only')[0]['fp_nll']),full_statistics_match=True))
        elif stage=='c':run_modes(source,specs,weights,scales,['cap0','cap1','cap2'])
        elif stage=='d':run_modes(source,specs,weights,scales,['pair_oracle','int10_oracle'])
        elif stage=='e':
            try:
                selected=recenter(source,specs,weights,scales);updated={}
                for n,s in specs.items():
                    sx=selected['selected'][n]['s_X'];updated[n]=dict(s,sx=sx,params=Profile().approximate(sx*scales[n]/s['s10']))
                run_modes(source,updated,weights,scales,['recentered'])
            except Exception as exc:failure('recentered',None,exc)
        elif stage=='f':run_modes(source,specs,weights,scales,['arbitrary_absmax'])
        report()


if __name__=='__main__':main()
