#!/usr/bin/env python3
"""Activation-only dynamic row PoT diagnostic; frozen down_proj channel factors."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import test_e2e_linear_quant as base
from static_quant.core import quantize_input
from static_quant.hardware import Profile, finite, positive_scale
from quant_diagnostic_math import integer_dot


def select_rows(x,sx,shift):
    """No weights, Linear outputs or reference inputs: only X and shift bounds.

    RNE for k0; ties prefer k0, then k0-1, then k0+1. Zero rows use k0=0.
    An empty feasible candidate set is an explicit failure, never a fallback.
    """
    x=finite(np.asarray(x,np.float64),'row activation');positive_scale(sx,'base scale')
    if x.ndim!=2:raise ValueError('expected rows by channels')
    maximum=np.abs(x).max(axis=1)
    ratio=np.where(maximum>0,maximum/(127*sx),1.)
    k0=np.rint(np.log2(ratio)).astype(np.int64)
    shifts=Profile().effective(shift)
    lo,hi=max(-8,int(shifts.max())-31),min(7,int(shifts.min()))
    best=np.full(len(x),np.inf);ks=np.zeros(len(x),np.int64);q=np.zeros(x.shape,np.int8)
    best_unrestricted=np.full(len(x),np.inf);ku=np.zeros(len(x),np.int64)
    rejected=np.zeros(len(x),np.int64);bounded=np.zeros(len(x),np.int64)
    bound_rejected=np.zeros(len(x),np.int64)
    for offset in [0,-1,1]:
        candidate=k0+offset
        bound_rejected+=(candidate < -8)|(candidate > 7)
        for k in np.unique(candidate):
            if not -8<=k<=7:continue
            rows=np.flatnonzero(candidate==k);bounded[rows]+=1
            scale=sx*2.**int(k);qq=quantize_input(x[rows],scale)
            error=np.square(qq.astype(np.float64)*scale-x[rows]).mean(axis=1)
            improve=error<best_unrestricted[rows]
            best_unrestricted[rows[improve]]=error[improve];ku[rows[improve]]=k
            if not lo<=k<=hi:
                rejected[rows]+=1;continue
            improve=error<best[rows];take=rows[improve]
            best[take]=error[improve];ks[take]=k;q[take]=qq[improve]
    failed=np.flatnonzero(~np.isfinite(best))
    if len(failed):
        raise ValueError(f'No feasible k0±1 candidate: rows={failed.tolist()}, k0={k0[failed].tolist()}, feasible k=[{lo},{hi}]')
    counts={}
    for group,sl in [('bos',slice(0,1)),('non_bos',slice(1,None))]:
        kg=ks[sl];xx=x[sl];qq=q[sl];scale=sx*np.exp2(kg)
        counts.update({f'{group}_k_{k}':int((kg==k).sum()) for k in range(-8,8)})
        counts.update({group+'_rows':len(kg),group+'_inputs':xx.size,
            group+'_activation_clips':int((np.abs(xx)>127*scale[:,None]).sum()),
            group+'_activation_zeros':int((qq==0).sum()),
            group+'_candidate_count':int(bounded[sl].sum()),
            group+'_shift_rejected_candidates':int(rejected[sl].sum()),
            group+'_shift_restricted_rows':int((rejected[sl]>0).sum()),
            group+'_shift_changed_choice_rows':int((kg!=ku[sl]).sum()),
            group+'_k_bound_rejected_candidates':int(bound_rejected[sl].sum())})
    counts.update(effective_shift_min=int(shifts.min()-ks.max()),effective_shift_max=int(shifts.max()-ks.min()))
    return q,ks,counts


def apply_rows(x,wq,sx,s10,params,k=None,chunk=512):
    """Reuse existing exact integer GEMM and hardware Profile; never approximate M/S."""
    x=np.asarray(x,np.float64)
    if x.ndim!=3 or x.shape[0]!=1:raise ValueError('unpadded batch-one prefill only')
    q,ks,counts=select_rows(x[0],sx,params['shift'])
    out=np.empty((len(q),len(wq)),np.float32)
    groups=[(kk,np.flatnonzero(ks==kk)) for kk in np.unique(ks)]
    counts.update(bos_int10_clips=0,non_bos_int10_clips=0,
                  bos_outputs=len(wq),non_bos_outputs=(len(q)-1)*len(wq))
    for start in range(0,len(wq),chunk):
        end=min(start+chunk,len(wq));acc=integer_dot(q,wq[start:end],'fp64-exact')
        raw=np.empty(acc.shape,np.int64)
        for kk,rows in groups:
            raw[rows]=Profile().apply(acc[rows],params['multiplier'][start:end],params['shift'][start:end]-kk,saturate=False)
        counts['bos_int10_clips']+=int(((raw[:1]<-512)|(raw[:1]>511)).sum())
        counts['non_bos_int10_clips']+=int(((raw[1:]<-512)|(raw[1:]>511)).sum())
        out[:,start:end]=(np.clip(raw,-512,511)*s10).astype(np.float32)
    return out[None],counts


def measure(torch,logits,states,norm,ref,ref_states,ref_norm,ids):
    layers=[]
    for layer in range(18):
        groups={}
        for g,sl in [('all',slice(None)),('bos',slice(0,1)),('non_bos',slice(1,None))]:
            m=base.Metric();m.add(states[layer][0,sl].numpy(),ref_states[layer][0,sl].numpy());groups[g]=vars(m)
        layers.append(groups)
    m=base.Metric();m.add(norm.numpy(),ref_norm.numpy())
    return dict(layers=layers,normalized_hidden=vars(m),logits=base.logit_metrics(torch,logits,ref,ids))


def check_baseline(actual,stored):
    """Compare all hidden/logit sufficient statistics, not rounded headline values."""
    def walk(a,b):
        if isinstance(a,dict):
            for key in a:walk(a[key],b[key])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for aa,bb in zip(a,b):walk(aa,bb)
        else:np.testing.assert_allclose(a,b,rtol=2e-12,atol=1e-12)
    for key in ['layers','normalized_hidden','logits']:walk(actual[key],stored[key])


def histogram_summary(hist):
    n=sum(hist.values());used=[k for k,v in hist.items() if v]
    if not n:raise ValueError('empty row histogram')
    def at(index):
        cumulative=0
        for k in sorted(hist):
            cumulative+=hist[k]
            if cumulative>index:return k
    return dict(rows=n,min_k=min(used),max_k=max(used),median_k=(at((n-1)//2)+at(n//2))/2,
                distinct_k=len(used),zero_fraction=hist.get(0,0)/n,
                nonzero_fraction=1-hist.get(0,0)/n,at_bounds_fraction=(hist.get(-8,0)+hist.get(7,0))/n,
                histogram={str(k):v for k,v in sorted(hist.items())})


def materialize(out):
    records=[json.loads(p.read_text()) for p in sorted(out.glob('sequence_*.json'))]
    if not records:return
    modes={};layer_rows=[];hist_rows=[];stats_rows=[]
    for tag in ['bos_only','general_pot']:
        rr=[r['modes'][tag] for r in records];ll=[r['logits'] for r in rr]
        tokens=sum(r['tokens'] for r in ll);targets=sum(r['targets'] for r in ll)
        for layer in range(18):
            groups={g:base.merge_metrics([r['layers'][layer][g] for r in rr]) for g in ['all','bos','non_bos']}
            layer_rows.append(dict(mode=tag,layer=layer,**groups['all'],bos_nmse=groups['bos']['nmse'],non_bos_nmse=groups['non_bos']['nmse']))
        modes[tag]=dict(final_hidden=layer_rows[-1],final_normalized_hidden=base.merge_metrics([r['normalized_hidden'] for r in rr]),
            logits=base.merge_metrics([r['metric'] for r in ll]),kl=sum(r['kl_sum'] for r in ll)/tokens,
            top1_agreement=sum(r['top1_count'] for r in ll)/tokens,top5_overlap=sum(r['top5_overlap_sum'] for r in ll)/tokens,
            nll=sum(r['nll_sum'] for r in ll)/targets,fp_nll=sum(r['fp_nll_sum'] for r in ll)/targets)
        modes[tag]['ppl']=math.exp(modes[tag]['nll'])
    operations=[r['modes']['general_pot']['operations'] for r in records]
    distributions={}
    for layer in list(range(18))+['all']:
        names=[f'model.layers.{l}.mlp.down_proj' for l in (range(18) if layer=='all' else [layer])]
        for group in ['bos','non_bos','all']:
            gs=['bos','non_bos'] if group=='all' else [group]
            def total(key):return sum(op[n][g+'_'+key] for op in operations for n in names for g in gs)
            hist={k:total(f'k_{k}') for k in range(-8,8)};d=histogram_summary(hist)
            d.update(activation_clip_rate=total('activation_clips')/total('inputs'),
                activation_zero_rate=total('activation_zeros')/total('inputs'),
                int10_clip_rate=total('int10_clips')/total('outputs'),
                shift_rejected_candidate_fraction=total('shift_rejected_candidates')/total('candidate_count'),
                shift_restricted_row_fraction=total('shift_restricted_rows')/total('rows'),
                shift_changed_choice_fraction=total('shift_changed_choice_rows')/total('rows'),
                k_bound_rejected_candidate_fraction=total('k_bound_rejected_candidates')/(3*total('rows')))
            stats_rows.append(dict(layer=layer,group=group,**{k:v for k,v in d.items() if k!='histogram'}))
            for k,count in hist.items():hist_rows.append(dict(layer=layer,group=group,k=k,count=count,fraction=count/d['rows']))
            if layer=='all':distributions[group]=d
    fp=math.exp(modes['bos_only']['fp_nll']);bos=modes['bos_only']['ppl'];pot=modes['general_pot']['ppl']
    summary=dict(sequences=len(records),tokens=tokens,next_token_targets=targets,fp_ppl=fp,modes=modes,
        ppl_degradation_recovery=(bos-pot)/(bos-fp),k_distribution=distributions,
        effective_shift_min=min(s['effective_shift_min'] for op in operations for s in op.values()),
        effective_shift_max=max(s['effective_shift_max'] for op in operations for s in op.values()),
        baseline_reproduced=all(r['baseline_reproduced'] for r in records),arbitrary_oracle='not measured',
        policy='frozen w_BOS=0.25 base scales and channel factors; activation-only row k selection',
        source_provenance='e2e_row_pot/provenance.json')
    base.write_json(out.parent/'e2e_row_pot_summary.json',summary)
    base.write_csv(out.parent/'e2e_row_pot_layers.csv',layer_rows)
    base.write_csv(out.parent/'e2e_row_pot_k_histogram.csv',hist_rows)
    base.write_csv(out/'row_statistics.csv',stats_rows)
    report=['# General-row PoT: down_proj-only E2E',
        f'{len(records)} held-out sequences / {tokens} valid tokens / {targets} next-token targets. FP and BOS-only are rerun on the same inputs; all stored per-sequence baseline statistics reproduce (rtol 2e-12, atol 1e-12).',
        'Only 18 down_proj operations are quantized. All other Linear and nonlinear operations, embedding and lm_head remain FP. The existing model driver and boundary-pointer propagation checks are reused. No activation reinjection, capture, calibration, parameter search, downloads or production modifications.',
        'Base sX, weight INT8, common s10 and per-channel M/S are frozen to global w_BOS=0.25. Each actual propagated input row selects only among raw k0±1 (k0=RNE(log2(absmax/(127*sX)))) in [-8,7], rejecting any S-k outside [0,31]. Selection uses activation reconstruction MSE only. Ties prefer k0, then k0-1, then k0+1; zero rows use k0=0. No feasible candidate causes an explicit error. One base channel multiplier is shared across rows.',
        '| mode | final block hidden NMSE % | logits NMSE % | cosine | KL nats | top1 % | top5 overlap % | PPL |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
        f'| FP | 0 | 0 | 1 | 0 | 100 | 100 | {fp:.6f} |']
    for tag,s in modes.items():report.append(f'| {tag} | {100*s["final_hidden"]["nmse"]:.6f} | {100*s["logits"]["nmse"]:.6f} | {s["logits"]["cosine"]:.8f} | {s["kl"]:.8f} | {100*s["top1_agreement"]:.4f} | {100*s["top5_overlap"]:.4f} | {s["ppl"]:.6f} |')
    report += [f'PPL degradation recovery: {100*summary["ppl_degradation_recovery"]:.6f}%. Hidden is block 17 before final RMSNorm. Cosine is mean token cosine; top5 is intersection size / 5. PPL uses shifted next-token labels, excluding the last position per sequence. Reference is FP32 recovered GGUF, not original BF16.',
        '| layer | mode | hidden NMSE % | cosine | MAE | BOS NMSE % | non-BOS NMSE % |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in layer_rows:report.append(f'| {r["layer"]} | {r["mode"]} | {100*r["nmse"]:.6f} | {r["cosine"]:.8f} | {r["mae"]:.8f} | {100*r["bos_nmse"]:.6f} | {100*r["non_bos_nmse"]:.6f} |')
    report += ['## Row distribution (row-operation observations)',json.dumps(distributions,indent=2),
        'Per-layer distinct-k counts and row/activation rates are in e2e_row_pot/row_statistics.csv. Histogram CSV has per-layer and pooled BOS/non-BOS/all groups. Rates are fractions. Shift-rejected candidate rate divides by candidate trials inside [-8,7]; restricted-row rate counts rows with any candidate rejected by shift feasibility; changed-choice rate compares to the same three-candidate activation-MSE winner without shift filtering. At-bounds means k=-8 or +7. No runtime scale is selected using Linear/reference outputs.',
        f'Observed effective shifts: {summary["effective_shift_min"]}..{summary["effective_shift_max"]}; violations: 0. Arbitrary-row oracle not run: it needs distinct scalar requantization, outside this focused shift-only implementation. No claim of closeness to arbitrary scaling is made.']
    if (out/'answers.md').exists():report.append((out/'answers.md').read_text())
    (out.parent/'e2e_row_pot_report.md').write_text('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--sequences',type=int,default=32)
    ap.add_argument('--model',default='/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf');args=ap.parse_args()
    out=ROOT/'diagnostics/e2e_row_pot';out.mkdir(exist_ok=True)
    previous=ROOT/'diagnostics/e2e_linear_quant';past=json.loads((previous/'provenance.json').read_text())
    assert base.sha256_file(ROOT/'scripts/test_e2e_linear_quant.py')==past['script_sha256']
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1';manifest=json.loads((art/'manifest.json').read_text())
    frozen_path=ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json';frozen=json.loads(frozen_path.read_text())
    assert base.sha256_file(frozen_path)==past['frozen_selection_sha256']
    source=base.RecordedGGUFSource(args.model,ROOT/'calibration_outputs/data-oasst1-v1',manifest)
    base.configure_blas('accelerate')
    filtered=dict(manifest,modules=[e for e in manifest['modules'] if e['module_name'].endswith('.down_proj')])
    specs,missing,hashes=base.load_artifacts(source,art,filtered,frozen);assert len(specs)==18 and not missing
    assert all(dict(s_X=s['sx'],s_10=s['s10'],k=s['k'])==past['policies'][n] for n,s in specs.items())
    provenance=dict(script_sha256=base.sha256_file(__file__),base_script_sha256=past['script_sha256'],
        previous_provenance_sha256=base.sha256_file(previous/'provenance.json'),frozen_selection_sha256=base.sha256_file(frozen_path),
        model_sha256=manifest['gguf']['sha256'],validation=manifest['datasets']['validation'],artifact_hashes=hashes,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        profile=Profile().metadata(),enabled=list(specs),base_policies={n:past['policies'][n] for n in specs},
        source_hashes=past['source_hashes'],selection='runtime activation-only three-candidate MSE; no calibration changes')
    for p,h in provenance['source_hashes'].items():assert base.sha256_file(ROOT/p)==h
    if (out/'provenance.json').exists():assert json.loads((out/'provenance.json').read_text())==provenance
    else:base.write_json(out/'provenance.json',provenance)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,time=time.time()))+'\n')
    ids_list=source.datasets['validation'];assert 1<=args.sequences<=len(ids_list)
    print('READY 18 frozen down_proj policies',flush=True)
    for i,ids in enumerate(ids_list[:args.sequences]):
        path=out/f'sequence_{i:03d}.json'
        if path.exists():continue
        started=time.monotonic();ref,ref_states,ref_norm=base.forward_prefill(source,ids)
        old=json.loads((previous/f'sequence_{i:03d}.json').read_text());assert len(ids)==old['tokens']
        record=dict(sequence=i,tokens=len(ids),modes={})
        for tag in ['bos_only','general_pot']:
            counters={}
            with patch.object(base,'quantized_linear',base.quantized_linear if tag=='bos_only' else apply_rows):
                with base.patch_linears(source,specs,'down_only',counters):
                    try:logits,states,norm=base.forward_prefill(source,ids)
                    except ValueError as exc:
                        base.write_json(out/'blocker.json',dict(sequence=i,mode=tag,error=str(exc),completed_operations=list(counters)))
                        raise
            assert len(counters)==18 and all(v['calls']==1 for v in counters.values())
            metrics=measure(source.torch,logits,states,norm,ref,ref_states,ref_norm,ids)
            metrics['operations']=counters;metrics['propagated_boundaries']=17
            if tag=='bos_only':check_baseline(metrics,old['modes']['down_only']);record['baseline_reproduced']=True
            record['modes'][tag]=metrics
            print('MODE',i,tag,'hidden',base.merge_metrics([metrics['layers'][-1]['all']])['nmse'],'seconds',time.monotonic()-started,flush=True)
            del logits,states,norm
        record['seconds']=time.monotonic()-started
        record['previous_sequence_sha256']=base.sha256_file(previous/f'sequence_{i:03d}.json')
        base.write_json(path,record);materialize(out)
        del ref,ref_states,ref_norm
        print('DONE',i,len(ids),record['seconds'],flush=True)
    materialize(out)


if __name__=='__main__':main()
