#!/usr/bin/env python3
"""Fixed 384-target fresh cached-decode selector confirmation, without hidden capture."""
import json
import subprocess
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import confirm_k_selector_fidelity as prior
final=prior.final
base=prior.base
response=prior.response
select_absmax=prior.compare.select_absmax
SUBSET=[0,3,6,9,12,15,18,21,2,8,14,20]
OLD_SUBSET=[0,3,6,9,12,15,18,21]
TARGETS=32
MODES=prior.MODES
OUT=ROOT/'diagnostics/k_selector_fidelity_384'
SEED=20261001


def decision(ms,hw):
    checks=dict(kl=hw['kl']<=1.10*ms['kl']+1e-15,
        nmse=hw['nmse']<=1.10*ms['nmse']+1e-15,
        top1=hw['top1']>=ms['top1']-.01-1e-15)
    return dict(**checks,overall='HW-ready' if all(checks.values()) else 'not-yet')


def classify(ms,hw):
    d=decision(ms,hw)
    if d['overall']=='HW-ready':return dict(case='A',label='HW-ready')
    if (hw['kl']<=1.15*ms['kl']+1e-15 and d['nmse']
            and hw['top1']>=ms['top1']-1e-15):
        return dict(case='B',label='HW-favorable, KL trade-off remains')
    if hw['kl']>1.15*ms['kl']+1e-15 or not d['nmse'] or not d['top1']:
        return dict(case='C',label='not-yet; test absmax-aware calibration next')
    # Matrix leaves a gap: KL +10..15% with Top1 degradation of 0..1 pp.
    return dict(case='uncovered',label='not-yet; predefined matrix does not cover this combination')


def bootstrap(rows,seed=SEED,resamples=10000):
    values=np.array([[r['delta_kl'],r['delta_nmse'],r['delta_top1']] for r in rows])
    indices=np.random.default_rng(seed).integers(0,len(rows),size=(resamples,len(rows)))
    means=values[indices].mean(axis=1)
    return dict(seed=seed,resamples=resamples,unit='paired conversation',
        note='Descriptive equal-conversation mean differences. NMSE CI uses per-conversation NMSE, not pooled-energy NMSE. Top1 units are fractions.',
        intervals={key:dict(mean=float(values[:,i].mean()),low=float(np.percentile(means[:,i],2.5)),
                            high=float(np.percentile(means[:,i],97.5)))
                   for i,key in enumerate(['kl','nmse','top1'])})


def run_conversation(source,specs,e,index):
    scores={m:response.Scores() for m in MODES}; direct=response.Scores()
    caches={}; logits={}
    assert len(e['targets'])>=TARGETS
    for mode in MODES:
        logits[mode],_,caches[mode]=prior.execute(source,specs,mode,e['prompt_ids'],None,False)
    prior.assert_independent([caches[m] for m in MODES])
    for t,target in enumerate(e['targets'][:TARGETS]):
        if t:
            for mode in MODES:
                logits[mode],_,caches[mode]=prior.execute(source,specs,mode,[e['targets'][t-1]],caches[mode],False)
        prior.assert_independent([caches[m] for m in MODES])
        assert all(caches[m].get_seq_length()==len(e['prompt_ids'])+t for m in MODES)
        for mode in MODES:
            assert bool(source.torch.isfinite(logits[mode]).all())
            scores[mode].add(source.torch,logits[mode],logits['FP'],[target])
        direct.add(source.torch,logits['HW absmax'],logits['Final MSE3'],[target])
    errors={}
    if index in OLD_SUBSET:
        old=json.loads((prior.OUT/f'conversation_{index:03d}.json').read_text())
        assert old['source_index']==e['index'] and old['targets']==TARGETS
        for m in MODES:
            errors[m]=scores[m].nll/TARGETS-old['scores'][m]['nll']/TARGETS
            assert abs(errors[m])<1e-6,(index,m,errors[m])
    return dict(index=index,source_index=e['index'],targets=TARGETS,
        scores={m:s.raw() for m,s in scores.items()},direct=direct.raw(),fresh_minus_previous_nll=errors)


def aggregate(records,provenance):
    assert [r['index'] for r in records]==SUBSET
    assert sum(r['targets'] for r in records)==384
    modes=[dict(mode=m,**response.merged([r['scores'][m] for r in records])) for m in MODES]
    for k,v in [('kl',0),('nmse',0),('cosine',1),('top1',1)]:assert abs(modes[0][k]-v)<1e-12
    rows=[]
    for r in records:
        fp,ms,hw=[response.merged([r['scores'][m]]) for m in MODES]
        dn=hw['nll']-ms['nll']; dk=hw['kl']-ms['kl']
        rows.append(dict(index=r['index'],source_index=r['source_index'],targets=32,
            fp_nll=fp['nll'],mse3_nll=ms['nll'],absmax_nll=hw['nll'],
            mse3_kl=ms['kl'],absmax_kl=hw['kl'],mse3_nmse=ms['nmse'],absmax_nmse=hw['nmse'],
            mse3_top1=ms['top1'],absmax_top1=hw['top1'],delta_kl=dk,
            delta_nmse=hw['nmse']-ms['nmse'],delta_top1=hw['top1']-ms['top1'],
            nll_outcome='equal' if abs(dn)<1e-3 else ('better' if dn<0 else 'worse'),
            kl_outcome='approximately_equal' if abs(dk)<1e-4 else ('absmax_lower' if dk<0 else 'absmax_higher')))
    boot=bootstrap(rows)
    direct=response.merged([r['direct'] for r in records])
    old_modes=[response.merged([r['scores'][m] for r in records[:8]]) for m in MODES]
    old=json.loads((prior.OUT/'summary.json').read_text())
    assert old['subset']==OLD_SUBSET
    for fresh,stored in zip(old_modes,old['modes']):assert abs(fresh['nll']-stored['nll'])<1e-6
    summary=dict(provenance=provenance,subset=SUBSET,targets=384,modes=modes,
        deltas=prior.deltas(modes[1],modes[2]),decision=decision(modes[1],modes[2]),
        classification=classify(modes[1],modes[2]),bootstrap=boot,
        direct_absmax_vs_mse3={k:direct[k] for k in ['nmse','cosine','top1']},
        nll_outcomes={k:sum(r['nll_outcome']==k for r in rows) for k in ['better','worse','equal']},
        kl_outcomes={k:sum(r['kl_outcome']==k for r in rows) for k in ['absmax_lower','absmax_higher','approximately_equal']},
        max_previous_nll_error=max(abs(v) for r in records for v in r['fresh_minus_previous_nll'].values()),
        old_subset_fresh_deltas=prior.deltas(old_modes[1],old_modes[2]))
    prior.finite_tree(summary);prior.finite_tree(rows)
    base.write_json(OUT/'summary.json',summary);base.write_json(OUT/'bootstrap.json',boot)
    base.write_csv(OUT/'per_conversation.csv',rows)
    lines=['# Fixed 384-target selector confirmation','',
        '| Mode | NLL | PPL | KL vs FP | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in modes:lines.append('| '+r['mode']+' | '+' | '.join(f'{r[k]:.9g}' for k in ['nll','ppl','kl','nmse','cosine','top1','in5','overlap'])+' |')
    lines+=['','| MSE3 → absmax | Delta |','|---|---:|']
    for k,v in summary['deltas'].items():lines.append(f"| {k} {'pp' if k=='top1_pp' else '%'} | {v*(1 if k=='top1_pp' else 100):.6f} |")
    lines+=['','| Predefined criterion | Result |','|---|---|']
    for label,k in [('KL degradation <=10%','kl'),('NMSE degradation <=10%','nmse'),('Top1 degradation <=1pp','top1')]:
        lines.append(f"| {label} | {'PASS' if summary['decision'][k] else 'FAIL'} |")
    lines+=[f"| Overall | {summary['decision']['overall']} |",'',
        f"Classification: {summary['classification']}. No follow-up calibration was run.",
        f"NLL outcomes: {summary['nll_outcomes']}. KL outcomes: {summary['kl_outcomes']}.",
        f"Direct absmax vs MSE3: {summary['direct_absmax_vs_mse3']}.",'',
        '| Paired bootstrap difference | Mean | 95% low | 95% high |','|---|---:|---:|---:|']
    for k,v in boot['intervals'].items():lines.append(f"| {k} | {v['mean']:.9g} | {v['low']:.9g} | {v['high']:.9g} |")
    lines+=['',boot['note'],f'Seed {SEED}; 10,000 paired resamples. CI does not change the decision rule.',
        f"The first 8 chats reproduce prior FP/MSE3/absmax NLL; maximum per-chat difference {summary['max_previous_nll_error']:.12g}. Their fresh KL relative delta is {100*summary['old_subset_fresh_deltas']['kl_relative']:.6f}%.",
        'All 3 paths freshly executed with independent caches, layers=False throughout; 12 chats × 32 = 384 response targets.',
        'Frozen fingerprints unchanged. Previous 256-target outputs, including their original not-yet verdict, were not modified.',
        'No parameter selection, histogram, hidden capture, long validation or generation.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'summary.json').exists(),'384-target results already exist'
    source,_,_,previous,_,_=final.load_source_and_policies()
    specs,_=final.restore_final(source,previous)
    assert set(specs)=={f'model.layers.{i}.mlp.down_proj' for i in range(18)}
    path=final.OUT/'decisions.json'; before=prior.frozen_fingerprint(specs,path.read_bytes())
    examples,_=final.validation_examples(source);assert len(examples)==24
    assert SUBSET[:8]==OLD_SUBSET and len(SUBSET)*TARGETS==384
    provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        before=before,script_sha256=base.sha256_file(__file__),command=sys.argv,
        prior_artifact_hashes={str(p.relative_to(ROOT)):base.sha256_file(p)
            for directory in [prior.OUT,prior.compare.OUT] for p in directory.rglob('*') if p.is_file()})
    records=[]
    for n,index in enumerate(SUBSET):
        r=run_conversation(source,specs,examples[index],index)
        assert prior.frozen_fingerprint(specs,path.read_bytes())==before
        prior.finite_tree(r);records.append(r);base.write_json(OUT/f'conversation_{index:03d}.json',r)
        print('CONFIRMED',n+1,12,index,flush=True)
    provenance['after']=prior.frozen_fingerprint(specs,path.read_bytes());assert provenance['after']==before
    for p,h in provenance['prior_artifact_hashes'].items():assert base.sha256_file(ROOT/p)==h
    aggregate(records,provenance)
    base.write_json(OUT/'verification.json',dict(subset=SUBSET,targets=384,
        fresh_modes=list(MODES),independent_caches=True,hidden_capture=False,histogram_capture=False,
        quantized_modules=sorted(specs),frozen_fingerprint_unchanged=True,
        previous_artifacts_unchanged=True,previous_nll_reproduced_all_three_modes=True,
        selector_function_identity=select_absmax is prior.compare.select_absmax,
        before=before,after=provenance['after'],all_metrics_finite=True))

if __name__=='__main__':main()
