#!/usr/bin/env python3
"""Fixed 8-chat cached-decode confirmation; no calibration or parameter search."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import compare_k_selectors_e2e as compare
final = compare.final
row = compare.row
response = final.response
base = final.base
select_absmax = compare.select_absmax
SUBSET = [0, 3, 6, 9, 12, 15, 18, 21]
TARGETS = 32
HIDDEN_POSITIONS = {0, 7, 31}
MODES = ('FP', 'Final MSE3', 'HW absmax')
OUT = ROOT/'diagnostics/k_selector_fidelity'


def assert_independent(caches):
    assert len(caches)==3 and all(c is not None for c in caches)
    assert len({id(c) for c in caches})==3


def frozen_fingerprint(specs, decisions):
    return dict(decisions_sha256=hashlib.sha256(decisions).hexdigest(),
                parameters_sha256=compare.fingerprint(specs))


def deltas(mse3, hw):
    return dict(kl_relative=(hw['kl']-mse3['kl'])/mse3['kl'],
        nmse_relative=(hw['nmse']-mse3['nmse'])/mse3['nmse'],
        ppl_relative=(hw['ppl']-mse3['ppl'])/mse3['ppl'],
        top1_pp=100*(hw['top1']-mse3['top1']))


def decision(mse3, hw):
    # Tiny arithmetic tolerance only handles binary representation of exact boundaries.
    checks=dict(kl=hw['kl'] <= 1.10*mse3['kl']+1e-15,
        nmse=hw['nmse'] <= 1.10*mse3['nmse']+1e-15,
        top1=abs(hw['top1']-mse3['top1']) <= .01+1e-15)
    return dict(**checks,overall='HW-ready' if all(checks.values()) else 'not-yet')


def execute(source, specs, mode, ids, cache, layers):
    if mode=='FP': return final.cached.run_fp(source,ids,cache,layers=layers)
    if mode=='Final MSE3':
        assert row.select_rows is compare.MSE3
        return final.cached.run_mode(source,specs,ids,cache,layers=layers)
    assert mode=='HW absmax'
    with patch.object(row,'select_rows',select_absmax):
        return final.cached.run_mode(source,specs,ids,cache,layers=layers)


def conversation(source,specs,example,subset_index,stored):
    assert len(example['targets'])>=TARGETS
    assert example['index']==stored['index'] and stored['targets']==TARGETS
    scores={m:response.Scores() for m in MODES}; direct=response.Scores()
    hidden={m:[base.Metric() for _ in range(18)] for m in MODES}
    caches={}; logits={}; states={}
    for mode in MODES:
        logits[mode],states[mode],caches[mode]=execute(source,specs,mode,example['prompt_ids'],None,True)
    assert_independent([caches[m] for m in MODES])
    for t,target in enumerate(example['targets'][:TARGETS]):
        if t:
            for mode in MODES:
                logits[mode],states[mode],caches[mode]=execute(source,specs,mode,
                    [example['targets'][t-1]],caches[mode],t in HIDDEN_POSITIONS)
        assert_independent([caches[m] for m in MODES])
        assert all(caches[m].get_seq_length()==len(example['prompt_ids'])+t for m in MODES)
        for mode in MODES:
            assert bool(source.torch.isfinite(logits[mode]).all())
            scores[mode].add(source.torch,logits[mode],logits['FP'],[target])
            if t in HIDDEN_POSITIONS:
                for layer in range(18):
                    hidden[mode][layer].add(states[mode][layer].numpy(),states['FP'][layer].numpy())
        direct.add(source.torch,logits['HW absmax'],logits['Final MSE3'],[target])
    errors={}
    for mode,key in [('FP','FP'),('Final MSE3','calibrated')]:
        previous=stored['scores'][key]['all']
        assert previous['n']==TARGETS
        errors[mode]=scores[mode].nll/TARGETS-previous['nll']/TARGETS
        assert abs(errors[mode])<1e-6, f'Fresh/stored baseline mismatch: chat={subset_index}, mode={mode}, delta={errors[mode]}'
    return dict(index=subset_index,source_index=example['index'],targets=TARGETS,
        scores={m:s.raw() for m,s in scores.items()},direct=direct.raw(),
        hidden={m:[vars(v) for v in values] for m,values in hidden.items()},
        fresh_minus_stored_nll=errors)


def finite_tree(value):
    if isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,list):
        for v in value: finite_tree(v)
    elif isinstance(value,float): assert math.isfinite(value)


def aggregate(records,provenance):
    assert [r['index'] for r in records]==SUBSET
    assert sum(r['targets'] for r in records)==256
    modes=[dict(mode=m,**response.merged([r['scores'][m] for r in records])) for m in MODES]
    for key,value in [('kl',0),('nmse',0),('cosine',1),('top1',1)]:
        assert abs(modes[0][key]-value)<1e-12
    changes=deltas(modes[1],modes[2]); rules=decision(modes[1],modes[2])
    direct=response.merged([r['direct'] for r in records])
    per=[]; layers=[]
    for r in records:
        metrics={m:response.merged([r['scores'][m]]) for m in MODES}
        fp,ms,hw=[metrics[m] for m in MODES]
        dn=hw['nll']-ms['nll']
        per.append(dict(index=r['index'],source_index=r['source_index'],targets=r['targets'],
            fp_nll=fp['nll'],mse3_nll=ms['nll'],absmax_nll=hw['nll'],mse3_kl=ms['kl'],absmax_kl=hw['kl'],
            mse3_nmse=ms['nmse'],absmax_nmse=hw['nmse'],mse3_top1=ms['top1'],absmax_top1=hw['top1'],
            nll_outcome='equal' if abs(dn)<1e-3 else ('better' if dn<0 else 'worse')))
    for mode in MODES:
        for l in range(18):
            values=base.merge_metrics([r['hidden'][mode][l] for r in records])
            layers.append(dict(mode=mode,layer=l,nmse=values['nmse'],cosine=values['cosine']))
    worst=max(range(18),key=lambda l:layers[36+l]['nmse']-layers[18+l]['nmse'])
    worst_info=dict(layer=worst,mse3_nmse=layers[18+worst]['nmse'],absmax_nmse=layers[36+worst]['nmse'],
        degradation=layers[36+worst]['nmse']-layers[18+worst]['nmse'])
    summary=dict(provenance=provenance,subset=SUBSET,targets=256,modes=modes,deltas=changes,
        direct_absmax_vs_mse3={k:direct[k] for k in ['nmse','cosine','top1']},decision=rules,
        worst_hidden_degradation=worst_info,
        conversation_outcomes={key:sum(r['nll_outcome']==key for r in per) for key in ['better','worse','equal']},
        max_fresh_stored_nll_error=max(abs(v) for r in records for v in r['fresh_minus_stored_nll'].values()))
    finite_tree(summary);finite_tree(layers);finite_tree(per)
    base.write_json(OUT/'summary.json',summary)
    base.write_csv(OUT/'per_conversation.csv',per);base.write_csv(OUT/'hidden_layers.csv',layers)
    lines=['# Fixed-subset selector fidelity confirmation',
        '| Mode | NLL | PPL | KL vs FP | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in modes:
        lines.append('| '+r['mode']+' | '+' | '.join(f'{r[k]:.9g}' for k in ['nll','ppl','kl','nmse','cosine','top1','in5','overlap'])+' |')
    lines+=['','| MSE3 → absmax | Delta |','|---|---:|']
    for name,key in [('KL relative %','kl_relative'),('NMSE relative %','nmse_relative'),('PPL relative %','ppl_relative'),('Top1 pp','top1_pp')]:
        lines.append(f"| {name} | {changes[key]*(1 if key=='top1_pp' else 100):.6f} |")
    lines+=['','| Decision criteria | Result |','|---|---|']
    for name,key in [('KL <= +10%','kl'),('NMSE <= +10%','nmse'),('abs(Top1 delta) <= 1 pp','top1')]:
        lines.append(f"| {name.replace('|','')} | {'PASS' if rules[key] else 'FAIL'} |")
    lines += [f"| Overall | {rules['overall']} |",'',
        f"Subset {SUBSET}: 8 conversations, 256 response targets. Each mode ran fresh with independent KV caches.",
        f"Direct absmax-vs-MSE3 logits comparison: {summary['direct_absmax_vs_mse3']}.",
        f"Conversation NLL outcomes: {summary['conversation_outcomes']}.",
        f"Largest hidden NMSE degradation: layer {worst}, MSE3 {100*worst_info['mse3_nmse']:.6f}%, absmax {100*worst_info['absmax_nmse']:.6f}% (delta {100*worst_info['degradation']:.6f} pp). Hidden positions only 0, 7, 31.",
        f"Fresh/stored FP and MSE3 NLL: all absolute differences <1e-6; maximum {summary['max_fresh_stored_nll_error']:.12g}.",
        'Frozen decisions, weights, sX/s10 and channel M/S fingerprints: unchanged. No recalibration, search, long validation or greedy generation.',
        'The HW-ready label applies only to the requested fixed-subset fidelity heuristic; PPL was excluded from the decision.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Never silently mix prior inference with this fresh aligned run.
    assert not (OUT/'summary.json').exists(), 'confirmation results already exist'
    source,_,_,previous,_,_=final.load_source_and_policies()
    specs,_=final.restore_final(source,previous)
    assert set(specs)=={f'model.layers.{i}.mlp.down_proj' for i in range(18)}
    frozen=final.OUT/'decisions.json'
    before=frozen_fingerprint(specs,frozen.read_bytes())
    examples,_=final.validation_examples(source);assert len(examples)==24
    assert len(SUBSET)*TARGETS==256
    provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        before=before,script_sha256=base.sha256_file(__file__),command=sys.argv,
        baseline_sha256={str(i):base.sha256_file(final.PROGRESS/f'short_{i:03d}.json') for i in SUBSET})
    records=[]
    for n,index in enumerate(SUBSET):
        stored=json.loads((final.PROGRESS/f'short_{index:03d}.json').read_text())
        record=conversation(source,specs,examples[index],index,stored)
        assert frozen_fingerprint(specs,frozen.read_bytes())==before
        finite_tree(record); records.append(record)
        base.write_json(OUT/f'conversation_{index:03d}.json',record)
        print('CONFIRMED',n+1,8,index,flush=True)
    provenance['after']=frozen_fingerprint(specs,frozen.read_bytes())
    assert provenance['after']==before
    aggregate(records,provenance)
    base.write_json(OUT/'verification.json',dict(frozen_fingerprint_unchanged=True,
        independent_caches=True,fresh_baselines_match_stored=True,
        subset=SUBSET,targets=256,quantized_operations=sorted(specs),
        hidden_positions=sorted(HIDDEN_POSITIONS),all_metrics_finite=True,
        selector_import_identity=select_absmax is compare.select_absmax,
        before=before,after=provenance['after']))

if __name__=='__main__':main()
