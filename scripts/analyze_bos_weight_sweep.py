#!/usr/bin/env python3
"""Rescore stored calibration candidates; optionally evaluate only missing choices."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import sys

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'diagnostics';OUT=D/'bos_weight_sweep';WEIGHTS=[i/20 for i in range(21)]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csvfile(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def frontier(rows):
    feasible=[r for r in rows if r['feasible']]
    return [r for r in feasible if not any(
        t['bos_nmse']<=r['bos_nmse'] and t['non_bos_nmse']<=r['non_bos_nmse'] and
        (t['bos_nmse']<r['bos_nmse'] or t['non_bos_nmse']<r['non_bos_nmse']) for t in feasible)]


def choose(rows,w):
    score=lambda r:w*r['bos_nmse']+(1-w)*r['non_bos_nmse']
    best=min(map(score,rows))
    tied=[r for r in rows if math.isclose(score(r),best,rel_tol=1e-6,abs_tol=1e-12)]
    return min(tied,key=lambda r:(r['worst_group_score'],r['balanced_clip_rate'],abs(math.log2(r['ratio'])),r['s_10']))


def aggregate(rows):
    result={g+'_'+key:f(r[g+'_nmse'] for r in rows) for g in ['bos','non_bos'] for key,f in [('median',stats.median),('max',max)]}
    result.update(mean_worst=stats.mean(max(r['bos_nmse'],r['non_bos_nmse']) for r in rows),
                  max_worst=max(max(r['bos_nmse'],r['non_bos_nmse']) for r in rows),
                  mean_balanced=stats.mean((r['bos_nmse']+r['non_bos_nmse'])/2 for r in rows))
    return result


def load_analysis():
    by={i:[] for i in range(18)};path=D/'s10_calibration/candidates.csv'
    for raw in csv.DictReader(path.open()):
        r={k:(v=='True' if k=='feasible' else float(v)) for k,v in raw.items()}
        by[int(r.pop('layer'))].append(r)
    inputs={str(path.relative_to(ROOT)):digest(path)};records={}
    for layer,rows in by.items():
        p=D/f's10_calibration/layer_{layer}.json';d=json.loads(p.read_text());records[layer]=d
        inputs[str(p.relative_to(ROOT))]=digest(p)
        assert len(rows)==33 and rows==d['search']['candidates']
        assert len({(r['s_X_normal'],r['k']) for r in rows})==1
        assert all(r['worst_group_score']==max(r['bos_nmse'],r['non_bos_nmse']) for r in rows)
    fronts={l:frontier(rows) for l,rows in by.items()}
    selected={w:[choose(fronts[l],w) for l in range(18)] for w in WEIGHTS}
    for l in by:
        assert selected[.5][l]==records[l]['search']['selected'],'50:50 reproduction failed'
        for w in WEIGHTS:
            assert selected[w][l] in fronts[l]
            assert selected[w][l]==choose([r for r in by[l] if r['feasible']],w)
    aggs={w:aggregate(rows) for w,rows in selected.items()}
    # Robust comparisons do not reuse the weight-dependent candidate objective.
    best=min(WEIGHTS,key=lambda w:(aggs[w]['mean_worst'],aggs[w]['max_worst'],aggs[w]['mean_balanced'],abs(w-.5),w))
    per_weights=[min(WEIGHTS,key=lambda w:(selected[w][l]['worst_group_score'],selected[w][l]['balanced_score'],abs(w-best),w)) for l in by]
    policies={'50:50':selected[.5],'best_global':selected[best],
              'per_op':[selected[w][l] for l,w in enumerate(per_weights)]}
    plateau=[w for w in WEIGHTS if aggs[w]['mean_worst']<=aggs[best]['mean_worst']*1.02]
    frozen=dict(selection_split='calibration',best_global_weight=best,plateau_weights=plateau,
                per_op_weights=per_weights,policies=policies,input_hashes=inputs,script_sha256=digest(__file__),
                tie_rule='candidate scores isclose rtol=1e-6 atol=1e-12; robust comparisons exact ties then specified criteria',
                global_criterion='mean per-op max(BOS NMSE, non-BOS NMSE)',
                note='local-NMSE criterion; not end-to-end model optimality')
    p=OUT/'frozen_selection.json'
    if p.exists():assert json.loads(p.read_text())==frozen,'frozen selection/source mismatch'
    else:save(p,frozen)
    return by,fronts,selected,aggs,frozen,records


def available_validation(records):
    result={l:{r['s_10']:r for r in d['evaluation']} for l,d in records.items()}
    for p in (OUT/'validation').glob('layer_*.json'):
        d=json.loads(p.read_text());result[d['layer']].update({r['s_10']:r for r in d['evaluation']})
    return result


def missing_choices(frozen,available):
    return {l:sorted({rows[l]['s_10'] for rows in frozen['policies'].values()}-available[l].keys()) for l in range(18)}


def evaluate_missing(frozen,records,missing):
    """Only validation rows of already frozen, previously unmeasured scale choices.

    No calibration GEMM, candidate search, model forward or activation capture.
    One shared reference/ACC per needed layer; stored validation rows are reused.
    """
    import os
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
    from test_s10_calibration import evaluate,Profile,quantize_weight,token_positions,configure_blas
    import numpy as np
    import gguf
    old=json.loads((D/'s10_calibration/provenance.json').read_text())
    assert Profile().metadata()==old['profile']
    model=Path('/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf')
    assert digest(model)==old['model_sha256']
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1';manifest=json.loads((art/'manifest.json').read_text())
    reader=gguf.GGUFReader(str(model));tensors={t.name:t for t in reader.tensors}
    pos=token_positions(reader,manifest,ROOT/'calibration_outputs/data-oasst1-v1')['validation']
    entries={e['module_name']:e for e in manifest['modules']};configure_blas('accelerate')
    for layer,scales in missing.items():
        if not scales:continue
        t=tensors[f'blk.{layer}.ffn_down.weight'];w=gguf.quants.dequantize(t.data,t.tensor_type).reshape(tuple(reversed(t.shape)))
        wq,sw=quantize_weight(w);name=f'model.layers.{layer}.mlp.down_proj';entry=entries[name]
        np.testing.assert_array_equal(wq,np.load(art/entry['int8_weight_file'],mmap_mode='r'))
        reference=records[layer]['evaluation'][1]
        cache=Path('/tmp/bos-aware-capture')/str(layer)/'validation'
        marker=json.loads((cache/'identity.json').read_text())
        assert marker==dict(model=old['model_sha256'],data=manifest['datasets']['validation']['file_sha256'],
                            source=digest(ROOT/'scripts/quant_diagnostic_source.py'),tokens=len(pos))
        path=cache/(name+'.input.npy');x=np.load(path,mmap_mode='r');assert len(x)==len(pos)
        policies={str(s):dict(s_X_normal=reference['s_X_normal'],k=reference['k'],old_s_10=reference['old_s_10'],s_10=s,ratio=s/reference['old_s_10']) for s in scales}
        # Output-only reproduction check adds no additional GEMM/reference product.
        policies['stored_check']={k:reference[k] for k in ['s_X_normal','k','old_s_10','s_10','ratio']}
        measured=evaluate(x,pos,w,sw,policies,backend='fp64-exact')
        for g in ['bos','non_bos']:
            for metric in ['mse','mae','nmse','int10_clip_rate']:
                np.testing.assert_allclose(measured[-1][g+'_'+metric],reference[g+'_'+metric],rtol=2e-12,atol=1e-15)
        save(OUT/'validation'/f'layer_{layer}.json',dict(layer=layer,evaluation=measured[:-1],
             frozen_selection_sha256=digest(OUT/'frozen_selection.json'),validation_only=True,
             source_cache=str(path),cache_mtime_ns=path.stat().st_mtime_ns,reproduction_check_passed=True))
        print('VALIDATED missing scales',layer,scales,flush=True)


def render(by,fronts,selected,aggs,frozen,records):
    best=frozen['best_global_weight'];policies=frozen['policies'];available=available_validation(records)
    missing=missing_choices(frozen,available);global_rows=[];per_rows=[];transitions=[];pareto=[]
    for w in WEIGHTS:
        global_rows.append(dict(w_BOS=w,split='calibration',**aggs[w],
            changed_ops_vs_50=sum(a['s_10']!=b['s_10'] for a,b in zip(selected[w],selected[.5])),
            best_global=w==best,in_two_percent_plateau=w in frozen['plateau_weights']))
    for l in by:
        global_r=policies['best_global'][l];r=policies['per_op'][l]
        distinct=len({selected[w][l]['s_10'] for w in WEIGHTS})
        gain=1-r['worst_group_score']/global_r['worst_group_score']
        row=dict(layer=l,best_w_BOS=frozen['per_op_weights'][l],s_10=r['s_10'],
                 calibration_bos_nmse=r['bos_nmse'],calibration_non_bos_nmse=r['non_bos_nmse'],
                 calibration_global_worst=global_r['worst_group_score'],calibration_per_op_worst=r['worst_group_score'],
                 relative_worst_reduction=gain,material_over_5percent=gain>.05,distinct_candidates=distinct,
                 sensitivity='stable' if distinct==1 else ('mildly sensitive' if distinct<=3 else 'strongly sensitive'),
                 pareto_count=len(fronts[l]),global_on_pareto=global_r in fronts[l],per_op_on_pareto=r in fronts[l])
        for tag in policies:
            v=available[l].get(policies[tag][l]['s_10'])
            if v:row.update({tag+'_validation_'+g+'_nmse':v[g+'_nmse'] for g in ['bos','non_bos']})
        per_rows.append(row)
        start=0
        for i in range(1,22):
            if i==21 or selected[WEIGHTS[i]][l]['s_10']!=selected[WEIGHTS[start]][l]['s_10']:
                c=selected[WEIGHTS[start]][l]
                transitions.append(dict(layer=l,w_from=WEIGHTS[start],w_to=WEIGHTS[i-1],s_10=c['s_10'],bos_nmse=c['bos_nmse'],non_bos_nmse=c['non_bos_nmse']))
                start=i
        for c in by[l]:
            pareto.append(dict(layer=l,s_10=c['s_10'],bos_nmse=c['bos_nmse'],non_bos_nmse=c['non_bos_nmse'],feasible=c['feasible'],on_frontier=c in fronts[l],selected_global=c==global_r,selected_per_op=c==r))
    csvfile(D/'bos_weight_sweep_global.csv',global_rows);csvfile(D/'bos_weight_sweep_per_op.csv',per_rows)
    csvfile(D/'bos_weight_sweep_transitions.csv',transitions);csvfile(D/'bos_weight_sweep_pareto.csv',pareto)
    comparisons=[dict(policy=tag,split='calibration',**aggregate(rr)) for tag,rr in policies.items()]
    if not any(missing.values()):
        comparisons += [dict(policy=tag,split='validation',**aggregate([available[l][r['s_10']] for l,r in enumerate(rr)])) for tag,rr in policies.items()]
    csvfile(OUT/'comparison.csv',comparisons)
    def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    def pct(v):return f'{100*v:.6f}%'
    report=['# BOS objective-weight sweep (candidate-table rescoring)',
        'Selection and the 2% plateau use calibration only. Existing 594 s10 candidates, s_X_normal, k and feasibility are unchanged. No new candidate scales or calibration GEMMs. '
        'The common weight may select a different precomputed s10 per operation. Global/per-op policy comparison uses the weight-independent mean per-op worst-group NMSE, not each weight’s own objective. This is local Linear robustness, not end-to-end optimality.',
        f'Best global w_BOS = {best:.2f}. Weights within 2% of the best calibration robust score: {frozen["plateau_weights"]}. '
        'Candidate tie handling reproduces the previous 50:50 policy exactly; robust-score ties follow the requested ordering. Dominated feasible candidates are removed before rescoring. All selected policies are Pareto-frontier members.',
        '## Global sweep — calibration',table(['w BOS','BOS median','BOS max','non-BOS median','non-BOS max','Mean worst','Changed ops vs 0.50'],
            [[r['w_BOS']]+[pct(r[k]) for k in ['bos_median','bos_max','non_bos_median','non_bos_max','mean_worst']]+[r['changed_ops_vs_50']] for r in global_rows]),
        '## Per-operation oracle — calibration',table(['Layer','Best w BOS','s10','BOS NMSE','non-BOS NMSE','Distinct candidates','Pareto count','Worst reduction vs global'],
            [[r['layer'],r['best_w_BOS'],f"{r['s_10']:.9g}",pct(r['calibration_bos_nmse']),pct(r['calibration_non_bos_nmse']),r['distinct_candidates'],r['pareto_count'],pct(r['relative_worst_reduction'])] for r in per_rows]),
        '## Frozen-policy comparison',
        table(['Split','Policy','BOS median','BOS max','non-BOS median','non-BOS max','Mean worst','Max worst'],
              [[r['split'],r['policy']]+[pct(r[k]) for k in ['bos_median','bos_max','non_bos_median','non_bos_max','mean_worst','max_worst']] for r in comparisons])]
    gains={}
    for split in ['calibration','validation']:
        rr={r['policy']:r for r in comparisons if r['split']==split}
        if rr:
            absolute=rr['best_global']['mean_worst']-rr['per_op']['mean_worst'];relative=absolute/rr['best_global']['mean_worst']
            gains[split]=dict(absolute_nmse_reduction=absolute,percentage_point_reduction=100*absolute,relative_reduction=relative)
            report.append(f'{split}: per-op versus best global mean-worst reduction = {100*absolute:.9f} percentage points ({100*relative:.6f}% relative). Negative means per-op is worse.')
    material=[r['layer'] for r in per_rows if r['material_over_5percent']];strong=[r['layer'] for r in per_rows if r['sensitivity']=='strongly sensitive']
    report.append(f'Material (>5% calibration worst-group reduction) override layers: {material}. Strongly weight-sensitive over the full 0..1 sweep: {strong}. '
                  'Sensitivity counts distinct candidates, not improvement size. Full contiguous transition intervals are in bos_weight_sweep_transitions.csv; a regime describes the 0.05 grid, not an exact continuous breakpoint.')
    report.append(f'Validation missing choices: {sum(map(len,missing.values()))}. Stored validation initially covers only frozen and 50:50 selections; missing frozen choices require minimal validation-only evaluation. '
                  'Existing validation is reused, one shared reference/ACC is computed per needed layer, and the 594 calibration candidates are never rerun. No inference, recapture or new model download.')
    if (OUT/'answers.md').exists():report.append((OUT/'answers.md').read_text())
    (D/'bos_weight_sweep_report.md').write_text('\n\n'.join(report)+'\n')
    save(OUT/'summary.json',dict(best_global_weight=best,plateau=frozen['plateau_weights'],gains=gains,material_override_layers=material,strongly_sensitive_layers=strong,
         validation_missing=sum(map(len,missing.values())),pareto_violations=0,distinct_counts={str(r['layer']):r['distinct_candidates'] for r in per_rows},comparisons=comparisons))
    print(json.dumps(dict(best_global=best,plateau=frozen['plateau_weights'],gains=gains,material_layers=material,missing_validation=sum(map(len,missing.values()))),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluate-missing',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);by,fronts,selected,aggs,frozen,records=load_analysis()
    missing=missing_choices(frozen,available_validation(records))
    save(OUT/'validation_plan.json',dict(missing={str(l):s for l,s in missing.items() if s},frozen_selection_sha256=digest(OUT/'frozen_selection.json')))
    if args.evaluate_missing and any(missing.values()):evaluate_missing(frozen,records,missing)
    render(by,fronts,selected,aggs,frozen,records)


if __name__=='__main__':main()
