#!/usr/bin/env python3
"""Offline factorized row-scalar experiment; cached down_proj only, no QB writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
from static_quant.core import Options, quantize_input, quantize_weight
from static_quant.hardware import Profile
from static_quant.sampling import RepresentativeRowSampler
from static_quant.search import activation_candidates, joint_scale_search
from static_quant.gemma import sha256_file, input_group
from quant_diagnostic_math import dot, integer_dot, configure_blas, ErrorMetric
from run_bos_aware_calibration import token_positions
from diagnose_linear_quant_error import write_json, write_csv


def row_requant(acc, params, correction=1., k=None):
    """Exactly round ACC * stored channel multiplier / 2**shift * scalar.

    Arbitrary FP64 C is interpreted as its exact binary rational. Python integers
    avoid intermediate overflow/double rounding. PoT uses that same exact math,
    including effective shifts outside UInt5; no channel array is regenerated.
    Returns unsaturated integers so saturation can be measured independently.
    """
    c = float(correction)
    if not np.isfinite(c) or c <= 0:
        raise ValueError('positive finite row correction required')
    if k is not None and (int(k) != k or c != 2.**k):
        raise ValueError('PoT correction must equal 2**k')
    profile = Profile()
    if c == 1.:
        return profile.apply(acc, params['multiplier'], params['shift'], saturate=False)
    # Reuse profile input/domain validation without using its rounded result.
    profile.apply(acc, params['multiplier'], params['shift'], saturate=False)
    numerator, denominator = c.as_integer_ratio()
    product = np.asarray(acc, np.int64) * params['multiplier']
    denominators = np.array([denominator << int(s) for s in params['shift']], dtype=object)
    scaled = product.astype(object)*numerator
    q = scaled // denominators
    remainder = scaled - q*denominators
    twice = remainder*2
    rounded = q + ((twice > denominators) | ((twice == denominators) & ((q & 1) != 0)))
    return np.asarray(rounded, np.int64)


def row_path(x, wq, sx_base, params, correction=1., k=None, backend='int64'):
    q = quantize_input(x, sx_base*correction)
    acc = integer_dot(q, wq, backend)
    return q, row_requant(acc, params, correction, k)


def choose_bos(x, w, wq, sx_base, s10, params, corrections, backend):
    """Only BOS calibration observations are accepted here; normal error absent."""
    ref = dot(x, w)
    table = []
    for c, k in corrections:
        q, raw = row_path(x, wq, sx_base, params, c, k, backend)
        metric = ErrorMetric(); metric.add(np.clip(raw, -512, 511)*s10, ref)
        table.append(dict(C=float(c), k=k, s_X_BOS=sx_base*c, **metric.result(),
                          activation_zero_rate=float((q == 0).mean()),
                          activation_clip_rate=float((np.abs(x) > 127*sx_base*c).mean()),
                          int10_clip_rate=float(((raw < -512) | (raw > 511)).mean())))
    return min(table, key=lambda row: row['mse']), table


def calibrate(x, positions, w, sw, baseline, backend='int64', rows=256):
    """No validation argument. Independently select normal and BOS static cases."""
    sampler = RepresentativeRowSampler(rows, 42, bos_rows=0, early_rows=16)
    maximum = 0.
    for start in range(0, len(x), 512):
        p = positions[start:start+512]; block = np.asarray(x[start:start+512])[p != 0]
        if len(block):
            maximum = max(maximum, float(np.abs(block).max()))
            sampler.add(block, p[p != 0])
    candidates = activation_candidates(maximum, np.abs(sampler.values).ravel())
    candidates.append(dict(method='frozen_baseline', s_X=baseline['s_X'], threshold=baseline['s_X']*127))
    s10 = baseline['s_10']
    sx, chosen_s10, params, search = joint_scale_search(
        lambda sl: w[sl], w.shape, sw, sampler, candidates, s10, Profile(),
        Options(mode='fixed', s10=s10, n_chunk=512),
        accumulator=lambda a,b: integer_dot(a,b,backend), matmul=dot)
    assert chosen_s10 == s10 and not np.any(sampler.selected()[1] == 0)
    bos = np.asarray(x[positions == 0], np.float64)
    wq, check = quantize_weight(w); np.testing.assert_array_equal(sw, check)
    # BOS-only percentiles and 16 log-spaced thresholds, plus the base endpoint.
    bos_candidates = activation_candidates(float(np.abs(bos).max()), np.abs(bos).ravel())
    positive = [c['threshold'] for c in bos_candidates if c['threshold'] > 0]
    grid = np.geomspace(min(positive), max(positive), 16) if positive else [127.]
    corrections = sorted(set([1.] + [c['s_X']/sx for c in bos_candidates] + [float(t)/127/sx for t in grid]))
    arbitrary, arbitrary_table = choose_bos(bos,w,wq,sx,s10,params,[(c,None) for c in corrections],backend)
    pot, pot_table = choose_bos(bos,w,wq,sx,s10,params,[(2.**k,k) for k in range(-8,9)],backend)
    # The unrestricted benchmark contains the restricted set, without repeating
    # identical arithmetic; it cannot lose solely because a PoT grid point was missed.
    arbitrary_table.extend(dict(row, k=None, candidate_source='included PoT point') for row in pot_table)
    arbitrary = min(arbitrary_table, key=lambda row: row['mse'])
    return dict(s_X_normal=sx, s_10=s10, arbitrary=arbitrary, pot=pot,
                normal_search=search, arbitrary_candidates=arbitrary_table, pot_candidates=pot_table,
                selection_split='calibration', bos_calibration_rows=len(bos),
                normal_candidate_estimator='percentiles of 256 selected non-BOS rows; full non-BOS absmax plus frozen baseline',
                objective='independent normal MSE and BOS MSE; no cross-group population weighting',
                channel_parameter_arrays=1)


def evaluate(x, positions, w, sw, baseline, selected, backend='int64'):
    """Full validation; B/C share the identical normal result and channel array."""
    s10 = baseline['s_10']; assert selected['s_10'] == s10
    base_params = Profile().approximate(baseline['s_X']*sw/s10)
    params = Profile().approximate(selected['s_X_normal']*sw/s10)
    assert np.all(params['status'] == 'ok') and np.all(base_params['status'] == 'ok')
    wq, actual_sw = quantize_weight(w); np.testing.assert_array_equal(sw, actual_sw)
    schemes = dict(baseline=(baseline['s_X'],1.,None,base_params),
                   arbitrary=(selected['s_X_normal'],selected['arbitrary']['C'],None,params),
                   pot=(selected['s_X_normal'],selected['pot']['C'],selected['pot']['k'],params))
    metrics = {tag:{g:ErrorMetric() for g in ['bos','non_bos','all']} for tag in schemes}
    counts = {tag:{g:dict(elements=0,outputs=0,zero=0,clip=0,q10clip=0) for g in ['bos','non_bos']} for tag in schemes}
    for start in range(0,len(x),512):
        block = np.asarray(x[start:start+512],np.float64); pos = positions[start:start+512]
        for group, mask in [('bos',pos == 0),('non_bos',pos != 0)]:
            if not mask.any(): continue
            xx = block[mask]; ref = dot(xx,w); shared = {}
            for tag,(sx,c,k,p) in schemes.items():
                if group == 'non_bos': c,k = 1.,None
                key=(sx,c)
                if key not in shared:
                    q,raw = row_path(xx,wq,sx,p,c,k,backend)
                    shared[key]=(q,raw,np.clip(raw,-512,511)*s10)
                q,raw,y = shared[key]
                metrics[tag][group].add(y,ref); metrics[tag]['all'].add(y,ref)
                tally=counts[tag][group];tally['elements']+=xx.size;tally['outputs']+=raw.size
                tally['zero']+=int((q == 0).sum());tally['clip']+=int((np.abs(xx)>127*sx*c).sum())
                tally['q10clip']+=int(((raw < -512)|(raw > 511)).sum())
    results=[]
    for tag,(sx,c,k,p) in schemes.items():
        row=dict(scheme=tag,split='validation',s_X_normal=sx,s_X_BOS=sx*c,C_BOS=c,k=k,
                 effective_shift_correction=None if k is None else -k,s_10=s10,
                 effective_shift_min=int(p['shift'].min())-(k or 0),
                 effective_shift_max=int(p['shift'].max())-(k or 0))
        for group in ['bos','non_bos','all']:
            row.update({group+'_'+key:value for key,value in metrics[tag][group].result().items()})
            if group != 'all':
                t=counts[tag][group]
                row.update({group+'_zero_rate':t['zero']/t['elements'],group+'_activation_clip_rate':t['clip']/t['elements'],
                            group+'_int10_clip_rate':t['q10clip']/t['outputs']})
        results.append(row)
    return results


def materialize(out):
    data=sorted([json.loads(p.read_text()) for p in out.glob('layer_*.json')],key=lambda d:d['layer'])
    rows=[dict(layer=d['layer'],**r) for d in data for r in d['evaluation']]
    write_csv(out.parent/'factorized_row_requant_3layer.csv',[r for r in rows if r['layer'] in (7,8,17)])
    if len(data)==18:write_csv(out.parent/'factorized_row_requant_all_down.csv',rows)
    def table(headers,records):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in records])
    report=['# Factorized row requantization — frozen common s_10 experiment',
        'Cached reference inputs reused; no model forward, recapture, download, RTL, LUT, ABI, QB or production changes. '
        'Baseline is the previously frozen stratified_64 main policy, not a validation-selected replacement. '
        'Parameters fit on calibration only; all metrics in the table use full validation. This is local Linear accuracy, not end-to-end model accuracy.',
        'Hardware interpretation: r[m,n] = C_row[m] * r_channel[n], where r_channel[n] is the ONE stored UInt16 multiplier / 2^UInt5 shift array. '
        'B/C share that array, s_W, weights, normal scale, common s_10 and validation rows. Normal C=1. '
        'Input quantization uses s_X_base*C as well. The row factor is applied before the single RNE/saturation, never after INT10. '
        'PoT C=2^k implies effective shift s[n]-k, without clamping to UInt5 or changing packing. Arbitrary C uses exact integer arithmetic for the stored FP64 scalar binary rational. '
        'A scalar shift adjustment might eventually use row_change / row index metadata; negative or extended shifts need implementation review. RTL cost is not measured.',
        'Non-BOS fitting: 256 representative calibration rows (early/general weighted only within non-BOS), sampled-row percentiles p99..p99.99, full non-BOS absmax, and the frozen baseline candidate. '
        'BOS fitting: all 128 calibration BOS rows, BOS-only percentile endpoints plus 16 log-spaced thresholds from p99 to absmax and C=1. '
        'PoT searches k=-8..8; these points are also included in the unrestricted benchmark without duplicate computation. BOS error is never weighted by its corpus frequency. The arbitrary result is a bounded-search software benchmark, not proof of a continuous global optimum.',
        table(['Layer','Scheme','sX normal','sX BOS','C BOS','k','BOS NMSE %','Non-BOS NMSE %','Total NMSE %'],
              [[r['layer'],r['scheme'],f"{r['s_X_normal']:.8g}",f"{r['s_X_BOS']:.8g}",f"{r['C_BOS']:.8g}",r['k'],f"{100*r['bos_nmse']:.6f}",f"{100*r['non_bos_nmse']:.6f}",f"{100*r['all_nmse']:.6f}"] for r in rows]),
        'CSV includes full MSE/MAE/NMSE, activation zero/clipping and INT10 clipping rates for both groups; JSON checkpoints contain calibration candidate tables and frozen choices.']
    gaps=[]
    for d in data:
        a,b,c=d['evaluation'];gaps.append([d['layer'],100*(c['bos_nmse']-b['bos_nmse']),c['bos_mse']/b['bos_mse'] if b['bos_mse'] else None])
    report.append(table(['Layer','PoT minus arbitrary BOS NMSE (percentage points)','PoT / arbitrary BOS MSE'],gaps))
    summaries=[]
    for tag in ['baseline','arbitrary','pot']:
        rr=[r for r in rows if r['scheme']==tag]
        summaries.append([tag]+[float(100*f([r[g+'_nmse'] for r in rr])) for g in ['bos','non_bos'] for f in [np.median,np.max]])
    report.append(table(['Scheme','Median BOS NMSE %','Max BOS NMSE %','Median non-BOS NMSE %','Max non-BOS NMSE %'],summaries))
    gate=out/'decision.json'
    if gate.exists(): report.append('## Three-layer decision\n\n'+json.loads(gate.read_text())['explanation'])
    report.append('Common s_10 optimization was not performed. Validation was not used to select scales. See provenance.json, commands.jsonl, tests.log and per-layer checkpoints in factorized_row_requant/.')
    (out.parent/'factorized_row_requant_report.md').write_text('\n\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers',default='7,8,17');parser.add_argument('--model',default='/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf')
    parser.add_argument('--cache-dir',default='/tmp/bos-aware-capture')
    args=parser.parse_args();out=ROOT/'diagnostics/factorized_row_requant';out.mkdir(exist_ok=True)
    layers=list(range(18)) if args.layers=='all' else [int(i) for i in args.layers.split(',')]
    if set(layers)-{7,8,17}:
        if not (out/'decision.json').exists() or not json.loads((out/'decision.json').read_text())['proceed']:
            raise ValueError('record a positive reviewed three-layer decision before expanding')
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1';prior=ROOT/'diagnostics/bos_aware'
    manifest=json.loads((art/'manifest.json').read_text());old_identity=json.loads((prior/'provenance.json').read_text())
    for file,digest in old_identity['source_hashes'].items():
        if sha256_file(ROOT/file)!=digest:raise ValueError('BOS-aware source changed: '+file)
    identity=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256_file(__file__),baseline='stratified_64',previous_provenance_sha256=sha256_file(prior/'provenance.json'),
        model_sha256=sha256_file(args.model),profile=Profile().metadata(),cache_dir=args.cache_dir,
        normal_rows=256,bos_log_points=16,k_range=[-8,8],seed=42,common_s10='frozen baseline',backend='fp64-exact bounded INT32 with INT64 oracle')
    assert identity['model_sha256']==manifest['gguf']['sha256'] and identity['profile']==manifest['hardware_profile']
    if (out/'provenance.json').exists():assert json.loads((out/'provenance.json').read_text())==identity
    write_json(out/'provenance.json',identity)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,time=time.time()))+'\n')
    import gguf
    reader=gguf.GGUFReader(args.model);positions=token_positions(reader,manifest,ROOT/'calibration_outputs/data-oasst1-v1')
    configure_blas('accelerate');tensors={t.name:t for t in reader.tensors};entries={e['module_name']:e for e in manifest['modules']}
    for layer in layers:
        if (out/f'layer_{layer}.json').exists():continue
        start=time.monotonic();name=f'model.layers.{layer}.mlp.down_proj';entry=entries[name]
        previous=json.loads((prior/'results'/(name+'.json')).read_text())
        baseline=next(r for r in previous['evaluation'] if r['policy']=='stratified_64' and r['split']=='validation')
        t=tensors[f'blk.{layer}.ffn_down.weight'];w=gguf.quants.dequantize(t.data,t.tensor_type).reshape(tuple(reversed(t.shape)))
        wq,sw=quantize_weight(w)
        np.testing.assert_array_equal(wq,np.load(art/entry['int8_weight_file'],mmap_mode='r'))
        with np.load(art/Path(entry['module_manifest']).parent/'scales.npz') as z:np.testing.assert_array_equal(sw,z['op0000.s_W'])
        cache_paths={}
        for split in ['calibration','validation']:
            cache=Path(args.cache_dir)/str(layer)/split
            expected=dict(model=identity['model_sha256'],data=manifest['datasets'][split]['file_sha256'],source=sha256_file(ROOT/'scripts/quant_diagnostic_source.py'),tokens=manifest['datasets'][split]['valid_tokens'])
            assert json.loads((cache/'identity.json').read_text())==expected
            cache_paths[split]=cache/(input_group(name)+'.npy')
        x=np.load(cache_paths['calibration'],mmap_mode='r');assert len(x)==len(positions['calibration'])
        selected=calibrate(x,positions['calibration'],w,sw,baseline,'fp64-exact')
        write_json(out/f'selected_{layer}.json',selected)
        print('SELECTED',layer,selected['s_X_normal'],selected['arbitrary']['C'],selected['pot']['k'],flush=True)
        # Validation values are first opened after selection is frozen on disk.
        xv=np.load(cache_paths['validation'],mmap_mode='r');assert len(xv)==len(positions['validation'])
        for sx in [baseline['s_X'],selected['s_X_normal'],selected['arbitrary']['s_X_BOS'],selected['pot']['s_X_BOS']]:
            q=quantize_input(xv[[0,len(xv)//2,-1]],sx)
            np.testing.assert_array_equal(integer_dot(q,wq[[0,len(wq)//2,-1]],'fp64-exact'),integer_dot(q,wq[[0,len(wq)//2,-1]],'int64'))
        evaluation=evaluate(xv,positions['validation'],w,sw,baseline,selected,'fp64-exact')
        for group in ['bos','non_bos','all']:
            assert np.isclose(evaluation[0][group+'_mse'],baseline[group+'_final_local_total_mse'],rtol=2e-12,atol=1e-15)
        for key in evaluation[1]:
            if key.startswith('non_bos_'):assert evaluation[1][key]==evaluation[2][key]
        write_json(out/f'layer_{layer}.json',dict(layer=layer,baseline_sha256=sha256_file(prior/'results'/(name+'.json')),
            cache_files={s:dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for s,p in cache_paths.items()},
            evaluation=evaluation,seconds=time.monotonic()-start))
        materialize(out);print('DONE',layer,[(r['scheme'],r['bos_nmse'],r['non_bos_nmse']) for r in evaluation],flush=True)
    materialize(out)


if __name__=='__main__':main()
