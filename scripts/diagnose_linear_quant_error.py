#!/usr/bin/env python3
"""Offline, resumable Linear error attribution. Does not change calibration policy.

Default exact INT64 backend prioritizes reference correctness. --integer-backend
fp64-exact uses bounded exact integer arithmetic in FP64 BLAS, with INT64 checks.
All floating counterfactual GEMMs and metric accumulations use FP64.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from static_quant.core import quantize_input, quantize_weight, Reservoir
from static_quant.gemma import sha256_file, input_group, select_names
from static_quant.hardware import get_profile
from quant_diagnostic_math import ErrorMetric, activation_stats, integer_dot, scale_from_threshold, configure_blas, dot


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    temporary=Path(path).with_suffix('.tmp')
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def flatten_metrics(row,metrics):
    for name,metric in metrics.items():
        row.update({name+'_'+k:v for k,v in metric.result().items()})
    return row


def evaluate(x,w,sx,sw,s10,profile,params,backend,m_chunk,n_chunk,
             input_candidates=None,output_candidates=None,reducer=None):
    """One common float reference for all independently computed variants.

    actual_requant is HW vs ideal INT10 in REAL units. final_local_total is HW
    vs float reference. requantization_only is HW vs dequantized INT32.
    """
    base_names=['input_only','weight_only','int8_pair','ideal_int10','actual_requant',
                'final_local_total','ideal_requantization_only','requantization_only']
    metrics={k:ErrorMetric() for k in base_names}
    sweep=[]
    for c in input_candidates or []:
        c=dict(c);c['params']=profile.approximate(c['s_X']*sw/s10)
        c['metrics']={k:ErrorMetric() for k in ['input_only','int8_pair','final_local_total','requantization_only']}
        c.update(output_clip_count=0,zero_count=0,input_clip_count=0,input_count=0)
        sweep.append(c)
    outputs=[]
    for s in output_candidates or []:
        outputs.append(dict(s_10=s,params=profile.approximate(sx*sw/s),
                            metrics={k:ErrorMetric() for k in ['final_local_total','requantization_only','ideal_requantization_only']},clip_count=0))
    clips=ideal_clips=zero=input_clip=input_count=output_count=0
    # Verify actual source weights -> NPU INT8 separately from GGUF source loss.
    w=np.asarray(w);wq,check_sw=quantize_weight(w)
    np.testing.assert_array_equal(sw,check_sw)
    # Independent INT64 oracle includes full K for sampled extreme and data rows.
    sample_x=quantize_input(np.asarray(x)[[0,len(x)//2,len(x)-1]],sx)
    sample_w=wq[[0,len(w)//2,len(w)-1]]
    np.testing.assert_array_equal(integer_dot(sample_x,sample_w,backend),integer_dot(sample_x,sample_w,'int64'))
    for start in range(0,len(x),m_chunk):
        xb=np.asarray(x[start:start+m_chunk],np.float64);xq=quantize_input(xb,sx);xh=xq.astype(np.float64)*sx
        zero+=int((xq==0).sum());input_clip+=int((np.abs(xb)>127*sx).sum());input_count+=xb.size
        candidate_inputs=[]
        for c in sweep:
            cq=quantize_input(xb,c['s_X']);candidate_inputs.append(cq)
            c['zero_count']+=int((cq==0).sum());c['input_clip_count']+=int((np.abs(xb)>127*c['s_X']).sum());c['input_count']+=xb.size
        for ns in range(0,len(w),n_chunk):
            sl=slice(ns,ns+n_chunk);wb=np.asarray(w[sl],np.float64);qw=wq[sl];sc=sw[sl]
            ref=dot(xb,wb)
            acc=integer_dot(xq,qw,backend);pair=acc*(sx*sc)
            raw=profile.apply(acc,params['multiplier'][sl],params['shift'][sl],saturate=False)
            hw=np.clip(raw,-512,511)*s10
            iraw=np.rint(acc*(sx*sc/s10));ideal=np.clip(iraw,-512,511)*s10
            input_y=dot(xh,wb);weight_y=dot(xb,qw.astype(np.float64)*sc[:,None])
            if reducer:
                reducer.add(metrics,ref,input_y,weight_y,pair,ideal,hw)
            else:
                values={'input_only':(input_y,ref),'weight_only':(weight_y,ref),
                        'int8_pair':(pair,ref),'ideal_int10':(ideal,ref),'actual_requant':(hw,ideal),
                        'final_local_total':(hw,ref),'ideal_requantization_only':(ideal,pair),'requantization_only':(hw,pair)}
                for key,(v,r) in values.items():metrics[key].add(v,r)
            clips+=int(((raw < -512)|(raw > 511)).sum());ideal_clips+=int(((iraw < -512)|(iraw > 511)).sum());output_count+=acc.size
            for c,cq in zip(sweep,candidate_inputs):
                ca=integer_dot(cq,qw,backend);cp=ca*(c['s_X']*sc)
                cr=profile.apply(ca,c['params']['multiplier'][sl],c['params']['shift'][sl],saturate=False)
                ch=np.clip(cr,-512,511)*s10
                cy=dot(cq.astype(np.float64)*c['s_X'],wb)
                if reducer:
                    reducer.add(c['metrics'],ref,cy,ref,cp,ch,ch)
                else:
                    for key,v,r in [('input_only',cy,ref),('int8_pair',cp,ref),('final_local_total',ch,ref),('requantization_only',ch,cp)]:
                        c['metrics'][key].add(v,r)
                c['output_clip_count']+=int(((cr < -512)|(cr > 511)).sum())
            for c in outputs:
                cr=profile.apply(acc,c['params']['multiplier'][sl],c['params']['shift'][sl],saturate=False)
                ch=np.clip(cr,-512,511)*c['s_10']
                ci=np.clip(np.rint(acc*(sx*sc/c['s_10'])),-512,511)*c['s_10']
                if reducer:
                    reducer.add(c['metrics'],ref,ref,ref,pair,ci,ch)
                else:
                    c['metrics']['final_local_total'].add(ch,ref);c['metrics']['requantization_only'].add(ch,pair)
                    c['metrics']['ideal_requantization_only'].add(ci,pair)
                c['clip_count']+=int(((cr < -512)|(cr > 511)).sum())
    base=flatten_metrics(dict(s_X=sx,s_10=s10,int10_clip_rate=clips/output_count,
            ideal_int10_clip_rate=ideal_clips/output_count,activation_int8_clip_rate=input_clip/input_count,
            activation_zero_rate=zero/input_count,valid_tokens=len(x),output_elements=output_count),metrics)
    rows=[]
    for c in sweep:
        rows.append(flatten_metrics(dict(policy=c['policy'],threshold=c['threshold'],s_X=c['s_X'],s_10=s10,
             input_clipping_fraction=c['input_clip_count']/c['input_count'],zero_after_quant_fraction=c['zero_count']/c['input_count'],
             int10_clip_rate=c['output_clip_count']/output_count,
             ratio_max_relative_error=float(c['params']['relative_error'].max()),
             bad_ratio_channels=int((c['params']['status']!='ok').sum())),c['metrics']))
    outrows=[flatten_metrics(dict(s_X=sx,s_10=c['s_10'],int10_clip_rate=c['clip_count']/output_count),c['metrics']) for c in outputs]
    return base,rows,outrows


def materialize(output):
    results=[json.loads(p.read_text()) for p in sorted((output/'results').glob('*.json'))]
    write_csv(output/'quant_error_attribution.csv',[r for d in results for r in d['attribution']])
    stats=[r for d in results for r in d['activation_stats']]
    write_csv(output/'down_proj_activation_stats.csv',[r for r in stats if r['module']=='down_proj'])
    write_csv(output/'mlp_activation_comparison.csv',stats)
    sweeps=[json.loads(p.read_text()) for p in sorted((output/'sweeps').glob('*.json'))]
    write_csv(output/'down_proj_scale_sweep.csv',[r for d in sweeps for r in d['input_sweep']])
    write_csv(output/'down_proj_output_scale_sweep.csv',[r for d in sweeps for r in d['output_sweep']])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',default='calibration_outputs/gguf-all-126-oasst1')
    p.add_argument('--model',required=True,help='existing local GGUF only; never downloaded')
    p.add_argument('--data-dir',default='calibration_outputs/data-oasst1-v1')
    p.add_argument('--output-dir',default='diagnostics')
    p.add_argument('--cache-dir',default='/tmp/quant-diagnosis-capture')
    p.add_argument('--layers',default='all');p.add_argument('--modules',default='all')
    p.add_argument('--integer-backend',choices=['int64','fp64-exact'],default='int64')
    p.add_argument('--m-chunk',type=int,default=512);p.add_argument('--n-chunk',type=int,default=512)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--blas',choices=['numpy','accelerate'],default='numpy')
    p.add_argument('--metric-backend',choices=['numpy','native'],default='numpy')
    p.add_argument('--phase',choices=['baseline','sweep','all'],default='all')
    args=p.parse_args(argv)
    configure_blas(args.blas)
    if args.metric_backend=='native':
        from quant_diagnostic_reduce import NativeMetrics
        reducer=NativeMetrics()
    else:reducer=None
    if min(args.m_chunk,args.n_chunk,args.threads)<1:raise ValueError('chunks/threads must be positive')
    art=Path(args.artifacts);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((art/'manifest.json').read_text());run=json.loads((art/'run.json').read_text())
    names=select_names(18,args.layers,args.modules)
    entries={e['module_name']:e for e in manifest['modules']}
    if any(n not in entries for n in names):raise ValueError('operation not in stored run')
    profile=get_profile(manifest['hardware_profile']['name'])
    if profile.metadata()!=manifest['hardware_profile']:raise ValueError('hardware profile mismatch')
    source_hashes={str(f.relative_to(ROOT)):sha256_file(f) for f in sorted((ROOT/'scripts').glob('quant_diagnostic*.py'))}
    source_hashes['scripts/diagnose_linear_quant_error.py']=sha256_file(__file__)
    source_hashes['scripts/quant_metric_reduce.c']=sha256_file(ROOT/'scripts/quant_metric_reduce.c')
    source_hashes.update({str(f.relative_to(ROOT)):sha256_file(f) for f in (ROOT/'src/static_quant').glob('*.py')})
    identity=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_hashes=source_hashes,stored_manifest_sha256=sha256_file(art/'manifest.json'),
                  stored_run_sha256=sha256_file(art/'run.json'),model_sha256=sha256_file(args.model),
                  datasets={s:sha256_file(Path(args.data_dir)/(s+'.jsonl')) for s in ['calibration','validation']},
                  integer_backend=args.integer_backend,blas=args.blas,metric_backend=args.metric_backend,m_chunk=args.m_chunk,n_chunk=args.n_chunk,threads=args.threads)
    provenance=out/'diagnosis_provenance.json'
    if provenance.exists():
        previous=json.loads(provenance.read_text())
        if previous['identity']!=identity:raise ValueError('resume identity changed; use a fresh output directory')
    else:
        write_json(provenance,dict(identity=identity,stored_source_commit=manifest['source_commit'],stored_source_dirty=manifest['source_tree_dirty'],
                   stored_code_sha256=run['identity']['code_sha256'],stored_model=manifest['gguf'],stored_datasets=manifest['datasets'],
                   profile=profile.metadata(),gguf_source_error='not measured: original checkpoint unavailable',
                   percentile_method='exact NumPy linear interpolation over all elements; calibration-only fitting',
                   input_sweep_output_policy='hold recorded s_10 fixed; recompute multiplier/shift for diagnostic s_X only',
                   metrics={'actual_requant':'HW vs ideal INT10, real units; denominator ideal output energy',
                            'final_local_total':'HW vs same float X@W.T','all_counterfactual_nmse':'MSE / mean(reference squared)',
                            'requantization_only':'HW vs dequantized INT32, denominator pair energy'}))
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,time=time.time(),selection=names,phase=args.phase))+'\n')
    # Stored artifacts must be verified before being trusted, even on resume.
    for name in names:
        entry=entries[name];module_dir=art/Path(entry['module_manifest']).parent
        mm=json.loads((module_dir/'manifest.json').read_text())
        for file,digest in mm['artifact_sha256'].items():
            if sha256_file(module_dir/file)!=digest:raise ValueError(f'artifact hash mismatch {module_dir/file}')
    source=None
    from quant_diagnostic_source import RecordedGGUFSource
    for phase in (['baseline','sweep'] if args.phase=='all' else [args.phase]):
        for layer in sorted({int(n.split('.')[2]) for n in names}):
            layer_names=[n for n in names if int(n.split('.')[2])==layer and (phase=='baseline' or n.endswith('down_proj'))]
            pending=[n for n in layer_names if not (out/('results' if phase=='baseline' else 'sweeps')/(n+'.json')).exists()]
            if not pending:continue
            if source is None:
                print('Loading and verifying local model/data...',flush=True)
                source=RecordedGGUFSource(args.model,args.data_dir,manifest,args.threads)
                print('Model and token IDs match recorded identity',flush=True)
            captures={}
            for split in ['calibration','validation']:
                cache=Path(args.cache_dir)/str(layer)/split;marker=cache/'identity.json'
                expected=dict(model=identity['model_sha256'],data=identity['datasets'][split],source=sha256_file(ROOT/'scripts/quant_diagnostic_source.py'),
                              tokens=manifest['datasets'][split]['valid_tokens'])
                if marker.exists() and json.loads(marker.read_text())==expected and all((cache/(input_group(n)+'.npy')).exists() for n in pending):
                    captures[split]={input_group(n):np.load(cache/(input_group(n)+'.npy'),mmap_mode='r') for n in pending}
                else:
                    print(f'Capture layer {layer} {split}',flush=True)
                    captures[split]=source.capture(layer,pending,split,cache)
                    write_json(marker,expected)
            for name in pending:
                started=time.monotonic();entry=entries[name];module_dir=art/Path(entry['module_manifest']).parent
                report=json.loads((art/entry['report_file']).read_text())['modules'][name]
                with np.load(module_dir/'scales.npz') as z:
                    sx=float(z['op0000.s_X']);s10=float(z['op0000.s_10']);sw=z['op0000.s_W'].copy()
                with np.load(module_dir/'qparams.npz') as z:
                    params={k:z['op0000.'+k].copy() for k in ['multiplier','shift','packed','zero_point','ratio']}
                np.testing.assert_array_equal(params['packed'],profile.pack(params['multiplier'],params['shift']))
                if np.any(params['zero_point']):raise ValueError('nonzero stored zero point')
                np.testing.assert_allclose(params['ratio'],sx*sw/s10,rtol=1e-14,atol=0)
                w=source.weight(name);wq,ws=quantize_weight(w)
                np.testing.assert_array_equal(sw,ws)
                savedw=np.load(art/entry['int8_weight_file'],mmap_mode='r')
                np.testing.assert_array_equal(wq,savedw);del wq,savedw
                xc=captures['calibration'][input_group(name)]
                recovered_sx=scale_from_threshold(float(np.max(np.abs(xc))),float(np.max(np.abs(xc))))
                if not np.isclose(recovered_sx,sx,rtol=2e-6):raise ValueError(f'replayed activation absmax differs: {name}: {recovered_sx} vs {sx}')
                prefix=dict(layer=layer,module=name.split('.')[-1],module_name=name)
                result=dict(attribution=[],activation_stats=[],input_sweep=[],output_sweep=[])
                candidates=None;oscales=None
                if phase=='sweep':
                    stat=activation_stats(xc,sx)
                    candidates=[dict(policy='absmax',threshold=127*sx,s_X=sx)]
                    candidates += [dict(policy=f'percentile_{q:g}',threshold=stat[f'p{q:g}'],s_X=scale_from_threshold(stat[f'p{q:g}'],stat['absmax'])) for q in [99,99.5,99.9,99.95,99.99]]
                    # Optional activation-reconstruction MSE search on deterministic 64-row calibration reservoir.
                    reservoir=Reservoir(64,manifest['seed'])
                    for start in range(0,len(xc),512):reservoir.add(xc[start:start+512])
                    sample=reservoir.values.astype(np.float64)
                    scored=[(float(np.square(quantize_input(sample,c['s_X']).astype(np.float64)*c['s_X']-sample).mean()),c) for c in candidates]
                    best=min(scored,key=lambda t:t[0])[1]
                    candidates.append(dict(best,policy='activation_mse_search_64_rows'))
                    oscales=sorted(set([s10,report['selection']['minmax_baseline']]+[c['s10'] for c in report['selection']['candidates']]))
                    result['activation_search']=dict(objective='X reconstruction MSE (not output MSE)',seed=manifest['seed'],rows=64,
                          candidates=[dict(policy=c['policy'],mse=score) for score,c in scored],chosen=best)
                for split in ['calibration','validation']:
                    print(f'{phase} {name} {split}',flush=True)
                    x=captures[split][input_group(name)]
                    base,sweep,output_sweep=evaluate(x,w,sx,sw,s10,profile,params,args.integer_backend,args.m_chunk,args.n_chunk,candidates,oscales,reducer)
                    # Independently reproduced totals must agree before attributing stored error.
                    expected_mse=report[split]['local_total']['mse']
                    if not np.isclose(base['final_local_total_mse'],expected_mse,rtol=2e-5,atol=1e-10):
                        raise ValueError(f'stored local MSE mismatch {name} {split}: {base["final_local_total_mse"]} vs {expected_mse}')
                    base.update(prefix,split=split,stored_local_total_mse=expected_mse,stored_mse_relative_difference=abs(base['final_local_total_mse']/expected_mse-1) if expected_mse else 0)
                    result['attribution'].append(base)
                    if phase=='baseline' and name.endswith(('gate_proj','up_proj','down_proj')):
                        stat=activation_stats(x,sx);stat.update(prefix,split=split)
                        for key in ['input_only_nmse','weight_only_nmse','int8_pair_nmse','final_local_total_nmse']:stat[key]=base[key]
                        result['activation_stats'].append(stat)
                    result['input_sweep'].extend(dict(r,**prefix,split=split) for r in sweep)
                    result['output_sweep'].extend(dict(r,**prefix,split=split,selected_current=bool(r['s_10']==s10),minmax=bool(r['s_10']==report['selection']['minmax_baseline'])) for r in output_sweep)
                result['seconds']=time.monotonic()-started
                write_json(out/('results' if phase=='baseline' else 'sweeps')/(name+'.json'),result)
                materialize(out)
                print(f'DONE {name} {phase}: {result["seconds"]:.1f}s',flush=True)
    materialize(out)


if __name__=='__main__':main()
