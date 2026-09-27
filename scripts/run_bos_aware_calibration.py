#!/usr/bin/env python3
"""Real recorded GGUF/OASST1 replay, stratified joint search and row sensitivity.

No synthetic fallback, no download, no hardware changes. Resumable per operation.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from static_quant.core import Options,quantize_input,quantize_weight
from static_quant.sampling import RepresentativeRowSampler
from static_quant.search import activation_candidates,joint_scale_search,DEFAULT_PERCENTILES
from static_quant.cli import operation_options
from static_quant.gemma import sha256_file,load_texts,input_group,select_names
from static_quant.hardware import Profile
from quant_diagnostic_math import dot,integer_dot,configure_blas
from quant_diagnostic_reduce import NativeMetrics
from quant_diagnostic_math import ErrorMetric
from diagnose_linear_quant_error import write_json,write_csv


def token_positions(reader,manifest,data_dir):
    from transformers import GemmaTokenizerFast
    from transformers.integrations.ggml import GGUFGemmaConverter,GGUF_TOKENIZER_MAPPING,_gguf_parse_value
    data={}
    for key,target in GGUF_TOKENIZER_MAPPING['tokenizer'].items():
        field=reader.fields.get('tokenizer.'+key)
        if field is not None:
            values=[_gguf_parse_value(field.parts[i],field.types) for i in field.data]
            data[target]=values if field.types[0]==9 else values[0]
    converter=GGUFGemmaConverter(data)
    tokenizer=GemmaTokenizerFast(tokenizer_object=converter.converted(),add_bos_token=True,add_eos_token=False,**converter.additional_kwargs)
    result={}
    for split in ['calibration','validation']:
        meta=manifest['datasets'][split];path=data_dir/(split+'.jsonl')
        if sha256_file(path)!=meta['file_sha256']:raise ValueError('dataset identity mismatch')
        texts,_=load_texts(path,meta['actual_samples'])
        ids=[tokenizer(t,add_special_tokens=True,truncation=True,max_length=512)['input_ids'] for t in texts]
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=meta['token_ids_sha256']:raise ValueError('token identity mismatch')
        if not all(row[0]==tokenizer.bos_token_id for row in ids):raise ValueError('position zero is not BOS')
        result[split]=np.concatenate([np.arange(len(row),dtype=np.int64) for row in ids])
    return result


def metrics_for_policies(x,pos,w,sw,policies,profile,m_chunk=512,n_chunk=512):
    """Share exact accumulators and float counterfactuals across equal input scales."""
    reducer=NativeMetrics();n=len(w);wq,check=quantize_weight(w);np.testing.assert_array_equal(sw,check)
    names=['input_only','weight_only','int8_pair','ideal_int10','actual_requant','final_local_total','ideal_requantization_only','requantization_only']
    groups=['bos','non_bos','early','general','all']
    unique={}
    for policy in policies:
        key=(policy['s_X'],policy['s_10'])
        if key not in unique:
            unique[key]=dict(metrics={g:{k:ErrorMetric() for k in names} for g in groups},
                             clips={g:0 for g in groups},counts={g:0 for g in groups},input_clip=0,zero=0,
                             params=profile.approximate(policy['s_X']*sw/policy['s_10']))
    scales=sorted({k[0] for k in unique})
    # Independent full-K integer oracle for the accelerated exact-integer path.
    for sx in scales:
        q=quantize_input(np.asarray(x)[[0,len(x)//2,-1]],sx);qw=wq[[0,n//2,-1]]
        np.testing.assert_array_equal(integer_dot(q,qw,'fp64-exact'),integer_dot(q,qw,'int64'))
    for start in range(0,len(x),m_chunk):
        xb=np.asarray(x[start:start+m_chunk],np.float64);pb=pos[start:start+len(xb)]
        masks=dict(bos=pb==0,non_bos=pb!=0,early=(pb>0)&(pb<=8),general=pb>8,all=np.ones(len(xb),bool))
        xqs={sx:quantize_input(xb,sx) for sx in scales}
        for (sx,s10),state in unique.items():
            state['input_clip']+=int((np.abs(xb)>127*sx).sum());state['zero']+=int((xqs[sx]==0).sum())
        for ns in range(0,n,n_chunk):
            sl=slice(ns,min(ns+n_chunk,n));wb=np.asarray(w[sl],np.float64);qw=wq[sl];sc=sw[sl]
            ref=dot(xb,wb);weight=dot(xb,qw.astype(np.float64)*sc[:,None])
            for sx,q in xqs.items():
                acc=integer_dot(q,qw,'fp64-exact');pair=acc*(sx*sc);input_y=dot(q.astype(np.float64)*sx,wb)
                for (psx,s10),state in unique.items():
                    if psx!=sx:continue
                    p=state['params'];raw=profile.apply(acc,p['multiplier'][sl],p['shift'][sl],saturate=False)
                    hw=np.clip(raw,-512,511)*s10;ideal=np.clip(np.rint(acc*(sx*sc/s10)),-512,511)*s10
                    clipped=(raw < -512)|(raw > 511)
                    for group,mask in masks.items():
                        if not mask.any():continue
                        arrays=[ref,input_y,weight,pair,ideal,hw]
                        reducer.add(state['metrics'][group],*[a if group=='all' else a[mask] for a in arrays])
                        state['clips'][group]+=int(clipped[mask].sum());state['counts'][group]+=int(mask.sum())*len(wb)
    results=[]
    for policy in policies:
        state=unique[policy['s_X'],policy['s_10']]
        row=dict(policy,activation_clip_rate=state['input_clip']/x.size,activation_zero_rate=state['zero']/x.size,
                 int10_clip_rate=state['clips']['all']/state['counts']['all'],
                 ratio_max_relative_error=float(state['params']['relative_error'].max()),
                 bad_ratio_channels=int((state['params']['status']!='ok').sum()))
        for group in groups:
            row[group+'_int10_clip_rate']=state['clips'][group]/state['counts'][group] if state['counts'][group] else None
            for path,metric in state['metrics'][group].items():
                values=metric.result() if metric.count else dict(mse=None,nmse=None)
                for key in ['mse','nmse']:row[group+'_'+path+'_'+key]=values[key]
        results.append(row)
    return results


def materialize(out):
    data=[json.loads(p.read_text()) for p in sorted((out/'results').glob('*.json'))]
    searches=[];evaluations=[]
    for d in data:
        for tag,search in d['searches'].items():
            for c in search['candidates']:
                searches.append(dict(layer=d['layer'],module=d['module'],policy=tag,search_rows=search['sampling']['retained'],
                                     search_BOS_rows=search['sampling']['bos_rows'],selected=c==search['selected'],**c))
        for r in d['evaluation']:
            evaluations.append(dict(layer=d['layer'],module=d['module'],**r))
    write_csv(out.parent/'bos_aware_scale_search.csv',searches)
    write_csv(out.parent/'bos_aware_evaluation.csv',evaluations)
    sensitivity=[];comparison=[]
    for d in data:
        if d['module']!='down_proj':continue
        old={r['split']:r for r in d['evaluation'] if r['policy']=='historical'}
        for tag,search in d['searches'].items():
            for split in ['calibration','validation']:
                r=next(r for r in d['evaluation'] if r['policy']==tag and r['split']==split)
                sensitivity.append(dict(layer=d['layer'],policy=tag,split=split,search_rows=search['sampling']['retained'],
                    bos_rows=search['sampling']['bos_rows'],early_rows=search['sampling']['early_rows'],general_rows=search['sampling']['general_rows'],
                    method=r['method'],threshold=r['threshold'],s_X=r['s_X'],s_10=r['s_10'],
                    mse=r['all_final_local_total_mse'],nmse=r['all_final_local_total_nmse'],
                    bos_nmse=r['bos_final_local_total_nmse'],non_bos_nmse=r['non_bos_final_local_total_nmse'],
                    search_seconds=search['seconds'],sample_activation_bytes=search['sampling']['retained']*d['K']*4))
                if tag=='stratified_64':
                    comparison.append(dict(layer=d['layer'],split=split,old_s_X=old[split]['s_X'],new_s_X=r['s_X'],
                        old_s_10=old[split]['s_10'],new_s_10=r['s_10'],selected_threshold_method=r['method'],
                        selected_threshold=r['threshold'],search_BOS_rows=search['sampling']['bos_rows'],
                        old_activation_clip_rate=old[split]['activation_clip_rate'],new_activation_clip_rate=r['activation_clip_rate'],
                        old_activation_zero_rate=old[split]['activation_zero_rate'],new_activation_zero_rate=r['activation_zero_rate'],
                        old_INT10_clip_rate=old[split]['int10_clip_rate'],new_INT10_clip_rate=r['int10_clip_rate'],
                        old_NMSE=old[split]['all_final_local_total_nmse'],new_NMSE=r['all_final_local_total_nmse'],
                        BOS_NMSE=r['bos_final_local_total_nmse'],non_BOS_NMSE=r['non_bos_final_local_total_nmse']))
    write_csv(out.parent/'search_row_sensitivity.csv',sensitivity)
    write_csv(out.parent/'down_proj_old_vs_new.csv',comparison)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    p.add_argument('--layers',default='all');p.add_argument('--modules',default='down_proj')
    p.add_argument('--cache-dir',default='/tmp/bos-aware-capture');p.add_argument('--output-dir',default='diagnostics/bos_aware')
    p.add_argument('--artifacts',default='calibration_outputs/gguf-all-126-oasst1');p.add_argument('--data-dir',default='calibration_outputs/data-oasst1-v1')
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);art=Path(args.artifacts)
    manifest=json.loads((art/'manifest.json').read_text());profile=Profile();configure_blas('accelerate')
    if profile.metadata()!=manifest['hardware_profile']:raise ValueError('profile identity mismatch')
    import gguf,torch,transformers
    for key,actual in [('torch',torch.__version__),('transformers',transformers.__version__),('numpy',np.__version__)]:
        if actual!=manifest['library_versions'][key]:raise ValueError('runtime version mismatch')
    sources=list((ROOT/'src/static_quant').glob('*.py'))+[Path(__file__)]+[ROOT/'scripts'/f for f in ['quant_diagnostic_math.py','quant_diagnostic_reduce.py','quant_metric_reduce.c','quant_diagnostic_source.py']]
    identity=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model_sha256=sha256_file(args.model),manifest_sha256=sha256_file(art/'manifest.json'),
        source_hashes={str(f.relative_to(ROOT)):sha256_file(f) for f in sources},seed=42,
        thresholds='exact full calibration percentiles',row_counts=[64,128,256],bos_quota=16,early_quota=16,early_limit=8,
        objective='population-weighted final output MSE',integer_backend='bounded fp64-exact with NumPy INT64 oracle checks',
        datasets=manifest['datasets'],versions=manifest['library_versions'])
    if identity['model_sha256']!=manifest['gguf']['sha256']:raise ValueError('model hash mismatch')
    ip=out/'provenance.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('resume identity mismatch')
    write_json(ip,identity)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,time=time.time()))+'\n')
    names=select_names(18,args.layers,args.modules);entries={e['module_name']:e for e in manifest['modules']}
    pending=[name for name in names if not (out/'results'/(name+'.json')).exists()]
    if not pending:materialize(out);return
    reader=gguf.GGUFReader(args.model);positions=token_positions(reader,manifest,Path(args.data_dir));tensors={t.name:t for t in reader.tensors}
    gguf_names=dict(q_proj='attn_q',k_proj='attn_k',v_proj='attn_v',o_proj='attn_output',gate_proj='ffn_gate',up_proj='ffn_up',down_proj='ffn_down')
    for name in pending:
        started=time.monotonic();layer=int(name.split('.')[2]);module=name.split('.')[-1];entry=entries[name]
        directory=art/Path(entry['module_manifest']).parent
        mm=json.loads((directory/'manifest.json').read_text())
        for file,digest in mm['artifact_sha256'].items():
            if sha256_file(directory/file)!=digest:raise ValueError('original artifact hash mismatch')
        stored=json.loads((art/entry['report_file']).read_text())['modules'][name]
        t=tensors[f'blk.{layer}.{gguf_names[module]}.weight']
        w=gguf.quants.dequantize(t.data,t.tensor_type).reshape(tuple(reversed(t.shape)))
        wq,sw=quantize_weight(w)
        np.testing.assert_array_equal(wq,np.load(art/entry['int8_weight_file'],mmap_mode='r'));del wq
        with np.load(directory/'scales.npz') as z:np.testing.assert_array_equal(sw,z['op0000.s_W'])
        captures={}
        for split in ['calibration','validation']:
            cache=Path(args.cache_dir)/str(layer)/split
            marker=json.loads((cache/'identity.json').read_text())
            expected=dict(model=identity['model_sha256'],data=manifest['datasets'][split]['file_sha256'],
                source=sha256_file(ROOT/'scripts/quant_diagnostic_source.py'),tokens=manifest['datasets'][split]['valid_tokens'])
            if marker!=expected:raise ValueError('capture identity mismatch')
            captures[split]=np.load(cache/(input_group(name)+'.npy'),mmap_mode='r')
            if len(captures[split])!=len(positions[split]):raise ValueError('position alignment mismatch')
        xc=captures['calibration'];absvalues=np.abs(xc).reshape(-1);amax=float(absvalues.max())
        percentiles=np.percentile(absvalues,DEFAULT_PERCENTILES,overwrite_input=True);del absvalues
        if not np.isclose(amax/127,entry['s_X'],rtol=2e-6):raise ValueError('calibration absmax mismatch')
        candidates=activation_candidates(amax,exact_percentiles=dict(zip(DEFAULT_PERCENTILES,percentiles)))
        baseline=stored['selection']['minmax_baseline'];searches={};policies=[dict(policy='historical',s_X=entry['s_X'],s_10=entry['s_10'],method='absmax',threshold=entry['s_X']*127)]
        configurations=[('stratified_64',64,True),('stratified_128',128,True),('stratified_256',256,True),('uniform_64',64,False)] if module=='down_proj' else [('stratified_64',64,True)]
        for tag,count,stratified in configurations:
            ts=time.monotonic();sample=RepresentativeRowSampler(count,42,representative=stratified)
            for start in range(0,len(xc),512):sample.add(xc[start:start+512],positions['calibration'][start:start+512])
            options=operation_options(name,Options(mode='mse',search_rows=count,seed=42,n_chunk=512))
            sx,s10,params,selection=joint_scale_search(lambda sl:w[sl],w.shape,sw,sample,candidates,baseline,profile,options,
                accumulator=lambda a,b:integer_dot(a,b,'fp64-exact'),matmul=dot)
            selection['seconds']=time.monotonic()-ts;searches[tag]=selection
            policies.append(dict(policy=tag,s_X=sx,s_10=s10,method=selection['selected']['method'],threshold=selection['selected']['threshold']))
            print(name,tag,selection['selected']['method'],sx,s10,selection['seconds'],flush=True)
            if tag=='stratified_64' and module=='down_proj':
                # Output-only control at unchanged absmax activation threshold.
                eligible=[r for r in selection['candidates'] if r['method']=='absmax' and r['bad_ratio_channels']==0]
                chosen=min(eligible,key=lambda r:r['mse'])
                policies.append(dict(policy='stratified_64_absmax',s_X=chosen['s_X'],s_10=chosen['s10'],method='absmax',threshold=chosen['threshold']))
        # Freeze all choices before touching validation values in evaluation.
        write_json(out/'selected'/(name+'.json'),dict(layer=layer,module=module,policies=policies,searches=searches))
        results=[]
        for split in ['calibration','validation']:
            print(name,'evaluate',split,flush=True)
            measured=metrics_for_policies(captures[split],positions[split],w,sw,policies,profile)
            old=measured[0]['all_final_local_total_mse'];expected=stored[split]['local_total']['mse']
            if not np.isclose(old,expected,rtol=2e-5,atol=1e-10):raise ValueError('historical MSE failed reproduction')
            results.extend(dict(r,split=split) for r in measured)
        result=dict(layer=layer,module=module,K=w.shape[1],N=w.shape[0],searches=searches,evaluation=results,seconds=time.monotonic()-started)
        write_json(out/'results'/(name+'.json'),result);materialize(out)
        print('DONE',name,result['seconds'],flush=True)


if __name__=='__main__':main()
