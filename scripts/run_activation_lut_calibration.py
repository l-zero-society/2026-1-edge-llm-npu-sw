"""Shared signed INT10 -> signed INT8 GELUTanh pilot; frozen down_proj."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import pilot_absmax_calibration as reuse
cal, base, row, final = reuse.cal, reuse.base, reuse.row, reuse.final
OUT = ROOT/'diagnostics/activation_lut_calibration_pilot'
WORK = OUT/'work'
PROGRESS = ROOT/'activation_lut_calibration_progress.txt'
CANON = ROOT/'calibration/frozen_linear_quant_v1'
SEED = 20261004
GROUPS = ('prefill', 'decode')
PERCENTILES = [99, 99.5, 99.9, 99.95, 99.99, 100]
MODES = ('FP_FULL', 'FROZEN_LINEAR_FP_ACT', 'FROZEN_LINEAR_SHARED_LUT')
START = time.monotonic()

def progress(message):
    with PROGRESS.open('a') as f:
        f.write(message+'\n'); f.flush()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def frozen_hashes():
    return {p.name: base.sha256_file(p) for p in sorted(CANON.glob('*.json'))}

def verify_hashes(before, after):
    if before != after: raise RuntimeError('frozen Linear identity changed')

def checkpoint(path, identity):
    if not path.exists(): return None
    value = json.loads(path.read_text())
    if value['identity'] != identity: raise RuntimeError('checkpoint fingerprint mismatch')
    return value['value']

def gelu(x):
    import torch
    a = np.asarray(x, np.float64)
    return torch.nn.functional.gelu(torch.from_numpy(a.copy()), approximate='tanh').numpy()

def qinput(x, scale):
    if not np.isfinite(scale) or scale <= 0: raise ValueError('invalid scale')
    return np.clip(np.rint(np.asarray(x)/scale), -512, 511).astype(np.int16)

def address(q):
    q = np.asarray(q)
    if np.any(q < -512) or np.any(q > 511): raise ValueError('invalid INT10 code')
    return q.astype(np.int64)+512

def qoutput(x, scale):
    if not np.isfinite(scale) or scale <= 0: raise ValueError('invalid scale')
    return np.clip(np.rint(np.asarray(x)/scale), -127, 127).astype(np.int8)

def table(sl, sa):
    return qoutput(gelu(np.arange(-512, 512)*sl), sa)

def scales(values, maximum):
    values = np.abs(np.asarray(values)).reshape(-1)
    return sorted(set(float(v)/maximum for v in np.percentile(values, PERCENTILES) if v > 0)) or [1.0]

def metric(value, ref):
    m = base.Metric(); m.add(value, ref); return m.result()

def balanced(rows, key):
    return float(np.mean([np.mean([r[key] for r in rows if r['layer']==layer])
                          for layer in sorted({r['layer'] for r in rows})]))

def gate(old, new):
    checks = dict(kl_pass=new['kl']<=1.05*old['kl'], nmse_pass=new['nmse']<=1.05*old['nmse'],
                  top1_pass=new['top1']>=old['top1']-.01)
    return dict(checks, passed=all(checks.values()))

def identity(e):
    return {k:e[k] for k in ('index','anchor_id','response_id','tree_id')}

def load_specs(source, previous):
    frozen,_ = final.restore_final(source, previous)
    selected = json.loads((CANON/'parameters.json').read_text())['layers']
    if set(selected)!=set(frozen): raise RuntimeError('module set mismatch')
    for name, item in selected.items():
        old = frozen[name]
        # Copy stored M/S verbatim; no approximation or parameter selection here.
        params = dict(old['params'], multiplier=np.asarray(item['multiplier'],dtype=old['params']['multiplier'].dtype),
                      shift=np.asarray(item['shift'],dtype=old['params']['shift'].dtype))
        frozen[name] = dict(old, sx=item['sx'], s10=item['s10'], params=params)
    return frozen

@contextlib.contextmanager
def activation_patch(source, sl=None, sa=None, lut=None):
    originals=[]
    try:
        if lut is not None:
            for layer in source.model.model.layers:
                act=layer.mlp.act_fn; originals.append((act,act.forward))
                def forward(x):
                    q=qinput(x.detach().numpy(),sl)
                    result=(lut[address(q)].astype(np.float64)*sa).astype(np.float32)
                    return source.torch.from_numpy(result).to(dtype=x.dtype,device=x.device)
                act.forward=forward
        yield
    finally:
        for act, original in originals: act.forward=original

def capture(source, specs, examples):
    reservoirs={l:{g:cal.Reservoir(64,SEED) for g in GROUPS} for l in range(18)}
    state={}; pending={}; handles=[]
    for l, block in enumerate(source.model.model.layers):
        def gate_hook(module,args,out,l=l): pending[l]=out.detach().numpy()[0].copy()
        def up_hook(module,args,out,l=l):
            g=pending.pop(l); u=out.detach().numpy()[0]
            for p,(gr,ur) in enumerate(zip(g,u)):
                reservoirs[l][state['phase']].add(np.concatenate([gr,ur]),l,state['phase'],state['conversation'],state['offset']+p)
        handles += [block.mlp.gate_proj.register_forward_hook(gate_hook),block.mlp.up_proj.register_forward_hook(up_hook)]
    try:
        with patch.object(row,'select_rows',reuse.select_absmax):
            for number,e in enumerate(examples,1):
                cache=None
                for t in range(min(16,len(e['targets']))):
                    state.update(phase='prefill' if t==0 else 'decode',conversation=e['index'],
                                 offset=0 if t==0 else len(e['prompt_ids'])+t-1)
                    ids=e['prompt_ids'] if t==0 else [e['targets'][t-1]]
                    _,_,cache=cal.run_mode(source,specs,ids,cache,layers=False)
                progress(f'CAPTURE {number}/12 DONE elapsed_sec={time.monotonic()-START:.3f}')
    finally:
        for h in handles:h.remove()
    return {f'{l}_{g}':reservoirs[l][g].arrays()[0] for l in range(18) for g in GROUPS}

def candidate_rows(captured,sl,sa):
    lut=table(sl,sa); result=[]
    for key,data in captured.items():
        l,g=key.split('_'); gate_x,up=np.split(data.astype(np.float64),2,axis=1)
        q=qinput(gate_x,sl); a=gelu(gate_x); aq=gelu(q*sl); codes=lut[address(q)]; al=codes.astype(np.float64)*sa
        r=dict(layer=int(l),group=g,rows=len(data),inputs=gate_x.size,
               input_clip_rate=float(np.mean((gate_x/sl < -512)|(gate_x/sl > 511))),
               boundary_hit_rate=float(np.mean((q==-512)|(q==511))),
               saturation_rate=float(np.mean(np.abs(np.rint(aq/sa))>127)),
               zero_fraction=float(np.mean(codes==0)),code_utilization=len(np.unique(codes))/255,
               min_code=int(codes.min()),max_code=int(codes.max()))
        for tag,value,ref in [('input',aq,a),('table',al,aq),('activation',al,a),('product',al*up,a*up)]:
            r.update({tag+'_'+k:v for k,v in metric(value,ref).items()})
        r['downproj_input_nmse']=r['product_nmse'];result.append(r)
    return result

def search(captured):
    pooled=np.concatenate([np.abs(np.split(a,2,axis=1)[0]).reshape(-1) for a in captured.values()])
    sls=scales(pooled,511); candidates=[]; details={}
    for sl in sls:
        # Distribution is pooled over captured addresses, not uniformly over the table.
        corresponding=gelu(qinput(pooled,sl)*sl)
        # Preserve negative gate values when deriving the actual output distribution.
        signed=np.concatenate([np.split(a,2,axis=1)[0].reshape(-1) for a in captured.values()])
        corresponding=gelu(qinput(signed,sl)*sl); del signed
        for sa in scales(corresponding,127):
            cid=len(candidates); rows=candidate_rows(captured,sl,sa);details[str(cid)]=rows
            layer_scores=[np.mean([r['product_nmse'] for r in rows if r['layer']==l]) for l in range(18)]
            candidates.append(dict(candidate=cid,sL=sl,sA=sa,score=balanced(rows,'product_nmse'),
                worst_layer_product_nmse=float(max(layer_scores)),input_clip_rate=balanced(rows,'input_clip_rate'),
                saturation_rate=balanced(rows,'saturation_rate'),activation_nmse=balanced(rows,'activation_nmse')))
    candidates.sort(key=lambda c:(c['score'],c['worst_layer_product_nmse'],c['input_clip_rate'],c['saturation_rate'],-c['sL'],-c['sA']))
    return candidates,details

def downstream(captured,specs,candidates,details):
    with patch.object(row,'select_rows',reuse.select_absmax):
        for key,data in captured.items():
            l,g=key.split('_'); x,u=np.split(data.astype(np.float64),2,axis=1)
            spec=specs[f'model.layers.{l}.mlp.down_proj']
            ref=cal.quantized_one((gelu(x)*u)[None],spec)[0]
            for c in candidates[:3]:
                lut=table(c['sL'],c['sA']); a=lut[address(qinput(x,c['sL']))]*c['sA']
                out=cal.quantized_one((a*u)[None],spec)[0]
                target=next(r for r in details[str(c['candidate'])] if r['layer']==int(l) and r['group']==g)
                target.update({'downstream_'+k:v for k,v in metric(out,ref).items()})
        for c in candidates[:3]: c['downstream_nmse']=balanced(details[str(c['candidate'])],'downstream_nmse')

def evaluate(source,specs,e,selected):
    lut=table(selected['sL'],selected['sA']);scores={m:reuse.response.Scores() for m in MODES};caches={m:None for m in MODES}
    for t,target in enumerate(e['targets'][:24]):
        ids=e['prompt_ids'] if t==0 else [e['targets'][t-1]];logits={}
        for mode in MODES:
            if mode=='FP_FULL': logits[mode],_,caches[mode]=cal.run_fp(source,ids,caches[mode],layers=False)
            else:
                with patch.object(row,'select_rows',reuse.select_absmax), activation_patch(source,selected['sL'],selected['sA'],lut if mode.endswith('SHARED_LUT') else None):
                    logits[mode],_,caches[mode]=cal.run_mode(source,specs,ids,caches[mode],layers=False)
        if len({id(v) for v in caches.values()})!=3:raise RuntimeError('KV cache alias')
        for mode in MODES:scores[mode].add(source.torch,logits[mode],logits['FP_FULL'],[target])
    return {m:s.raw() for m,s in scores.items()}

def main():
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);progress('RUN_RESUME_AFTER_CONTRACT')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='lut_cali':raise RuntimeError('wrong branch')
    before=frozen_hashes(); discovery=json.loads((OUT/'discovery.json').read_text())
    discovery['resolution']=dict(source='explicit user architecture contract',fp16_expansion_required=False,
        lut_input_dtype='signed_int10',lut_input_codes=[-512,511],lut_depth=1024,address_mapping='q + 512',
        activation='gelu_tanh',lut_output_dtype='signed_int8',lut_output_codes=[-127,127],
        input_scale_policy='global_common_sL',output_scale_policy='global_common_sA',
        table_sharing='one_common_table_image_all_layers_and_lanes',interpolation=False)
    discovery['status']='RESOLVED_BY_EXPLICIT_CONTRACT';base.write_json(OUT/'discovery.json',discovery)
    progress('CONTRACT_APPLIED');progress('UNIT_TEST_START')
    tests=[sys.executable,'-m','unittest','tests.test_activation_lut_calibration','tests.test_frozen_linear_quant_v1']
    with (OUT/'unit_test.log').open('w') as f:
        code=subprocess.run(tests,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPYCACHEPREFIX='/tmp/activation-lut')).returncode
    if code:progress('UNIT_TEST_FAILED');raise RuntimeError('unit tests failed')
    progress('UNIT_TEST_DONE')
    source,manifest,_,previous,_,_=final.load_source_and_policies();specs=load_specs(source,previous)
    param_fp=reuse.compare.fingerprint(specs)
    excluded=json.loads((ROOT/'diagnostics/frozen_cross_layer_confirmation/confirmation_dataset_manifest.json').read_text())['examples']
    excluded_ids={e[k] for e in excluded for k in ('anchor_id','response_id','tree_id') if e.get(k)}
    def examples(split,count,targets):
        pairs,_=cal.build_pairs(split);es,_=cal.tokenize_pairs(source,pairs)
        es=[e for e in es if len(e['targets'])>=targets and not any(e[k] in excluded_ids for k in ('anchor_id','response_id','tree_id'))][:count]
        if len(es)!=count:raise RuntimeError('insufficient non-confirmation examples')
        return es
    capture_examples=examples('calibration',12,1);e2e_examples=examples('validation',8,24)
    cm=dict(examples=[identity(e) for e in capture_examples],response_limit=16,reservoir_capacity=64,seed=SEED,trajectory='frozen_linear_quant_v1 + FP GELUTanh')
    em=dict(examples=[identity(e) for e in e2e_examples],targets=192,role='selection only')
    base.write_json(OUT/'capture_manifest.json',cm);base.write_json(OUT/'e2e_dataset_manifest.json',em)
    run_id=digest(dict(frozen=before,parameters=param_fp,model=manifest['gguf'],capture=cm,e2e=em,
        script=base.sha256_file(__file__),activation='torch.nn.functional.gelu:tanh',contract=discovery['resolution']))
    progress('CAPTURE_START');meta=WORK/'capture.json'
    saved=checkpoint(meta,run_id)
    if saved is None:
        captured=capture(source,specs,capture_examples);np.savez_compressed(WORK/'capture.npz',**captured)
        base.write_json(meta,dict(identity=run_id,value=dict(sha256=base.sha256_file(WORK/'capture.npz'))))
    else:
        if saved['sha256']!=base.sha256_file(WORK/'capture.npz'):raise RuntimeError('capture hash mismatch')
        with np.load(WORK/'capture.npz') as z:captured={k:z[k] for k in z.files}
    progress(f'CAPTURE_DONE samples={sum(len(x) for x in captured.values())}')
    progress('GLOBAL_SEARCH_START');cached=checkpoint(WORK/'search.json',run_id)
    if cached is None:
        candidates,details=search(captured);base.write_json(WORK/'search.json',dict(identity=run_id,value=dict(candidates=candidates,details=details)))
    else:candidates,details=cached['candidates'],cached['details']
    progress(f'GLOBAL_SEARCH_DONE candidates={len(candidates)}');progress('DOWNPROJ_LOCAL_START')
    cached=checkpoint(WORK/'downstream.json',run_id)
    if cached is None:
        downstream(captured,specs,candidates,details);base.write_json(WORK/'downstream.json',dict(identity=run_id,value=dict(candidates=candidates,details=details)))
    else:candidates,details=cached['candidates'],cached['details']
    progress('DOWNPROJ_LOCAL_DONE candidates=3')
    selected=min(candidates[:3],key=lambda c:(c['downstream_nmse'],c['score'],c['worst_layer_product_nmse'],c['input_clip_rate'],c['saturation_rate'],-c['sL'],-c['sA']))
    base.write_csv(OUT/'global_candidates.csv',[dict(c,downstream_nmse=c.get('downstream_nmse')) for c in candidates])
    base.write_json(OUT/'global_selection.json',dict(selected=selected,shortlist=candidates[:3],policy='product-balanced top3, then frozen-downproj-balanced NMSE; no E2E reselection',refinement=False))
    base.write_csv(OUT/'layer_summary.csv',details[str(selected['candidate'])])
    oracle=[]
    for l in range(18):
        values=[(float(np.mean([r['product_nmse'] for r in details[str(c['candidate'])] if r['layer']==l])),c['candidate']) for c in candidates]
        score,cid=min(values);shared=float(np.mean([r['product_nmse'] for r in details[str(selected['candidate'])] if r['layer']==l]))
        oracle.append(dict(layer=l,candidate=cid,score=score,shared_score=shared,gap=shared-score))
    base.write_json(OUT/'oracle_summary.json',dict(layers=oracle,shared_score=selected['score'],mean_oracle_score=float(np.mean([x['score'] for x in oracle])),worst_layer_gap=max(x['gap'] for x in oracle)))
    tables=OUT/'tables';tables.mkdir(exist_ok=True);lut=table(selected['sL'],selected['sA']);(tables/'gelu_lut.bin').write_bytes(lut.tobytes())
    base.write_json(tables/'gelu_lut.json',dict(sL=selected['sL'],sA=selected['sA'],depth=1024,dtype='int8',range=[-127,127],address='q+512',sha256=base.sha256_file(tables/'gelu_lut.bin'),status='PILOT'))
    progress('E2E_START modes=3 sequences=8 targets=192');records=[]
    for i,e in enumerate(e2e_examples):
        key=digest(dict(run=run_id,selected=selected,example=identity(e)));path=WORK/f'e2e_{i:02d}.json';record=checkpoint(path,key)
        if record is None:
            record=evaluate(source,specs,e,selected);base.write_json(path,dict(identity=key,value=record))
        records.append(record);progress(f'SEQUENCE {i+1}/8 DONE elapsed_sec={time.monotonic()-START:.3f}')
    progress('E2E_DONE');metrics={m:reuse.response.merged([r[m] for r in records]) for m in MODES}
    if any(m['tokens']!=192 for m in metrics.values()):raise RuntimeError('target count')
    old,new=metrics[MODES[1]],metrics[MODES[2]];check=gate(old,new)
    deltas={k:dict(absolute=new[k]-old[k],relative=(new[k]-old[k])/old[k] if old[k] else None) for k in ('kl','nmse','ppl','top1')}
    summary=dict(modes=metrics,deltas=deltas,gate=check,classification='PILOT_PASS' if check['passed'] else 'PILOT_REGRESSION')
    base.write_json(OUT/'e2e_summary.json',summary)
    after=frozen_hashes();verify_hashes(before,after)
    if reuse.compare.fingerprint(specs)!=param_fp:raise RuntimeError('Linear parameters mutated')
    verification=dict(frozen_linear_before=before,frozen_linear_after=after,frozen_linear_unchanged=True,
        activation_function='gelu_tanh',activation_implementation='torch.nn.functional.gelu(approximate=tanh)',
        previous_fp16_lut_found=False,previous_fp16_lut_reference=None,lut_input_semantics_source='explicit user contract',
        lut_table_depth=1024,lut_output_dtype='signed_int8',lut_output_bits=8,capture_dataset_fingerprint=digest(cm),e2e_dataset_fingerprint=digest(em),
        final_linear_confirmation_overlap=False,linear_recalibration_performed=False,sX_modified=False,s10_modified=False,RS_modified=False,
        model_inference_per_candidate=False,local_candidates_evaluated=len(candidates),downstream_candidates=3,e2e_modes_evaluated=list(MODES),
        independent_kv_caches=True,test_command=tests,test_return_code=code,elapsed_sec=time.monotonic()-START)
    base.write_json(OUT/'verification.json',verification)
    lines=['# Shared GELU Activation LUT pilot','','## 1. Scope','frozen_linear_quant_v1 was not modified. This is a pilot, not a final freeze.',
        '## 2. Activation graph','gate=gate_proj(x); up=up_proj(x); act=GELUTanh(gate); h=act*up; y=down_proj(h).',
        '## 3. Previous FP16 reference','No previous FP16 expansion implementation was located. The explicit architecture contract resolved this; no legacy unsigned LUT was used.',
        '## 4. Integer LUT contract','Global sL; signed INT10 input [-512,511]; address=q+512; 1024 signed INT8 entries [-127,127]; global sA; RNE; no interpolation. One table shared by all layers and lanes.',
        f'Selected sL={selected["sL"]!r}, sA={selected["sA"]!r}.',
        '## 5. Data','12 deterministic previously available calibration chats; at most 16 response positions and 64 paired rows/group/layer. E2E selection: 8 chats × 24 targets. The 1024-target confirmation identities were excluded.',
        '## 6. Search',f'{len(candidates)} coarse percentile pairs; no refinement. Equal weight per layer and available prefill/decode group. Product NMSE ranks top three; frozen down_proj output NMSE selects among them before E2E.',
        '## 7. Per-layer local results','See layer_summary.csv for each group and global_candidates.csv for the full coarse pool.',
        '## 8. Error decomposition']
    rr=details[str(selected['candidate'])]
    for k in ('input_nmse','table_nmse','activation_nmse','product_nmse','downstream_nmse'):
        lines.append(f'- Balanced {k}: {balanced(rr,k):.9g}')
    lines+=['Input error compares GELU(Q10(g)) to GELU(g); table error compares reconstructed LUT output to GELU(Q10(g)); combined error compares LUT to GELU(g). Downstream reference uses the same frozen quantized down_proj on the exact-activation product.',
        '## 9. E2E comparison','| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in metrics.items():lines.append('| '+m+' | '+' | '.join(f'{v[k]:.9g}' for k in ('nll','ppl','kl','nmse','mse','mae','cosine','flattened_cosine','top1','in5','overlap'))+' |')
    lines+=['',f'Incremental LUT deltas: `{json.dumps(deltas)}`.','## 10. FP16 comparison','Not applicable under the supplied direct INT10 contract.',
        '## 11. Gate',f'Predeclared +5% KL/+5% NMSE/-1pp Top1 gate: **{summary["classification"]}**. `{json.dumps(check)}`.',
        '## 12. Limitations','This emulates future gate output INT10 quantization while gate/up MACs stay FP. Only down_proj is quantized. Selection data are not unseen confirmation; no final adoption is implied.',
        '## 13. Next experiment recommendation','Review the recorded input/table/product/downstream decomposition and the fixed pilot gate before designing any larger experiment. No follow-up was run.']
    report = '\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n'
    (OUT/'report.md').write_text(report)
    progress('ARTIFACT_WRITE_DONE');progress('FINAL_VERIFY_DONE');progress('RUN_COMPLETE')

if __name__=='__main__':
    try:main()
    except BaseException:progress('RUN_FAILED');raise
