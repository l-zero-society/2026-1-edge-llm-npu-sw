"""Actual-activation RMSNorm debugging; background-owned execution and publication."""
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_vpu10_e2e_pilot as old
w = old.width
base, cal, final, response, mixed = w.base, w.cal, w.final, w.response, w.mixed
OUT = ROOT / 'diagnostics/norm10_debug_pilot'
WORK = OUT / 'work'
STAGE = 'INITIALIZING'
SOURCE_HEAD = None
SOURCE_HASHES = {}
START = time.monotonic()
MODES = ('FP_FULL', 'GPALU10_FP_VPU2', 'GPALU10_NORM10_OLD_EMULATOR',
         'GPALU10_NORM10_RTL_EXACT_COEFF16')
ARTIFACTS = ('baseline_manifest.json', 'rtl_exact_trace.json', 'old_vs_rtl_exact.csv',
             'norm_scale_drift.csv', 'single_norm_insertion.csv', 'cumulative_norm_insertion.csv',
             'first_divergence.json', 'first_divergence_recalibration.json', 'e2e_summary.json',
             'verification.json', 'report.md', 'unit_test.log', 'run.log')
SOURCE_FILES = ('scripts/run_norm10_debug_pilot.py', 'scripts/launch_norm10_debug_pilot.sh',
                'tests/test_norm10_debug_pilot.py')

def save(name, value): old.atomic_json(OUT / name, value)
def stage(name):
    global STAGE
    STAGE = name
    save('status.json', dict(state='RUNNING', stage=name, elapsed_seconds=time.monotonic()-START))
    print(name, flush=True)
def write_csv(name, rows):
    if not rows: raise RuntimeError('empty artifact: '+name)
    with (OUT/name).open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
def inv_n(n): return int(math.floor((1 << 20)/n + .5))
def epsilon_q(eps, scale): return int(np.rint(eps / (scale*scale)))
def mean_sq(q): return (sum(int(x)*int(x) for x in q) * inv_n(len(q))) >> 20

def rsqrt_table(frac=14):
    # Scala math.round positive values = floor(x+0.5).
    m=np.maximum(np.arange(1024),512)/512.
    return np.floor((1/np.sqrt(m))*(1<<frac)+.5).astype(np.int64)

def corrected_scale(raw, exponent, frac=14):
    odd=int(math.floor((1/math.sqrt(2))*(1<<frac)+.5))
    return ((int(raw)*odd)>>frac if exponent&1 else int(raw)) >> (exponent>>1)

def phase2(q, scale, frac=14):
    product=np.asarray(q,np.int64)*int(scale)
    raw=product>>frac
    return np.clip(raw,-512,511).astype(np.int16), product, raw

def cosine(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else float(np.array_equal(a,b))

def reference(x, weight, eps):
    x=np.asarray(x,np.float64)
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*(1+weight)

def exact(x, spec, weight, frac=14, trace=False):
    """Literal requested RTL core. Gamma remains an explicit FP model adapter.

    No hidden 1/s_out gain is inserted into the RTL scale. The core returns
    q_out*s_out, then Gemma's (1+weight) is applied outside the RTL unit.
    This distinction is deliberately measured, not silently repaired.
    """
    x=np.asarray(x,np.float64);shape=x.shape;rows=x.reshape(-1,shape[-1])
    q,iclip=old.quantize10(rows,spec['input_scale']);table=rsqrt_table(frac)
    eq=epsilon_q(spec['eps'],spec['input_scale']);output=[];traces=[];clipped=0
    for r,(xr,qr) in enumerate(zip(rows,q)):
        ss=sum(int(a)*int(a) for a in qr);ms=(ss*inv_n(len(qr)))>>20
        stat=ms+eq
        if stat >= (1<<old.accumulator_bits()): raise RuntimeError('stat overflow')
        idx,exp=old.normalized_index(stat);raw_scale=int(table[idx]);adj=corrected_scale(raw_scale,exp,frac)
        code,product,raw=phase2(qr,adj,frac)
        clipped+=int(np.count_nonzero((raw< -512)|(raw>511)))
        y=code.astype(np.float64)*spec['output_scale']*(1+weight)
        output.append(y)
        if trace:
            fp=reference(xr,weight,spec['eps']);ideal=1/math.sqrt(stat) if stat else None
            traces.append(dict(row_id=r,N=len(qr),input_fp_absmax=float(abs(xr).max()),s_in=spec['input_scale'],s_out=spec['output_scale'],
                q_input_min=int(qr.min()),q_input_max=int(qr.max()),input_clipping_count=int(np.count_nonzero((np.rint(xr/spec['input_scale'])< -512)|(np.rint(xr/spec['input_scale'])>511))),
                input_clipping_rate=float(np.mean((np.rint(xr/spec['input_scale'])< -512)|(np.rint(xr/spec['input_scale'])>511))),
                sq_sum=ss,invN_Q=inv_n(len(qr)),meanSq=ms,epsilon_fp=spec['eps'],epsilon_q=eq,stat=stat,
                normalized_index=idx,exponent=exp,lut_raw=raw_scale,adjusted_scale_integer=adj,adjusted_scale_real=adj/(1<<frac),
                exact_reference_scale=ideal,relative_scale_error=(adj/(1<<frac)-ideal)/ideal if ideal else None,
                phase2_product_min=int(product.min()),phase2_product_max=int(product.max()),output_code_min=int(code.min()),output_code_max=int(code.max()),
                output_clipping_count=int(np.count_nonzero((raw< -512)|(raw>511))),output_clipping_rate=float(np.mean((raw< -512)|(raw>511))),
                reconstructed_output_nmse=mixed.nmse(y,fp),cosine=cosine(y,fp),gamma_adapter='FP after RTL core reconstruction'))
    return np.stack(output).reshape(shape),dict(input_clipping_rate=iclip,output_clipping_rate=clipped/q.size,traces=traces)

def mismatch(a,b,tol=1e-6): return mixed.nmse(a,b)>tol

def first_jump(rows):
    previous=rows[0]
    for row in rows[1:]:
        reasons=[]
        if row['nmse']>2*max(previous['nmse'],1e-12):reasons.append('logits_nmse_gt_2x')
        if previous['top1']-row['top1']>=.1-1e-12:reasons.append('top1_drop_ge_10pp')
        if row['input_clip']>1e-4 or row['output_clip']>1e-4:reasons.append('clip_gt_1e-4')
        if row['hidden_nmse']>10*max(previous['hidden_nmse'],1e-12):reasons.append('hidden_nmse_gt_10x')
        if reasons:return dict(found=True,module=row['newly_enabled_module'],number_of_int10_norms=row['number_of_int10_norms'],reasons=reasons)
        previous=row
    return dict(found=False,module=None)

def done_allowed(tests,report,pushed):return tests and report and pushed

class Experiment:
    def __init__(self,source,gate,up,luts,gpalu,specs):
        self.source=source;self.gate=gate;self.up=up;self.luts=luts;self.gpalu=gpalu;self.specs=specs
        self.names=old.module_names(source);self.modules=dict(source.model.named_modules())
        self.weights={n:self.modules[n].weight.detach().cpu().numpy().astype(np.float64) for n in self.names}
        self.samples={};self.telemetry={};self.group='prefill';self.label='';self.conversation=None;self.offset=0
    def observe(self,name,x,y,fp,details):
        # Deterministically retain only bounded actual rows, with their provenance.
        key=(self.label,name,self.group)
        stored=self.samples.setdefault(key,[])
        for i,row in enumerate(x.reshape(-1,x.shape[-1])):
            if len(stored)>=mixed.RESERVOIR:break
            stored.append((row.copy(),dict(conversation=self.conversation,position=self.offset+i,group=self.group)))
        key=(self.label,name)
        record=self.telemetry.setdefault(key,dict(n=0,ic=0.,oc=0.,error=0.,energy=0.,max_in=0.,max_out=0.))
        record['n']+=x.size;record['ic']+=details.get('input_clipping_rate',0)*x.size;record['oc']+=details.get('output_clipping_rate',0)*x.size
        record['error']+=float(np.square(y-fp).sum());record['energy']+=float(np.square(fp).sum())
        record['max_in']=max(record['max_in'],float(abs(x).max()));record['max_out']=max(record['max_out'],float(abs(fp).max()))
    @contextlib.contextmanager
    def patch(self,enabled=(),old_emulator=False,capture=False,gpalu=True):
        originals=[];enabled=set(enabled)
        with (w.patch_width(self.source,self.gate,self.up,self.luts,self.gpalu) if gpalu else contextlib.nullcontext()):
            try:
                for name in self.names:
                    if name not in enabled and not capture:continue
                    module=self.modules[name];original=module.forward;originals.append((module,original))
                    def forward(value,name=name,original=original):
                        x=value.detach().cpu().float().numpy();spec=self.specs[name];weight=self.weights[name]
                        fp=reference(x,weight,spec['eps'])
                        if name in enabled:
                            if old_emulator:y,details=old.rms_hw_numpy(x,spec['input_scale'],spec['output_scale'],weight,spec['eps'],old.norm_tables(14))
                            else:y,details=exact(x,spec,weight)
                            result=value.new_tensor(y)
                        else:
                            result=original(value);y=result.detach().cpu().float().numpy();q,clip=old.quantize10(x,spec['input_scale']);details=dict(input_clipping_rate=clip,output_clipping_rate=0)
                        if capture:self.observe(name,x,y,fp,details)
                        return result
                    module.forward=forward
                yield
            finally:
                for module,fn in originals:module.forward=fn
    def forward(self,ids,cache=None,layers=False):
        hidden,states,cache=response.forward_hidden(self.source,ids,[len(ids)-1],{},cache,True,layers)
        with self.source.torch.inference_mode():logits=self.source.model.lm_head(hidden)
        if layers: states['final_norm']=hidden.detach().clone()
        return logits,states,cache
    def diagnostic(self,example,enabled=(),label='',capture=False):
        self.label=label;self.conversation=example['index'];outputs=[];states=[];cache=None
        with self.patch(enabled,capture=capture):
            for t in range(2):
                self.group='prefill' if t==0 else 'decode';self.offset=0 if t==0 else len(example['prompt_ids'])+t-1
                logits,hidden,cache=self.forward(example['prompt_ids'] if t==0 else [example['targets'][t-1]],cache,True)
                outputs.append(logits);states.append(hidden)
        return outputs,states

def measure(exp,result,ref,example,module=None):
    scores=response.Scores()
    for t,(value,baseline) in enumerate(zip(result[0],ref[0])):scores.add(exp.source.torch,value,baseline,[example['targets'][t]])
    metrics=response.merged([scores.raw()])
    layer='final_norm' if module=='model.norm' else 17 if module is None else int(module.split('.')[2])
    metrics['hidden_nmse']=mixed.nmse(np.concatenate([x[layer].numpy() for x in result[1]]),np.concatenate([x[layer].numpy() for x in ref[1]]))
    return metrics

def load():
    source,*_=final.load_source_and_policies()
    if str(next(source.model.parameters()).device)!='cpu':raise RuntimeError('CPU required')
    historical=mixed.load_historical(source,json.loads((mixed.ART/'manifest.json').read_text()))
    specs=w.pot.load_current_specs(w.fusion.load_baseline_specs(source,historical));luts=w.pot.load_current_luts()
    gpalu={int(x['layer']):x for x in json.loads((old.GPALU/'int10/gpalu_parameters.json').read_text())['layers']}
    norm=json.loads((old.OUT/'norm10_coeff16_parameters.json').read_text())['specs']
    return Experiment(source,{i:specs[f'model.layers.{i}.mlp.gate_proj'] for i in range(18)},
                      {i:specs[f'model.layers.{i}.mlp.up_proj'] for i in range(18)},luts,gpalu,norm)

def trace_rows(exp,names,label):
    traces=[];comparisons=[]
    for name in names:
        for group in ('prefill','decode'):
            for x,identity in exp.samples.get((label,name,group),[])[:2]:
                spec=exp.specs[name];weight=exp.weights[name];fp=reference(x,weight,spec['eps'])
                new,d=exact(x[None],spec,weight,trace=True);prior,_=old.rms_hw_numpy(x[None],spec['input_scale'],spec['output_scale'],weight,spec['eps'],old.norm_tables(14))
                tr=d['traces'][0];tr.update(module=name,row_identity=identity);traces.append(tr)
                q,_=old.quantize10(x,spec['input_scale']);old_mean=float(np.mean(q.astype(np.int64)**2));old_stat=int(np.rint(old_mean+spec['eps']/spec['input_scale']**2))
                idx,e=old.normalized_index(old_stat);old_scale=old.adjust_rsqrt(old.norm_tables(14)['rsqrt'][idx],e,14)
                first='mean_square_rounding' if int(np.rint(old_mean))!=tr['meanSq'] else 'epsilon_rounding' if old_stat!=tr['stat'] else 'phase2_shift_and_output_scale_domain'
                comparisons.append(dict(module=name,row_id=json.dumps(identity,sort_keys=True),old_output_nmse_vs_fp=mixed.nmse(prior,fp),rtl_output_nmse_vs_fp=mixed.nmse(new,fp),
                    old_vs_rtl_output_nmse=mixed.nmse(prior,new),old_meanSq=old_mean,rtl_meanSq=tr['meanSq'],old_scale=old_scale,rtl_scale=tr['adjusted_scale_integer'],
                    old_epsilon_integer='not independently quantized',rtl_epsilon_integer=tr['epsilon_q'],first_differing_stage=first,
                    flag='EMULATOR_CONTRACT_MISMATCH' if mismatch(prior,new) else 'MATCH'))
    return traces,comparisons

def capture(exp):
    # Once, on the original FP predecessor path, same four chats/eight positions.
    for number,example in enumerate(mixed.calibration_examples(exp.source)):
        exp.label='calibration';exp.conversation=example['index'];cache=None
        with exp.patch(capture=True,gpalu=False):
            for t in range(min(mixed.CAL_LIMIT,len(example['targets']))):
                exp.group='prefill' if t==0 else 'decode';exp.offset=0 if t==0 else len(example['prompt_ids'])+t-1
                _,_,cache=exp.forward(example['prompt_ids'] if t==0 else [example['targets'][t-1]],cache)
    arrays={};ids={}
    for i,(key,rows) in enumerate(exp.samples.items()):
        arrays[str(i)]=np.stack([x for x,_ in rows]);ids[str(i)]=dict(key=list(key),identities=[ident for _,ident in rows])
    np.savez_compressed(WORK/'actual_capture.npz',**arrays)
    save('work/actual_capture_manifest.json',dict(samples=ids,calibration_positions=list(mixed.CAL_POSITIONS),response_limit=mixed.CAL_LIMIT,reservoir_per_group=mixed.RESERVOIR))

def telemetry_rates(exp,label):
    records=[r for (l,n),r in exp.telemetry.items() if l==label];n=sum(r['n'] for r in records)
    return (sum(r['ic'] for r in records)/n,sum(r['oc'] for r in records)/n) if n else (0.,0.)

def sweeps(exp,example):
    baseline=exp.diagnostic(example,label='diagnostic_fp_norm',capture=True)
    single=[];cumulative=[dict(number_of_int10_norms=0,newly_enabled_module='',nmse=0.,kl=0.,top1=1.,hidden_nmse=0.,input_clip=0.,output_clip=0.)]
    stage('SINGLE_NORM_SWEEP')
    for i,name in enumerate(exp.names):
        label=f'single_{i}';result=exp.diagnostic(example,[name],label,True);m=measure(exp,result,baseline,example,name);r=exp.telemetry[label,name]
        single.append(dict(module=name,local_output_nmse=r['error']/max(r['energy'],1e-30),downstream_block_nmse=m['hidden_nmse'],logits_nmse=m['nmse'],top1=m['top1'],input_clip=r['ic']/r['n'],output_clip=r['oc']/r['n']))
        save(f'work/single_{i:02}.json',single[-1])
    stage('CUMULATIVE_SWEEP')
    for i,name in enumerate(exp.names):
        label=f'cumulative_{i}';result=exp.diagnostic(example,exp.names[:i+1],label,True);m=measure(exp,result,baseline,example,name);ic,oc=telemetry_rates(exp,label)
        cumulative.append(dict(number_of_int10_norms=i+1,newly_enabled_module=name,nmse=m['nmse'],kl=m['kl'],top1=m['top1'],hidden_nmse=m['hidden_nmse'],input_clip=ic,output_clip=oc))
        save(f'work/cumulative_{i:02}.json',cumulative[-1])
    write_csv('single_norm_insertion.csv',single);write_csv('cumulative_norm_insertion.csv',cumulative)
    divergence=first_jump(cumulative);save('first_divergence.json',divergence)
    return divergence,single,cumulative

def drift_and_recalibration(exp,divergence):
    rows=[]
    for i,name in enumerate(exp.names):
        # When module i is first enabled its input has passed all earlier INT10 Norms.
        samples=[x for group in ('prefill','decode') for x,_ in exp.samples.get((f'cumulative_{i}',name,group),[])]
        calibration=[x for group in ('prefill','decode') for x,_ in exp.samples.get(('calibration',name,group),[])]
        x=np.stack(samples);c=np.stack(calibration);spec=exp.specs[name];weight=exp.weights[name]
        ref=reference(x,weight,spec['eps']);cref=reference(c,weight,spec['eps']);_,detail=exact(x,spec,weight)
        ci=float(abs(c).max());co=float(abs(cref).max());pi=float(abs(x).max());po=float(abs(ref).max())
        rows.append(dict(module=name,layer_index=i,calibration_input_absmax=ci,propagated_input_absmax=pi,s_in=spec['input_scale'],representable_input_max=511*spec['input_scale'],
            calibration_output_absmax=co,propagated_output_absmax=po,s_out=spec['output_scale'],representable_output_max=511*spec['output_scale'],
            input_clipping_rate=detail['input_clipping_rate'],output_clipping_rate=detail['output_clipping_rate'],p99_abs=float(np.percentile(abs(x),99)),p999_abs=float(np.percentile(abs(x),99.9)),max_abs=pi,
            input_scale_utilization=pi/(511*spec['input_scale']),output_scale_utilization=po/(511*spec['output_scale']),propagated_to_calibration_absmax=pi/max(ci,1e-30)))
    write_csv('norm_scale_drift.csv',rows)
    result=dict(performed=False,reason='no first divergence')
    if divergence['found']:
        name=divergence['module'];i=exp.names.index(name);spec=exp.specs[name];weight=exp.weights[name]
        x=np.stack([x for group in ('prefill','decode') for x,_ in exp.samples.get((f'cumulative_{i}',name,group),[])])
        ref=reference(x,weight,spec['eps']);original,_=exact(x,spec,weight)
        proposed=dict(spec,input_scale=max(float(abs(x).max())/511,1e-12),output_scale=max(float(abs(ref).max())/511,1e-12))
        changed,_=exact(x,proposed,weight)
        result=dict(performed=True,module=name,original_spec=spec,propagated_spec=proposed,original_nmse=mixed.nmse(original,ref),propagation_aware_nmse=mixed.nmse(changed,ref),selection_scope='local diagnostic only; never installed into E2E')
    save('first_divergence_recalibration.json',result)
    return rows,result

def e2e(exp):
    examples=mixed.e2e_examples(exp.source)
    if len(examples)!=8 or any(len(e['targets'])<24 for e in examples):raise RuntimeError('wrong E2E population')
    records=[]
    for i,example in enumerate(examples):
        accum={m:response.Scores() for m in MODES};caches={m:None for m in MODES}
        for t,target in enumerate(example['targets'][:24]):
            ids=example['prompt_ids'] if not t else [example['targets'][t-1]]
            fp,_,caches['FP_FULL']=cal.run_fp(exp.source,ids,caches['FP_FULL'],layers=False);accum['FP_FULL'].add(exp.source.torch,fp,fp,[target])
            for mode in MODES[1:]:
                enabled=() if mode=='GPALU10_FP_VPU2' else exp.names
                with exp.patch(enabled,old_emulator=mode.endswith('OLD_EMULATOR')):
                    logits,_,caches[mode]=exp.forward(ids,caches[mode])
                accum[mode].add(exp.source.torch,logits,fp,[target])
            if len(set(map(id,caches.values())))!=len(MODES):raise RuntimeError('KV alias')
        records.append({m:a.raw() for m,a in accum.items()});save(f'work/e2e_{i:02}.json',records[-1])
        # Baseline reproduction on each completed conversation, before continuing.
        prior_path=old.GPALU/'work'/f'e2e_{i:02}.json'
        if prior_path.exists():
            prior=response.merged([json.loads(prior_path.read_text())['value']['GPALU_INT10']]);fresh=response.merged([records[-1]['GPALU10_FP_VPU2']])
            if any(abs(prior[k]-fresh[k])>1e-6 for k in ('nll','kl','nmse','top1')):raise RuntimeError('baseline reproduction failed')
    metrics={m:response.merged([r[m] for r in records]) for m in MODES}
    if any(m['tokens']!=192 or not all(math.isfinite(v) for v in m.values()) for m in metrics.values()):raise RuntimeError('invalid E2E count or nonfinite metric')
    prior=json.loads((old.GPALU/'e2e_summary.json').read_text())['modes']['GPALU_INT10']
    if any(abs(prior[k]-metrics['GPALU10_FP_VPU2'][k])>1e-6 for k in ('nll','kl','nmse','top1')):raise RuntimeError('aggregate baseline reproduction failed')
    save('e2e_summary.json',dict(modes=metrics,tokens=192,examples=[mixed.identity(e) for e in examples],softmax='FP in every mode',rope='FP in every mode',coeff10='not run: literal core output scale/gamma domain is not a validated model interface'))
    return metrics

def publish(tests_ok):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=SOURCE_HEAD:
        raise RuntimeError('HEAD changed during experiment; publication stopped safely')
    if {p:old.sha(ROOT/p) for p in SOURCE_FILES}!=SOURCE_HASHES:
        raise RuntimeError('runner source changed during experiment')
    if not tests_ok or not all((OUT/p).is_file() for p in ARTIFACTS):raise RuntimeError('incomplete artifacts/tests')
    intended=list(SOURCE_FILES)+[str((OUT/p).relative_to(ROOT)) for p in ARTIFACTS]
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if set(staged)-set(intended):raise RuntimeError('unrelated staged files; publication stopped safely')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='lut_cali':raise RuntimeError('branch changed')
    subprocess.run(['git','add','--',*intended],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','test: debug INT10 normalizer propagation'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','lut_cali'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not done_allowed(tests_ok,(OUT/'report.md').exists(),True):raise RuntimeError('completion guard')
    save('status.json',dict(state='DONE',stage='COMPLETE',final_commit=commit,report='diagnostics/norm10_debug_pilot/report.md'))
    (OUT/'DONE').write_text(f'COMPLETE\ncommit={commit}\nreport=diagnostics/norm10_debug_pilot/report.md\n')

def main():
    global SOURCE_HEAD, SOURCE_HASHES
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True);stage('INITIALIZING')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='lut_cali':raise RuntimeError('wrong branch')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    SOURCE_HEAD=head;SOURCE_HASHES={p:old.sha(ROOT/p) for p in SOURCE_FILES}
    if subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip():raise RuntimeError('unrelated staged files at start')
    if subprocess.check_output(['git','-C',str(old.RTL),'branch','--show-current'],text=True).strip()!='feat/tpu-rocc-driver':raise RuntimeError('wrong RTL branch')
    subprocess.run(['git','merge-base','--is-ancestor','b7daf5ad1fa7bde89e92bbf6d66d692d5ad77108','HEAD'],check=True)
    paths=[old.GPALU/'int10/gpalu_parameters.json',old.GPALU/'e2e_summary.json',old.OUT/'norm10_coeff16_parameters.json',old.OUT/'e2e_summary.json']
    protected={str(p):old.sha(p) for p in paths}
    rtl={p:old.sha(old.RTL/p) for p in old.RTL_FILES if 'Rope' not in p}
    save('baseline_manifest.json',dict(source_HEAD=head,source_hashes=SOURCE_HASHES,hashes=protected,rtl_hashes=rtl,diagnostic_subset='first calibration chat, first two response predictors',gamma_adapter='FP multiply (1+weight) after literal RTL output reconstruction',softmax='FP, deliberately isolated from previous combined Norm+Softmax experiment'))
    stage('UNIT_TESTS');cmd=[sys.executable,'-m','unittest','tests.test_norm10_debug_pilot','tests.test_vpu10_e2e_pilot','tests.test_gpalu_width_sweep_pilot']
    with (OUT/'unit_test.log').open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc:raise RuntimeError('unit tests failed')
    stage('MODEL_LOAD');exp=load()
    stage('ACTUAL_CAPTURE');capture(exp)
    stage('RTL_TRACE');traces,comparisons=trace_rows(exp,exp.names,'calibration')
    save('rtl_exact_trace.json',dict(epsilon_rounding='RNE independently in q^2 domain',invN_rounding='positive half up (Scala math.round)',samples=traces));write_csv('old_vs_rtl_exact.csv',comparisons)
    example=mixed.calibration_examples(exp.source)[0];divergence,single,cumulative=sweeps(exp,example)
    drift,recal=drift_and_recalibration(exp,divergence)
    if divergence['found']:
        extra,_=trace_rows(exp,[divergence['module']],f"cumulative_{exp.names.index(divergence['module'])}");traces+=extra
        save('rtl_exact_trace.json',dict(epsilon_rounding='RNE independently in q^2 domain',invN_rounding='positive half up (Scala math.round)',samples=traces))
    stage('E2E_CONFIRMATION');metrics=e2e(exp)
    stage('REPORT')
    if {str(p):old.sha(p) for p in paths}!=protected:raise RuntimeError('baseline changed')
    # Literal RTL lacks both learned gamma and an arbitrary output-scale conversion.
    # Thus absence of a model-interface gain prevents a claim of a precision limit.
    diagnosis='MIXED_OR_INCONCLUSIVE'
    save('verification.json',dict(source_HEAD=head,branch='lut_cali',execution_device='CPU',MPS_used=False,ANE_used=False,CUDA_used=False,
        actual_activations_only=True,synthetic_conclusions=False,accBits=old.accumulator_bits(),baseline_unchanged=True,rtl_hashes=rtl,
        test_command=cmd,test_return_code=rc,independent_caches=True,diagnostic_targets=2,final_targets=192,diagnosis=diagnosis,
        no_e2e_scale_selection=True,softmax_remains_fp=True,gamma='FP adapter after core output',scale_domain_validated=False,
        elapsed_seconds=time.monotonic()-START))
    prior=json.loads((old.OUT/'e2e_summary.json').read_text())['modes']
    sections=[('1. Scope','CPU-only actual-activation debugging of the literal specified RMSNorm integer path; no architecture decision.'),
      ('2. Baselines reproduced','GPALU INT10 is checked against the preceding width sweep with 1e-6 absolute metric tolerance. The OLD_EMULATOR mode here replaces RMSNorm only. The previous ~6–8% result also replaced Softmax and cannot be attributed to RMSNorm alone.'),
      ('3. RTL exact RMSNorm arithmetic','Input RNE INT10; sum squares; invN=round_half_up(2^20/N); (sum*invN)>>20; separately RNE epsilon/s_in^2; normalizedIndex; Q2.14 rsqrt; odd exponent correction; q*scale>>14; INT10 saturation; reconstruction q_out*s_out. Gemma learned gamma is applied afterward in FP, outside the RTL core. No implicit 1/s_out gain is inserted.'),
      ('4. Old emulator vs RTL-exact mismatch',f"{sum(x['flag']=='EMULATOR_CONTRACT_MISMATCH' for x in comparisons)}/{len(comparisons)} actual sampled rows mismatch. See old_vs_rtl_exact.csv for first differing stages. The old implementation keeps fractional normalized values, multiplies gamma, then requantizes by s_out; RTL truncates before reconstruction. These are different numerical domains."),
      ('5. Actual activation clipping',f"Worst propagated input clipping {max(x['input_clipping_rate'] for x in drift):.6g}; output clipping {max(x['output_clipping_rate'] for x in drift):.6g}. Actual bounded rows only; full-forward aggregate clipping is also in insertion artifacts."),
      ('6. Static scale drift under propagation','norm_scale_drift.csv compares FP calibration against the input seen when each Norm is cumulatively enabled. Percentiles use deterministic bounded actual samples.'),
      ('7. Single-Norm insertion results',f"Worst diagnostic logits NMSE module: {max(single,key=lambda x:x['logits_nmse'])['module']}. All 37 insertions use identical two response predictors from one calibration conversation, not 37 full validations."),
      ('8. Cumulative insertion results','37 execution-order checkpoints. Hidden comparison is the output of the associated transformer block; model.norm uses its actual final normalized hidden output. First-jump rules are fixed before measurement.'),
      ('9. First divergence',json.dumps(divergence)),('10. Propagation-aware recalibration diagnostic',json.dumps(recal)),
      ('11. Final 192-token E2E','All RMSNorm modes keep Softmax and RoPE FP. Coeff10 was skipped because the literal core-to-model scale contract remains unvalidated.'),
      ('12. Primary diagnosis',diagnosis+'. Previous ~6–8% Top1 is not a trustworthy isolated RMSNorm precision conclusion: it combines Softmax with RMSNorm and uses synthetic local probes. The new exact core differs in phase-2 truncation and scale/gamma adaptation. This task does not silently introduce a missing gain. There is insufficient evidence that INT10 itself is the problem.'),
      ('13. Implication for next calibration / RTL work','Resolve the real output-scale and learned-gamma interface and separate Softmax attribution before interpreting bit-width adequacy. No further experiment is automatically run.'),
      ('14. Limitations','Two diagnostic targets and 192 evaluation targets; actual capture is four existing chats, eight response positions, eight rows per group/module. Literal RTL core may intentionally expose interface incompatibility. No simulator/RTL co-simulation was run; arithmetic is translated from golden semantics.')]
    lines=['# INT10 Normalizer Debug Pilot','']
    for title,body in sections:
        lines += ['## '+title,'',body,'']
        if title.startswith('11.'):
            keys=('nll','ppl','kl','nmse','mse','mae','cosine','flattened_cosine','top1','in5','overlap')
            lines += ['| Mode | '+' | '.join(keys)+' |','|---|'+'---:|'*len(keys)]
            for mode,m in metrics.items():lines.append('| '+mode+' | '+' | '.join(f'{m[k]:.9g}' for k in keys)+' |')
            lines.append('')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    stage('GIT');publish(rc==0)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        traceback.print_exc();OUT.mkdir(parents=True,exist_ok=True)
        save('status.json',dict(state='FAILED',stage=STAGE,exception_type=type(exc).__name__,error=str(exc)))
        (OUT/'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        (OUT/'DONE').unlink(missing_ok=True)
        sys.exit(1)
