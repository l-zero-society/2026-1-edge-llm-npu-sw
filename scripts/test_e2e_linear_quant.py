#!/usr/bin/env python3
"""Frozen-artifact, Linear-only propagated prefill diagnostic. No calibration."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
from static_quant.core import quantize_input, quantize_weight
from static_quant.hardware import Profile
from quant_diagnostic_math import integer_dot, configure_blas
from quant_diagnostic_source import RecordedGGUFSource
from test_s10_calibration import parameters
from static_quant.gemma import sha256_file
from diagnose_linear_quant_error import write_json, write_csv

MODES = ['down_only', 'all_linear']


class Metric:
    def __init__(self):
        self.n = self.rows = 0
        self.se = self.ae = self.re = self.qe = self.cross = self.cos = 0.

    def add(self, value, ref):
        q, r = np.asarray(value, np.float64), np.asarray(ref, np.float64)
        if q.shape != r.shape or not np.isfinite(q).all() or not np.isfinite(r).all():
            raise ValueError('invalid metric operands')
        e = q-r
        self.n += r.size; self.rows += len(r)
        self.se += float(np.square(e).sum()); self.ae += float(np.abs(e).sum())
        re = np.square(r).sum(axis=-1); qe = np.square(q).sum(axis=-1)
        cross = (q*r).sum(axis=-1); den = np.sqrt(re*qe)
        self.re += float(re.sum()); self.qe += float(qe.sum()); self.cross += float(cross.sum())
        self.cos += float(np.divide(cross, den, out=np.zeros_like(cross), where=den!=0).sum())

    def result(self):
        return dict(nmse=self.se/self.re if self.re else (0. if not self.se else None),
                    mae=self.ae/self.n, mse=self.se/self.n, cosine=self.cos/self.rows,
                    flattened_cosine=self.cross/math.sqrt(self.re*self.qe) if self.re*self.qe else 0.,
                    fp_energy=self.re/self.n, quant_energy=self.qe/self.n)


def quantized_linear(x, wq, sx, s10, params, k, chunk=512):
    """Single unpadded prefill sequence. Position 0 correction only, exact ACC."""
    x = np.asarray(x, np.float64)
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError('only batch-one, unpadded full prefill is supported')
    x = x[0]
    if not np.all((params['shift']-k >= 0) & (params['shift']-k <= 31)):
        raise ValueError('unsupported BOS effective shift')
    q = quantize_input(x, sx)
    q[0:1] = quantize_input(x[0:1], sx*2.**k)
    out = np.empty((len(x), len(wq)), np.float32)
    clipped = [0, 0]
    for start in range(0, len(wq), chunk):
        end = min(start+chunk, len(wq))
        acc = integer_dot(q, wq[start:end], 'fp64-exact')
        m, s = params['multiplier'][start:end], params['shift'][start:end]
        raw = Profile().apply(acc, m, s, saturate=False)
        # SAME channel multiplier; shift adjustment before rounding/saturation.
        raw[0:1] = Profile().apply(acc[0:1], m, s-k, saturate=False)
        clipped[0] += int(((raw[0] < -512) | (raw[0] > 511)).sum())
        clipped[1] += int(((raw[1:] < -512) | (raw[1:] > 511)).sum())
        out[:, start:end] = (np.clip(raw, -512, 511)*s10).astype(np.float32)
    counts = dict(bos_outputs=len(wq), non_bos_outputs=(len(x)-1)*len(wq),
                  bos_int10_clips=clipped[0], non_bos_int10_clips=clipped[1],
                  bos_inputs=x.shape[1], non_bos_inputs=(len(x)-1)*x.shape[1],
                  bos_input_clips=int((np.abs(x[:1])>127*sx*2.**k).sum()),
                  non_bos_input_clips=int((np.abs(x[1:])>127*sx).sum()))
    return out[None], counts


def load_artifacts(source, art, manifest, frozen):
    assert frozen['selection_split'] == 'calibration' and frozen['best_global_weight'] == .25
    assert len(frozen['policies']['best_global']) == 18
    for p, h in frozen['input_hashes'].items():
        assert sha256_file(ROOT/p) == h, p
    assert manifest['hardware_profile'] == Profile().metadata()
    specs, missing, hashes = {}, [], {}
    for entry in manifest['modules']:
        name = entry['module_name']; module = source.modules[name]
        md = art/Path(entry['module_manifest']).parent
        module_manifest = json.loads((md/'manifest.json').read_text())
        key = module_manifest['modules'][0]['key']
        paths = [art/entry['int8_weight_file'], md/'scales.npz', md/'qparams.npz']
        if not all(p.exists() for p in paths):
            missing.append(dict(module=name, reason='missing weight/scales/qparams')); continue
        try:
            wq = np.load(paths[0], mmap_mode='r')
            with np.load(paths[1]) as z:
                sx, s10, sw = float(z[key+'.s_X']), float(z[key+'.s_10']), z[key+'.s_W'].copy()
            with np.load(paths[2]) as z:
                params = {k:z[key+'.'+k].copy() for k in ['multiplier', 'shift', 'status']}
                assert not z[key+'.zero_point'].any()
                np.testing.assert_array_equal(Profile().pack(params['multiplier'], params['shift']), z[key+'.packed'])
            assert sx == entry['s_X'] and s10 == entry['s_10'] and sx > 0 and s10 > 0
            assert np.isfinite(sw).all() and (sw>0).all()
            assert wq.dtype == np.int8 and wq.shape == tuple(module.weight.shape) and module.bias is None
            # Compare recovered FP weight against existing Wq/sW, chunked in N.
            for start in range(0, len(wq), 512):
                expected, scales = quantize_weight(source.weight(name)[start:start+512])
                np.testing.assert_array_equal(expected, wq[start:start+512])
                np.testing.assert_array_equal(scales, sw[start:start+512])
            check = Profile().approximate(sx*sw/s10)
            for field in ['multiplier','shift','status']:
                np.testing.assert_array_equal(params[field], check[field])
            k = 0
            if name.endswith('.down_proj'):
                choice = frozen['policies']['best_global'][int(name.split('.')[2])]
                sx, s10, k = choice['s_X_normal'], choice['s_10'], int(choice['k'])
                params, feasibility = parameters(sx, sw, s10, k)
                assert feasibility['feasible'] and choice['feasible']
            assert np.all(params['status']=='ok')
            Profile().pack(params['multiplier'], params['shift'])
            specs[name] = dict(wq=wq, sx=sx, s10=s10, k=k, params=params)
            hashes.update({str(p.relative_to(ROOT)):sha256_file(p) for p in paths})
        except (AssertionError, ValueError, KeyError) as exc:
            missing.append(dict(module=name, reason=str(exc) or 'artifact consistency check failed'))
    if len([n for n in specs if n.endswith('.down_proj')]) != 18:
        raise ValueError('down-only requires all 18 valid frozen down_proj policies')
    return specs, missing, hashes


@contextmanager
def patch_linears(source, specs, mode, counters):
    originals = {}
    try:
        for name, spec in specs.items():
            if mode == 'down_only' and not name.endswith('.down_proj'):
                continue
            module = source.modules[name]; originals[name] = module.forward
            def forward(x, name=name, spec=spec):
                y, counts = quantized_linear(x.detach().cpu().numpy(), **spec)
                stat = counters.setdefault(name, dict(calls=0))
                stat['calls'] += 1
                for key, value in counts.items(): stat[key] = stat.get(key, 0)+value
                return source.torch.from_numpy(y).to(dtype=x.dtype, device=x.device)
            module.forward = forward
        yield list(originals)
    finally:
        for name, forward in originals.items(): source.modules[name].forward = forward


def forward_prefill(source, ids):
    """Use the original model driver. Hooks observe boundaries, never replace inputs."""
    torch = source.torch; states = {}; handles = []; checks = []
    for layer, block in enumerate(source.model.model.layers):
        def pre(module, args, kwargs, layer=layer):
            x = args[0] if args else kwargs['hidden_states']
            if layer:
                # The exact preceding block tensor must feed the next block.
                assert x.data_ptr() == states[layer-1].data_ptr()
                checks.append(layer)
        def post(module, args, output, layer=layer): states[layer] = output.detach()
        handles.extend([block.register_forward_pre_hook(pre, with_kwargs=True), block.register_forward_hook(post)])
    x = torch.tensor([ids], dtype=torch.int64)
    try:
        with torch.inference_mode():
            output = source.model(input_ids=x, attention_mask=torch.ones_like(x),
                                  position_ids=torch.arange(len(ids))[None], use_cache=False,
                                  output_hidden_states=True)
    finally:
        for handle in handles: handle.remove()
    assert len(checks) == 17 and len(states) == 18
    return output.logits[0], states, output.hidden_states[-1][0]


def logit_metrics(torch, q, ref, ids):
    metric = Metric(); kl = nll = ref_nll = top1 = overlap = setmatch = top1in5 = 0.
    for start in range(0, len(ids), 16):
        end = min(start+16, len(ids))
        a, b = q[start:end].double(), ref[start:end].double()
        metric.add(a.numpy(), b.numpy())
        lp, lq = torch.log_softmax(b, -1), torch.log_softmax(a, -1)
        kl += float((lp.exp()*(lp-lq)).sum())
        rt, qt = b.topk(5, dim=-1).indices, a.topk(5, dim=-1).indices
        top1 += int((rt[:,0]==qt[:,0]).sum())
        matches = (rt[:,:,None]==qt[:,None,:]).any(-1).sum(-1)
        overlap += float((matches/5.).sum()); setmatch += int((matches==5).sum())
        top1in5 += int((rt[:,0,None]==qt).any(-1).sum())
        valid = min(end,len(ids)-1)-start
        if valid > 0:
            targets = torch.tensor(ids[start+1:start+1+valid])
            nll -= float(lq[torch.arange(valid),targets].sum())
            ref_nll -= float(lp[torch.arange(valid),targets].sum())
    return dict(metric=vars(metric), kl_sum=kl, top1_count=top1, top5_overlap_sum=overlap,
                top5_exact_count=setmatch, fp_top1_in_quant_top5_count=top1in5,
                nll_sum=nll, fp_nll_sum=ref_nll, tokens=len(ids), targets=len(ids)-1)


def merge_metrics(dicts):
    m = Metric()
    for d in dicts:
        for key, value in d.items(): setattr(m, key, getattr(m,key)+value)
    return m.result()


def materialize(out, provenance):
    records = [json.loads(p.read_text()) for p in sorted(out.glob('sequence_*.json'))]
    if not records: return
    layers, summaries = [], {}
    for mode in MODES:
        rr = [r['modes'][mode] for r in records]
        for layer in range(18):
            groups = {g:merge_metrics([r['layers'][layer][g] for r in rr]) for g in ['all','bos','non_bos']}
            layers.append(dict(layer=layer, mode=mode, **groups['all'],
                               bos_nmse=groups['bos']['nmse'], non_bos_nmse=groups['non_bos']['nmse']))
        lm = [r['logits'] for r in rr]; n = sum(r['tokens'] for r in lm); targets = sum(r['targets'] for r in lm)
        metrics = merge_metrics([r['metric'] for r in lm])
        local = [r for r in layers if r['mode']==mode]
        summaries[mode] = dict(logits=metrics,
            final_block_hidden=local[-1], final_normalized_hidden=merge_metrics([r['normalized_hidden'] for r in rr]),
            kl=sum(r['kl_sum'] for r in lm)/n, top1_agreement=sum(r['top1_count'] for r in lm)/n,
            top5_agreement=sum(r['top5_overlap_sum'] for r in lm)/n,
            top5_exact_set_agreement=sum(r['top5_exact_count'] for r in lm)/n,
            fp_top1_in_quant_top5=sum(r['fp_top1_in_quant_top5_count'] for r in lm)/n,
            nll=sum(r['nll_sum'] for r in lm)/targets, fp_nll=sum(r['fp_nll_sum'] for r in lm)/targets,
            first_layer_above_1percent=next((r['layer'] for r in local if r['nmse']>.01),None),
            first_layer_above_5percent=next((r['layer'] for r in local if r['nmse']>.05),None),
            worst_layer=max(local,key=lambda r:r['nmse'])['layer'],
            worst_hidden_nmse=max(r['nmse'] for r in local))
        summaries[mode]['ppl'] = math.exp(summaries[mode]['nll'])
        summaries[mode]['fp_ppl'] = math.exp(summaries[mode]['fp_nll'])
    summary = dict(sequences=len(records),tokens=sum(r['tokens'] for r in records),
        next_token_targets=sum(r['tokens']-1 for r in records), modes=summaries,
        provenance=provenance, scope='FP32 recovered GGUF model; only Linear INT8/INT10 propagated',
        top5_definition='mean |FP top5 intersection quant top5| / 5; exact set agreement also recorded',
        cosine_definition='mean per-token cosine; flattened cosine also recorded',
        decode='not run; this focused experiment uses complete unpadded prefill, use_cache=False')
    write_json(out.parent/'e2e_linear_quant_summary.json',summary)
    write_csv(out.parent/'e2e_linear_quant_layers.csv',layers)
    headers='| mode | final block hidden NMSE % | logits NMSE % | logits cosine | KL nats | top1 % | top5 overlap % | ppl |'
    report=['# End-to-end Linear-only prefill validation',
        f'{len(records)} held-out validation sequences, {summary["tokens"]} valid tokens, {summary["next_token_targets"]} next-token targets. Selection is frozen from calibration; no recalibration.',
        'Reference is the existing FP32 GGUF-recovered model, not original BF16. Quantized Linear outputs feed the original model driver and subsequent layers; transformer boundary pointers are asserted equal. All nonlinear math, embedding and lm_head remain the original FP implementation. No LUT approximation. INT8 GEMMs use the existing exact FP64-integer backend (K*127² < INT32 bound), checked against INT64. Output is the existing Profile RNE/saturation with BOS effective shift S-k; all effective shifts are in 0..31.',
        f'FP NLL / perplexity: {summaries[MODES[0]]["fp_nll"]:.8f} / {summaries[MODES[0]]["fp_ppl"]:.6f}. Logits metrics include all valid positions; NLL shifts targets by one and excludes the final position. Cosine is mean per-token cosine. Top5 is set overlap / 5, not exact-set agreement.',
        headers,'| --- | --- | --- | --- | --- | --- | --- | --- |']
    for mode,s in summaries.items():
        report.append(f'| {mode} | {100*s["final_block_hidden"]["nmse"]:.6f} | {100*s["logits"]["nmse"]:.6f} | {s["logits"]["cosine"]:.8f} | {s["kl"]:.8f} | {100*s["top1_agreement"]:.4f} | {100*s["top5_agreement"]:.4f} | {s["ppl"]:.6f} |')
    report += ['', '| layer | mode | hidden NMSE % | cosine | BOS NMSE % | non-BOS NMSE % | FP energy | quant energy |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in layers:
        report.append(f'| {r["layer"]} | {r["mode"]} | {100*r["nmse"]:.6f} | {r["cosine"]:.8f} | {100*r["bos_nmse"]:.6f} | {100*r["non_bos_nmse"]:.6f} | {r["fp_energy"]:.8g} | {r["quant_energy"]:.8g} |')
    for mode,s in summaries.items():
        report.append(f'{mode}: first hidden NMSE >1%: layer {s["first_layer_above_1percent"]}; >5%: layer {s["first_layer_above_5percent"]}; worst: layer {s["worst_layer"]} ({100*s["worst_hidden_nmse"]:.6f}%). Final post-RMSNorm hidden NMSE: {100*s["final_normalized_hidden"]["nmse"]:.6f}%.')
    report.append('Exact enabled module names, source/artifact hashes and frozen policy values are in e2e_linear_quant/provenance.json; per-sequence counters and metrics are resumable JSON records. Missing artifacts: '+json.dumps(provenance['missing_artifacts']))
    if (out/'answers.md').exists(): report.append((out/'answers.md').read_text())
    (out.parent/'e2e_linear_quant_report.md').write_text('\n\n'.join(report).replace('|\n\n|', '|\n|')+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequences',type=int,default=8)
    ap.add_argument('--model',default='/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf')
    args=ap.parse_args(); out=ROOT/'diagnostics/e2e_linear_quant';out.mkdir(exist_ok=True)
    art=ROOT/'calibration_outputs/gguf-all-126-oasst1'; manifest=json.loads((art/'manifest.json').read_text())
    fp=ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json';frozen=json.loads(fp.read_text())
    started=time.monotonic()
    source=RecordedGGUFSource(args.model,ROOT/'calibration_outputs/data-oasst1-v1',manifest)
    configure_blas('accelerate')
    specs,missing,hashes=load_artifacts(source,art,manifest,frozen)
    enabled={mode:[n for n in specs if mode=='all_linear' or n.endswith('.down_proj')] for mode in MODES}
    provenance=dict(script_sha256=sha256_file(__file__),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model_sha256=manifest['gguf']['sha256'],manifest_sha256=sha256_file(art/'manifest.json'),
        frozen_selection_sha256=sha256_file(fp),global_w_BOS=.25,selection_split='calibration',
        validation_dataset=manifest['datasets']['validation'],profile=Profile().metadata(),
        enabled=enabled,missing_artifacts=missing,artifact_hashes=hashes,
        policies={n:dict(s_X=s['sx'],s_10=s['s10'],k=s['k']) for n,s in specs.items()},
        source_hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in [ROOT/'scripts/quant_diagnostic_source.py',ROOT/'scripts/quant_diagnostic_math.py',ROOT/'src/static_quant/hardware.py',ROOT/'src/static_quant/core.py']})
    if (out/'provenance.json').exists(): assert json.loads((out/'provenance.json').read_text())==provenance
    else: write_json(out/'provenance.json',provenance)
    with (out/'commands.jsonl').open('a') as f: f.write(json.dumps(dict(argv=sys.argv,time=time.time()))+'\n')
    # Exact accelerated ACC and corrected shifts agree with the existing references.
    from test_factorized_row_requant import row_requant
    for s in specs.values():
        q=np.tile(np.array([-127,0,127],np.int8),s['wq'].shape[1]//3+1)[:s['wq'].shape[1]][None]
        w=s['wq'][:2]; exact=integer_dot(q,w,'int64');fast=integer_dot(q,w,'fp64-exact')
        np.testing.assert_array_equal(exact,fast)
        p={key:s['params'][key][:2] for key in ['multiplier','shift']}
        np.testing.assert_array_equal(Profile().apply(exact,p['multiplier'],p['shift']-s['k'],saturate=False),row_requant(exact,p,2.**s['k'],s['k']))
    print('READY',len(specs),'valid operations;',len(missing),'missing;',time.monotonic()-started,'seconds',flush=True)
    ids_list=source.datasets['validation']
    if not 1<=args.sequences<=len(ids_list): raise ValueError('invalid validation subset size')
    torch=source.torch
    # Instrumentation-free original logits versus observer-only prefill, short held-out prefix.
    if not (out/'sanity.json').exists():
        ids=ids_list[0][:8];x=torch.tensor([ids])
        with torch.inference_mode(): original=source.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0]
        observed,_,_=forward_prefill(source,ids)
        assert torch.equal(original,observed)
        write_json(out/'sanity.json',dict(fp_logits_bit_exact=True,quantized_ops_checked=len(specs),
            exact_accumulator_and_pot_reference_match=True,boundary_pointer_checks_per_forward=17,
            nonlinear_methods_unchanged=True,fp_embedding_lm_head=True))
        del original,observed
    for i,ids in enumerate(ids_list[:args.sequences]):
        path=out/f'sequence_{i:03d}.json'
        if path.exists(): continue
        begin=time.monotonic();ref,ref_states,ref_norm=forward_prefill(source,ids)
        record=dict(sequence=i,tokens=len(ids),modes={})
        for mode in MODES:
            counters={}
            with patch_linears(source,specs,mode,counters) as names:
                assert names==enabled[mode]
                logits,states,norm=forward_prefill(source,ids)
            assert set(counters)==set(names) and all(r['calls']==1 for r in counters.values())
            layers=[]
            for layer in range(18):
                groups={}
                for group,sl in [('all',slice(None)),('bos',slice(0,1)),('non_bos',slice(1,None))]:
                    metric=Metric();metric.add(states[layer][0,sl].numpy(),ref_states[layer][0,sl].numpy());groups[group]=vars(metric)
                layers.append(groups)
            metric=Metric();metric.add(norm.numpy(),ref_norm.numpy())
            record['modes'][mode]=dict(layers=layers,normalized_hidden=vars(metric),
                logits=logit_metrics(torch,logits,ref,ids),operations=counters,
                propagated_boundaries=17,enabled=names)
            print('MODE',i,len(ids),mode,'hidden_nmse',merge_metrics([layers[-1]['all']])['nmse'],
                  'seconds',time.monotonic()-begin,flush=True)
            del logits,states,norm
        record['seconds']=time.monotonic()-begin
        write_json(path,record);materialize(out,provenance)
        del ref,ref_states,ref_norm
        print('DONE',i,'seconds',record['seconds'],flush=True)
    materialize(out,provenance)


if __name__=='__main__': main()
