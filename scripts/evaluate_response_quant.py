#!/usr/bin/env python3
"""Offline chat-response evaluation of frozen down_proj quantization only."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import test_e2e_row_pot as row
base = row.base
OUT = ROOT / 'diagnostics/response_quant_eval'
MODES = ['FP', 'bos_only', 'general_pot']
BUCKETS = [(0, 1, '0'), (1, 2, '1'), (2, 4, '2-3'), (4, 8, '4-7'),
           (8, 16, '8-15'), (16, 32, '16-31'), (32, 10**9, '>=32')]
MODEL = '/Users/mac/lzero_llama_workspace/llama.cpp/models/gemma-2b-it.Q8_0.gguf'
ARCHIVE = ROOT / 'calibration_outputs/data-oasst1-128-32/2023-04-12_oasst_ready.messages.jsonl.gz'
DATA = ROOT / 'calibration_outputs/data-oasst1-v1'


def response_indices(offsets, start, end):
    """Content-only targets; reject ambiguous header/content boundary tokens."""
    selected = []
    for i, (a, b) in enumerate(offsets):
        if b > a and a < end and b > start:
            if a < start or b > end:
                raise ValueError('token crosses content/template boundary')
            selected.append(i)
    if not selected or selected[0] == 0:
        raise ValueError('empty response or missing first predictor')
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError('noncontiguous response content')
    return selected


def build_example(tokenizer, messages, max_length):
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    content = messages[-1]['content'].strip()  # Native GGUF template applies trim.
    if not content or not full.startswith(prompt + content):
        raise ValueError('native template does not expose an exact content prefix')
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    targets = response_indices(enc['offset_mapping'], len(prompt), len(prompt) + len(content))
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    if targets[0] != len(prompt_ids) or ids[:targets[0]] != prompt_ids:
        raise ValueError('first content token does not follow exact tokenized assistant prefix')
    if len(ids) > max_length:
        raise ValueError(f'full chat exceeds model context: {len(ids)} > {max_length}')
    if any(ids[i] in tokenizer.all_special_ids for i in targets):
        raise ValueError('control/special token inside response content')
    return dict(ids=ids, prompt_ids=prompt_ids, target_positions=targets,
                predictor_positions=[i-1 for i in targets], targets=[ids[i] for i in targets],
                prompt_tokens=len(prompt_ids), response_tokens=len(targets), total_tokens=len(ids))


def pairs_from_archive():
    validation = [json.loads(l) for l in (DATA / 'validation.jsonl').read_text().splitlines()]
    calibration = [json.loads(l) for l in (DATA / 'calibration.jsonl').read_text().splitlines()]
    trees = {r['message_tree_id'] for r in validation}
    assert not trees & {r['message_tree_id'] for r in calibration}
    with gzip.open(ARCHIVE, 'rt') as f:
        records = {r['message_id']: r for line in f
                   if (r := json.loads(line)).get('message_tree_id') in trees}
    pairs, skipped = [], []
    for index, anchor in enumerate(validation):
        try:
            if anchor['message_id'] not in records:
                raise ValueError('anchor absent from locally available ready-tree archive')
            target = records[anchor['message_id']]
            assert target['text'] == anchor['text'] and target['role'] == anchor['role']
            if target['role'] == 'prompter':
                children = [r for r in records.values() if r.get('parent_id') == target['message_id']
                            and r['role'] == 'assistant' and not r.get('deleted')
                            and r.get('review_result') and not r.get('synthetic')]
                if not children:
                    raise ValueError('no valid human assistant child in local archive')
                target = min(children, key=lambda r: (r.get('rank') if r.get('rank') is not None else 10**9,
                                                       r['message_id']))
            chain, seen = [target], {target['message_id']}
            while chain[-1].get('parent_id'):
                parent = chain[-1]['parent_id']
                if parent not in records or parent in seen:
                    raise ValueError('missing or cyclic ancestor')
                chain.append(records[parent]); seen.add(parent)
            chain.reverse()
            if len(chain) % 2 or any(r['role'] != ('prompter' if i % 2 == 0 else 'assistant')
                                     for i, r in enumerate(chain)):
                raise ValueError('invalid conversation role sequence')
            if any(r['message_tree_id'] != anchor['message_tree_id'] or not r['text'].strip() for r in chain):
                raise ValueError('invalid ancestor text/tree')
            pairs.append(dict(index=index, anchor_id=anchor['message_id'], response_id=target['message_id'],
                              tree_id=anchor['message_tree_id'], message_ids=[r['message_id'] for r in chain],
                              messages=[dict(role='user' if r['role']=='prompter' else 'assistant', content=r['text']) for r in chain]))
        except (ValueError, KeyError) as exc:
            skipped.append(dict(index=index, anchor_id=anchor['message_id'], reason=str(exc)))
    return pairs, skipped


def native_template(model_path):
    import gguf
    from transformers.integrations.ggml import _gguf_parse_value
    reader = gguf.GGUFReader(model_path)
    field = reader.fields.get('tokenizer.chat_template')
    if field is None:
        raise ValueError('local GGUF has no native chat template; no template will be invented')
    return _gguf_parse_value(field.parts[field.data[0]], field.types)


def quant_call(x, spec, mode, offset, selected, histogram):
    if mode == 'bos_only':
        # The first row of a decode call is NOT position 0.
        return base.quantized_linear(x, **dict(spec, k=spec['k'] if offset == 0 else 0))[0]
    original = row.select_rows
    def observe(*args, **kwargs):
        q, ks, counts = original(*args, **kwargs)
        if selected is not None:
            ids = np.asarray(selected, dtype=np.int64)
            ids = ids[offset + ids > 0]
            histogram[:] += np.bincount(ks[ids] + 8, minlength=16)
        return q, ks, counts
    with patch.object(row, 'select_rows', observe):
        return row.apply_rows(x, **spec)[0]


@contextmanager
def quantized(source, specs, mode, context):
    originals = {}
    try:
        if mode != 'FP':
            for name, spec in specs.items():
                module = source.modules[name]; originals[name] = module.forward
                def forward(x, name=name, spec=spec):
                    hist = context['hist'].setdefault(name, np.zeros(16, np.int64))
                    y = quant_call(x.detach().numpy(), spec, mode, context['offset'], context['selected'], hist)
                    context['calls'] += 1
                    return source.torch.from_numpy(y).to(x.dtype)
                module.forward = forward
        yield
    finally:
        for name, fn in originals.items():
            source.modules[name].forward = fn


def forward_hidden(source, ids, positions, context, cache=None, use_cache=False, layers=True):
    """Original HF transformer and FP lm_head, evaluated separately to bound RAM.

    GemmaForCausalLM performs exactly model(...) followed by lm_head(last_hidden_state).
    Head evaluation is delayed/chunked, allowing reference reuse without logits dumps.
    """
    torch = source.torch
    offset = cache.get_seq_length() if cache is not None else 0
    context.update(offset=offset, selected=positions, calls=0, hist={})
    states, handles, previous, checks = {}, [], [None], []
    for layer, block in enumerate(source.model.model.layers):
        def pre(module, args, kwargs, layer=layer):
            x = args[0] if args else kwargs['hidden_states']
            if layer:
                assert x.data_ptr() == previous[0]
                checks.append(layer)
        def post(module, args, output, layer=layer):
            previous[0] = output.data_ptr()
            if layers:
                states[layer] = output[0, positions].detach().clone()
        handles.extend([block.register_forward_pre_hook(pre, with_kwargs=True), block.register_forward_hook(post)])
    x = torch.tensor([ids], dtype=torch.int64)
    try:
        with torch.inference_mode():
            result = source.model.model(input_ids=x, attention_mask=torch.ones((1, offset+len(ids)), dtype=torch.int64),
                                        position_ids=torch.arange(offset, offset+len(ids))[None],
                                        cache_position=torch.arange(offset, offset+len(ids)),
                                        past_key_values=cache, use_cache=use_cache)
            selected = result.last_hidden_state[0, positions].clone()
    finally:
        for h in handles: h.remove()
    assert len(checks) == 17
    assert context['calls'] in (0, 18)
    if use_cache:
        assert result.past_key_values.get_seq_length() == offset + len(ids)
    return selected, states, result.past_key_values


class Scores:
    def __init__(self):
        self.metric = base.Metric()
        self.n = 0
        self.nll = self.kl = self.top1 = self.in5 = self.overlap = 0.

    def add(self, torch, q, ref, targets):
        if not len(targets): return
        a, b = q.double(), ref.double()
        self.metric.add(a.numpy(), b.numpy())
        lq, lp = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        target = torch.tensor(targets, dtype=torch.int64)
        nll = -lq[torch.arange(len(targets)), target]
        rt, qt = b.topk(5, dim=-1).indices, a.topk(5, dim=-1).indices
        self.n += len(targets); self.nll += float(nll.sum())
        self.kl += float((lp.exp()*(lp-lq)).sum())
        self.top1 += int((rt[:, 0] == qt[:, 0]).sum())
        self.in5 += int((rt[:, 0, None] == qt).any(-1).sum())
        self.overlap += float((rt[:, :, None] == qt[:, None, :]).any(-1).sum()/5)
        return dict(nll=nll.tolist(), top1=qt[:, 0].tolist())

    def raw(self):
        return dict(metric=vars(self.metric), **{k:v for k,v in vars(self).items() if k != 'metric'})


def merged(records):
    n = sum(r['n'] for r in records)
    if not n: return None
    result = dict(tokens=n, **base.merge_metrics([r['metric'] for r in records]))
    result.update({k:sum(r[k] for r in records)/n for k in ['nll', 'kl', 'top1', 'in5', 'overlap']})
    result['ppl'] = math.exp(result['nll'])
    return result


def compare_heads(source, hidden, targets, window=32):
    torch = source.torch
    accum = {m:Scores() for m in MODES}
    buckets = {m:{b:Scores() for _,_,b in BUCKETS} for m in MODES}
    windows = {m:Scores() for m in MODES}
    details = {m:dict(nll=[], top1=[]) for m in MODES}
    with torch.inference_mode():
        for start in range(0, len(targets), 16):
            end = min(start+16, len(targets))
            ref = source.model.lm_head(hidden['FP'][start:end])
            for mode in MODES:
                q = ref if mode == 'FP' else source.model.lm_head(hidden[mode][start:end])
                detail = accum[mode].add(torch, q, ref, targets[start:end])
                for key in details[mode]: details[mode][key].extend(detail[key])
                for lo, hi, label in BUCKETS:
                    a, b = max(start, lo), min(end, hi)
                    if b > a: buckets[mode][label].add(torch, q[a-start:b-start], ref[a-start:b-start], targets[a:b])
                b = min(end, window)
                if b > start: windows[mode].add(torch, q[:b-start], ref[:b-start], targets[start:b])
    return {m:dict(total=accum[m].raw(), buckets={b:a.raw() for b,a in buckets[m].items() if a.n},
                   window=windows[m].raw(), window_details={k:v[:window] for k,v in details[m].items()}) for m in MODES}


def layer_metrics(states):
    result = {}
    for mode in MODES:
        result[mode] = []
        for layer in range(18):
            metric = base.Metric(); metric.add(states[mode][layer].numpy(), states['FP'][layer].numpy())
            result[mode].append(vars(metric))
    return result


def run_prefill(source, specs, examples, window):
    for e in examples:
        path = OUT / f'example_{e["index"]:03d}.json'
        if path.exists(): continue
        started = time.monotonic(); hidden = {'first':{}, 'response':{}}; states = {'first':{}, 'response':{}}; hist = {}
        for mode in MODES:
            context = {}
            try:
                with quantized(source, specs, mode, context):
                    h, ll, _ = forward_hidden(source, e['prompt_ids'], [len(e['prompt_ids'])-1], context)
                    hidden['first'][mode] = h; states['first'][mode] = ll
                    h, ll, _ = forward_hidden(source, e['ids'], e['predictor_positions'], context)
                    hidden['response'][mode] = h; states['response'][mode] = ll
                    if mode == 'general_pot': hist = {k:v.tolist() for k,v in context['hist'].items()}
            except Exception as exc:
                if mode == 'FP': raise
                import traceback
                base.write_json(OUT/f'failure_{e["index"]:03d}_{mode}.json',
                                dict(index=e['index'],mode=mode,reason=str(exc),traceback=traceback.format_exc()))
                for stage in hidden:
                    hidden[stage].pop(mode,None); states[stage].pop(mode,None)
                print('FAILED PREFILL',e['index'],mode,str(exc),flush=True)
                continue
            print('PREFILL', e['index'], mode, flush=True)
        if len(hidden['response']) != 3:
            continue  # Never silently compare policies on different target populations.
        scores = {stage:compare_heads(source, hidden[stage], e['targets'][:1] if stage=='first' else e['targets'], window)
                  for stage in ['first','response']}
        lm = {stage:layer_metrics(states[stage]) for stage in states}
        first_parity = {}
        for mode in MODES:
            m = base.Metric(); m.add(hidden['first'][mode].numpy(), hidden['response'][mode][:1].numpy())
            first_parity[mode] = m.result()
        assert all(math.isfinite(scores['response'][m]['total']['nll']) for m in MODES)
        base.write_json(path, dict(index=e['index'], scores=scores, layers=lm, k_histogram=hist,
                                   prefix_vs_full_normalized_hidden=first_parity, seconds=time.monotonic()-started))
        print('DONE PREFILL', e['index'], {m:merged([scores['response'][m]['total']])['ppl'] for m in MODES}, flush=True)


def run_cached(source, specs, examples, count, limit):
    seeds = {}; torch = source.torch
    for e in examples[:count]:
        started = time.monotonic(); caches = {}; hidden = {}; scores = {m:Scores() for m in MODES}; details = {m:dict(nll=[], top1=[]) for m in MODES}
        n = min(limit, len(e['targets']))
        for mode in MODES:
            context = {}
            with quantized(source, specs, mode, context):
                h, _, c = forward_hidden(source, e['prompt_ids'], [len(e['prompt_ids'])-1], context, use_cache=True, layers=False)
            caches[mode] = c; hidden[mode] = h
            seeds[e['index'], mode] = dict(cache=copy.deepcopy(c), first_hidden=h)
        assert len({id(c) for c in caches.values()}) == 3
        for t in range(n):
            if t:
                for mode in MODES:
                    context = {}
                    with quantized(source, specs, mode, context):
                        h, _, c = forward_hidden(source, [e['targets'][t-1]], [0], context, caches[mode], True, False)
                    assert context['offset'] == len(e['prompt_ids']) + t-1
                    caches[mode] = c; hidden[mode] = h
            with torch.inference_mode():
                ref = source.model.lm_head(hidden['FP'])
                for mode in MODES:
                    q = ref if mode=='FP' else source.model.lm_head(hidden[mode])
                    d = scores[mode].add(torch, q, ref, [e['targets'][t]])
                    for key in details[mode]: details[mode][key].extend(d[key])
                    if not t: seeds[e['index'], mode]['first_token'] = d['top1'][0]
        full = json.loads((OUT/f'example_{e["index"]:03d}.json').read_text())
        parity = {}
        for mode in MODES:
            previous = full['scores']['response'][mode]['window_details']
            parity[mode] = dict(top1_agreement=float(np.mean(np.array(details[mode]['top1'])==previous['top1'][:n])),
                                max_abs_token_nll_delta=float(np.max(np.abs(np.array(details[mode]['nll'])-previous['nll'][:n]))))
        base.write_json(OUT/f'decode_{e["index"]:03d}.json', dict(index=e['index'], targets=n,
                        modes={m:scores[m].raw() for m in MODES}, parity=parity, seconds=time.monotonic()-started))
        print('DONE CACHE', e['index'], n, 'seconds',time.monotonic()-started, flush=True)
    return seeds


def generation_comparison(ref, q, eos_ids):
    common = 0
    for a,b in zip(ref,q):
        if a != b: break
        common += 1
    equal = ref == q
    a = next((i for i,t in enumerate(ref) if t in eos_ids), None)
    b = next((i for i,t in enumerate(q) if t in eos_ids), None)
    return dict(shared_prefix=common, first_divergence=None if equal else common,
                token_agreement=sum(a==b for a,b in zip(ref,q))/max(len(ref),len(q)),
                fp_length=len(ref), length=len(q), length_difference=len(q)-len(ref), fp_eos_step=a, eos_step=b,
                eos_step_difference=None if a is None or b is None else b-a,
                eos_comparison_censored=a is None or b is None)


def run_greedy(source, specs, examples, seeds, count):
    eos = source.model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos,int) else list(eos)
    rows = []; torch = source.torch
    for e in examples[:count]:
        generated = {}
        for mode in MODES:
            seed = seeds.pop((e['index'], mode)); cache = seed['cache']; tokens = [seed['first_token']]
            while len(tokens) < 32 and tokens[-1] not in eos_ids:
                context = {}
                with quantized(source, specs, mode, context):
                    h, _, cache = forward_hidden(source, [tokens[-1]], [0], context, cache, True, False)
                with torch.inference_mode(): token = int(source.model.lm_head(h).argmax(-1)[0])
                tokens.append(token)
            generated[mode] = tokens
        for mode in MODES:
            rows.append(dict(index=e['index'], mode=mode, **generation_comparison(generated['FP'], generated[mode], eos_ids)))
        base.write_json(OUT/f'generation_{e["index"]:03d}.json', dict(index=e['index'], tokens=generated, eos_ids=eos_ids))
        print('DONE GREEDY',e['index'], {r['mode']:r['shared_prefix'] for r in rows[-3:]}, flush=True)
    base.write_csv(OUT.parent/'response_quant_eval_generation.csv', rows)
    return rows


def report(protocol, decode_status=None, generation_status=None):
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob('example_*.json'))]
    if not records: return
    modes, positions, layers = [], [], []
    for mode in MODES:
        first = merged([r['scores']['first'][mode]['total'] for r in records])
        response = merged([r['scores']['response'][mode]['total'] for r in records])
        modes.append(dict(mode=mode, first_token_nll=first['nll'], **response))
        for stage in ['first','response']:
            for layer in range(18):
                layers.append(dict(mode=mode, stage=stage, layer=layer, **base.merge_metrics([r['layers'][stage][mode][layer] for r in records])))
        for _,_,label in BUCKETS:
            values = [r['scores']['response'][mode]['buckets'][label] for r in records if label in r['scores']['response'][mode]['buckets']]
            if values: positions.append(dict(mode=mode, bucket=label, **merged(values)))
    evaluated={r['index'] for r in records}
    examples=json.loads((OUT/'examples.json').read_text())
    prompt_tokens=sum(e['prompt_tokens'] for e in examples if e['index'] in evaluated)
    response_tokens=sum(e['response_tokens'] for e in examples if e['index'] in evaluated)
    summary = dict(protocol=protocol, completed_examples=len(records), prompt_tokens=prompt_tokens,response_tokens=response_tokens,modes=modes,
                   evaluation_failures=[json.loads(p.read_text()) for p in sorted(OUT.glob('failure_*.json'))],
                   first_token={m:merged([r['scores']['first'][m]['total'] for r in records]) for m in MODES},
                   localization={}, response_k_histogram={}, cached_decode=decode_status, generation=generation_status)
    for mode in MODES:
        ll=[r for r in layers if r['mode']==mode and r['stage']=='response']
        summary['localization'][mode] = dict(first_above_1percent=next((r['layer'] for r in ll if r['nmse']>.01),None),
            first_above_5percent=next((r['layer'] for r in ll if r['nmse']>.05),None), worst_layer=max(ll,key=lambda r:r['nmse'])['layer'],final_nmse=ll[-1]['nmse'])
    pooled=np.zeros(16,np.int64)
    for layer in range(18):
        name=f'model.layers.{layer}.mlp.down_proj';hist=np.sum([r['k_histogram'][name] for r in records],axis=0);pooled+=hist
        summary['response_k_histogram'][str(layer)]=row.histogram_summary({k-8:int(v) for k,v in enumerate(hist)})
    summary['response_k_histogram']['all']=row.histogram_summary({k-8:int(v) for k,v in enumerate(pooled)})
    dec=[json.loads(p.read_text()) for p in sorted(OUT.glob('decode_*.json'))]
    if dec:
        dr=[]
        for mode in MODES:
            cache=merged([r['modes'][mode] for r in dec]);ix={r['index'] for r in dec}
            full=merged([r['scores']['response'][mode]['window'] for r in records if r['index'] in ix])
            dr.append(dict(mode=mode, **cache, matched_full_nll=full['nll'], matched_full_ppl=full['ppl'],
                           cached_minus_full_nll=cache['nll']-full['nll'],
                           cached_vs_full_top1=sum(r['parity'][mode]['top1_agreement']*r['targets'] for r in dec)/cache['tokens']))
        summary['decode_metrics']=dr
        base.write_csv(OUT.parent/'response_quant_eval_decode.csv',dr)
    base.write_json(OUT.parent/'response_quant_eval_summary.json',summary)
    base.write_csv(OUT.parent/'response_quant_eval_modes.csv',modes)
    base.write_csv(OUT.parent/'response_quant_eval_layers.csv',layers)
    base.write_csv(OUT.parent/'response_quant_eval_positions.csv',positions)
    lines=['# Assistant-response quantization evaluation',
           'Official metrics score assistant content targets only. All modes use identical chat tokens and masks. Prompt/template/BOS predictions, turn-ending control tokens and padding are excluded. The first assistant content token is included.',
           f'{len(records)} examples; {prompt_tokens} prompt tokens; {response_tokens} response targets. Native tokenizer.chat_template from the existing GGUF, rendered using apply_chat_template; full ancestor history preserved, no truncation or invented text.',
           '| mode | first-token NLL | response NLL | response PPL | KL | logits NMSE % | top1 % | FP top1 in Q top5 % | top5 overlap % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in modes:lines.append(f'| {r["mode"]} | {r["first_token_nll"]:.6f} | {r["nll"]:.6f} | {r["ppl"]:.6f} | {r["kl"]:.6f} | {100*r["nmse"]:.6f} | {100*r["top1"]:.3f} | {100*r["in5"]:.3f} | {100*r["overlap"]:.3f} |')
    lines += ['Source recovery: '+json.dumps(protocol['source_recovery']),
              'Only 18 down_proj are quantized. Frozen global w_BOS=0.25 sX/s10/Wq/sW/M/S are reused; no parameter search. All remaining Linears, RMSNorm, GeLU, RoPE, attention/softmax, residuals and lm_head remain FP. Every forward checks all 17 transformer propagation boundaries. Decode BOS correction depends on absolute position, not position within a one-token call.',
              'The native HF Gemma transformer is followed by its original FP lm_head in 16-row chunks, exactly the existing causal-LM composition. This bounds RAM and shares FP head results across modes without saving full logits or activations. For each example Stage 1/2 run FP, BOS-only, General in that order; then score shared references.',
              'Stage 3 is a bounded cache parity diagnostic on the first 16 valid prompts and up to 32 ground-truth response targets per prompt. Its PPL is compared to exactly those Stage-2 targets, not the full-response PPL. Each mode has its own propagated cache. Stage 4 reuses separate untouched prompt-cache copies from Stage 3; 16 prompts, greedy, at most 32 new tokens, existing generation_config EOS IDs.',
              'Cached decode status: '+json.dumps(decode_status), 'Generation status: '+json.dumps(generation_status),
              'Raw full-sequence PPL is not an official metric here. Reference weights are recovered local Q8_0 GGUF, not an original BF16 checkpoint. Source hashes, masks, target IDs, skip reasons and exact commands are under response_quant_eval/.']
    if (OUT/'answers.md').exists(): lines.append((OUT/'answers.md').read_text())
    (OUT.parent/'response_quant_eval_report.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--decode-examples',type=int,default=16);ap.add_argument('--decode-tokens',type=int,default=32);ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.report_only:
        status=json.loads((OUT/'stage_status.json').read_text());report(json.loads((OUT/'protocol.json').read_text()),**status);return
    started=time.time(); art=ROOT/'calibration_outputs/gguf-all-126-oasst1';manifest=json.loads((art/'manifest.json').read_text())
    pairs,skipped=pairs_from_archive()
    if not pairs:raise RuntimeError('No valid local prompt/assistant pairs')
    source=base.RecordedGGUFSource(MODEL,DATA,manifest);base.configure_blas('accelerate')
    source.tokenizer.chat_template=native_template(MODEL)
    examples=[]
    for pair in pairs:
        try:
            e=build_example(source.tokenizer,pair['messages'],source.model.config.max_position_embeddings)
            e.update({k:v for k,v in pair.items() if k!='messages'});examples.append(e)
        except ValueError as exc:skipped.append(dict(index=pair['index'],anchor_id=pair['anchor_id'],reason=str(exc)))
    if not examples:raise RuntimeError('No valid chat-templated examples')
    protocol=dict(examples=len(examples),prompt_tokens=sum(e['prompt_tokens'] for e in examples),response_tokens=sum(e['response_tokens'] for e in examples),
                  total_tokens=sum(e['total_tokens'] for e in examples),source_recovery=dict(original_anchors=32,valid=len(examples),skipped=skipped,
                  rule='original assistant anchor or ranked valid human assistant child of original user anchor; preserve all ancestors; no calibration-tree overlap'),
                  template=source.tokenizer.chat_template,template_source='local GGUF tokenizer.chat_template',
                  decode_examples=min(args.decode_examples,len(examples)),decode_token_limit=args.decode_tokens,
                  scoring='target content token at t uses logits/hidden at t-1; first content token included; no prompt/template/control targets')
    frozen=json.loads((ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json').read_text())
    filtered=dict(manifest,modules=[e for e in manifest['modules'] if e['module_name'].endswith('.down_proj')])
    specs,missing,hashes=base.load_artifacts(source,art,filtered,frozen);assert not missing and len(specs)==18
    provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=base.sha256_file(__file__),
                    model_sha256=manifest['gguf']['sha256'],archive_sha256=base.sha256_file(ARCHIVE),validation_sha256=base.sha256_file(DATA/'validation.jsonl'),
                    frozen_selection_sha256=base.sha256_file(ROOT/'diagnostics/bos_weight_sweep/frozen_selection.json'),artifact_hashes=hashes,
                    source_hashes={p:base.sha256_file(ROOT/p) for p in ['scripts/test_e2e_linear_quant.py','scripts/test_e2e_row_pot.py','scripts/quant_diagnostic_source.py']},
                    tokenization_sha256=hashlib.sha256(json.dumps(examples,sort_keys=True).encode()).hexdigest(),profile=base.Profile().metadata(),
                    library_versions=manifest['library_versions'],decode_examples=args.decode_examples,decode_tokens=args.decode_tokens)
    if (OUT/'provenance.json').exists():assert json.loads((OUT/'provenance.json').read_text())==provenance
    else:base.write_json(OUT/'provenance.json',provenance)
    base.write_json(OUT/'protocol.json',protocol);base.write_json(OUT/'examples.json',examples)
    with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=sys.argv,start_time=started))+'\n')
    # Required mask/pure-math tests before any evaluation forward.
    import io
    import unittest
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_response_eval_math.py')
    test=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'mask_tests.log').write_text(stream.getvalue())
    if not test.wasSuccessful():raise RuntimeError('response mask/math tests failed')
    print('PROTOCOL',protocol['examples'],protocol['prompt_tokens'],protocol['response_tokens'],'skipped',len(skipped),flush=True)
    run_prefill(source,specs,examples,args.decode_tokens);report(protocol)
    status=dict(decode_status=None,generation_status=None)
    try:
        seeds=run_cached(source,specs,examples,args.decode_examples,args.decode_tokens)
        status['decode_status']=dict(status='complete',examples=min(args.decode_examples,len(examples)),max_response_targets=args.decode_tokens)
    except Exception as exc:
        import traceback
        status['decode_status']=dict(status='failed',reason=str(exc),traceback=traceback.format_exc())
        status['generation_status']=dict(status='skipped',reason='Stage 3 did not complete')
    base.write_json(OUT/'stage_status.json',status);report(protocol,**status)
    if status['decode_status']['status']=='complete':
        try:
            rows=run_greedy(source,specs,examples,seeds,args.decode_examples)
            status['generation_status']=dict(status='complete',examples=len(rows)//3,max_new_tokens=32)
        except Exception as exc:
            import traceback
            status['generation_status']=dict(status='failed',reason=str(exc),traceback=traceback.format_exc())
    base.write_json(OUT/'stage_status.json',status);report(protocol,**status)
    print('COMPLETE',json.dumps(status),flush=True)


if __name__=='__main__':main()
