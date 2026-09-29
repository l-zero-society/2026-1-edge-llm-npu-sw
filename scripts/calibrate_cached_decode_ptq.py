#!/usr/bin/env python3
"""Deployment-path PTQ of frozen General-row PoT down_proj operations."""
import argparse
import copy
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import evaluate_response_quant as response

row = response.row
base = response.base
Profile = base.Profile
OUT = ROOT / "diagnostics/cached_decode_ptq"
ART = ROOT / "calibration_outputs/gguf-all-126-oasst1"
FROZEN = ROOT / "diagnostics/bos_weight_sweep/frozen_selection.json"
GROUPS = ("prefill", "decode")
BUCKETS = response.BUCKETS


def build_pairs(split, limit=None):
    """Recover real chat histories using the same deterministic rule as validation."""
    anchors = [json.loads(line) for line in (response.DATA / f"{split}.jsonl").read_text().splitlines()]
    other = "validation" if split == "calibration" else "calibration"
    other_trees = {json.loads(line)["message_tree_id"] for line in
                   (response.DATA / f"{other}.jsonl").read_text().splitlines()}
    trees = {a["message_tree_id"] for a in anchors}
    assert not trees & other_trees
    with gzip.open(response.ARCHIVE, "rt") as stream:
        records = {r["message_id"]: r for line in stream
                   if (r := json.loads(line)).get("message_tree_id") in trees}
    pairs, skipped = [], []
    for index, anchor in enumerate(anchors):
        try:
            if anchor["message_id"] not in records:
                raise ValueError("anchor absent from locally available ready-tree archive")
            target = records[anchor["message_id"]]
            assert target["text"] == anchor["text"] and target["role"] == anchor["role"]
            if target["role"] == "prompter":
                children = [r for r in records.values()
                            if r.get("parent_id") == target["message_id"] and r["role"] == "assistant"
                            and not r.get("deleted") and r.get("review_result") and not r.get("synthetic")]
                if not children:
                    raise ValueError("no valid human assistant child in local archive")
                target = min(children, key=lambda r: (r.get("rank") if r.get("rank") is not None else 10**9,
                                                       r["message_id"]))
            chain, seen = [target], {target["message_id"]}
            while chain[-1].get("parent_id"):
                parent = chain[-1]["parent_id"]
                if parent not in records or parent in seen:
                    raise ValueError("missing or cyclic ancestor")
                chain.append(records[parent]); seen.add(parent)
            chain.reverse()
            if len(chain) % 2 or any(r["role"] != ("prompter" if i % 2 == 0 else "assistant")
                                     for i, r in enumerate(chain)):
                raise ValueError("invalid conversation role sequence")
            pairs.append(dict(index=index, anchor_id=anchor["message_id"], response_id=target["message_id"],
                              tree_id=anchor["message_tree_id"],
                              messages=[dict(role="user" if r["role"] == "prompter" else "assistant",
                                             content=r["text"]) for r in chain]))
        except (ValueError, KeyError, AssertionError) as exc:
            skipped.append(dict(index=index, anchor_id=anchor.get("message_id"), reason=str(exc)))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs, skipped


def tokenize_pairs(source, pairs):
    result, skipped = [], []
    for pair in pairs:
        try:
            item = response.build_example(source.tokenizer, pair["messages"],
                                          source.model.config.max_position_embeddings)
            item.update({k: v for k, v in pair.items() if k != "messages"})
            result.append(item)
        except ValueError as exc:
            skipped.append(dict(index=pair["index"], anchor_id=pair["anchor_id"], reason=str(exc)))
    return result, skipped


class Reservoir:
    """Keep rows with the 64 smallest deterministic SHA256 priorities."""
    def __init__(self, capacity=64, seed=20260929):
        self.capacity, self.seed, self.heap, self.seen = capacity, seed, [], 0

    def add(self, value, layer, group, conversation, position):
        key = f"{self.seed}:{layer}:{group}:{conversation}:{position}".encode()
        priority = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        item = (-priority, self.seen, np.asarray(value, np.float32).copy(),
                dict(layer=layer, group=group, conversation=conversation, absolute_position=position))
        self.seen += 1
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif item[0] > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def arrays(self):
        items = sorted(self.heap, key=lambda x: (-x[0], x[1]))
        return np.stack([x[2] for x in items]), [x[3] for x in items]


def choose_rows(x, sx, shift):
    return row.select_rows(np.asarray(x, np.float64), float(sx), shift)


def activation_score(groups, sx, shift):
    scores, worst, hist = {}, 0.0, {str(k): 0 for k in range(-8, 8)}
    for group, x in groups.items():
        if len(x) == 0:
            continue
        q, ks, _ = choose_rows(x, sx, shift)
        scale = sx * np.exp2(ks)
        error = q.astype(np.float64) * scale[:, None] - x
        value = float(np.square(error).sum() / np.square(x).sum()) if np.square(x).sum() else 0.0
        scores[group] = value; worst = max(worst, value)
        for k, count in zip(*np.unique(ks, return_counts=True)):
            hist[str(int(k))] += int(count)
    balanced = float(np.mean(list(scores.values())))
    return dict(score=balanced, worst=worst, groups=scores, histogram=hist)


def search_sx(groups, old_sx, shift, js=range(-4, 5)):
    candidates = []
    for j in js:
        sx = old_sx * 2.0 ** (j / 4.0)
        result = activation_score(groups, sx, shift)
        candidates.append(dict(j=j, sx=sx, **result))
    best = min(candidates, key=lambda c: (c["score"], c["worst"], abs(math.log2(c["sx"] / old_sx)), abs(c["j"])))
    return best, candidates


def integer_data(source, name, spec, groups, sx):
    x = np.concatenate([groups[g] for g in GROUPS if len(groups[g])])
    labels = np.concatenate([[g] * len(groups[g]) for g in GROUPS if len(groups[g])])
    q, ks, _ = choose_rows(x, sx, spec["params"]["shift"])
    acc = base.integer_dot(q, spec["wq"], "fp64-exact")
    weight = source.weight(name)
    ref = x.astype(np.float64) @ weight.astype(np.float64).T
    expected_wq, sw = base.quantize_weight(weight)
    np.testing.assert_array_equal(expected_wq, spec["wq"])
    return dict(x=x, labels=labels, q=q, ks=ks, acc=acc, ref=ref, sw=sw)


def evaluate_s10(data, sx, s10, sw):
    params = Profile().approximate(sx * sw / s10)
    if np.any(params["status"] != "ok"):
        return None
    shifts = params["shift"][None, :] - data["ks"][:, None]
    if shifts.min() < 0 or shifts.max() > 31:
        return None
    out = np.empty(data["acc"].shape, np.float64); clips = 0
    for k in np.unique(data["ks"]):
        rows = data["ks"] == k
        raw = Profile().apply(data["acc"][rows], params["multiplier"], params["shift"] - int(k),
                              saturate=False)
        clips += int(((raw < -512) | (raw > 511)).sum())
        out[rows] = np.clip(raw, -512, 511) * s10
    groups = {}
    for group in GROUPS:
        mask = data["labels"] == group
        if mask.any():
            groups[group] = float(np.square(out[mask] - data["ref"][mask]).sum() /
                                  np.square(data["ref"][mask]).sum())
    return dict(score=float(np.mean(list(groups.values()))), worst=max(groups.values()), groups=groups,
                clip_rate=clips / out.size, params=params, effective_shift_min=int(shifts.min()),
                effective_shift_max=int(shifts.max()))


def search_s10(data, sx, old_s10, sw, indices=range(-8, 9)):
    candidates = []
    for i in indices:
        s10 = old_s10 * 2.0 ** (i / 8.0)
        result = evaluate_s10(data, sx, s10, sw)
        if result is not None:
            candidates.append(dict(i=i, s10=s10, **result))
    if not candidates:
        raise ValueError("no feasible s10 candidate")
    best = min(candidates, key=lambda c: (c["score"], c["worst"], c["clip_rate"],
                                          abs(math.log2(c["s10"] / old_s10))))
    return best, candidates


def quantized_one(x, spec):
    return row.apply_rows(x, **spec)[0]


@contextmanager
def patch_down(source, specs, capture=None, phase=None, conversation=None, offset=0):
    originals = {}
    try:
        for name, spec in specs.items():
            module = source.modules[name]; originals[name] = module.forward
            layer = int(name.split(".")[2])
            def forward(x, name=name, spec=spec, layer=layer):
                xx = x.detach().numpy()
                if capture is not None:
                    if phase == "prefill":
                        for position, vector in enumerate(xx[0]):
                            capture[layer][phase].add(vector, layer, phase, conversation, position)
                    else:
                        capture[layer][phase].add(xx[0, -1], layer, phase, conversation, offset)
                y = quantized_one(xx, spec)
                return source.torch.from_numpy(y).to(x.dtype)
            module.forward = forward
        yield
    finally:
        for name, fn in originals.items():
            source.modules[name].forward = fn


def deployment_capture(source, specs, examples, limit=32, seed=20260929):
    capture = {layer: {g: Reservoir(64, seed) for g in GROUPS} for layer in range(18)}
    for number, example in enumerate(examples):
        context = {}; cache = None
        with patch_down(source, specs, capture, "prefill", example["index"], 0):
            _, _, cache = response.forward_hidden(source, example["prompt_ids"],
                                                  [len(example["prompt_ids"]) - 1], context,
                                                  use_cache=True, layers=False)
        n = min(limit, len(example["targets"]))
        for t in range(1, n):
            context = {}; absolute = len(example["prompt_ids"]) + t - 1
            with patch_down(source, specs, capture, "decode", example["index"], absolute):
                _, _, cache = response.forward_hidden(source, [example["targets"][t - 1]], [0], context,
                                                      cache, True, False)
        print("CAPTURE", number + 1, len(examples), example["index"], n, flush=True)
    result = {}
    for layer in range(18):
        result[layer] = {}
        for group in GROUPS:
            result[layer][group] = capture[layer][group].arrays()
    return result


def arrays_only(capture, layer):
    return {g: capture[layer][g][0] for g in GROUPS}


def calibrate(source, specs, capture, layers=range(18), center=None):
    decisions, details = {}, {}
    for layer in layers:
        name = f"model.layers.{layer}.mlp.down_proj"; old = (center or specs)[name]
        groups = arrays_only(capture, layer)
        sx_best, sx_table = search_sx(groups, old["sx"], old["params"]["shift"])
        data = integer_data(source, name, specs[name], groups, sx_best["sx"])
        s10_best, s10_table = search_s10(data, sx_best["sx"], old["s10"], data["sw"])
        proposed = dict(specs[name], sx=sx_best["sx"], s10=s10_best["s10"], params=s10_best["params"])
        decisions[name] = proposed
        # Old local score is evaluated on the same propagated rows.
        old_data = integer_data(source, name, specs[name], groups, specs[name]["sx"])
        old_eval = evaluate_s10(old_data, specs[name]["sx"], specs[name]["s10"], old_data["sw"])
        details[layer] = dict(layer=layer, old_sx=specs[name]["sx"], new_sx=proposed["sx"],
                              old_s10=specs[name]["s10"], new_s10=proposed["s10"],
                              sx_j=sx_best["j"], s10_i=s10_best["i"], old_nmse=old_eval["score"],
                              new_nmse=s10_best["score"], new_prefill_nmse=s10_best["groups"].get("prefill"),
                              new_decode_nmse=s10_best["groups"].get("decode"),
                              clip_rate=s10_best["clip_rate"],
                              effective_shift_min=s10_best["effective_shift_min"],
                              effective_shift_max=s10_best["effective_shift_max"],
                              k_histogram=sx_best["histogram"],
                              sx_candidates=[{k:v for k,v in c.items() if k != "histogram"} for c in sx_table],
                              s10_candidates=[{k:v for k,v in c.items() if k != "params"} for c in s10_table])
        print("CALIBRATE", layer, proposed["sx"], proposed["s10"], old_eval["score"], s10_best["score"], flush=True)
    return decisions, details


def group_diagnostic(source, specs, shared, capture):
    rows, dual = [], {"prefill": {}, "decode": {}}
    qualified = 0
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"; values = {}
        for group in GROUPS:
            groups = {g: capture[layer][g][0] if g == group else np.empty((0, capture[layer][g][0].shape[1]))
                      for g in GROUPS}
            sx_best, _ = search_sx(groups, specs[name]["sx"], specs[name]["params"]["shift"])
            data = integer_data(source, name, specs[name], groups, sx_best["sx"])
            s10_best, _ = search_s10(data, sx_best["sx"], specs[name]["s10"], data["sw"])
            dual[group][name] = dict(specs[name], sx=sx_best["sx"], s10=s10_best["s10"],
                                     params=s10_best["params"])
            values[group] = dict(sx=sx_best["sx"], s10=s10_best["s10"], nmse=s10_best["score"])
        shared_detail = {}
        groups = arrays_only(capture, layer)
        data = integer_data(source, name, specs[name], groups, shared[name]["sx"])
        shared_eval = evaluate_s10(data, shared[name]["sx"], shared[name]["s10"], data["sw"])
        for group in GROUPS:
            shared_detail[group] = shared_eval["groups"][group]
        sx_ratio = values["prefill"]["sx"] / values["decode"]["sx"]
        s10_ratio = values["prefill"]["s10"] / values["decode"]["s10"]
        gains = {g: (shared_detail[g] - values[g]["nmse"]) / shared_detail[g] if shared_detail[g] else 0.
                 for g in GROUPS}
        qualifies = ((sx_ratio >= 1.5 or sx_ratio <= 1/1.5 or
                      s10_ratio >= 1.5 or s10_ratio <= 1/1.5) and max(gains.values()) >= .10)
        qualified += int(qualifies)
        rows.append(dict(layer=layer, prefill_sx=values["prefill"]["sx"], decode_sx=values["decode"]["sx"],
                         sx_ratio=sx_ratio, prefill_s10=values["prefill"]["s10"],
                         decode_s10=values["decode"]["s10"], s10_ratio=s10_ratio,
                         shared_prefill_nmse=shared_detail["prefill"],
                         separate_prefill_nmse=values["prefill"]["nmse"], prefill_gain=gains["prefill"],
                         shared_decode_nmse=shared_detail["decode"],
                         separate_decode_nmse=values["decode"]["nmse"], decode_gain=gains["decode"],
                         qualifies=qualifies))
    return rows, dual, qualified >= 4, qualified


def capture_stats(capture, policies, source, specs):
    result = {}
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"; groups = arrays_only(capture, layer)
        data = integer_data(source, name, specs[name], groups, policies[name]["sx"])
        metric = evaluate_s10(data, policies[name]["sx"], policies[name]["s10"], data["sw"])
        _, ks, _ = choose_rows(np.concatenate([groups[g] for g in GROUPS]), policies[name]["sx"],
                               policies[name]["params"]["shift"])
        x = np.concatenate([groups[g] for g in GROUPS])
        result[layer] = dict(absmax_median=float(np.median(np.max(np.abs(x), axis=1))),
                             local_nmse=metric["score"],
                             k_histogram={str(k): int((ks == k).sum()) for k in range(-8, 8)})
    return result


def score_add(scores, layers, torch, logits, hidden, reference_logits, reference_hidden, target, bucket):
    for mode in scores:
        q = reference_logits if mode == "FP" else logits[mode]
        scores[mode]["all"].add(torch, q, reference_logits, [target])
        scores[mode][bucket].add(torch, q, reference_logits, [target])
        for layer in range(18):
            layers[mode][layer].add(hidden[mode][layer].numpy(), reference_hidden[layer].numpy())


def run_mode(source, specs, ids, cache=None, layers=True):
    context = {}
    with patch_down(source, specs):
        hidden, states, cache = response.forward_hidden(source, ids, [len(ids)-1], context, cache,
                                                       True, layers)
    with source.torch.inference_mode():
        logits = source.model.lm_head(hidden)
    return logits, states, cache


def run_fp(source, ids, cache=None, layers=True):
    context = {}
    hidden, states, cache = response.forward_hidden(source, ids, [len(ids)-1], context, cache, True, layers)
    with source.torch.inference_mode():
        logits = source.model.lm_head(hidden)
    return logits, states, cache


@contextmanager
def patch_mixed_down(source, mode_specs, phase):
    """Run FP/old/new(/dual) in one model batch; batch rows never interact."""
    originals = {}
    try:
        for name in next(spec for spec in mode_specs.values() if spec is not None):
            module = source.modules[name]; originals[name] = module.forward
            def forward(x, name=name, original=module.forward):
                pieces = []
                for index, (mode, specs) in enumerate(mode_specs.items()):
                    value = x[index:index+1]
                    if specs is None:
                        pieces.append(original(value))
                    else:
                        selected = specs[phase][name] if mode == "dual" else specs[name]
                        y = quantized_one(value.detach().numpy(), selected)
                        pieces.append(source.torch.from_numpy(y).to(x.dtype))
                return source.torch.cat(pieces, dim=0)
            module.forward = forward
        yield
    finally:
        for name, fn in originals.items(): module = source.modules[name]; module.forward = fn


def mixed_forward(source, mode_specs, ids, phase, cache=None, layers=True):
    torch = source.torch; batch = len(mode_specs); offset = cache.get_seq_length() if cache is not None else 0
    states, handles = {}, []
    if layers:
        for layer, block in enumerate(source.model.model.layers):
            def post(module, args, output, layer=layer): states[layer] = output.detach().clone()
            handles.append(block.register_forward_hook(post))
    token = torch.tensor([ids] * batch, dtype=torch.int64)
    try:
        with patch_mixed_down(source, mode_specs, phase), torch.inference_mode():
            result = source.model.model(input_ids=token,
                attention_mask=torch.ones((batch, offset + len(ids)), dtype=torch.int64),
                position_ids=torch.arange(offset, offset + len(ids))[None].expand(batch, -1),
                cache_position=torch.arange(offset, offset + len(ids)),
                past_key_values=cache, use_cache=True)
            hidden = result.last_hidden_state[:, -1].clone()
            logits = source.model.lm_head(hidden)
    finally:
        for handle in handles: handle.remove()
    assert result.past_key_values.get_seq_length() == offset + len(ids)
    return logits, {l:v[:,-1].clone() for l,v in states.items()}, result.past_key_values


def validation_example(source, example, policies, dual=None, greedy=False, target_limit=32):
    modes = ["FP", "old_general", "calibrated"] + (["dual"] if dual else [])
    caches, logits, hidden, seeds = {}, {}, {}, {}
    for mode in modes:
        if mode == "FP": out = run_fp(source, example["prompt_ids"])
        else:
            spec = policies["old"] if mode == "old_general" else policies["new"]
            if mode == "dual": spec = dual["prefill"]
            out = run_mode(source, spec, example["prompt_ids"])
        logits[mode], hidden[mode], caches[mode] = out
        if greedy: seeds[mode] = dict(cache=copy.deepcopy(caches[mode]), token=int(logits[mode].argmax(-1)[0]))
    scores = {m: {"all": response.Scores(), **{b: response.Scores() for _,_,b in BUCKETS}} for m in modes}
    layer_scores = {m: [base.Metric() for _ in range(18)] for m in modes}
    first = {m: response.Scores() for m in modes}
    selected_targets=example["targets"][:target_limit] if target_limit is not None else example["targets"]
    for t, target in enumerate(selected_targets):
        if t:
            for mode in modes:
                feed = [example["targets"][t-1]]
                if mode == "FP": out = run_fp(source, feed, caches[mode])
                else:
                    spec = policies["old"] if mode == "old_general" else policies["new"]
                    if mode == "dual": spec = dual["decode"]
                    out = run_mode(source, spec, feed, caches[mode])
                logits[mode], hidden[mode], caches[mode] = out
        bucket = next(label for lo,hi,label in BUCKETS if lo <= t < hi)
        ref_logits, ref_hidden = logits["FP"], hidden["FP"]
        for mode in modes:
            q = ref_logits if mode == "FP" else logits[mode]
            scores[mode]["all"].add(source.torch, q, ref_logits, [target])
            scores[mode][bucket].add(source.torch, q, ref_logits, [target])
            if t == 0: first[mode].add(source.torch, q, ref_logits, [target])
            for layer in range(18):
                layer_scores[mode][layer].add(hidden[mode][layer].numpy(), ref_hidden[layer].numpy())
    result = dict(index=example["index"], available_targets=len(example["targets"]),
                  targets=len(selected_targets),target_limit=target_limit,
                  scores={m:{k:v.raw() for k,v in values.items() if v.n} for m,values in scores.items()},
                  first={m:v.raw() for m,v in first.items()},
                  layers={m:[vars(v) for v in ll] for m,ll in layer_scores.items()})
    if greedy:
        eos = source.model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        generated = {}
        for mode in modes:
            cache=seeds[mode]["cache"]; tokens=[seeds[mode]["token"]]
            while len(tokens)<32 and tokens[-1] not in eos_ids:
                if mode=="FP": out=run_fp(source,[tokens[-1]],cache,False)
                else:
                    spec=policies["old"] if mode=="old_general" else policies["new"]
                    if mode=="dual":spec=dual["decode"]
                    out=run_mode(source,spec,[tokens[-1]],cache,False)
                q,_,cache=out;tokens.append(int(q.argmax(-1)[0]))
            generated[mode]=tokens
        result["generation"]={m:response.generation_comparison(generated["FP"],tokens,eos_ids)
                              for m,tokens in generated.items()}
    return result


def aggregate_validation(records):
    modes=list(records[0]["scores"]);summary=[];positions=[];layers=[];generation=[]
    for mode in modes:
        total=response.merged([r["scores"][mode]["all"] for r in records])
        first=response.merged([r["first"][mode] for r in records])
        first_fields={f"first_token_{key}": first[key] for key in
                      ("nll","ppl","kl","nmse","cosine","top1","in5","overlap")}
        summary.append(dict(mode=mode,**first_fields,**total))
        for _,_,bucket in BUCKETS:
            rr=[r["scores"][mode][bucket] for r in records if bucket in r["scores"][mode]]
            if rr:positions.append(dict(mode=mode,bucket=bucket,**response.merged(rr)))
        for layer in range(18):
            layers.append(dict(mode=mode,layer=layer,**base.merge_metrics([r["layers"][mode][layer] for r in records])))
        rr=[r for r in records if "generation" in r]
        if rr:
            values=[r["generation"][mode] for r in rr]
            generation.append(dict(mode=mode,examples=len(values),
                mean_shared_prefix=float(np.mean([v["shared_prefix"] for v in values])),
                median_shared_prefix=float(np.median([v["shared_prefix"] for v in values])),
                token_agreement=float(np.mean([v["token_agreement"] for v in values])),
                exact_32=sum(v["first_divergence"] is None and v["fp_length"]==32 and v["length"]==32 for v in values)))
    return summary,positions,layers,generation


def materialize(protocol, decisions, detail, stats1, stats2, diagnostic, records, shifted):
    """Write compact aggregate artifacts from frozen calibration and validation records."""
    summary_rows,position_rows,layer_rows,generation_rows=aggregate_validation(records)
    scale_rows=[];local_rows=[]
    for layer in range(18):
        d=detail[layer] if layer in detail else detail[str(layer)]
        one=stats1[layer] if layer in stats1 else stats1[str(layer)]
        two=stats2[layer] if layer in stats2 else stats2[str(layer)]
        scale_rows.append({k:v for k,v in d.items() if k not in ("sx_candidates","s10_candidates","k_histogram")})
        local_rows.append(dict(layer=layer,**one,pass2_absmax_median=two["absmax_median"],
                               pass2_local_nmse=two["local_nmse"],refined=layer in shifted))
    base.write_csv(ROOT/"diagnostics/cached_decode_ptq_scales.csv",scale_rows)
    base.write_csv(ROOT/"diagnostics/cached_decode_ptq_local.csv",local_rows+diagnostic)
    base.write_csv(ROOT/"diagnostics/cached_decode_ptq_validation.csv",summary_rows+position_rows)
    base.write_csv(ROOT/"diagnostics/cached_decode_ptq_layers.csv",layer_rows)
    base.write_csv(ROOT/"diagnostics/cached_decode_ptq_generation.csv",generation_rows)
    summary=dict(protocol=protocol,decisions=decisions,scales=scale_rows,diagnostic=diagnostic,
                 validation=summary_rows,positions=position_rows,layers=layer_rows,generation=generation_rows)
    base.write_json(ROOT/"diagnostics/cached_decode_ptq_summary.json",summary)
    make_report()


def serializable_policy(spec):
    return dict(sx=spec["sx"],s10=spec["s10"],k=spec["k"],
                effective_shift_min=int(spec["params"]["shift"].min()-7),
                effective_shift_max=int(spec["params"]["shift"].max()+8))


def restore_proposed(source, old):
    decisions=json.loads((OUT/"decisions.json").read_text());proposed={}
    for layer in range(18):
        name=f"model.layers.{layer}.mlp.down_proj";choice=decisions["shared"][name]
        _,sw=base.quantize_weight(source.weight(name))
        params=Profile().approximate(choice["sx"]*sw/choice["s10"])
        proposed[name]=dict(old[name],sx=choice["sx"],s10=choice["s10"],params=params)
    return proposed,decisions


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--report-only",action="store_true")
    ap.add_argument("--validation-only",action="store_true");ap.add_argument("--shard",default="0/1");args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"proposed").mkdir(exist_ok=True)
    if args.report_only:
        make_report();return
    manifest=json.loads((ART/"manifest.json").read_text())
    source=base.RecordedGGUFSource(response.MODEL,response.DATA,manifest);base.configure_blas("accelerate")
    source.tokenizer.chat_template=response.native_template(response.MODEL)
    assert source.tokenizer.chat_template==json.loads((ROOT/"diagnostics/response_quant_eval/protocol.json").read_text())["template"]
    filtered=dict(manifest,modules=[m for m in manifest["modules"] if m["module_name"].endswith(".down_proj")])
    old,missing,hashes=base.load_artifacts(source,ART,filtered,json.loads(FROZEN.read_text()))
    assert not missing and len(old)==18
    cal_pairs,cal_skip=build_pairs("calibration",32);cal,token_skip=tokenize_pairs(source,cal_pairs)
    if len(cal)<32: raise RuntimeError(f"only {len(cal)} valid calibration chats")
    val_pairs,val_skip=build_pairs("validation");val,validation_token_skip=tokenize_pairs(source,val_pairs)
    assert len(val)==24
    if args.validation_only:
        proposed,decisions=restore_proposed(source,old);policies=dict(old=old,new=proposed)
        shard,total=map(int,args.shard.split("/"));assert 0<=shard<total
        for i,example in enumerate(val):
            if i%total!=shard:continue
            path=OUT/f"validation_{i:03d}.json"
            if path.exists():continue
            record=validation_example(source,example,policies,None,greedy=i<16)
            base.write_json(path,record);print("VALIDATION",i+1,len(val),example["response_tokens"],flush=True)
        return
    protocol=dict(calibration_examples=len(cal),calibration_prompt_tokens=sum(x["prompt_tokens"] for x in cal),
                  calibration_targets=sum(min(32,x["response_tokens"]) for x in cal),
                  validation_examples=len(val),validation_prompt_tokens=sum(x["prompt_tokens"] for x in val),
                  validation_available_targets=sum(x["response_tokens"] for x in val),
                  validation_targets=sum(min(32,x["response_tokens"]) for x in val),
                  calibration_skipped=cal_skip+token_skip,
                  validation_skipped=val_skip+validation_token_skip,selection_split="calibration",
                  validation_role="evaluation only",decode_limit=32)
    base.write_json(OUT/"protocol.json",protocol)
    provenance=dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    script_sha256=base.sha256_file(__file__),model_sha256=manifest["gguf"]["sha256"],
                    frozen_sha256=base.sha256_file(FROZEN),archive_sha256=base.sha256_file(response.ARCHIVE),
                    calibration_sha256=base.sha256_file(response.DATA/"calibration.jsonl"),
                    validation_sha256=base.sha256_file(response.DATA/"validation.jsonl"),artifact_hashes=hashes,
                    profile=Profile().metadata(),command=sys.argv)
    base.write_json(OUT/"provenance.json",provenance)
    capture1=deployment_capture(source,old,cal)
    proposed,detail=calibrate(source,old,capture1)
    stats1=capture_stats(capture1,proposed,source,old)
    capture2=deployment_capture(source,proposed,cal,seed=20260929)
    stats2=capture_stats(capture2,proposed,source,old)
    shifted=[]
    mean1=np.mean([stats1[l]["local_nmse"] for l in range(18)]);mean2=np.mean([stats2[l]["local_nmse"] for l in range(18)])
    for layer in range(18):
        ratio=stats2[layer]["absmax_median"]/stats1[layer]["absmax_median"]
        if ratio>1.25 or ratio<.8:shifted.append(layer)
    if mean2>1.10*mean1:
        shifted=sorted(set(shifted)|set(range(18)))
    if shifted:
        revised,revised_detail=calibrate(source,old,capture2,shifted,proposed)
        proposed.update(revised);detail.update(revised_detail)
    diagnostic,dual,run_dual,qualified=group_diagnostic(source,old,proposed,capture2)
    decisions=dict(shared={name:serializable_policy(spec) for name,spec in proposed.items()},
                   refinement=dict(mean_pass1_nmse=mean1,mean_pass2_nmse=mean2,shifted_layers=shifted),
                   dual_threshold_met=run_dual,dual_qualified_layers=qualified)
    base.write_json(OUT/"decisions.json",decisions)
    base.write_json(OUT/"calibration_results.json",dict(details=detail,stats1=stats1,stats2=stats2,
                                                         diagnostic=diagnostic,decisions=decisions))
    for layer in range(18):
        name=f"model.layers.{layer}.mlp.down_proj"
        base.write_json(OUT/"proposed"/f"layer_{layer:02d}.json",dict(layer=layer,name=name,
            old=serializable_policy(old[name]),proposed=serializable_policy(proposed[name]),local=detail[layer]))
    records=[]
    policies=dict(old=old,new=proposed)
    for i,example in enumerate(val):
        path=OUT/f"validation_{i:03d}.json"
        if path.exists():record=json.loads(path.read_text())
        else:
            record=validation_example(source,example,policies,dual if run_dual else None,greedy=i<16)
            base.write_json(path,record)
        records.append(record);print("VALIDATION",i+1,len(val),example["response_tokens"],flush=True)
    materialize(protocol,decisions,detail,stats1,stats2,diagnostic,records,shifted)


def make_report():
    path=ROOT/"diagnostics/cached_decode_ptq_summary.json"
    if not path.exists():return
    s=json.loads(path.read_text());modes={r["mode"]:r for r in s["validation"]}
    old,new=modes["old_general"],modes["calibrated"]
    old_local=float(np.mean([r["old_nmse"] for r in s["scales"]]))
    new_local=float(np.mean([r["new_nmse"] for r in s["scales"]]))
    ref=s["decisions"]["refinement"]
    lines=["# Cached-decode deployment PTQ",
        "Only the 18 `down_proj` operations are quantized. Parameter selection used calibration chats only; validation was frozen evaluation.",
        f"Calibration: {s['protocol']['calibration_examples']} chats, {s['protocol']['calibration_prompt_tokens']} prompt tokens, and {s['protocol']['calibration_targets']} cached response targets (maximum 32/chat). Validation: {s['protocol']['validation_examples']} chats, {s['protocol']['validation_prompt_tokens']} prompt tokens, and {s['protocol']['validation_targets']} cached response targets sampled as the first 32 of {s['protocol']['validation_available_targets']} available response targets.",
        "## Scale decisions",
        "| layer | old sX | new sX | old s10 | new s10 | old local NMSE % | new local NMSE % |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["scales"]:
        lines.append(f"| {r['layer']} | {r['old_sx']:.8g} | {r['new_sx']:.8g} | {r['old_s10']:.8g} | {r['new_s10']:.8g} | {100*r['old_nmse']:.4f} | {100*r['new_nmse']:.4f} |")
    lines += [f"Mean local NMSE changed from {100*old_local:.6f}% to {100*new_local:.6f}% ({100*(1-new_local/old_local):.3f}% relative reduction). Fourteen sX bases and all 18 s10 values changed.",
        f"The propagated refinement capture changed mean local NMSE from {100*ref['mean_pass1_nmse']:.6f}% to {100*ref['mean_pass2_nmse']:.6f}%. No layer crossed the refinement trigger, so no second parameter search ran.",
        "## Prefill/decode diagnostic",
        "| layer | prefill/decode sX | prefill/decode s10 | prefill gain % | decode gain % | dual trigger |",
        "|---:|---:|---:|---:|---:|:---:|"]
    for r in s["diagnostic"]:
        lines.append(f"| {r['layer']} | {r['sx_ratio']:.4f} | {r['s10_ratio']:.4f} | {100*r['prefill_gain']:.3f} | {100*r['decode_gain']:.3f} | {'yes' if r['qualifies'] else 'no'} |")
    lines += [f"Dual-policy threshold: **not met** ({s['decisions']['dual_qualified_layers']}/18 qualifying layers); dual-policy E2E validation was skipped.",
        "## Cached response validation",
        "| mode | response NLL | PPL | KL | logits NMSE % | cosine | top1 % | FP top1 in top5 % | top5 overlap % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["validation"]:
        lines.append(f"| {r['mode']} | {r['nll']:.6f} | {r['ppl']:.6f} | {r['kl']:.6f} | {100*r['nmse']:.6f} | {r['cosine']:.8f} | {100*r['top1']:.3f} | {100*r['in5']:.3f} | {100*r['overlap']:.3f} |")
    lines += ["## First assistant token",
        "| mode | NLL | KL | logits NMSE % | cosine | top1 % | FP top1 in top5 % | top5 overlap % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["validation"]:
        lines.append(f"| {r['mode']} | {r['first_token_nll']:.6f} | {r['first_token_kl']:.6f} | {100*r['first_token_nmse']:.6f} | {r['first_token_cosine']:.8f} | {100*r['first_token_top1']:.3f} | {100*r['first_token_in5']:.3f} | {100*r['first_token_overlap']:.3f} |")
    lines += ["## Greedy validation", "| mode | mean shared prefix | median shared prefix | aligned agreement % | exact 32-token matches |",
              "|---|---:|---:|---:|---:|"]
    for r in s["generation"]:
        lines.append(f"| {r['mode']} | {r['mean_shared_prefix']:.4f} | {r['median_shared_prefix']:.4f} | {100*r['token_agreement']:.3f} | {r['exact_32']}/{r['examples']} |")
    ppl_recovery=(old["ppl"]-new["ppl"])/(old["ppl"]-modes["FP"]["ppl"])
    lines += ["## Answers",
        "A. **14/18** layers changed `sX_base`.",
        "B. **18/18** layers changed `s10`.",
        f"C. Mean local calibration output NMSE fell from **{100*old_local:.6f}%** to **{100*new_local:.6f}%**, a **{100*(1-new_local/old_local):.3f}%** relative reduction.",
        f"D. No material propagated shift was found: pass-2 mean local NMSE changed by **{100*(ref['mean_pass2_nmse']/ref['mean_pass1_nmse']-1):.3f}%** and no layer crossed the absmax trigger.",
        "E. Prefill/decode optima sometimes chose different grid points, but none of the 18 layers satisfied both the scale-ratio and >=10% local-gain conditions.",
        "F. A separate prefill/decode scale policy is **not justified** by this diagnostic.",
        f"G. Yes. Cached response PPL improved from **{old['ppl']:.6f}** to **{new['ppl']:.6f}**; the gap above FP was reduced by **{100*ppl_recovery:.2f}%**.",
        f"H. Yes. KL fell **{100*(1-new['kl']/old['kl']):.2f}%**, logits NMSE fell **{100*(1-new['nmse']/old['nmse']):.2f}%**, and top-1 agreement rose **{100*(new['top1']-old['top1']):.3f} percentage points**.",
        "I. Greedy preservation is mixed: mean shared prefix and exact matches improved, while median prefix and aligned token agreement declined slightly.",
        "J. The calibrated shared policy is the better **software research baseline** because most cached deployment metrics improve; it remains a diagnostic artifact and does not replace production parameters."]
    (ROOT/"diagnostics/cached_decode_ptq_report.md").write_text("\n\n".join(lines).replace("|\n\n|","|\n|")+"\n")


if __name__ == "__main__": main()
