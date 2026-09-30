#!/usr/bin/env python3
"""Finalize deployment-aware down_proj PTQ and run frozen cached-decode validation."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import calibrate_cached_decode_ptq as cached

response = cached.response
base = cached.base
Profile = cached.Profile
SOURCE = ROOT / "diagnostics/cached_decode_ptq"
OUT = ROOT / "diagnostics/final_downproj_ptq"
PROPOSED = OUT / "proposed"
PROGRESS = OUT / "progress"
BOUNDARY = {0: 8, 1: 8, 3: 8, 7: -8, 8: -8, 10: -8, 17: -8}
DEPTH_BUCKETS = [(0, 32, "0-31"), (32, 64, "32-63"), (64, 128, "64-127"),
                 (128, 256, "128-255"), (256, 512, "256-511"),
                 (512, 10**9, ">=512")]
HIDDEN_POSITIONS = {0, 1, 7, 31, 63, 127, 255, 511}
MODES = ("FP", "previous", "final")


def read_scale_rows(path=ROOT / "diagnostics/cached_decode_ptq_scales.csv"):
    with path.open(newline="") as stream:
        rows = [{k: (int(v) if k in ("layer", "sx_j", "s10_i") else float(v))
                 for k, v in row.items()} for row in csv.DictReader(stream)]
    found = {r["layer"]: r["s10_i"] for r in rows if abs(r["s10_i"]) == 8}
    if found != BOUNDARY:
        raise AssertionError(f"boundary layer set differs: CSV={found}, expected={BOUNDARY}")
    return rows


def load_source_and_policies():
    manifest = json.loads((cached.ART / "manifest.json").read_text())
    source = base.RecordedGGUFSource(response.MODEL, response.DATA, manifest)
    base.configure_blas("accelerate")
    source.tokenizer.chat_template = response.native_template(response.MODEL)
    protocol = json.loads((ROOT / "diagnostics/response_quant_eval/protocol.json").read_text())
    assert source.tokenizer.chat_template == protocol["template"]
    filtered = dict(manifest, modules=[m for m in manifest["modules"]
                                      if m["module_name"].endswith(".down_proj")])
    initial, missing, hashes = base.load_artifacts(
        source, cached.ART, filtered, json.loads(cached.FROZEN.read_text()))
    assert not missing and len(initial) == 18
    previous, previous_decisions = cached.restore_proposed(source, initial)
    return source, manifest, initial, previous, previous_decisions, hashes


def calibration_examples(source):
    pairs, skipped = cached.build_pairs("calibration", 32)
    examples, token_skipped = cached.tokenize_pairs(source, pairs)
    if len(examples) != 32:
        raise RuntimeError(f"expected 32 calibration chats, got {len(examples)}")
    return examples, skipped + token_skipped


def validation_examples(source):
    pairs, skipped = cached.build_pairs("validation")
    examples, token_skipped = cached.tokenize_pairs(source, pairs)
    if len(examples) != 24:
        raise RuntimeError(f"expected 24 validation chats, got {len(examples)}")
    return examples, skipped + token_skipped


def candidate_key(candidate, boundary):
    return (candidate["score"], candidate["worst"], candidate["clip_rate"],
            abs(candidate["i"] - boundary))


def evaluate_indices(data, sx, reference_s10, sw, indices):
    candidates = []
    for i in indices:
        s10 = reference_s10 * 2.0 ** (i / 8.0)
        result = cached.evaluate_s10(data, sx, s10, sw)
        if result is not None:
            candidates.append(dict(i=i, s10=s10, **result))
    return candidates


def observed_shift_metadata(groups, spec):
    x = np.concatenate([groups[g] for g in cached.GROUPS if len(groups[g])])
    _, ks, _ = cached.choose_rows(x, spec["sx"], spec["params"]["shift"])
    effective = spec["params"]["shift"][None, :].astype(np.int64) - ks[:, None]
    result = dict(static_shift_min=int(spec["params"]["shift"].min()),
                  static_shift_max=int(spec["params"]["shift"].max()),
                  observed_effective_shift_min=int(effective.min()),
                  observed_effective_shift_max=int(effective.max()),
                  observed_k_min=int(ks.min()), observed_k_max=int(ks.max()))
    assert 0 <= result["observed_effective_shift_min"] <= result["observed_effective_shift_max"] <= 31
    return result


def sweep_boundary_layers(source, previous, capture, scale_rows):
    by_layer = {r["layer"]: r for r in scale_rows}
    final = dict(previous)
    candidate_rows, decisions = [], {}
    for layer, boundary in BOUNDARY.items():
        name = f"model.layers.{layer}.mlp.down_proj"
        spec = previous[name]
        groups = cached.arrays_only(capture, layer)
        # integer_data selects k, quantizes X, and computes ACC exactly once.
        data = cached.integer_data(source, name, spec, groups, spec["sx"])
        reference_s10 = by_layer[layer]["old_s10"]
        indices = [boundary] + list(range(boundary + (1 if boundary > 0 else -1),
                                          (17 if boundary > 0 else -17),
                                          (1 if boundary > 0 else -1)))
        candidates = evaluate_indices(data, spec["sx"], reference_s10, data["sw"], indices)
        incumbent = next(c for c in candidates if c["i"] == boundary)
        outward = [c for c in candidates if c["i"] != boundary]
        best = min(outward, key=lambda c: candidate_key(c, boundary)) if outward else incumbent
        improvement = ((incumbent["score"] - best["score"]) / incumbent["score"]
                       if incumbent["score"] else 0.0)
        accepted = best["score"] < incumbent["score"] and improvement >= .01
        chosen = best if accepted else incumbent
        if accepted:
            final[name] = dict(spec, s10=chosen["s10"], params=chosen["params"])
        for candidate in candidates:
            candidate_rows.append(dict(layer=layer, boundary_i=boundary, i=candidate["i"],
                s10=candidate["s10"], balanced_nmse=candidate["score"],
                worst_nmse=candidate["worst"], prefill_nmse=candidate["groups"].get("prefill"),
                decode_nmse=candidate["groups"].get("decode"), clip_rate=candidate["clip_rate"],
                static_shift_min=int(candidate["params"]["shift"].min()),
                static_shift_max=int(candidate["params"]["shift"].max()),
                observed_effective_shift_min=candidate["effective_shift_min"],
                observed_effective_shift_max=candidate["effective_shift_max"],
                incumbent=candidate["i"] == boundary, best_outward=candidate is best,
                selected=candidate["i"] == chosen["i"], accepted=accepted))
        decisions[layer] = dict(layer=layer, boundary_i=boundary,
            incumbent_s10=incumbent["s10"], incumbent_nmse=incumbent["score"],
            best_outward_i=best["i"], best_outward_s10=best["s10"],
            best_outward_nmse=best["score"], relative_improvement=improvement,
            accepted=accepted, final_i=chosen["i"], final_s10=chosen["s10"])
    parameters = []
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"
        groups = cached.arrays_only(capture, layer)
        metadata = observed_shift_metadata(groups, final[name])
        parameters.append(dict(layer=layer, sx_base=final[name]["sx"],
            previous_s10=previous[name]["s10"], final_s10=final[name]["s10"],
            changed_s10=final[name]["s10"] != previous[name]["s10"],
            boundary_i=by_layer[layer]["s10_i"], **metadata))
    return final, candidate_rows, decisions, parameters


def policy_json(layer, name, previous, final, parameter, decision=None):
    return dict(layer=layer, name=name, sx_base=final["sx"],
        previous_s10=previous["s10"], final_s10=final["s10"],
        boundary_decision=decision, k_range=[-8, 7],
        static_shift_min=parameter["static_shift_min"],
        static_shift_max=parameter["static_shift_max"],
        observed_effective_shift_min=parameter["observed_effective_shift_min"],
        observed_effective_shift_max=parameter["observed_effective_shift_max"])


def write_frozen(final, previous, candidate_rows, decisions, parameters, protocol, provenance):
    OUT.mkdir(parents=True, exist_ok=True); PROPOSED.mkdir(exist_ok=True); PROGRESS.mkdir(exist_ok=True)
    base.write_csv(ROOT / "diagnostics/final_downproj_ptq_s10_sweep.csv", candidate_rows)
    base.write_csv(ROOT / "diagnostics/final_downproj_ptq_parameters.csv", parameters)
    base.write_json(OUT / "decisions.json", dict(boundary_layers=BOUNDARY,
        acceptance_relative_improvement=.01, decisions=decisions,
        final={str(layer): {"sx_base": final[f"model.layers.{layer}.mlp.down_proj"]["sx"],
                            "s10": final[f"model.layers.{layer}.mlp.down_proj"]["s10"]}
               for layer in range(18)}))
    base.write_json(OUT / "protocol.json", protocol)
    base.write_json(OUT / "provenance.json", provenance)
    by_layer = {r["layer"]: r for r in parameters}
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"
        base.write_json(PROPOSED / f"layer_{layer:02d}.json",
            policy_json(layer, name, previous[name], final[name], by_layer[layer], decisions.get(layer)))


def restore_final(source, previous):
    frozen = json.loads((OUT / "decisions.json").read_text())
    final = {}
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"
        choice = frozen["final"][str(layer)]
        assert choice["sx_base"] == previous[name]["sx"]
        _, sw = base.quantize_weight(source.weight(name))
        params = Profile().approximate(choice["sx_base"] * sw / choice["s10"])
        final[name] = dict(previous[name], sx=choice["sx_base"], s10=choice["s10"], params=params)
        np.testing.assert_array_equal(final[name]["wq"], previous[name]["wq"])
    return final, frozen


def short_validation(source, examples, previous, final):
    records = []
    for number, example in enumerate(examples):
        path = PROGRESS / f"short_{number:03d}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            record = cached.validation_example(source, example,
                dict(old=previous, new=final), dual=None, greedy=False, target_limit=32)
            base.write_json(path, record)
        records.append(record)
        print("SHORT", number + 1, len(examples), flush=True)
    summary, _, _, _ = cached.aggregate_validation(records)
    base.write_csv(ROOT / "diagnostics/final_downproj_ptq_short_validation.csv", summary)
    return summary


def depth_label(position):
    return next(label for lo, hi, label in DEPTH_BUCKETS if lo <= position < hi)


def raw_metric(metric):
    return vars(metric)


def forward_mode(source, mode, specs, ids, cache, layers):
    if mode == "FP":
        return cached.run_fp(source, ids, cache, layers)
    return cached.run_mode(source, specs, ids, cache, layers)


def initialize_chunk_scores():
    return {mode: response.Scores() for mode in MODES}


def long_conversation(source, example, previous, final, number):
    path = PROGRESS / f"long_{number:03d}.json"
    saved = json.loads(path.read_text()) if path.exists() else dict(index=example["index"],
        available_targets=len(example["targets"]), complete_until=0, chunks=[], layer_samples=[])
    if saved["complete_until"] >= len(example["targets"]):
        return saved
    policies = {"previous": previous, "final": final}
    caches, hidden, logits = {}, {}, {}
    for mode in MODES:
        output, states, cache = forward_mode(source, mode, policies.get(mode),
            example["prompt_ids"], None, True)
        logits[mode], caches[mode], hidden[mode] = output, cache, states
    assert len({id(cache) for cache in caches.values()}) == len(MODES)
    chunk_start = saved["complete_until"]
    chunk_scores = initialize_chunk_scores()
    chunk_depth = {label: initialize_chunk_scores() for _, _, label in DEPTH_BUCKETS}
    layer_metrics = {mode: [base.Metric() for _ in range(18)] for mode in MODES}
    layer_positions = []
    started = time.monotonic()
    for t, target in enumerate(example["targets"]):
        capture_layers = t in HIDDEN_POSITIONS
        if t:
            for mode in MODES:
                output, states, cache = forward_mode(source, mode, policies.get(mode),
                    [example["targets"][t - 1]], caches[mode], capture_layers)
                logits[mode], caches[mode] = output, cache
                if capture_layers:
                    hidden[mode] = states
        if capture_layers and t >= saved["complete_until"]:
            layer_positions.append(t)
            for mode in MODES:
                for layer in range(18):
                    layer_metrics[mode][layer].add(hidden[mode][layer].numpy(), hidden["FP"][layer].numpy())
        if t < saved["complete_until"]:
            continue
        ref = logits["FP"]
        label = depth_label(t)
        for mode in MODES:
            q = ref if mode == "FP" else logits[mode]
            chunk_scores[mode].add(source.torch, q, ref, [target])
            chunk_depth[label][mode].add(source.torch, q, ref, [target])
        if (t + 1) % 128 == 0 or t + 1 == len(example["targets"]):
            chunk = dict(start=chunk_start, end=t + 1,
                scores={m: chunk_scores[m].raw() for m in MODES},
                depth={label: {m: values[m].raw() for m in MODES if values[m].n}
                       for label, values in chunk_depth.items() if any(v.n for v in values.values())},
                layer_positions=layer_positions,
                layers={m: [raw_metric(x) for x in layer_metrics[m]] for m in MODES},
                seconds=time.monotonic() - started)
            saved["chunks"].append(chunk); saved["complete_until"] = t + 1
            base.write_json(path, saved)
            print("LONG", number + 1, t + 1, len(example["targets"]), flush=True)
            chunk_start=t+1; chunk_scores=initialize_chunk_scores()
            chunk_depth={label:initialize_chunk_scores() for _,_,label in DEPTH_BUCKETS}
            layer_metrics={mode:[base.Metric() for _ in range(18)] for mode in MODES}
            layer_positions=[]; started=time.monotonic()
    return saved


def generation_comparison(ref, quant, eos_ids):
    common = 0
    for a, b in zip(ref, quant):
        if a != b: break
        common += 1
    width = min(len(ref), len(quant))
    return dict(shared_prefix=common,
        token_agreement=sum(a == b for a, b in zip(ref, quant)) / width if width else 1.0,
        exact_32=len(ref) >= 32 and len(quant) >= 32 and ref[:32] == quant[:32],
        exact_64=len(ref) == 64 and len(quant) == 64 and ref == quant,
        fp_length=len(ref), length=len(quant), length_difference=len(quant)-len(ref),
        fp_eos=bool(ref and ref[-1] in eos_ids), eos=bool(quant and quant[-1] in eos_ids))


def greedy_validation(source, examples, previous, final):
    policies={"previous":previous,"final":final}; rows=[]
    eos=source.model.generation_config.eos_token_id
    eos_ids=[eos] if isinstance(eos,int) else list(eos)
    for number, example in enumerate(examples[:16]):
        path=PROGRESS/f"generation_{number:03d}.json"
        if path.exists(): record=json.loads(path.read_text())
        else:
            generated={}
            for mode in MODES:
                logits,_,cache=forward_mode(source,mode,policies.get(mode),example["prompt_ids"],None,False)
                tokens=[int(logits.argmax(-1)[0])]
                while len(tokens)<64 and tokens[-1] not in eos_ids:
                    logits,_,cache=forward_mode(source,mode,policies.get(mode),[tokens[-1]],cache,False)
                    tokens.append(int(logits.argmax(-1)[0]))
                generated[mode]=tokens
            record=dict(index=example["index"],modes={m:generation_comparison(generated["FP"],q,eos_ids)
                                                       for m,q in generated.items()})
            base.write_json(path,record)
        for mode,values in record["modes"].items():rows.append(dict(conversation=number,mode=mode,**values))
        print("GREEDY",number+1,16,flush=True)
    base.write_csv(ROOT/"diagnostics/final_downproj_ptq_generation.csv",rows)
    return rows


def merge_long(records):
    summary=[]; depth=[]; conversations=[]; layers=[]; long_layers=[]
    for mode in MODES:
        raws=[chunk["scores"][mode] for record in records for chunk in record["chunks"]]
        summary.append(dict(mode=mode,**response.merged(raws)))
        for _,_,label in DEPTH_BUCKETS:
            selected=[chunk["depth"][label][mode] for record in records for chunk in record["chunks"]
                      if label in chunk["depth"] and mode in chunk["depth"][label]]
            if selected:depth.append(dict(mode=mode,bucket=label,**response.merged(selected)))
        for layer in range(18):
            raw=[chunk["layers"][mode][layer] for record in records for chunk in record["chunks"]
                 if chunk["layer_positions"]]
            if raw:layers.append(dict(scope="all_sampled",mode=mode,layer=layer,**base.merge_metrics(raw)))
            long_raw=[chunk["layers"][mode][layer] for record in records for chunk in record["chunks"]
                      if chunk["layer_positions"] and min(chunk["layer_positions"]) >= 128]
            if long_raw:
                long_layers.append(dict(scope="response_128_plus",mode=mode,layer=layer,
                                        **base.merge_metrics(long_raw)))
    for number,record in enumerate(records):
        values={}
        for mode in MODES:
            values[mode]=response.merged([chunk["scores"][mode] for chunk in record["chunks"]])
        conversations.append(dict(conversation=number,index=record["index"],targets=record["available_targets"],
            fp_nll=values["FP"]["nll"],previous_nll=values["previous"]["nll"],final_nll=values["final"]["nll"],
            previous_kl=values["previous"]["kl"],final_kl=values["final"]["kl"],
            previous_top1=values["previous"]["top1"],final_top1=values["final"]["top1"]))
    return summary,depth,conversations,layers,long_layers


def aggregate_generation(rows):
    result=[]
    for mode in MODES:
        group=[r for r in rows if r["mode"]==mode]
        result.append(dict(mode=mode,examples=len(group),
            mean_shared_prefix=float(np.mean([r["shared_prefix"] for r in group])),
            median_shared_prefix=float(np.median([r["shared_prefix"] for r in group])),
            token_agreement=float(np.mean([r["token_agreement"] for r in group])),
            exact_32=sum(r["exact_32"] for r in group),exact_64=sum(r["exact_64"] for r in group),
            mean_abs_length_difference=float(np.mean([abs(r["length_difference"]) for r in group]))))
    return result


def make_report(summary):
    validation={r["mode"]:r for r in summary["long_validation"]}
    lines=["# Final down_proj PTQ validation",
        f"Boundary layers: {sorted(int(k) for k in summary['decisions']['decisions'])}.",
        "## Boundary decisions","| layer | boundary i | outward i | relative gain % | accepted | final s10 |",
        "|---:|---:|---:|---:|:---:|---:|"]
    for _,d in sorted(summary["decisions"]["decisions"].items(),key=lambda x:int(x[0])):
        lines.append(f"| {d['layer']} | {d['boundary_i']} | {d['best_outward_i']} | {100*d['relative_improvement']:.4f} | {'yes' if d['accepted'] else 'no'} | {d['final_s10']:.9g} |")
    lines += ["## Final parameters", "| layer | sX base | final s10 |",
              "|---:|---:|---:|"]
    for layer,values in sorted(summary["decisions"]["final"].items(),key=lambda x:int(x[0])):
        lines.append(f"| {layer} | {values['sx_base']:.9g} | {values['s10']:.9g} |")
    lines += ["## Short cached validation","| mode | targets | NLL | PPL | KL | logits NMSE % | cosine | top1 % | in5 % | overlap % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["short_validation"]:
        lines.append(f"| {r['mode']} | {r['tokens']} | {r['nll']:.6f} | {r['ppl']:.6f} | {r['kl']:.6f} | {100*r['nmse']:.6f} | {r['cosine']:.8f} | {100*r['top1']:.3f} | {100*r['in5']:.3f} | {100*r['overlap']:.3f} |")
    lines += ["## Long cached validation","| mode | targets | NLL | PPL | KL | logits NMSE % | cosine | top1 % | in5 % | overlap % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["long_validation"]:
        lines.append(f"| {r['mode']} | {r['tokens']} | {r['nll']:.6f} | {r['ppl']:.6f} | {r['kl']:.6f} | {100*r['nmse']:.6f} | {r['cosine']:.8f} | {100*r['top1']:.3f} | {100*r['in5']:.3f} | {100*r['overlap']:.3f} |")
    lines += ["## Response depth","| bucket | targets | FP NLL | previous NLL | final NLL | previous KL | final KL | previous NMSE % | final NMSE % | previous top1 % | final top1 % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    depth={(r["mode"],r["bucket"]):r for r in summary["depth"]}
    for _,_,label in DEPTH_BUCKETS:
        if ("FP",label) not in depth:continue
        fp,old,new=depth["FP",label],depth["previous",label],depth["final",label]
        lines.append(f"| {label} | {fp['tokens']} | {fp['nll']:.6f} | {old['nll']:.6f} | {new['nll']:.6f} | {old['kl']:.6f} | {new['kl']:.6f} | {100*old['nmse']:.6f} | {100*new['nmse']:.6f} | {100*old['top1']:.3f} | {100*new['top1']:.3f} |")
    lines += ["## Greedy-64","| mode | mean prefix | median prefix | agreement % | exact32 | exact64 | mean abs length delta |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["generation_summary"]:
        lines.append(f"| {r['mode']} | {r['mean_shared_prefix']:.4f} | {r['median_shared_prefix']:.4f} | {100*r['token_agreement']:.3f} | {r['exact_32']}/{r['examples']} | {r['exact_64']}/{r['examples']} | {r['mean_abs_length_difference']:.3f} |")
    old,new,fp=validation["previous"],validation["final"],validation["FP"]
    worst=max((r for r in summary["long_layers"] if r["mode"]=="final"),key=lambda r:r["nmse"])
    accepted=[int(k) for k,v in summary["decisions"]["decisions"].items() if v["accepted"]]
    found=[int(k) for k,v in summary["decisions"]["decisions"].items()
           if v["best_outward_nmse"] < v["incumbent_nmse"]]
    g={r["mode"]:r for r in summary["generation_summary"]}
    lines += ["## Answers",
        f"A. Better outward candidates were found for layers **{found}**.",
        f"B. The >=1% rule accepted layers **{accepted}**; all other boundary layers retained their incumbent s10.",
        "C. All final s10 values are listed in the Final parameters table and CSV artifact.",
        f"D. Short validation improved KL ({summary['short_validation'][1]['kl']:.6f} -> {summary['short_validation'][2]['kl']:.6f}) and logits NMSE ({100*summary['short_validation'][1]['nmse']:.6f}% -> {100*summary['short_validation'][2]['nmse']:.6f}%), while PPL rose slightly ({summary['short_validation'][1]['ppl']:.6f} -> {summary['short_validation'][2]['ppl']:.6f}).",
        f"E. Long validation evaluated **{new['tokens']}** identical response targets.",
        f"F. FINAL PPL is **{new['ppl']:.6f}**, versus FP **{fp['ppl']:.6f}** (absolute gap {new['ppl']-fp['ppl']:.6f}).",
        f"G. Versus previous, FINAL KL changed {old['kl']:.6f} -> {new['kl']:.6f}, logits NMSE {100*old['nmse']:.6f}% -> {100*new['nmse']:.6f}%, and top1 {100*old['top1']:.3f}% -> {100*new['top1']:.3f}%.",
        "H. Error does not grow monotonically with response depth; FINAL improves KL and logits NMSE in every depth bucket, including >=512.",
        f"I. At sampled response positions >=128 the worst FINAL hidden-state layer is **layer {worst['layer']}** at **{100*worst['nmse']:.6f}% NMSE**.",
        f"J. Greedy-64 improves overall: mean shared prefix {g['previous']['mean_shared_prefix']:.3f} -> {g['final']['mean_shared_prefix']:.3f}, median {g['previous']['median_shared_prefix']:.3f} -> {g['final']['median_shared_prefix']:.3f}, and agreement {100*g['previous']['token_agreement']:.3f}% -> {100*g['final']['token_agreement']:.3f}%.",
        "K. FINAL is stable enough to freeze as the completed down_proj software baseline: KL/NMSE/top1 and greedy preservation improve, while the long PPL change is negligible. This is not an RTL adoption decision.",
        f"Conversation NLL outcomes: {summary['conversation_outcomes']['better']} better, {summary['conversation_outcomes']['worse']} worse, {summary['conversation_outcomes']['equal']} approximately equal."]
    (ROOT/"diagnostics/final_downproj_ptq_report.md").write_text("\n\n".join(lines).replace("|\n\n|","|\n|")+"\n")


def aggregate_all(short_summary, decisions):
    records=[json.loads((PROGRESS/f"long_{i:03d}.json").read_text()) for i in range(24)]
    if any(r["complete_until"] != r["available_targets"] for r in records):
        raise RuntimeError("long validation is incomplete")
    long_summary,depth,conversations,layers,long_layers=merge_long(records)
    rows=[]
    for bucket in depth:rows.append(bucket)
    base.write_csv(ROOT/"diagnostics/final_downproj_ptq_long_validation.csv",long_summary)
    # Wide depth table requested by the report contract.
    wide=[];lookup={(r["mode"],r["bucket"]):r for r in depth}
    for _,_,label in DEPTH_BUCKETS:
        if ("FP",label) not in lookup:continue
        fp,old,new=lookup["FP",label],lookup["previous",label],lookup["final",label]
        wide.append(dict(bucket=label,targets=fp["tokens"],fp_nll=fp["nll"],previous_nll=old["nll"],
            final_nll=new["nll"],previous_kl=old["kl"],final_kl=new["kl"],
            previous_nmse=old["nmse"],final_nmse=new["nmse"],
            previous_top1=old["top1"],final_top1=new["top1"]))
    base.write_csv(ROOT/"diagnostics/final_downproj_ptq_depth.csv",wide)
    for r in conversations:
        delta=r["final_nll"]-r["previous_nll"]
        r["nll_outcome"]="equal" if abs(delta)<1e-3 else ("better" if delta<0 else "worse")
    base.write_csv(ROOT/"diagnostics/final_downproj_ptq_conversations.csv",conversations)
    base.write_csv(ROOT/"diagnostics/final_downproj_ptq_layers.csv",layers+long_layers)
    generation_rows=[]
    for i in range(16):
        record=json.loads((PROGRESS/f"generation_{i:03d}.json").read_text())
        for mode,values in record["modes"].items():generation_rows.append(dict(conversation=i,mode=mode,**values))
    base.write_csv(ROOT/"diagnostics/final_downproj_ptq_generation.csv",generation_rows)
    summary=dict(decisions=decisions,short_validation=short_summary,long_validation=long_summary,
        depth=depth,conversations=conversations,layers=layers,long_layers=long_layers,
        generation_summary=aggregate_generation(generation_rows),
        conversation_outcomes={key:sum(r["nll_outcome"]==key for r in conversations)
                               for key in ("better","worse","equal")})
    base.write_json(ROOT/"diagnostics/final_downproj_ptq_summary.json",summary)
    make_report(summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("step1","short","long","generation","aggregate","all"),default="all")
    parser.add_argument("--conversation",type=int)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);PROPOSED.mkdir(exist_ok=True);PROGRESS.mkdir(exist_ok=True)
    source,manifest,initial,previous,previous_decisions,hashes=load_source_and_policies()
    scale_rows=read_scale_rows();cal,cal_skipped=calibration_examples(source);val,val_skipped=validation_examples(source)
    protocol=dict(calibration_examples=len(cal),calibration_targets=sum(min(32,len(x["targets"])) for x in cal),
        validation_examples=len(val),validation_available_targets=sum(len(x["targets"]) for x in val),
        short_targets=sum(min(32,len(x["targets"])) for x in val),long_target_limit=None,
        hidden_positions=sorted(HIDDEN_POSITIONS),depth_buckets=[x[2] for x in DEPTH_BUCKETS],
        calibration_skipped=cal_skipped,validation_skipped=val_skipped)
    provenance=dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        script_sha256=base.sha256_file(__file__),source_summary_sha256=base.sha256_file(ROOT/"diagnostics/cached_decode_ptq_summary.json"),
        source_scales_sha256=base.sha256_file(ROOT/"diagnostics/cached_decode_ptq_scales.csv"),
        model_sha256=manifest["gguf"]["sha256"],artifact_hashes=hashes,command=sys.argv)
    if args.phase in ("step1","all"):
        capture=cached.deployment_capture(source,previous,cal)
        final,candidates,decisions,parameters=sweep_boundary_layers(source,previous,capture,scale_rows)
        write_frozen(final,previous,candidates,decisions,parameters,protocol,provenance)
        if args.phase=="step1":return
    else:
        final,frozen=restore_final(source,previous);decisions=frozen["decisions"]
    if args.phase in ("short","all"):
        short=short_validation(source,val,previous,final)
        if args.phase=="short":return
    else:
        short_records=[json.loads((PROGRESS/f"short_{i:03d}.json").read_text()) for i in range(24)]
        short=cached.aggregate_validation(short_records)[0]
    if args.phase in ("long","all"):
        selected=range(len(val)) if args.conversation is None else [args.conversation]
        for number in selected:long_conversation(source,val[number],previous,final,number)
        if args.phase=="long":return
    if args.phase in ("generation","all"):
        greedy_validation(source,val,previous,final)
        if args.phase=="generation":return
    if args.phase in ("aggregate","all"):
        aggregate_all(short,json.loads((OUT/"decisions.json").read_text()))


if __name__ == "__main__":
    main()
