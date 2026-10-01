#!/usr/bin/env python3
"""Small, fixed-scope absmax-aware calibration pilot for 18 down_proj ops."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import calibrate_cached_decode_ptq as cal
import compare_k_selectors_e2e as compare
import confirm_k_selector_fidelity as fidelity

final = compare.final
response = final.response
base = final.base
row = compare.row
Profile = compare.Profile
select_absmax = compare.select_absmax

OUT = ROOT / "diagnostics/absmax_calibration_pilot"
CAL_POSITIONS = [0, 5, 10, 15, 20, 25]
VAL_POSITIONS = [0, 4, 8, 12, 16, 20]
CAL_RESPONSE_LIMIT = 16
VAL_TARGETS = 24
RESERVOIR_CAPACITY = 32
SEED = 20261001
SX_JS = [-2, -1, 0, 1, 2]
S10_IS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
MODES = ("FP", "old_absmax", "calibrated_absmax")


def select_examples(source, split, positions):
    pairs, skipped = cal.build_pairs(split)
    examples, token_skipped = cal.tokenize_pairs(source, pairs)
    if max(positions) >= len(examples):
        raise RuntimeError(f"{split} fixed position unavailable: have {len(examples)}")
    chosen = [examples[i] for i in positions]
    return chosen, skipped + token_skipped


def capture_absmax(source, specs, examples):
    capture = {layer: {group: cal.Reservoir(RESERVOIR_CAPACITY, SEED)
                       for group in cal.GROUPS} for layer in range(18)}
    with patch.object(row, "select_rows", select_absmax):
        for number, example in enumerate(examples):
            context = {}
            with cal.patch_down(source, specs, capture, "prefill", example["index"], 0):
                _, _, cache = response.forward_hidden(
                    source, example["prompt_ids"], [len(example["prompt_ids"]) - 1],
                    context, use_cache=True, layers=False)
            if len(example["targets"]) < CAL_RESPONSE_LIMIT:
                raise RuntimeError(f"calibration response too short at fixed position {CAL_POSITIONS[number]}")
            for t in range(1, CAL_RESPONSE_LIMIT):
                context = {}
                absolute = len(example["prompt_ids"]) + t - 1
                with cal.patch_down(source, specs, capture, "decode", example["index"], absolute):
                    _, _, cache = response.forward_hidden(
                        source, [example["targets"][t - 1]], [0], context,
                        cache, True, False)
            print("CAPTURE", number + 1, len(examples), flush=True)
    result = {}
    for layer in range(18):
        result[layer] = {group: capture[layer][group].arrays() for group in cal.GROUPS}
    return result


def activation_search(groups, old_sx, shift):
    candidates = []
    with patch.object(row, "select_rows", select_absmax):
        for j in SX_JS:
            sx = old_sx * 2.0 ** (j / 4.0)
            result = cal.activation_score(groups, sx, shift)
            candidates.append(dict(j=j, sx=sx, **result))
    best = min(candidates, key=lambda c: (
        c["score"], c["worst"], abs(math.log2(c["sx"] / old_sx)), abs(c["j"])))
    return best, candidates


def prepare_rows(source, name, groups):
    arrays = [groups[g] for g in cal.GROUPS if len(groups[g])]
    x = np.concatenate(arrays)
    labels = np.concatenate([[g] * len(groups[g]) for g in cal.GROUPS if len(groups[g])])
    weight = source.weight(name)
    ref = x.astype(np.float64) @ weight.astype(np.float64).T
    expected_wq, sw = base.quantize_weight(weight)
    return x, labels, ref, expected_wq, sw


def evaluate_s10_exact(x, labels, ref, wq, sw, sx, s10, acc_cache):
    params = Profile().approximate(sx * sw / s10)
    if np.any(params["status"] != "ok"):
        return None
    q, ks, info = select_absmax(x, sx, params["shift"])
    effective = params["shift"][None, :] - ks[:, None]
    if effective.min() < 0 or effective.max() > 31:
        return None
    key = hashlib.sha256(q.tobytes() + ks.tobytes()).digest()
    if key not in acc_cache:
        acc_cache[key] = base.integer_dot(q, wq, "fp64-exact")
    acc = acc_cache[key]
    out = np.empty(acc.shape, np.float64)
    clips = 0
    for k in np.unique(ks):
        mask = ks == k
        raw = Profile().apply(acc[mask], params["multiplier"],
                              params["shift"] - int(k), saturate=False)
        clips += int(((raw < -512) | (raw > 511)).sum())
        out[mask] = np.clip(raw, -512, 511) * s10
    scores = {}
    for group in cal.GROUPS:
        mask = labels == group
        if mask.any():
            denominator = np.square(ref[mask]).sum()
            scores[group] = float(np.square(out[mask] - ref[mask]).sum() / denominator) if denominator else 0.0
    return dict(score=float(np.mean(list(scores.values()))), worst=max(scores.values()),
                groups=scores, clip_rate=clips / out.size, params=params,
                effective_shift_min=int(effective.min()),
                effective_shift_max=int(effective.max()), k=ks, selector_info=info)


def calibrate_layers(source, frozen, capture):
    proposed = {}
    details = []
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"
        old = frozen[name]
        groups = {g: capture[layer][g][0] for g in cal.GROUPS}
        sx_best, _ = activation_search(groups, old["sx"], old["params"]["shift"])
        x, labels, ref, expected_wq, sw = prepare_rows(source, name, groups)
        np.testing.assert_array_equal(expected_wq, old["wq"])
        acc_cache = {}
        candidates = []
        for i in S10_IS:
            s10 = old["s10"] * 2.0 ** (i / 8.0)
            result = evaluate_s10_exact(x, labels, ref, old["wq"], sw,
                                        sx_best["sx"], s10, acc_cache)
            if result is not None:
                candidates.append(dict(i=i, s10=s10, **result))
        if not candidates:
            raise RuntimeError(f"no feasible s10 candidate at layer {layer}")
        best = min(candidates, key=lambda c: (
            c["score"], c["worst"], c["clip_rate"],
            abs(math.log2(c["s10"] / old["s10"]))))
        old_eval = evaluate_s10_exact(x, labels, ref, old["wq"], sw,
                                      old["sx"], old["s10"], {})
        if old_eval is None:
            raise RuntimeError(f"frozen policy infeasible at layer {layer}")
        spec = dict(old, sx=sx_best["sx"], s10=best["s10"], params=best["params"])
        np.testing.assert_array_equal(spec["wq"], old["wq"])
        proposed[name] = spec
        relative = (best["score"] - old_eval["score"]) / old_eval["score"]
        details.append(dict(layer=layer, old_sx=old["sx"], new_sx=spec["sx"],
            sx_j=sx_best["j"], old_s10=old["s10"], new_s10=spec["s10"],
            s10_i=best["i"], old_local_nmse=old_eval["score"],
            new_local_nmse=best["score"], relative_local_nmse_change=relative,
            prefill_nmse=best["groups"].get("prefill"),
            decode_nmse=best["groups"].get("decode"), clip_rate=best["clip_rate"],
            effective_shift_min=best["effective_shift_min"],
            effective_shift_max=best["effective_shift_max"],
            sx_boundary_hit=sx_best["j"] in (min(SX_JS), max(SX_JS)),
            s10_boundary_hit=best["i"] in (min(S10_IS), max(S10_IS))))
        print("CALIBRATE", layer, sx_best["j"], best["i"], relative, flush=True)
    return proposed, details


def independent(caches):
    assert len(caches) == 3 and all(c is not None for c in caches)
    assert len({id(c) for c in caches}) == 3


def forward(source, specs, mode, ids, cache):
    if mode == "FP":
        return final.cached.run_fp(source, ids, cache, layers=False)
    selected = specs[mode]
    with patch.object(row, "select_rows", select_absmax):
        return final.cached.run_mode(source, selected, ids, cache, layers=False)


def validate(source, old, proposed, examples):
    specs = {"old_absmax": old, "calibrated_absmax": proposed}
    records = []
    for number, example in enumerate(examples):
        if len(example["targets"]) < VAL_TARGETS:
            raise RuntimeError(f"validation response too short at fixed position {VAL_POSITIONS[number]}")
        scores = {mode: response.Scores() for mode in MODES}
        caches, logits = {}, {}
        for mode in MODES:
            logits[mode], _, caches[mode] = forward(source, specs, mode,
                                                    example["prompt_ids"], None)
        independent([caches[m] for m in MODES])
        for t, target in enumerate(example["targets"][:VAL_TARGETS]):
            if t:
                for mode in MODES:
                    logits[mode], _, caches[mode] = forward(
                        source, specs, mode, [example["targets"][t - 1]], caches[mode])
            independent([caches[m] for m in MODES])
            for mode in MODES:
                scores[mode].add(source.torch, logits[mode], logits["FP"], [target])
        records.append(dict(index=VAL_POSITIONS[number], source_index=example["index"],
                            targets=VAL_TARGETS,
                            scores={m: s.raw() for m, s in scores.items()}))
        print("VALIDATE", number + 1, len(examples), flush=True)
    return records


def classify(old, new, mean_local_improvement):
    safety = dict(kl=new["kl"] <= 1.05 * old["kl"] + 1e-15,
                  nmse=new["nmse"] <= 1.05 * old["nmse"] + 1e-15,
                  top1=new["top1"] >= old["top1"] - .01 - 1e-15)
    regression = (new["kl"] > 1.05 * old["kl"] + 1e-15 or
                  new["nmse"] > 1.05 * old["nmse"] + 1e-15 or
                  new["top1"] < old["top1"] - .01 - 1e-15)
    benefit = ((old["kl"] - new["kl"]) / old["kl"] >= .03 or
               (old["nmse"] - new["nmse"]) / old["nmse"] >= .03 or
               mean_local_improvement >= .05)
    if regression:
        label = "REGRESSION"
    elif all(safety.values()) and benefit:
        label = "CLEAR_BENEFIT"
    else:
        label = "NEUTRAL_BUT_SAFE"
    return dict(label=label, safety=safety, benefit=benefit)


def fingerprint_tree(paths):
    h = hashlib.sha256()
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def write_outputs(source, frozen, proposed, details, records, provenance):
    modes = [dict(mode=m, **response.merged([r["scores"][m] for r in records]))
             for m in MODES]
    assert all(r["tokens"] == 144 for r in modes)
    old, new = modes[1], modes[2]
    deltas = dict(kl_relative=(new["kl"] - old["kl"]) / old["kl"],
                  nmse_relative=(new["nmse"] - old["nmse"]) / old["nmse"],
                  top1_pp=100 * (new["top1"] - old["top1"]),
                  ppl_relative=(new["ppl"] - old["ppl"]) / old["ppl"])
    mean_change = float(np.mean([r["relative_local_nmse_change"] for r in details]))
    aggregate = dict(changed_sx=sum(r["sx_j"] != 0 for r in details),
        changed_s10=sum(r["s10_i"] != 0 for r in details),
        sx_boundary_hits=sum(r["sx_boundary_hit"] for r in details),
        s10_boundary_hits=sum(r["s10_boundary_hit"] for r in details),
        boundary_hit_layers=sum(r["sx_boundary_hit"] or r["s10_boundary_hit"] for r in details),
        mean_relative_local_nmse_change=mean_change,
        mean_local_nmse_improvement=-mean_change,
        worst_relative_local_nmse_regression=max(r["relative_local_nmse_change"] for r in details))
    interpretation = classify(old, new, aggregate["mean_local_nmse_improvement"])
    per = []
    for record in records:
        metrics = {m: response.merged([record["scores"][m]]) for m in MODES}
        per.append(dict(index=record["index"], source_index=record["source_index"],
            targets=record["targets"], fp_nll=metrics["FP"]["nll"],
            old_absmax_nll=metrics["old_absmax"]["nll"],
            calibrated_absmax_nll=metrics["calibrated_absmax"]["nll"],
            old_absmax_kl=metrics["old_absmax"]["kl"],
            calibrated_absmax_kl=metrics["calibrated_absmax"]["kl"],
            old_absmax_nmse=metrics["old_absmax"]["nmse"],
            calibrated_absmax_nmse=metrics["calibrated_absmax"]["nmse"],
            old_absmax_top1=metrics["old_absmax"]["top1"],
            calibrated_absmax_top1=metrics["calibrated_absmax"]["top1"]))
    proposed_json = {name: dict(sx_base=spec["sx"], s10=spec["s10"],
        multiplier=spec["params"]["multiplier"].tolist(),
        shift=spec["params"]["shift"].tolist()) for name, spec in proposed.items()}
    base.write_json(OUT / "proposed_parameters.json", proposed_json)
    base.write_csv(OUT / "calibration_layers.csv", details)
    base.write_csv(OUT / "validation_per_conversation.csv", per)
    summary = dict(provenance=provenance, calibration=aggregate, modes=modes,
                   deltas=deltas, classification=interpretation)
    fidelity.finite_tree(summary)
    base.write_json(OUT / "validation_summary.json", summary)
    lines = ["# Absmax-aware calibration pilot", "",
        f"Calibration: changed sX {aggregate['changed_sx']}/18, changed s10 {aggregate['changed_s10']}/18; boundary-hit layers {aggregate['boundary_hit_layers']}/18.",
        f"Mean local NMSE change: {100*mean_change:.6f}% relative.", "",
        "| Mode | NLL | PPL | KL vs FP | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in modes:
        lines.append("| " + r["mode"] + " | " + " | ".join(
            f"{r[k]:.9g}" for k in ["nll", "ppl", "kl", "nmse", "cosine", "top1", "in5", "overlap"]) + " |")
    lines += ["", f"Deltas old→calibrated: {deltas}.",
              f"Pilot classification: {interpretation['label']}.",
              "This is a fixed small pilot; no statistical claim or automatic large calibration follows.",
              "Frozen source artifacts and parameters remained unchanged."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "validation_summary.json").exists():
        raise RuntimeError("pilot output already exists")
    source, _, _, previous, _, _ = final.load_source_and_policies()
    frozen, _ = final.restore_final(source, previous)
    expected = {f"model.layers.{i}.mlp.down_proj" for i in range(18)}
    assert set(frozen) == expected
    decisions = final.OUT / "decisions.json"
    relevant = [final.OUT, compare.OUT, fidelity.OUT,
                ROOT / "diagnostics/k_selector_fidelity_384"]
    before = dict(decisions_sha256=base.sha256_file(decisions),
                  parameters_sha256=compare.fingerprint(frozen),
                  artifacts_sha256=fingerprint_tree(relevant))
    cal_examples, cal_skipped = select_examples(source, "calibration", CAL_POSITIONS)
    assert len(cal_examples) == 6
    capture = capture_absmax(source, frozen, cal_examples)
    proposed, details = calibrate_layers(source, frozen, capture)
    assert set(proposed) == expected
    val_examples, val_skipped = select_examples(source, "validation", VAL_POSITIONS)
    assert len(val_examples) == 6 and len(val_examples) * VAL_TARGETS == 144
    records = validate(source, frozen, proposed, val_examples)
    after = dict(decisions_sha256=base.sha256_file(decisions),
                 parameters_sha256=compare.fingerprint(frozen),
                 artifacts_sha256=fingerprint_tree(relevant))
    assert before == after
    provenance = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        command=sys.argv, selector="compare_k_selectors_e2e.select_absmax",
        calibration_positions=CAL_POSITIONS, validation_positions=VAL_POSITIONS,
        calibration_skipped=cal_skipped, validation_skipped=val_skipped,
        before=before, after=after)
    summary = write_outputs(source, frozen, proposed, details, records, provenance)
    base.write_json(OUT / "verification.json", dict(
        calibration_examples=6, calibration_response_limit=CAL_RESPONSE_LIMIT,
        reservoir_capacity=RESERVOIR_CAPACITY, validation_examples=6,
        validation_targets=144, selector_identity=select_absmax is compare.select_absmax,
        independent_validation_caches=True, effective_shifts_feasible=all(
            0 <= r["effective_shift_min"] <= r["effective_shift_max"] <= 31 for r in details),
        fixed_candidate_sets=dict(sx=SX_JS, s10=S10_IS), no_boundary_expansion=True,
        quantized_modules=sorted(expected), frozen_unchanged=before == after,
        classification=summary["classification"]["label"]))


if __name__ == "__main__":
    main()
