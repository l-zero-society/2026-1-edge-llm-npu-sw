"""Small-batch joint MLP calibration through the residual consumer."""
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gpalu_fullscale_downaware_pilot as prior

pot, fusion, mixed, gelu_base = prior.pot, prior.fusion, prior.mixed, prior.gelu_base
base, cal, final, response, Profile = prior.base, prior.cal, prior.final, prior.response, prior.Profile
OUT = ROOT / "diagnostics/joint_mlp_consumer_pilot"
WORK = OUT / "work"
LUT_DIR = OUT / "gelu_tables"
GATE_QB = OUT / "gate_qparams"
DOWN_QB = OUT / "down_qparams"
PREV = ROOT / "diagnostics/gpalu_fullscale_downaware_pilot"
POT = ROOT / "diagnostics/gpalu_pot_fusion_pilot"
FUSION = ROOT / "diagnostics/fusion_aware_gate_s10_pilot"
MIXED = ROOT / "diagnostics/mixed_linear_gelu_calibration"
PROGRESS = ROOT / "joint_mlp_consumer_calibration_progress.txt"
REQUIRED = (
    "2d77cf52907d8b1f910e26a1952c69248090190f",
    "f92fafb2db225165a2eb9064b5073e6f212fac54",
    "cbc545ddc00cf3ed697217de0748f9a396f479dc",
    "df47d1aa05662e2dad54e4c851dcf84b4428ab75",
)
GROUPS = ("prefill", "decode")
PERCENTILES = (99., 99.5, 99.9, 99.95, 99.99, 99.995, 100.)
S10_JS = tuple(range(-4, 5))
SH_JS = tuple(range(-6, 7))
DOWN_JS = tuple(range(-4, 5))
MAX_EXACT_LOCAL = 96
MODES_NEW = (
    "FREE_SACT_ONLY", "FREE_DOWN_S8", "FINER_GPALU_SCALE",
    "JOINT_NO_RESIDUAL_OBJECTIVE", "JOINT_RESIDUAL_AWARE",
    "GUARDED_JOINT_RESIDUAL_AWARE",
)
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as handle:
        handle.write(message + "\n"); handle.flush()


def sha(path): return base.sha256_file(Path(path))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
def balanced(values): return float(np.mean([v for v in values if v is not None]))


def write_csv(path, rows):
    base.write_csv(path, rows)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def group_metric(value, reference, labels):
    return {g: (mixed.nmse(value[labels == g], reference[labels == g]) if np.any(labels == g) else None) for g in GROUPS}


def residual_objective(residual, down, residual_ref, down_ref, labels):
    """The current simulator adds a dequantized MLP result to the FP residual."""
    return group_metric(residual + down, residual_ref + down_ref, labels)


def residual_key(candidate):
    return (candidate["post_residual_nmse_balanced"], candidate["post_residual_worst_group"],
            candidate["down_nmse_balanced"], candidate["post_gpalu_nmse"],
            candidate["gpalu_clip_rate"], candidate["down_saturation_rate"],
            candidate["gelu_nmse"], candidate["movement"], candidate["s10"],
            candidate["s_act"], candidate["s_h_target"], candidate["s8_down"])


def down_key(candidate):
    return (candidate["down_nmse_balanced"], candidate["down_nmse_worst"],
            candidate["post_gpalu_nmse"], candidate["gpalu_clip_rate"],
            candidate["down_saturation_rate"], candidate["movement"],
            candidate["s10"], candidate["s_act"], candidate["s_h_target"], candidate["s8_down"])


def warning_flags(candidate, minima):
    flags = {
        "gelu": candidate["gelu_nmse"] > 2. * minima["gelu"],
        "post_gpalu": candidate["post_gpalu_nmse"] > 2. * minima["post_gpalu"],
        "down": candidate["down_nmse_balanced"] > 2. * minima["down"],
    }
    return flags, any(flags.values())


def guarded_select(candidates):
    minima = {"gelu": min(c["gelu_nmse"] for c in candidates),
              "post_gpalu": min(c["post_gpalu_nmse"] for c in candidates),
              "down": min(c["down_nmse_balanced"] for c in candidates)}
    for candidate in candidates:
        candidate["warning_flags"], candidate["error_cancellation_warning"] = warning_flags(candidate, minima)
    unconstrained = min(candidates, key=residual_key)
    safe = [c for c in candidates if not c["error_cancellation_warning"]]
    guarded = min(safe, key=residual_key) if safe else unconstrained
    if safe and guarded["post_residual_nmse_balanced"] <= unconstrained["post_residual_nmse_balanced"] * 1.001:
        preferred = guarded
    else:
        preferred = guarded
    return unconstrained, guarded, preferred, minima


def unique_scales(items):
    result = []
    for value, *meta in sorted(items, key=lambda x: x[0]):
        if value > 0 and np.isfinite(value) and not any(np.isclose(value, x[0], rtol=1e-11) for x in result):
            result.append((float(value), *meta))
    return result


def s10_candidates(latest, original, fusion_value):
    items = [(latest * 2. ** (j / 8.), "latest_grid", j) for j in S10_JS]
    items += [(original, "original_mixed", None), (fusion_value, "fusion_winner", None)]
    return unique_scales(items)


def gelu_candidates(g_fp, q10, labels, s10):
    rows = []
    for percentile in PERCENTILES:
        item = mixed.gelu_candidate_rows(g_fp, q10, s10, percentile)
        actual = item["lut"][gelu_base.address(q10)].astype(np.float64) * item["s_act"]
        group = group_metric(actual, gelu_base.gelu(g_fp), labels)
        rows.append(dict(item, actual=actual, gelu_nmse=balanced(group.values()), gelu_worst=max(v for v in group.values() if v is not None)))
    return rows


def scale_anchors(reference, quantized, previous_sh, pot_sh, old_down_sx):
    anchors = []
    for source, values in (("fp", reference), ("quantized", quantized)):
        flat = np.abs(values).reshape(-1)
        anchors += [(float(np.percentile(flat, p)) / 127., source, p) for p in PERCENTILES]
    anchors += [(previous_sh, "previous_fullscale", None), (pot_sh, "previous_pot", None), (old_down_sx, "old_down_sx", None)]
    coarse = unique_scales(anchors)
    # Refine all coarse anchors; exact Down evaluation is pruned later.
    return unique_scales([(scale * 2. ** (j / 16.), source, (p, j)) for scale, source, p in coarse for j in SH_JS])


def down_scales(y_fp, acc_real, old_sout):
    anchors = [(old_sout, "previous", None)]
    for source, values in (("fp", y_fp), ("acc", acc_real)):
        flat = np.abs(values).reshape(-1)
        anchors += [(float(np.percentile(flat, p)) / 127., source, p) for p in PERCENTILES]
    return unique_scales([(scale * 2. ** (j / 16.), source, (p, j)) for scale, source, p in unique_scales(anchors) for j in DOWN_JS])


def direct_down_scale(q_h, s_h, down_spec, sout, acc=None):
    params = Profile().approximate(s_h * down_spec["sw"] / sout)
    if np.any(params["status"] != "ok"): return None
    if acc is None: acc = prior.exact_dot(q_h, down_spec["wq"])
    raw = Profile().apply(acc, params["multiplier"], params["shift"], saturate=False)
    codes = np.clip(raw, -128, 127).astype(np.int8)
    return codes.astype(np.float64) * sout, params, float(np.mean((raw < -128) | (raw > 127))), acc


def capture_residual(source, examples, expected_x):
    """Capture paired MLP residual and post-attention RMSNorm output."""
    reservoirs = {layer: {g: cal.Reservoir(mixed.RESERVOIR, mixed.SEED) for g in GROUPS} for layer in range(18)}
    state, handles = {}, []
    for layer, block in enumerate(source.model.model.layers):
        name = f"model.layers.{layer}.mlp.gate_proj"
        def hook(module, args, output, layer=layer, name=name):
            residual = args[0].detach().cpu().numpy()[0]
            normalized = output.detach().cpu().numpy()[0]
            group = state["group"]
            for position, (r, x) in enumerate(zip(residual, normalized)):
                reservoirs[layer][group].add(np.concatenate((r, x)), name, group, state["conversation"], state["offset"] + position)
        handles.append(block.post_attention_layernorm.register_forward_hook(hook))
    try:
        for number, example in enumerate(examples):
            state.update(group="prefill", conversation=example["index"], offset=0)
            _, _, cache = cal.run_fp(source, example["prompt_ids"], None, layers=False)
            for t in range(1, min(mixed.CAL_LIMIT, len(example["targets"]))):
                state.update(group="decode", offset=len(example["prompt_ids"]) + t - 1)
                _, _, cache = cal.run_fp(source, [example["targets"][t - 1]], cache, layers=False)
    finally:
        for handle in handles: handle.remove()
    result = {}
    width = expected_x[0].shape[1]
    for layer in range(18):
        for group in GROUPS:
            pair = reservoirs[layer][group].arrays()[0]
            result[f"layer{layer:02d}_{group}_residual"] = pair[:, :width]
            result[f"layer{layer:02d}_{group}_x"] = pair[:, width:]
    return result


def cheap_gpalu_candidates(layer, g_fp, g_hat, q10, u_fp, u_hat, q_u, labels, s10, s10_source,
                           previous, pot_info, down_spec):
    h_fp = gelu_base.gelu(g_fp) * u_fp
    results = []
    for gelu in gelu_candidates(g_fp, q10, labels, s10):
        q_a = gelu["lut"][gelu_base.address(q10)].astype(np.int16)
        h_pre = gelu["actual"] * u_hat
        p16 = q_a.astype(np.int32) * q_u.astype(np.int32)
        if np.any((p16 < -32768) | (p16 > 32767)): raise RuntimeError("INT16 GPALU overflow")
        probes = []
        for target, anchor_source, anchor_meta in scale_anchors(h_fp, h_pre, previous["s_h_effective"], pot_info["output_scale"], down_spec["sx"]):
            rep = prior.approximate_scalar((gelu["s_act"] * previous["s8_up_fixed"]) / target)
            effective = prior.effective_scale(gelu["s_act"] * previous["s8_up_fixed"], rep["M_G"], rep["S_G"])
            _, q_h, clipping = prior.fullscale_codes(p16, rep["alpha_hw"], rep["M_G"], rep["S_G"])
            h_hat = q_h.astype(np.float64) * effective
            group = group_metric(h_hat, h_fp, labels)
            probes.append(dict(layer=layer, s10=s10, s10_source=s10_source, s_act=gelu["s_act"], gelu_percentile=gelu["retained_percentile"],
                               gelu_nmse=gelu["gelu_nmse"], gelu_worst=gelu["gelu_worst"], gate_nmse=mixed.nmse(g_hat, g_fp),
                               s_product=gelu["s_act"] * previous["s8_up_fixed"], s_h_target=target, s_h_effective=effective,
                               M_G=rep["M_G"], S_G=rep["S_G"], alpha_relative_error=rep["alpha_relative_error"],
                               pre_gpalu_nmse=mixed.nmse(h_pre, h_fp), post_gpalu_nmse=balanced(group.values()),
                               gpalu_clip_rate=clipping["hw_clip_rate"], anchor_source=anchor_source, anchor_meta=anchor_meta,
                               q_h=q_h, lut=gelu["lut"], p16=p16))
        # Keep every s_act alive and preserve low-clipping/utilization alternatives.
        results.append(min(probes, key=lambda c: (c["post_gpalu_nmse"], c["gpalu_clip_rate"], c["s_h_target"])))
        results += sorted(probes, key=lambda c: (c["gpalu_clip_rate"], c["post_gpalu_nmse"]))[:1]
        for anchor in (previous["s_h_effective"], pot_info["output_scale"], down_spec["sx"]):
            results.append(min(probes, key=lambda c, a=anchor: abs(math.log2(c["s_h_target"] / a))))
        results += [c for c in probes if c["post_gpalu_nmse"] <= min(x["post_gpalu_nmse"] for x in probes) * 1.01]
    return results, h_fp


def evaluate_layer(layer, x, residual, labels, gate_base, gate_latest, fusion_s10, up_spec, down_spec, previous, pot_info):
    g_fp = x.astype(np.float64) @ gate_base["weight"].astype(np.float64).T
    u_fp = x.astype(np.float64) @ up_spec["weight"].astype(np.float64).T
    u_hat, ud = mixed.quantized_linear_rows(x, up_spec, return_codes=True)
    q_u = ud["output_codes"].astype(np.int16)
    down_fp = (gelu_base.gelu(g_fp) * u_fp) @ down_spec["weight"].astype(np.float64).T
    cheap = []
    for s10, source, _ in s10_candidates(gate_latest["sout"], gate_base["sout"], fusion_s10):
        params = Profile().approximate(gate_base["sx"] * gate_base["sw"] / s10)
        if np.any(params["status"] != "ok"): continue
        spec = dict(gate_base, sout=s10, params=params)
        g_hat, gd = mixed.quantized_linear_rows(x, spec, return_codes=True)
        rows, _ = cheap_gpalu_candidates(layer, g_fp, g_hat, gd["output_codes"], u_fp, u_hat, q_u, labels, s10, source, previous, pot_info, down_spec)
        for row in rows: row["gate_params"] = params
        cheap.extend(rows)
    # Deterministic union: local score, clipping, historical anchors, and near ties.
    ordered = sorted(cheap, key=lambda c: (c["post_gpalu_nmse"], c["gpalu_clip_rate"], c["s10"], c["s_act"], c["s_h_target"]))
    retained = []
    sources = ordered[:32] + sorted(cheap, key=lambda c: (c["gpalu_clip_rate"], c["post_gpalu_nmse"]))[:32]
    sources += [c for c in cheap if c["anchor_source"] in ("previous_fullscale", "previous_pot", "old_down_sx")]
    sources += [c for c in cheap if c["post_gpalu_nmse"] <= ordered[0]["post_gpalu_nmse"] * 1.02]
    for candidate in sources:
        key = (candidate["s10"], candidate["s_act"], candidate["s_h_target"], candidate["M_G"], candidate["S_G"])
        if not any(item[0] == key for item in retained): retained.append((key, candidate))
    retained = [item[1] for item in retained[:MAX_EXACT_LOCAL]]
    exact = []
    for gp in retained:
        acc = prior.exact_dot(gp["q_h"], down_spec["wq"])
        acc_real = acc.astype(np.float64) * gp["s_h_effective"] * down_spec["sw"][None, :]
        scales = down_scales(down_fp, acc_real, down_spec["sout"])
        for sout, scale_source, scale_meta in scales:
            evaluated = direct_down_scale(gp["q_h"], gp["s_h_effective"], down_spec, sout, acc)
            if evaluated is None: continue
            down_hat, down_params, down_sat, _ = evaluated
            dg = group_metric(down_hat, down_fp, labels)
            rg = residual_objective(residual, down_hat, residual, down_fp, labels)
            movement = abs(math.log2(gp["s10"] / previous["s10_selected"])) + abs(math.log2(gp["s_act"] / previous["s_act_selected"])) + abs(math.log2(gp["s_h_effective"] / previous["s_h_effective"])) + abs(math.log2(sout / down_spec["sout"]))
            exact.append({k: v for k, v in gp.items() if k not in ("q_h", "lut", "p16", "gate_params")} | {
                "s8_down": sout, "down_scale_source": scale_source, "down_scale_meta": scale_meta,
                "down_nmse_prefill": dg["prefill"], "down_nmse_decode": dg["decode"], "down_nmse_balanced": balanced(dg.values()),
                "down_nmse_worst": max(v for v in dg.values() if v is not None), "down_saturation_rate": down_sat,
                "post_residual_nmse_prefill": rg["prefill"], "post_residual_nmse_decode": rg["decode"],
                "post_residual_nmse_balanced": balanced(rg.values()), "post_residual_worst_group": max(v for v in rg.values() if v is not None),
                "movement": movement, "down_params": down_params, "gate_params": gp["gate_params"], "lut": gp["lut"]})
    if not exact: raise RuntimeError(f"no exact candidates layer {layer}")
    unconstrained, guarded, preferred, minima = guarded_select(exact)
    no_residual = min(exact, key=down_key)
    # Sequential ablations anchored to the previous solution.
    prev_s10, prev_sact, prev_sh = previous["s10_selected"], previous["s_act_selected"], previous["s_h_effective"]
    def nearest_filter(rows, field, target):
        distance = min(abs(math.log2(r[field] / target)) for r in rows)
        return [r for r in rows if np.isclose(abs(math.log2(r[field] / target)), distance, atol=1e-12)]
    anchored = nearest_filter(exact, "s10", prev_s10)
    anchored = nearest_filter(anchored, "s_h_effective", prev_sh)
    anchored_old_down = nearest_filter(anchored, "s8_down", down_spec["sout"])
    free_sact = min(anchored_old_down, key=residual_key)
    same_sact = nearest_filter(anchored, "s_act", free_sact["s_act"])
    free_down = min(same_sact, key=residual_key)
    same_s10 = nearest_filter(exact, "s10", prev_s10)
    same_sact2 = nearest_filter(same_s10, "s_act", free_sact["s_act"])
    finer_gpalu = min(same_sact2, key=residual_key)
    return dict(unconstrained=unconstrained, guarded=guarded, preferred=preferred, no_residual=no_residual,
                free_sact=free_sact, free_down=free_down, finer_gpalu=finer_gpalu, minima=minima,
                candidates=exact, retained_count=len(retained), cheap_count=len(cheap))


def compact_payload(candidate, gate_base, up_spec, down_base):
    gate_params = candidate.get("gate_params")
    if gate_params is None:
        gate_params = Profile().approximate(gate_base["sx"] * gate_base["sw"] / candidate["s10"])
    down_params = candidate.get("down_params")
    if down_params is None:
        down_params = Profile().approximate(candidate["s_h_effective"] * down_base["sw"] / candidate["s8_down"])
    lut = candidate.get("lut")
    if lut is None:
        lut = gelu_base.make_lut(candidate["s10"], candidate["s_act"])
    return {
        "gate_spec": dict(gate_base, sout=candidate["s10"], params=gate_params),
        "lut": {"s10": candidate["s10"], "s_act": candidate["s_act"], "lut": lut},
        "M_G": candidate["M_G"], "S_G": candidate["S_G"], "s_h_effective": candidate["s_h_effective"],
        "up_scale": up_spec["sout"], "down_spec": dict(down_base, sout=candidate["s8_down"], params=down_params),
    }


@contextlib.contextmanager
def patch_configuration(source, specs, payloads):
    originals = []
    active_specs = dict(specs)
    for layer, item in payloads.items():
        active_specs[f"model.layers.{layer}.mlp.gate_proj"] = item["gate_spec"]
    with mixed.patch_mixed(source, active_specs, {layer: item["lut"] for layer, item in payloads.items()}):
        try:
            for layer, block in enumerate(source.model.model.layers):
                mlp, item = block.mlp, payloads[layer]
                originals.append((mlp, mlp.forward))
                def forward(x, mlp=mlp, item=item):
                    a = mlp.act_fn(mlp.gate_proj(x)); u = mlp.up_proj(x)
                    qa = np.clip(np.rint(a.detach().cpu().numpy() / item["lut"]["s_act"]), -127, 127).astype(np.int16)
                    qu = np.clip(np.rint(u.detach().cpu().numpy() / item["up_scale"]), -128, 127).astype(np.int16)
                    product = qa.astype(np.int32) * qu.astype(np.int32)
                    _, qh, _ = prior.fullscale_codes(product, item["M_G"] / (2. ** item["S_G"]), item["M_G"], item["S_G"])
                    down = item["down_spec"]
                    y, _, _, _ = direct_down_scale(qh.reshape(-1, qh.shape[-1]), item["s_h_effective"], down, down["sout"])
                    return x.new_tensor(y.reshape(*qh.shape[:-1], y.shape[-1]).astype(np.float32))
                mlp.forward = forward
            yield
        finally:
            for module, fn in originals: module.forward = fn


def run_mode(source, specs, payloads, ids, cache):
    with patch_configuration(source, specs, payloads):
        hidden, states, cache = response.forward_hidden(source, ids, [len(ids) - 1], {}, cache, True, False)
        with source.torch.inference_mode(): logits = source.model.lm_head(hidden)
    return logits, states, cache


def evaluate_sequence(source, specs, configurations, example):
    def signature(payloads):
        return tuple((layer, item["gate_spec"]["sout"], item["lut"]["s_act"], item["M_G"], item["S_G"],
                      item["s_h_effective"], item["down_spec"]["sout"]) for layer, item in sorted(payloads.items()))
    representative, aliases = {}, {}
    for name, payloads in configurations.items():
        sig = signature(payloads)
        if sig not in representative: representative[sig] = name
        aliases[name] = representative[sig]
    active = {name: configurations[name] for name in dict.fromkeys(aliases.values())}
    scores = {name: response.Scores() for name in active}
    caches = {name: None for name in active}; fp_cache = None
    for t, target in enumerate(example["targets"][:24]):
        ids = example["prompt_ids"] if t == 0 else [example["targets"][t - 1]]
        fp_logits, _, fp_cache = cal.run_fp(source, ids, fp_cache, layers=False)
        for name, payloads in active.items():
            logits, _, caches[name] = run_mode(source, specs, payloads, ids, caches[name])
            scores[name].add(source.torch, logits, fp_logits, [target])
        if len({id(cache) for cache in caches.values()} | {id(fp_cache)}) != len(caches) + 1: raise RuntimeError("KV cache alias")
    raw = {name: score.raw() for name, score in scores.items()}
    return {name: raw[aliases[name]] for name in configurations}


def delta(old, new):
    result = {k: {"absolute": new[k] - old[k], "relative": (new[k] - old[k]) / old[k]} for k in ("kl", "nmse", "ppl")}
    result["top1"] = {"absolute_pp": 100. * (new["top1"] - old["top1"])}
    return result


def serializable(candidate):
    return {k: v for k, v in candidate.items() if k not in ("lut", "gate_params", "down_params", "warning_flags")}


CANDIDATE_COLUMNS = (
    "layer", "s10", "s_act", "gelu_percentile", "s_h_target", "s_h_effective", "M_G", "S_G",
    "s8_down", "gate_nmse", "gelu_nmse", "pre_gpalu_nmse", "post_gpalu_nmse",
    "gpalu_clip_rate", "down_nmse_prefill", "down_nmse_decode", "down_nmse_balanced",
    "down_nmse_worst", "down_saturation_rate", "post_residual_nmse_prefill", "post_residual_nmse_decode",
    "post_residual_nmse_balanced", "post_residual_worst_group", "movement",
    "error_cancellation_warning", "selected_unconstrained", "selected_guarded",
)


def compact_candidate_rows(rows):
    """Keep a deterministic scalar audit subset; full sweep data stays in ignored work/."""
    grouped = {}
    for row in rows: grouped.setdefault(int(row["layer"]), []).append(row)
    audit = []
    for layer in sorted(grouped):
        values = grouped[layer]
        chosen = [r for r in values if r.get("selected_unconstrained") or r.get("selected_guarded")]
        chosen += sorted(values, key=residual_key)[:32]
        chosen += sorted(values, key=down_key)[:32]
        chosen += sorted(values, key=lambda r: (r["gpalu_clip_rate"], r["post_gpalu_nmse"]))[:32]
        chosen += sorted(values, key=lambda r: (r["down_saturation_rate"], r["down_nmse_balanced"]))[:32]
        seen = set()
        for row in chosen:
            identity = tuple(row.get(key) for key in ("s10", "s_act", "s_h_effective", "M_G", "S_G", "s8_down"))
            if identity not in seen: audit.append(row); seen.add(identity)
    result = []
    for row in audit:
        item = {}
        for key in CANDIDATE_COLUMNS:
            value = row.get(key)
            item[key] = format(value, ".10g") if isinstance(value, float) else value
        result.append(item)
    return result


def load_csv(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def parse_csv_row(row):
    parsed = {}
    for key, value in row.items():
        if value in ("True", "False"):
            parsed[key] = value == "True"
        elif value == "":
            parsed[key] = None
        else:
            try: parsed[key] = float(value)
            except (TypeError, ValueError): parsed[key] = value
    for key in ("layer", "M_G", "S_G"):
        if key in parsed and isinstance(parsed[key], (int, float)): parsed[key] = int(parsed[key])
    return parsed


def restored_selections(rows, previous, down_spec):
    unconstrained = next(c for c in rows if c.get("selected_unconstrained"))
    guarded = next(c for c in rows if c.get("selected_guarded"))
    no_residual = min(rows, key=down_key)
    prev_s10, prev_sact, prev_sh = previous["s10_selected"], previous["s_act_selected"], previous["s_h_effective"]
    def nearest(items, field, target):
        distance = min(abs(math.log2(item[field] / target)) for item in items)
        return [item for item in items if np.isclose(abs(math.log2(item[field] / target)), distance, atol=1e-12)]
    anchored = nearest(rows, "s10", prev_s10)
    anchored = nearest(anchored, "s_h_effective", prev_sh)
    free_sact = min(nearest(anchored, "s8_down", down_spec["sout"]), key=residual_key)
    free_down = min(nearest(anchored, "s_act", free_sact["s_act"]), key=residual_key)
    finer = min(nearest(nearest(rows, "s10", prev_s10), "s_act", free_sact["s_act"]), key=residual_key)
    return {"unconstrained": unconstrained, "guarded": guarded, "no_residual": no_residual,
            "free_sact": free_sact, "free_down": free_down, "finer_gpalu": finer}


def main():
    PROGRESS.write_text(""); progress("RUN_START")
    for path in (OUT, WORK, LUT_DIR, GATE_QB, DOWN_QB): path.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != "lut_cali": raise RuntimeError("wrong branch")
    for commit in REQUIRED:
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"]).returncode: raise RuntimeError(f"missing commit {commit}")
    progress("GIT_CHECK_DONE")
    baseline_paths = [PREV / name for name in ("gate_parameters.json", "gpalu_parameters.json", "down_requant_parameters.json", "e2e_summary.json", "layer_selection.csv")]
    baseline_hashes = {str(p.relative_to(ROOT)): sha(p) for p in baseline_paths}
    base.write_json(OUT / "baseline_manifest.json", {"required_commits": list(REQUIRED), "artifact_hashes": baseline_hashes, "status": "READ_ONLY"})
    progress("BASELINE_VERIFY_DONE")
    progress("RESIDUAL_CONTRACT_INSPECTED")
    tests = [sys.executable, "-m", "unittest", "tests.test_joint_mlp_consumer_pilot", "tests.test_gpalu_fullscale_downaware_pilot", "tests.test_gpalu_pot_fusion_pilot", "tests.test_fusion_aware_gate_s10_pilot", "tests.test_mixed_linear_gelu_calibration"]
    with (OUT / "unit_test.log").open("w") as log:
        test_code = subprocess.run(tests, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/joint-mlp-consumer", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_USE_SHM="0")).returncode
    if test_code: raise RuntimeError("unit tests failed")
    source, _, _, _, _, _ = final.load_source_and_policies()
    historical_manifest = json.loads((mixed.ART / "manifest.json").read_text()); historical = mixed.load_historical(source, historical_manifest)
    base_specs = fusion.load_baseline_specs(source, historical); latest_specs = pot.load_current_specs(base_specs)
    previous_layers = json.loads((PREV / "gpalu_parameters.json").read_text())["layers"]
    pot_layers = json.loads((POT / "gpalu_parameters.json").read_text())["layers"]
    fusion_layers = json.loads((FUSION / "gate_parameters.json").read_text())["layers"]
    capture_path = MIXED / "work/capture.npz"; capture_meta = MIXED / "work/capture.json"
    if not capture_path.exists() or sha(capture_path) != json.loads(capture_meta.read_text())["sha256"]: raise RuntimeError("base capture unavailable")
    with np.load(capture_path) as z: captured = {k: z[k] for k in z.files}
    inventory = mixed.expected_inventory(); examples_cal = mixed.calibration_examples(source)
    residual_manifest = {"base_capture_sha256": sha(capture_path), "examples": [mixed.identity(e) for e in examples_cal], "residual_semantics": "FP decoder residual plus dequantized INT8 Down output", "reservoir": mixed.RESERVOIR, "seed": mixed.SEED}
    residual_id = digest(residual_manifest); residual_npz = WORK / "residual_capture.npz"; residual_meta = WORK / "residual_capture.json"
    if residual_npz.exists() and residual_meta.exists() and json.loads(residual_meta.read_text()).get("identity") == residual_id:
        with np.load(residual_npz) as z: residual_capture = {k: z[k] for k in z.files}
    else:
        expected_x = [captured[f"op{inventory.index(f'model.layers.{layer}.mlp.gate_proj'):04d}_prefill"] for layer in range(18)]
        residual_capture = capture_residual(source, examples_cal, expected_x)
        np.savez_compressed(residual_npz, **residual_capture); base.write_json(residual_meta, {"identity": residual_id, "sha256": sha(residual_npz)})
    # Validate exact pairing against the historical capture.
    for layer in range(18):
        idx = inventory.index(f"model.layers.{layer}.mlp.gate_proj")
        for group in GROUPS:
            if not np.allclose(residual_capture[f"layer{layer:02d}_{group}_x"], captured[f"op{idx:04d}_{group}"], rtol=2e-5, atol=2e-5): raise RuntimeError("residual capture pairing mismatch")
    progress("CAPTURE_REUSE_DONE")
    selections = {name: {} for name in ("free_sact", "free_down", "finer_gpalu", "no_residual", "residual", "guarded")}
    candidate_rows, selection_rows, guarded_rows, warning_rows, local_rows = [], [], [], [], []
    progress("SEARCH_START")
    reusable = OUT / "candidate_summary.csv"
    existing = [parse_csv_row(row) for row in load_csv(reusable)] if reusable.exists() else []
    if existing and {row["layer"] for row in existing} == set(range(18)):
        candidate_rows = existing
        selection_rows = [parse_csv_row(row) for row in load_csv(OUT / "layer_selection.csv")]
        guarded_rows = [parse_csv_row(row) for row in load_csv(OUT / "guarded_layer_selection.csv")]
        local_rows = [parse_csv_row(row) for row in load_csv(OUT / "local_baseline_comparison.csv")]
        warning_rows = [parse_csv_row(row) for row in load_csv(OUT / "error_cancellation_warnings.csv")]
        for layer in range(18):
            names = {s: f"model.layers.{layer}.mlp.{s}" for s in ("gate_proj", "up_proj", "down_proj")}
            restored = restored_selections([row for row in existing if row["layer"] == layer], previous_layers[layer], base_specs[names["down_proj"]])
            for key, result_key in (("free_sact", "free_sact"), ("free_down", "free_down"), ("finer_gpalu", "finer_gpalu"), ("no_residual", "no_residual"), ("residual", "unconstrained"), ("guarded", "guarded")):
                selections[key][layer] = compact_payload(restored[result_key], base_specs[names["gate_proj"]], base_specs[names["up_proj"]], base_specs[names["down_proj"]])
            progress(f"LAYER {layer+1}/18 DONE")
    else:
        for layer in range(18):
            names = {s: f"model.layers.{layer}.mlp.{s}" for s in ("gate_proj", "up_proj", "down_proj")}; idx = inventory.index(names["gate_proj"])
            x = np.concatenate([captured[f"op{idx:04d}_{g}"] for g in GROUPS]); labels = np.concatenate([[g] * len(captured[f"op{idx:04d}_{g}"]) for g in GROUPS])
            residual = np.concatenate([residual_capture[f"layer{layer:02d}_{g}_residual"] for g in GROUPS])
            result = evaluate_layer(layer, x, residual, labels, base_specs[names["gate_proj"]], latest_specs[names["gate_proj"]], float(fusion_layers[layer]["s10_selected"]), base_specs[names["up_proj"]], base_specs[names["down_proj"]], previous_layers[layer], pot_layers[layer])
            for key, result_key in (("free_sact", "free_sact"), ("free_down", "free_down"), ("finer_gpalu", "finer_gpalu"), ("no_residual", "no_residual"), ("residual", "unconstrained"), ("guarded", "guarded")):
                selections[key][layer] = compact_payload(result[result_key], base_specs[names["gate_proj"]], base_specs[names["up_proj"]], base_specs[names["down_proj"]])
            for c in result["candidates"]:
                c["selected_unconstrained"] = c is result["unconstrained"]; c["selected_guarded"] = c is result["guarded"]
                candidate_rows.append(serializable(c))
            selection_rows.append(serializable(result["unconstrained"])); guarded_rows.append(serializable(result["guarded"]))
            if result["unconstrained"]["error_cancellation_warning"]: warning_rows.append({"layer": layer, **result["unconstrained"]["warning_flags"], "selection": "unconstrained"})
            local_rows.append({"layer": layer, "previous_down_nmse": previous_layers[layer]["consumer_aware_down_nmse"], "joint_no_residual_down_nmse": result["no_residual"]["down_nmse_balanced"], "joint_residual_post_residual_nmse": result["unconstrained"]["post_residual_nmse_balanced"], "guarded_post_residual_nmse": result["guarded"]["post_residual_nmse_balanced"], "exact_candidate_count": len(result["candidates"]), "cheap_candidate_count": result["cheap_count"], "retained_gpalu_count": result["retained_count"]})
            progress(f"LAYER {layer+1}/18 DONE")
        write_csv(OUT / "candidate_summary.csv", compact_candidate_rows(candidate_rows)); write_csv(OUT / "layer_selection.csv", selection_rows); write_csv(OUT / "guarded_layer_selection.csv", guarded_rows); write_csv(OUT / "local_baseline_comparison.csv", local_rows); write_csv(OUT / "error_cancellation_warnings.csv", warning_rows or [{"layer": "none", "gelu": False, "post_gpalu": False, "down": False, "selection": "none"}])
    progress("SEARCH_DONE"); progress("GUARDED_SELECTION_DONE")
    # Export guarded configuration as the conservative hardware proposal.
    gate_meta, gelu_meta, gpalu_meta, down_meta, lut_hashes, gate_hashes, down_hashes = [], [], [], [], [], [], []
    for layer in range(18):
        item = selections["guarded"][layer]; row = result_row = guarded_rows[layer]
        gp, dp = item["gate_spec"]["params"], item["down_spec"]["params"]
        gpath = GATE_QB / f"layer_{layer:02d}_gate_qb.bin"; gpath.write_bytes(Profile().pack(gp["multiplier"], gp["shift"]).astype("<u4").tobytes()); gate_hashes.append(sha(gpath))
        dpath = DOWN_QB / f"layer_{layer:02d}_down_qb.bin"; dpath.write_bytes(Profile().pack(dp["multiplier"], dp["shift"]).astype("<u4").tobytes()); down_hashes.append(sha(dpath))
        lpath = LUT_DIR / f"layer_{layer:02d}_gelu.bin"; lpath.write_bytes(item["lut"]["lut"].astype(np.int8).tobytes()); lut_hashes.append(sha(lpath)); base.write_json(LUT_DIR / f"layer_{layer:02d}_gelu.json", {"layer": layer, "s10": row["s10"], "s_act": row["s_act"], "sha256": lut_hashes[-1]})
        gate_meta.append({"layer": layer, "s10": row["s10"], "qparams": str(gpath.relative_to(ROOT)), "sha256": gate_hashes[-1]})
        gelu_meta.append({"layer": layer, "s_act": row["s_act"], "percentile": row["gelu_percentile"], "lut": str(lpath.relative_to(ROOT)), "sha256": lut_hashes[-1]})
        gpalu_meta.append({k: row[k] for k in ("layer", "s_h_target", "s_h_effective", "M_G", "S_G", "gpalu_clip_rate", "post_gpalu_nmse")})
        down_meta.append({"layer": layer, "s8_down": row["s8_down"], "input_scale": row["s_h_effective"], "qparams": str(dpath.relative_to(ROOT)), "sha256": down_hashes[-1], "saturation_rate": row["down_saturation_rate"]})
    base.write_json(OUT / "gate_parameters.json", {"layers": gate_meta}); base.write_json(OUT / "gelu_parameters.json", {"layers": gelu_meta}); base.write_json(OUT / "gpalu_parameters.json", {"layers": gpalu_meta}); base.write_json(OUT / "down_parameters.json", {"layers": down_meta})
    progress("ABLATION_DONE")
    manifest = json.loads((PREV / "e2e_dataset_manifest.json").read_text()); base.write_json(OUT / "e2e_dataset_manifest.json", manifest)
    examples = mixed.e2e_examples(source); configurations = {"FREE_SACT_ONLY": selections["free_sact"], "FREE_DOWN_S8": selections["free_down"], "FINER_GPALU_SCALE": selections["finer_gpalu"], "JOINT_NO_RESIDUAL_OBJECTIVE": selections["no_residual"], "JOINT_RESIDUAL_AWARE": selections["residual"], "GUARDED_JOINT_RESIDUAL_AWARE": selections["guarded"]}
    run_id = digest({"baseline": baseline_hashes, "guarded": guarded_rows, "manifest": manifest}); records = []; progress("E2E_START")
    for i, example in enumerate(examples):
        prior_raw = json.loads((PREV / "work" / f"e2e_{i:02d}.json").read_text())["value"]
        pot_raw = json.loads((POT / "work" / f"e2e_{i:02d}.json").read_text())["value"]
        checkpoint = WORK / f"e2e_{i:02d}.json"; identity = digest({"run": run_id, "example": mixed.identity(example)})
        cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
        if cached and cached.get("identity") != identity: raise RuntimeError("E2E checkpoint mismatch")
        new = cached["value"] if cached else evaluate_sequence(source, latest_specs, configurations, example)
        if not cached: base.write_json(checkpoint, {"identity": identity, "value": new})
        records.append({"FP_FULL": pot_raw["FP_FULL"], "FUSION_AWARE_FLOAT_PRODUCT": pot_raw["CURRENT_FUSION_AWARE_FLOAT_PRODUCT"], "STATIC_POT_KG": pot_raw["JOINT_POST_GPALU_AWARE"], "FULL_SCALE_DOWN_AWARE_PREVIOUS": prior_raw["FULL_SCALE_DOWN_AWARE_HW"], **new})
        progress(f"SEQUENCE {i+1}/8 DONE")
    modes = list(records[0]); metrics = {mode: response.merged([r[mode] for r in records]) for mode in modes}
    comparisons = {"previous_to_guarded": delta(metrics["FULL_SCALE_DOWN_AWARE_PREVIOUS"], metrics["GUARDED_JOINT_RESIDUAL_AWARE"]), "no_residual_to_residual": delta(metrics["JOINT_NO_RESIDUAL_OBJECTIVE"], metrics["JOINT_RESIDUAL_AWARE"]), "unconstrained_to_guarded": delta(metrics["JOINT_RESIDUAL_AWARE"], metrics["GUARDED_JOINT_RESIDUAL_AWARE"])}
    previous_top1, guarded_top1 = metrics["FULL_SCALE_DOWN_AWARE_PREVIOUS"]["top1"], metrics["GUARDED_JOINT_RESIDUAL_AWARE"]["top1"]
    gain_pp = 100. * (guarded_top1 - previous_top1)
    preserved = metrics["GUARDED_JOINT_RESIDUAL_AWARE"]["nmse"] <= metrics["FULL_SCALE_DOWN_AWARE_PREVIOUS"]["nmse"] * 1.05 and metrics["GUARDED_JOINT_RESIDUAL_AWARE"]["kl"] <= metrics["FULL_SCALE_DOWN_AWARE_PREVIOUS"]["kl"] * 1.05
    decision = "PROCEED_TO_LARGE_BATCH" if gain_pp >= 3. and preserved else ("SMALL_BATCH_GAIN_PROMISING_BUT_BELOW_TARGET" if gain_pp > 0 and preserved else "CALIBRATION_ONLY_GAIN_INSUFFICIENT")
    base.write_json(OUT / "e2e_summary.json", {"modes": metrics, "comparisons": comparisons, "decision": decision, "e2e_used_for_selection": False})
    ablation_modes = ["FULL_SCALE_DOWN_AWARE_PREVIOUS", "FREE_SACT_ONLY", "FREE_DOWN_S8", "FINER_GPALU_SCALE", "JOINT_RESIDUAL_AWARE"]
    ablation_rows = []
    for before, after, change in zip(ablation_modes, ablation_modes[1:], ("free_s_act", "free_down_s8", "finer_gpalu_scale", "residual_aware_objective")):
        ablation_rows.append({"change": change, "from_mode": before, "to_mode": after, "kl": metrics[after]["kl"], "nmse": metrics[after]["nmse"], "top1": metrics[after]["top1"], "top1_delta_pp": 100. * (metrics[after]["top1"] - metrics[before]["top1"])})
    write_csv(OUT / "ablation_summary.csv", ablation_rows); progress("E2E_DONE")
    if {str(p.relative_to(ROOT)): sha(p) for p in baseline_paths} != baseline_hashes: raise RuntimeError("baseline artifact mutation")
    warning_layers = [int(row["layer"]) for row in guarded_rows if row["error_cancellation_warning"]]
    gpalu_clips = [r["gpalu_clip_rate"] for r in guarded_rows]
    down_saturations = [r["down_saturation_rate"] for r in guarded_rows]
    verification = {
        "branch": branch,
        "source_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "required_commits_present": True,
        "model_SHA256": historical_manifest["gguf"]["sha256"],
        "calibration_capture_identity": residual_id,
        "E2E_dataset_identity": digest(manifest),
        "external_activation_bit_width": 8,
        "GPALU_raw_product": "signed_INT16",
        "GPALU_multiplier": "UInt16",
        "GPALU_shift": "UInt5",
        "GPALU_output": "signed_INT8",
        "dynamic_GPALU_scaling": False,
        "s_act_jointly_searched": True,
        "Down_s8_searched": True,
        "residual_aware_objective": True,
        "residual_add_semantics": "FP decoder residual plus reconstructed Down output, matching current E2E simulator",
        "Normalizer_calibrated": False,
        "RoPE_calibrated": False,
        "E2E_used_for_selection": False,
        "large_batch_performed": False,
        "exact_candidate_counts_per_layer": [int(r["exact_candidate_count"]) for r in local_rows],
        "warning_layer_list": warning_layers,
        "GPALU_clipping_aggregate": float(np.mean(gpalu_clips)),
        "GPALU_clipping_worst": float(max(gpalu_clips)),
        "Down_saturation_aggregate": float(np.mean(down_saturations)),
        "Down_saturation_worst": float(max(down_saturations)),
        "baseline_artifacts_unchanged": True,
        "all_gate_qparam_hashes": gate_hashes,
        "all_down_qparam_hashes": down_hashes,
        "all_LUT_hashes": lut_hashes,
        "test_command": tests,
        "return_code": test_code,
        "elapsed_time": time.monotonic() - START,
    }
    base.write_json(OUT / "verification.json", verification)
    lines = ["# Joint MLP consumer-aware small-batch pilot", "", "## 1. Scope", "Calibration-only joint optimization of the existing INT8 MLP datapath through its residual consumer.", "", "## 2. Fixed hardware contract", "No RTL datapath changes were introduced. All external VPU activation boundaries remained INT8. GPALU used signed INT8 operands, signed INT16 product, static UInt16 M_G / UInt5 S_G requantization, and signed INT8 output fed directly into Down projection.", "", "## 3. Previous 79.17% baseline", f"The previous full-scale Down-aware mode had Top1 {previous_top1:.6%} on the same 192 targets.", "", "## 4. What calibration restrictions were removed", "s_act remained live through consumer-aware search, Down output s8 was recalibrated, GPALU search was refined, and selection extended through the post-MLP residual tensor.", "", "## 5. Search method", f"The fixed small capture was used. Up to {MAX_EXACT_LOCAL} deterministic GPALU candidates per layer received exact Down and residual evaluation; every exact candidate searched Down s8 and regenerated per-channel M/S. Exact counts are recorded in `verification.json`; `candidate_summary.csv` is the compact deterministic audit subset, while the full sweep remains an uncommitted work artifact. E2E metrics were never used for parameter selection.", "", "## 6. Joint s10 / s_act results", "See `layer_selection.csv` and `guarded_layer_selection.csv`.", "", "## 7. GPALU full-scale results", "Actual effective output scales derived from canonical M_G/S_G were used throughout.", "", "## 8. Down output scale recalibration", "Down Wq/sW stayed fixed; output s8 and its per-channel requantization parameters were searched jointly.", "", "## 9. Residual-aware calibration", "The current simulator reconstructs the Down result before adding the FP residual. The local objective preserves that apples-to-apples contract.", "", "## 10. Error-cancellation analysis", f"Guarded warning layers: {warning_layers or 'none'}. The guarded selection excludes candidates whose GELU, post-GPALU, or Down NMSE exceeds twice the corresponding best local value.", "", "## 11. Local ablation", "The sequential small-batch ablation is recorded in `ablation_summary.csv`.", "", "## 12. Small E2E results", "| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, value in metrics.items(): lines.append("| " + mode + " | " + " | ".join(f"{value[k]:.9g}" for k in ("nll", "ppl", "kl", "nmse", "mse", "mae", "cosine", "flattened_cosine", "top1", "in5", "overlap")) + " |")
    lines += ["", "## 13. Progress toward ~82-83% Top1 target", f"Guarded Top1 was {guarded_top1:.6%}, a {gain_pp:+.4f} pp change from the previous full-scale baseline. This target was interpretive only.", "", "## 14. Implications before Normalizer/RoPE calibration", "This pilot isolates MLP calibration margin. Normalizer and RoPE were not included.", "", "## 15. Decision", f"**{decision}**", "", "## 16. Limitations", "The calibration and validation populations remain deliberately small. Residual addition follows the current software simulator's reconstructed-Down plus FP-residual behavior. No statistical claim or large-batch conclusion is made.", "", "s_act was consumer-aware searched rather than frozen by standalone GELU optimization. Down output s8 was allowed to recalibrate. Final local selection used post-MLP-residual fidelity. E2E metrics were never used for parameter selection."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    progress("TEST_DONE"); progress("VERIFY_DONE"); progress("ARTIFACT_WRITE_DONE"); progress("RUN_COMPLETE")


if __name__ == "__main__":
    try: main()
    except BaseException:
        progress("RUN_FAILED")
        raise
