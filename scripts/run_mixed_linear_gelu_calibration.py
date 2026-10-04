"""Calibrate consumer-aware 126-Linear contracts and per-op GELUTanh LUTs."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import calibrate_cached_decode_ptq as cal
import compare_k_selectors_e2e as selector
import run_gelu_standalone_calibration as gelu_base

final = selector.final
response, base, row = final.response, selector.base, selector.row
Profile = selector.Profile
OUT = ROOT / "diagnostics/mixed_linear_gelu_calibration"
WORK = OUT / "work"
TABLES = OUT / "gelu_tables"
PROGRESS = ROOT / "mixed_linear_gelu_calibration_progress.txt"
ART = ROOT / "calibration_outputs/gguf-all-126-oasst1"
LINEAR_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ORDINARY_SUFFIXES = tuple(x for x in LINEAR_SUFFIXES if x != "mlp.gate_proj")
SX_JS = (-2, -1, 0, 1, 2)
SOUT_IS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
GELU_PERCENTILES = (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 100.0)
GROUPS = ("prefill", "decode")
CAL_POSITIONS = (0, 5, 10, 15)
CAL_LIMIT = 8
RESERVOIR = 8
SEED = 20261005
MODES = ("FP_FULL", "MIXED_LINEAR_FP_GELU", "MIXED_LINEAR_GELU_LUT")
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as f:
        f.write(message + "\n")
        f.flush()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha(path):
    return base.sha256_file(Path(path))


def write_csv(path, rows):
    base.write_csv(path, rows)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def classify_op(name):
    if name.endswith("mlp.gate_proj"):
        return "GATE_PREACT_INT10"
    if any(name.endswith(suffix) for suffix in ORDINARY_SUFFIXES):
        return "ORDINARY_INT8"
    raise ValueError(f"unsupported Linear op: {name}")


def expected_inventory():
    return [f"model.layers.{layer}.{suffix}" for layer in range(18) for suffix in LINEAR_SUFFIXES]


def output_range(contract):
    return (-512, 511) if contract == "GATE_PREACT_INT10" else (-128, 127)


def balanced(values):
    present = [v for v in values if v is not None]
    if not present:
        raise ValueError("no metric groups")
    return float(np.mean(present))


def nmse(value, reference):
    value, reference = np.asarray(value, np.float64), np.asarray(reference, np.float64)
    den = float(np.square(reference).sum())
    err = float(np.square(value - reference).sum())
    return err / den if den else (0.0 if err == 0 else math.inf)


def candidate_key(candidate, sx_anchor, sout_anchor):
    return (candidate["score"], candidate["worst_nmse"], candidate["output_saturation_rate"],
            candidate["input_clip_rate"], abs(math.log2(candidate["sx_base"] / sx_anchor)),
            abs(math.log2(candidate["output_scale"] / sout_anchor)),
            candidate["sx_base"], candidate["output_scale"])


def unique_scales(values):
    return sorted({float(v) for v in values if np.isfinite(v) and v > 0})


def scale_grid(anchor, offsets, incumbent):
    return unique_scales([anchor * 2.0 ** (offset / (4 if offsets is SX_JS else 8)) for offset in offsets] + [incumbent])


def effective_bounds(shift):
    shifts = Profile().effective(shift)
    lo, hi = max(-8, int(shifts.max()) - 31), min(7, int(shifts.min()))
    return lo, hi


def quantized_linear_rows(x, spec, return_codes=False):
    x = np.asarray(x, np.float64)
    q, ks, info = selector.select_absmax(x, spec["sx"], spec["params"]["shift"])
    acc = base.integer_dot(q, spec["wq"], "fp64-exact")
    lo, hi = output_range(spec["contract"])
    codes = np.empty(acc.shape, np.int16)
    raw_clips = 0
    for k in np.unique(ks):
        mask = ks == k
        raw = Profile().apply(acc[mask], spec["params"]["multiplier"],
                              spec["params"]["shift"] - int(k), saturate=False)
        raw_clips += int(((raw < lo) | (raw > hi)).sum())
        codes[mask] = np.clip(raw, lo, hi).astype(np.int16)
    output = codes.astype(np.float32) * spec["sout"]
    details = dict(k=ks, input_codes=q, output_codes=codes,
                   output_saturation_rate=raw_clips / codes.size,
                   input_clip_rate=float(np.mean(np.abs(x) > 127 * spec["sx"] * np.exp2(ks)[:, None])),
                   effective_shift_min=info["effective_shift_min"],
                   effective_shift_max=info["effective_shift_max"])
    return (output, details) if return_codes else output


def evaluate_candidate(x, labels, ref, wq, sw, sx, sout, contract, acc_cache):
    params = Profile().approximate(sx * sw / sout)
    result = dict(sx_base=sx, output_scale=sout, feasible=False)
    if np.any(params["status"] != "ok"):
        result["rejection_reason"] = "invalid_profile_approximation"
        return result
    lo_k, hi_k = effective_bounds(params["shift"])
    if lo_k > hi_k:
        result["rejection_reason"] = "empty_k_interval"
        return result
    q, ks, info = selector.select_absmax(x, sx, params["shift"])
    effective = params["shift"][None, :] - ks[:, None]
    if effective.min() < 0 or effective.max() > 31:
        result["rejection_reason"] = "effective_shift"
        return result
    qkey = hashlib.sha256(q.tobytes() + wq.shape.__repr__().encode() + wq[:1].tobytes()).hexdigest()
    if qkey not in acc_cache:
        acc_cache[qkey] = base.integer_dot(q, wq, "fp64-exact")
    acc = acc_cache[qkey]
    lo, hi = output_range(contract)
    out = np.empty(acc.shape, np.float64)
    clips = 0
    for k in np.unique(ks):
        mask = ks == k
        raw = Profile().apply(acc[mask], params["multiplier"], params["shift"] - int(k), saturate=False)
        clips += int(((raw < lo) | (raw > hi)).sum())
        out[mask] = np.clip(raw, lo, hi) * sout
    scores = {}
    for group in GROUPS:
        mask = labels == group
        scores[group] = nmse(out[mask], ref[mask]) if mask.any() else None
    result.update(feasible=True, params=params, score=balanced(scores.values()),
                  prefill_nmse=scores["prefill"], decode_nmse=scores["decode"],
                  worst_nmse=max(v for v in scores.values() if v is not None),
                  output_saturation_rate=clips / out.size,
                  input_clip_rate=float(np.mean(np.abs(x) > 127 * sx * np.exp2(ks)[:, None])),
                  feasible_k_min=lo_k, feasible_k_max=hi_k,
                  observed_k_min=int(ks.min()), observed_k_max=int(ks.max()),
                  effective_shift_min=int(effective.min()), effective_shift_max=int(effective.max()),
                  unique_acc_count=len(acc_cache))
    return result


def search_linear(x, labels, weight, wq, sw, historical_sx, historical_sout, contract):
    ref = x.astype(np.float64) @ weight.astype(np.float64).T
    row_absmax = np.max(np.abs(x), axis=1)
    positive = row_absmax[row_absmax > 0]
    sx_anchor = float(np.median(positive) / 127.0) if len(positive) else historical_sx
    limit = 511 if contract == "GATE_PREACT_INT10" else 127
    sout_anchor = max(float(np.max(np.abs(ref))) / limit, np.finfo(np.float64).tiny)
    sx_values = unique_scales([sx_anchor * 2.0 ** (j / 4) for j in SX_JS] + [historical_sx])
    sout_values = unique_scales([sout_anchor * 2.0 ** (i / 8) for i in SOUT_IS] + [historical_sout])
    candidates, cache = [], {}
    for sx in sx_values:
        for sout in sout_values:
            candidate = evaluate_candidate(x, labels, ref, wq, sw, sx, sout, contract, cache)
            candidate.update(sx_anchor=sx_anchor, output_anchor=sout_anchor,
                             historical_pair=(sx == historical_sx and sout == historical_sout))
            candidates.append(candidate)
    feasible = [c for c in candidates if c["feasible"]]
    if not feasible:
        raise RuntimeError("no feasible joint Linear candidate")
    selected = min(feasible, key=lambda c: candidate_key(c, sx_anchor, sout_anchor))
    return selected, candidates, ref


def load_historical(source, manifest):
    entries = {entry["module_name"]: entry for entry in manifest["modules"]}
    result = {}
    for name in expected_inventory():
        entry = entries[name]
        directory = ART / Path(entry["module_manifest"]).parent
        module_manifest = json.loads((directory / "manifest.json").read_text())
        key = module_manifest["modules"][0]["key"]
        wq_path = ART / entry["int8_weight_file"]
        wq = np.load(wq_path, mmap_mode="r")
        with np.load(directory / "scales.npz") as scales:
            sx, sout, sw = float(scales[key + ".s_X"]), float(scales[key + ".s_10"]), scales[key + ".s_W"].copy()
        expected_wq, expected_sw = base.quantize_weight(source.weight(name))
        np.testing.assert_array_equal(wq, expected_wq)
        np.testing.assert_array_equal(sw, expected_sw)
        result[name] = dict(entry=entry, historical_sx=sx, historical_sout=sout,
                            wq=wq, sw=sw, wq_path=str(wq_path.relative_to(ROOT)),
                            wq_sha256=sha(wq_path))
    return result


def identity(example):
    return {k: example[k] for k in ("index", "anchor_id", "response_id", "tree_id")}


def calibration_examples(source):
    pairs, _ = cal.build_pairs("calibration")
    examples, _ = cal.tokenize_pairs(source, pairs)
    if max(CAL_POSITIONS) >= len(examples):
        raise RuntimeError("calibration subset unavailable")
    return [examples[i] for i in CAL_POSITIONS]


def e2e_examples(source):
    wanted = json.loads((ROOT / "diagnostics/activation_lut_calibration_pilot/e2e_dataset_manifest.json").read_text())["examples"]
    pairs, _ = cal.build_pairs("validation")
    examples, _ = cal.tokenize_pairs(source, pairs)
    by_id = {e["anchor_id"]: e for e in examples if len(e["targets"]) >= 24}
    selected = [by_id[item["anchor_id"]] for item in wanted]
    if len(selected) != 8:
        raise RuntimeError("E2E subset unavailable")
    return selected


def capture(source, examples, inventory):
    reservoirs = {name: {group: cal.Reservoir(RESERVOIR, SEED) for group in GROUPS} for name in inventory}
    state, handles = {}, []
    for name in inventory:
        module = source.modules[name]
        def hook(module, args, name=name):
            value = args[0].detach().cpu().numpy()[0]
            group = state["group"]
            for position, vector in enumerate(value):
                reservoirs[name][group].add(vector, name, group, state["conversation"], state["offset"] + position)
        handles.append(module.register_forward_pre_hook(hook))
    try:
        for number, example in enumerate(examples):
            state.update(group="prefill", conversation=example["index"], offset=0)
            _, _, cache = cal.run_fp(source, example["prompt_ids"], None, layers=False)
            for t in range(1, min(CAL_LIMIT, len(example["targets"]))):
                state.update(group="decode", offset=len(example["prompt_ids"]) + t - 1)
                _, _, cache = cal.run_fp(source, [example["targets"][t - 1]], cache, layers=False)
            progress(f"CAPTURE {number+1}/{len(examples)} DONE")
    finally:
        for handle in handles:
            handle.remove()
    arrays = {}
    for index, name in enumerate(inventory):
        for group in GROUPS:
            arrays[f"op{index:04d}_{group}"] = reservoirs[name][group].arrays()[0]
    return arrays


def gelu_candidate_rows(g_fp, q10_codes, s10, percentile):
    gq = q10_codes.astype(np.float64) * s10
    ref = gelu_base.gelu(g_fp)
    qref = gelu_base.gelu(gq)
    threshold = float(np.percentile(np.abs(qref), percentile))
    s_act = threshold / 127.0
    lut = gelu_base.make_lut(s10, s_act)
    actual_codes = lut[gelu_base.address(q10_codes)]
    actual = actual_codes.astype(np.float64) * s_act
    return dict(retained_percentile=percentile, clipping_fraction=(100.0-percentile)/100.0,
                s_act=s_act, input_nmse=nmse(qref, ref), table_nmse=nmse(actual, qref),
                total_nmse=nmse(actual, ref), saturation_rate=float(np.mean(np.abs(np.rint(qref/s_act)) > 127)),
                code_utilization=float(len(np.unique(actual_codes))/255.0), zero_fraction=float(np.mean(actual_codes == 0)),
                lut=lut)


def gelu_key(candidate):
    return (candidate["score"], candidate["table_nmse"], candidate["worst_total_nmse"],
            candidate["clipping_fraction"], candidate["saturation_rate"], candidate["s_act"])


def calibrate_gelu(groups, spec):
    per_percentile = []
    group_data = {}
    for group, x in groups.items():
        g_fp = x.astype(np.float64) @ spec["weight"].astype(np.float64).T
        _, details = quantized_linear_rows(x, spec, return_codes=True)
        group_data[group] = (g_fp, details["output_codes"])
    for percentile in GELU_PERCENTILES:
        rows = {group: gelu_candidate_rows(g, q, spec["sout"], percentile)
                for group, (g, q) in group_data.items()}
        candidate = dict(retained_percentile=percentile,
                         selection_policy="ABSMAX" if percentile == 100.0 else "PERCENTILE_CLIPPING",
                         clipping_fraction=(100.0-percentile)/100.0,
                         s_act=rows[next(iter(rows))]["s_act"])
        # A shared candidate scale within one op must use pooled q-input distribution.
        pooled = np.concatenate([gelu_base.gelu(q.astype(np.float64)*spec["sout"]).reshape(-1)
                                 for _, q in group_data.values()])
        candidate["s_act"] = float(np.percentile(np.abs(pooled), percentile))/127.0
        lut = gelu_base.make_lut(spec["sout"], candidate["s_act"])
        total, table, inp, sat = {}, {}, {}, {}
        for group, (g_fp, q) in group_data.items():
            gq = q.astype(np.float64)*spec["sout"]
            ref, qref = gelu_base.gelu(g_fp), gelu_base.gelu(gq)
            codes = lut[gelu_base.address(q)]
            actual = codes.astype(np.float64)*candidate["s_act"]
            inp[group], table[group], total[group] = nmse(qref, ref), nmse(actual, qref), nmse(actual, ref)
            sat[group] = float(np.mean(np.abs(np.rint(qref/candidate["s_act"])) > 127))
        candidate.update(input_nmse=balanced(inp.values()), table_nmse=balanced(table.values()),
                         total_nmse=balanced(total.values()), score=balanced(total.values()),
                         prefill_total_nmse=total.get("prefill"), decode_total_nmse=total.get("decode"),
                         worst_total_nmse=max(total.values()), saturation_rate=balanced(sat.values()), lut=lut)
        per_percentile.append(candidate)
    return min(per_percentile, key=gelu_key), per_percentile


def serialize_params(specs):
    layers = {}
    for name, spec in specs.items():
        packed = Profile().pack(spec["params"]["multiplier"], spec["params"]["shift"])
        layers[name] = dict(contract=spec["contract"], sx_base=spec["sx"],
                            output_scale=spec["sout"], output_scale_name="s10" if spec["contract"].endswith("INT10") else "s8",
                            multiplier=spec["params"]["multiplier"].tolist(), shift=spec["params"]["shift"].tolist(),
                            zero_point=0, packed_qb=packed.tolist(),
                            wq_path=spec["wq_path"], wq_sha256=spec["wq_sha256"], parameter_hash=spec["parameter_hash"])
    return {"status": "MIXED_LINEAR_GELU_BASELINE", "layers": layers}


def apply_tensor(x, spec):
    array = x.detach().cpu().numpy()
    shape = array.shape
    out = quantized_linear_rows(array.reshape(-1, shape[-1]), spec)
    return x.new_tensor(out.reshape(*shape[:-1], out.shape[-1]))


@contextlib.contextmanager
def patch_mixed(source, specs, gelu_tables=None):
    originals = []
    try:
        for name, spec in specs.items():
            module = source.modules[name]
            originals.append((module, module.forward))
            module.forward = lambda x, spec=spec: apply_tensor(x, spec)
        if gelu_tables is not None:
            for layer, block in enumerate(source.model.model.layers):
                act = block.mlp.act_fn
                originals.append((act, act.forward))
                item = gelu_tables[layer]
                def forward(x, item=item):
                    q = np.clip(np.rint(x.detach().cpu().numpy()/item["s10"]), -512, 511).astype(np.int16)
                    value = item["lut"][gelu_base.address(q)].astype(np.float32)*item["s_act"]
                    return x.new_tensor(value)
                act.forward = forward
        yield
    finally:
        for module, original in originals:
            module.forward = original


def run_quant(source, specs, ids, cache, gelu_tables=None):
    with patch_mixed(source, specs, gelu_tables):
        context = {}
        hidden, states, cache = response.forward_hidden(source, ids, [len(ids)-1], context, cache, True, False)
        with source.torch.inference_mode():
            logits = source.model.lm_head(hidden)
    return logits, states, cache


def evaluate_sequence(source, specs, gelu_tables, example):
    scores = {mode: response.Scores() for mode in MODES}
    caches = {mode: None for mode in MODES}
    for t, target in enumerate(example["targets"][:24]):
        ids = example["prompt_ids"] if t == 0 else [example["targets"][t-1]]
        logits = {}
        logits[MODES[0]], _, caches[MODES[0]] = cal.run_fp(source, ids, caches[MODES[0]], layers=False)
        logits[MODES[1]], _, caches[MODES[1]] = run_quant(source, specs, ids, caches[MODES[1]])
        logits[MODES[2]], _, caches[MODES[2]] = run_quant(source, specs, ids, caches[MODES[2]], gelu_tables)
        if len({id(v) for v in caches.values()}) != 3:
            raise RuntimeError("KV cache alias")
        for mode in MODES:
            scores[mode].add(source.torch, logits[mode], logits[MODES[0]], [target])
    return {mode: value.raw() for mode, value in scores.items()}


def main():
    PROGRESS.write_text("")
    progress("RUN_START")
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != "lut_cali": raise RuntimeError("wrong branch")
    progress("GIT_CHECK_DONE")
    inventory = expected_inventory()
    contracts = {name: classify_op(name) for name in inventory}
    counts = {"total": len(inventory), "ordinary": sum(v == "ORDINARY_INT8" for v in contracts.values()), "gate": sum(v == "GATE_PREACT_INT10" for v in contracts.values())}
    if counts != {"total":126,"ordinary":108,"gate":18}: raise RuntimeError("op inventory mismatch")
    base.write_json(OUT/"op_inventory.json", {"counts":counts,"operations":[{"name":n,"contract":contracts[n]} for n in inventory]})
    progress("OP_INVENTORY_DONE total=126 ordinary=108 gate=18")

    progress("UNIT_TEST_START")
    tests=[sys.executable,"-m","unittest","tests.test_mixed_linear_gelu_calibration","tests.test_gelu_standalone_calibration"]
    with (OUT/"unit_test.log").open("w") as log:
        test_code=subprocess.run(tests,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPYCACHEPREFIX="/tmp/mixed-linear-gelu",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",KMP_USE_SHM="0")).returncode
    if test_code: progress("UNIT_TEST_FAILED"); raise RuntimeError("unit tests failed")
    progress("UNIT_TEST_DONE")

    source, model_manifest, _, _, _, _ = final.load_source_and_policies()
    historical_manifest=json.loads((ART/"manifest.json").read_text())
    history=load_historical(source,historical_manifest)
    cal_examples=calibration_examples(source)
    capture_manifest={"examples":[identity(e) for e in cal_examples],"response_limit":CAL_LIMIT,"reservoir_capacity":RESERVOIR,"seed":SEED,
                      "role":"FP Linear inputs","model_sha256":historical_manifest["gguf"]["sha256"],"inventory_hash":digest(inventory),
                      "script_sha256":sha(__file__),"row_selector":"compare_k_selectors_e2e.select_absmax",
                      "profile":Profile().metadata(),"weight_quantization":"static_quant.core.quantize_weight"}
    base.write_json(OUT/"capture_manifest.json",capture_manifest)
    capture_identity=digest(capture_manifest)
    capture_npz=WORK/"capture.npz"; capture_meta=WORK/"capture.json"
    progress("CAPTURE_REUSE_CHECK")
    if capture_npz.exists() and capture_meta.exists() and json.loads(capture_meta.read_text()).get("identity")==capture_identity:
        if sha(capture_npz)!=json.loads(capture_meta.read_text())["sha256"]: raise RuntimeError("capture checksum")
        with np.load(capture_npz) as z: captured={k:z[k] for k in z.files}
        progress("CAPTURE_REUSED")
    else:
        progress("CAPTURE_START")
        captured=capture(source,cal_examples,inventory)
        np.savez_compressed(capture_npz,**captured)
        base.write_json(capture_meta,{"identity":capture_identity,"sha256":sha(capture_npz)})
        progress("CAPTURE_DONE")

    progress("LINEAR_SEARCH_START")
    specs, summaries, candidate_rows = {}, [], []
    total_infeasible=0
    for layer in range(18):
        progress(f"LAYER {layer+1}/18 START")
        for suffix in LINEAR_SUFFIXES:
            name=f"model.layers.{layer}.{suffix}"; index=inventory.index(name)
            groups={g:captured[f"op{index:04d}_{g}"] for g in GROUPS}
            x=np.concatenate([groups[g] for g in GROUPS]); labels=np.concatenate([[g]*len(groups[g]) for g in GROUPS])
            old=history[name]; weight=source.weight(name)
            selected,candidates,_=search_linear(x,labels,weight,old["wq"],old["sw"],old["historical_sx"],old["historical_sout"],contracts[name])
            total_infeasible += sum(not c["feasible"] for c in candidates)
            params=selected["params"]
            parameter_hash=digest({"name":name,"sx":selected["sx_base"],"sout":selected["output_scale"],"m":params["multiplier"].tolist(),"s":params["shift"].tolist(),"wq":old["wq_sha256"]})
            specs[name]=dict(contract=contracts[name],sx=selected["sx_base"],sout=selected["output_scale"],params=params,wq=old["wq"],weight=weight,
                             sw=old["sw"],wq_path=old["wq_path"],wq_sha256=old["wq_sha256"],parameter_hash=parameter_hash)
            summary=dict(layer=layer,module_name=name,contract=contracts[name],sx_base=selected["sx_base"],output_scale=selected["output_scale"],
                         output_scale_name="s10" if contracts[name].endswith("INT10") else "s8",weight_scale_min=float(old["sw"].min()),weight_scale_max=float(old["sw"].max()),
                         multiplier_min=int(params["multiplier"].min()),multiplier_max=int(params["multiplier"].max()),shift_min=int(params["shift"].min()),shift_max=int(params["shift"].max()),
                         feasible_k_min=selected["feasible_k_min"],feasible_k_max=selected["feasible_k_max"],observed_k_min=selected["observed_k_min"],observed_k_max=selected["observed_k_max"],
                         effective_shift_min=selected["effective_shift_min"],effective_shift_max=selected["effective_shift_max"],input_clip_rate=selected["input_clip_rate"],
                         output_saturation_rate=selected["output_saturation_rate"],prefill_nmse=selected["prefill_nmse"],decode_nmse=selected["decode_nmse"],balanced_nmse=selected["score"],parameter_hash=parameter_hash)
            summaries.append(summary)
            for candidate in candidates:
                candidate_rows.append({
                    "layer":layer,"module_name":name,"contract":contracts[name],
                    "sx_base":candidate["sx_base"],"output_scale":candidate["output_scale"],
                    "feasible":candidate["feasible"],"score":candidate.get("score"),
                    "worst_nmse":candidate.get("worst_nmse"),
                    "output_saturation_rate":candidate.get("output_saturation_rate"),
                    "input_clip_rate":candidate.get("input_clip_rate"),
                    "prefill_nmse":candidate.get("prefill_nmse"),"decode_nmse":candidate.get("decode_nmse"),
                    "historical_pair":candidate["historical_pair"],
                    "rejection_reason":candidate.get("rejection_reason"),
                })
        progress(f"LAYER {layer+1}/18 LINEAR_DONE")
        base.write_json(WORK/f"linear_layer_{layer:02d}.json",{"summaries":[r for r in summaries if r["layer"]==layer]})
    write_csv(OUT/"linear_candidates_summary.csv",candidate_rows)
    write_csv(OUT/"linear_summary.csv",summaries)
    write_csv(OUT/"ordinary_gemm_summary.csv",[r for r in summaries if r["contract"]=="ORDINARY_INT8"])
    write_csv(OUT/"gate_gemm_summary.csv",[r for r in summaries if r["contract"]=="GATE_PREACT_INT10"])
    base.write_json(OUT/"linear_parameters.json",serialize_params(specs))
    progress("LINEAR_SEARCH_DONE ordinary=108 gate=18")

    progress("GELU_SEARCH_START")
    gelu_candidates,gelu_selected,gelu_tables,gelu_hashes=[],[],{},[]
    for layer in range(18):
        name=f"model.layers.{layer}.mlp.gate_proj"; index=inventory.index(name)
        groups={g:captured[f"op{index:04d}_{g}"] for g in GROUPS}
        selected,candidates=calibrate_gelu(groups,specs[name])
        for c in candidates: gelu_candidates.append({"layer":layer,**{k:v for k,v in c.items() if k!="lut"}})
        binary=TABLES/f"layer_{layer:02d}_gelu.bin"; binary.write_bytes(selected["lut"].tobytes())
        h=sha(binary);gelu_hashes.append(h)
        meta={"layer":layer,"gate_op_name":name,"s10":specs[name]["sout"],"s_act":selected["s_act"],"retained_percentile":selected["retained_percentile"],
              "clipping_fraction":selected["clipping_fraction"],"input_q_range":[-512,511],"output_q_range":[-127,127],"entries":1024,"lut_sha256":h}
        base.write_json(TABLES/f"layer_{layer:02d}_gelu.json",meta)
        row_sel={**{k:v for k,v in selected.items() if k!="lut"},**meta};gelu_selected.append(row_sel)
        gelu_tables[layer]={"s10":specs[name]["sout"],"s_act":selected["s_act"],"lut":selected["lut"]}
        progress(f"GELU_LAYER {layer+1}/18 DONE")
    write_csv(OUT/"gelu_candidates.csv",gelu_candidates);write_csv(OUT/"gelu_selection.csv",gelu_selected)
    pairs=[{"layer":r["layer"],"s10":r["s10"],"s_act":r["s_act"],"log2_s10":math.log2(r["s10"]),"log2_s_act":math.log2(r["s_act"]),
            "selection_policy":"ABSMAX" if r["retained_percentile"]==100 else "PERCENTILE_CLIPPING","percentile":r["retained_percentile"],"clipping_fraction":r["clipping_fraction"],"lut_hash":r["lut_sha256"]} for r in gelu_selected]
    write_csv(OUT/"gelu_scale_pairs.csv",pairs);base.write_json(OUT/"gelu_scale_pairs.json",{"clustering_performed":False,"pairs":pairs})
    progress("GELU_SEARCH_DONE");progress("LUT_WRITE_DONE")

    examples=e2e_examples(source);e2e_manifest={"examples":[identity(e) for e in examples],"sequences":8,"targets_per_sequence":24,"targets":192,"role":"evaluation only","final_confirmation_overlap":False}
    base.write_json(OUT/"e2e_dataset_manifest.json",e2e_manifest)
    run_id=digest({"params":[s["parameter_hash"] for s in specs.values()],"luts":gelu_hashes,"data":e2e_manifest})
    progress("E2E_START modes=3 sequences=8 targets=192")
    records=[]
    for number,example in enumerate(examples):
        path=WORK/f"e2e_{number:02d}.json";key=digest({"run":run_id,"example":identity(example)})
        cached=json.loads(path.read_text()) if path.exists() else None
        if cached is not None and cached.get("identity")!=key: raise RuntimeError("E2E checkpoint mismatch")
        if cached is None:
            value=evaluate_sequence(source,specs,gelu_tables,example);base.write_json(path,{"identity":key,"value":value})
        else:value=cached["value"]
        records.append(value);progress(f"SEQUENCE {number+1}/8 DONE")
    metrics={mode:response.merged([r[mode] for r in records]) for mode in MODES}
    if any(v["tokens"]!=192 for v in metrics.values()):raise RuntimeError("E2E target mismatch")
    def delta(a,b):return {k:{"absolute":b[k]-a[k],"relative":((b[k]-a[k])/a[k] if a[k] else None)} for k in ("kl","nmse","ppl","top1")}
    e2e_summary={"status":"MIXED_LINEAR_GELU_BASELINE","modes":metrics,"fp_to_mixed_linear":delta(metrics[MODES[0]],metrics[MODES[1]]),
                 "incremental_gelu_lut":delta(metrics[MODES[1]],metrics[MODES[2]]),"parameter_selection_used_e2e":False}
    base.write_json(OUT/"e2e_summary.json",e2e_summary);progress("E2E_DONE")

    parameter_hashes={"linear_parameters.json":sha(OUT/"linear_parameters.json"),"gelu_scale_pairs.json":sha(OUT/"gelu_scale_pairs.json")}
    verification={"branch":branch,"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"model_identity":model_manifest["gguf"],"GGUF_identity":historical_manifest["gguf"],
                  "total_linear_op_count":126,"ordinary_op_count":108,"gate_op_count":18,"ordinary_contract":{"output":"INT8","range":[-128,127]},
                  "gate_contract":{"output":"INT10","range":[-512,511]},"gelu_contract":{"input":"INT10","entries":1024,"output":"INT8","range":[-127,127]},
                  "row_selector_identity":"compare_k_selectors_e2e.select_absmax","quant_profile_identity":Profile().metadata(),"parameter_artifact_hashes":parameter_hashes,"all_lut_hashes":gelu_hashes,
                  "observed_effective_shift_min":min(r["effective_shift_min"] for r in summaries),"observed_effective_shift_max":max(r["effective_shift_max"] for r in summaries),
                  "infeasible_candidate_count":total_infeasible,"capture_reused":False,"clustering_performed":False,"fusion_tuning_performed":False,
                  "gpalu_product_calibration_performed":False,"e2e_used_for_parameter_selection":False,"final_confirmation_overlap":False,
                  "test_command":tests,"test_return_code":test_code,"elapsed_sec":time.monotonic()-START}
    base.write_json(OUT/"verification.json",verification)

    lines=["# Mixed Linear/GELU calibration baseline","","## 1. Scope","Consumer-aware calibration of 126 Linear operations followed by standalone per-layer GELUTanh LUT calibration. Status: `MIXED_LINEAR_GELU_BASELINE`; no production freeze was created.",
           "## 2. Consumer-aware Linear quantization contract","Ordinary GEMMs were calibrated for INT8 outputs. Gate GEMMs were calibrated for INT10 outputs because they feed the activation LUT. Inputs and weights are signed INT8; accumulation is INT32; per-channel M/S and runtime absmax row exponents are used.",
           "## 3. 126-op inventory","18 layers × seven projections: 108 ordinary INT8-output operations and 18 gate INT10-output operations.",
           "## 4. Ordinary GEMM INT8 calibration policy","Joint `(sX_base,s8)` output-domain search with signed hardware output clamp `[-128,127]`; historical scales were evaluated but output anchors were derived from captured FP outputs.",
           "## 5. Gate GEMM INT10 calibration policy","Joint `(sX_base,s10)` output-domain search with clamp `[-512,511]`. The old all-126 s10=0.1 values were not treated as authoritative.",
           "## 6. Runtime row-scale policy","`compare_k_selectors_e2e.select_absmax`; operation-specific feasible k bounds enforce `0 <= S-k <= 31`.",
           "## 7. Linear local calibration results",f"Selected 126 operations from bounded joint grids; {total_infeasible} candidates were infeasible. Mean balanced NMSE: {np.mean([r['balanced_nmse'] for r in summaries]):.9g}. Details are in the three summary CSVs.",
           "## 8. GELU INT10->INT8 LUT calibration policy","GELU LUT calibration used the newly calibrated per-op gate s10. Each table has 1024 signed INT8 entries and uses GELU tanh approximation; per-layer s_act selection compares ABSMAX with fixed tiny-tail percentiles.",
           "## 9. GELU error decomposition",f"Mean input/table/total NMSE: {np.mean([r['input_nmse'] for r in gelu_selected]):.9g} / {np.mean([r['table_nmse'] for r in gelu_selected]):.9g} / {np.mean([r['total_nmse'] for r in gelu_selected]):.9g}.",
           "## 10. E2E evaluation","| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Top5 overlap |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode,value in metrics.items(): lines.append("| "+mode+" | "+" | ".join(f"{value[k]:.9g}" for k in ("nll","ppl","kl","nmse","mse","mae","cosine","flattened_cosine","top1","in5","overlap"))+" |")
    lines += ["## 11. Historical comparison where valid","Historical down-only and prior LUT results are retained in their original diagnostics. They quantize different scopes, so no direct-equivalence claim is made.",
              "## 12. Limitations","No GPALU product scaling was calibrated. No GeGLU fusion-aware tuning was performed. No LUT clustering was performed. GeGLU multiplication during E2E evaluation was performed in floating point on reconstructed branch values solely to isolate pre-fusion numerical error. The calibration/E2E populations are small and are not a final unseen confirmation.",
              "## 13. Inputs for future GeGLU fusion-aware calibration","`linear_parameters.json`, gate INT10 scales, per-layer LUT tables, and `gelu_scale_pairs.*` define the pre-fusion baseline. Future work may calibrate INT8 GELU × INT8 Up product scaling without changing this experiment's records."]
    (OUT/"report.md").write_text("\n\n".join(lines).replace("|\n\n|","|\n|")+"\n")
    progress("TEST_DONE");progress("FINAL_VERIFY_DONE");progress("ARTIFACT_WRITE_DONE");progress("RUN_COMPLETE")


if __name__=="__main__":
    try:main()
    except BaseException:progress("RUN_FAILED");raise
