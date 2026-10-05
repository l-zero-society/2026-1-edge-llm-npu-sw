"""Small-batch fusion-aware Gate s10 pilot on the mixed Linear/GELU baseline."""
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
import run_mixed_linear_gelu_calibration as mixed
import run_gelu_standalone_calibration as gelu_base

base, cal, final, response = mixed.base, mixed.cal, mixed.final, mixed.response
Profile = mixed.Profile
OUT = ROOT / "diagnostics/fusion_aware_gate_s10_pilot"
WORK = OUT / "work"
LUT_DIR = OUT / "gelu_tables"
QB_DIR = OUT / "gate_qparams"
BASELINE = ROOT / "diagnostics/mixed_linear_gelu_calibration"
PROGRESS = ROOT / "fusion_aware_gate_s10_progress.txt"
BASELINE_COMMIT = "2d77cf52907d8b1f910e26a1952c69248090190f"
J_VALUES = tuple(range(-4, 5))
GROUPS = ("prefill", "decode")
MODES = ("FP_FULL", "CURRENT_MIXED_BASELINE", "FUSION_AWARE_GATE_S10")
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as f:
        f.write(message + "\n"); f.flush()


def sha(path):
    return base.sha256_file(Path(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_csv(path, rows):
    base.write_csv(path, rows)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def s10_grid(incumbent):
    result = [(j, incumbent * 2.0 ** (j / 8.0)) for j in J_VALUES]
    if len(result) != 9 or result[4] != (0, incumbent):
        raise RuntimeError("invalid s10 grid")
    return result


def balanced(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values))


def group_nmse(value, reference, labels):
    result = {}
    for group in GROUPS:
        mask = labels == group
        result[group] = mixed.nmse(value[mask], reference[mask]) if mask.any() else None
    return result


def inner_sact(g_fp, q10, labels, s10):
    gq = q10.astype(np.float64) * s10
    a_fp, a_qref = gelu_base.gelu(g_fp), gelu_base.gelu(gq)
    candidates = []
    for percentile in mixed.GELU_PERCENTILES:
        s_act = float(np.percentile(np.abs(a_qref), percentile)) / 127.0
        lut = gelu_base.make_lut(s10, s_act)
        a_hat = lut[gelu_base.address(q10)].astype(np.float64) * s_act
        total, table = group_nmse(a_hat, a_fp, labels), group_nmse(a_hat, a_qref, labels)
        candidate = dict(s_act=s_act, retained_percentile=percentile,
                         clipping_fraction=(100.0-percentile)/100.0,
                         total_nmse_prefill=total["prefill"], total_nmse_decode=total["decode"],
                         total_nmse_balanced=balanced(total.values()),
                         table_nmse_balanced=balanced(table.values()),
                         worst_total_nmse=max(v for v in total.values() if v is not None),
                         saturation_rate=float(np.mean(np.abs(np.rint(a_qref/s_act)) > 127)),
                         lut=lut, a_hat=a_hat)
        candidates.append(candidate)
    # Standalone objective only. No fusion value enters this key.
    selected = min(candidates, key=lambda c:(c["total_nmse_balanced"],c["table_nmse_balanced"],
                                             c["worst_total_nmse"],c["clipping_fraction"],
                                             c["saturation_rate"],c["s_act"]))
    return selected, candidates


def outer_key(candidate, incumbent):
    return (candidate["fusion_full_nmse_balanced"], candidate["fusion_full_worst_group"],
            candidate["fusion_gate_only_nmse_balanced"], candidate["gelu_nmse_balanced"],
            candidate["gate_nmse_balanced"], candidate["gate_output_saturation_rate"],
            abs(math.log2(candidate["s10_candidate"] / incumbent)), candidate["s10_candidate"])


def select_outer(candidates, incumbent):
    if not any(c["candidate_j"] == 0 and c["s10_candidate"] == incumbent for c in candidates):
        raise RuntimeError("incumbent missing")
    selected = min(candidates, key=lambda c: outer_key(c, incumbent))
    old = next(c for c in candidates if c["candidate_j"] == 0)
    if selected["fusion_full_nmse_balanced"] > old["fusion_full_nmse_balanced"] + 1e-12:
        raise RuntimeError("fusion-aware selection regressed against incumbent")
    return selected, old


def load_baseline_specs(source, historical):
    artifact = json.loads((BASELINE / "linear_parameters.json").read_text())["layers"]
    expected = set(mixed.expected_inventory())
    if set(artifact) != expected:
        raise RuntimeError("baseline Linear inventory mismatch")
    specs = {}
    for name, item in artifact.items():
        old = historical[name]
        params = dict(multiplier=np.asarray(item["multiplier"], np.int64),
                      shift=np.asarray(item["shift"], np.int64))
        packed = Profile().pack(params["multiplier"], params["shift"])
        np.testing.assert_array_equal(packed, np.asarray(item["packed_qb"], np.uint32))
        specs[name] = dict(contract=item["contract"], sx=float(item["sx_base"]),
                           sout=float(item["output_scale"]), params=params,
                           wq=old["wq"], sw=old["sw"], weight=source.weight(name),
                           wq_path=old["wq_path"], wq_sha256=old["wq_sha256"],
                           parameter_hash=item["parameter_hash"])
    return specs


def load_baseline_luts(specs):
    rows = {}
    with (BASELINE / "gelu_selection.csv").open() as f:
        import csv
        for row in csv.DictReader(f): rows[int(row["layer"])] = row
    if set(rows) != set(range(18)):
        raise RuntimeError("baseline GELU inventory mismatch")
    tables = {}
    for layer in range(18):
        path = BASELINE / "gelu_tables" / f"layer_{layer:02d}_gelu.bin"
        lut = np.frombuffer(path.read_bytes(), dtype=np.int8).copy()
        if len(lut) != 1024: raise RuntimeError("baseline LUT size")
        tables[layer] = dict(s10=specs[f"model.layers.{layer}.mlp.gate_proj"]["sout"],
                             s_act=float(rows[layer]["s_act"]),lut=lut,sha256=sha(path),
                             percentile=float(rows[layer]["retained_percentile"]))
    return tables


def evaluate_layer(layer, x, labels, gate_spec, up_spec, incumbent_lut):
    g_fp = x.astype(np.float64) @ gate_spec["weight"].astype(np.float64).T
    u_fp = x.astype(np.float64) @ up_spec["weight"].astype(np.float64).T
    u_hat, up_details = mixed.quantized_linear_rows(x, up_spec, return_codes=True)
    a_fp = gelu_base.gelu(g_fp); h_fp = a_fp * u_fp
    up_nmse = group_nmse(u_hat, u_fp, labels)
    candidates = []
    for j, s10 in s10_grid(gate_spec["sout"]):
        params = Profile().approximate(gate_spec["sx"] * gate_spec["sw"] / s10)
        if np.any(params["status"] != "ok"): raise RuntimeError(f"infeasible profile layer={layer} j={j}")
        candidate_spec = dict(gate_spec, sout=s10, params=params)
        g_hat, gate_details = mixed.quantized_linear_rows(x, candidate_spec, return_codes=True)
        q10 = gate_details["output_codes"]
        inner, _ = inner_sact(g_fp, q10, labels, s10)
        a_hat = inner["a_hat"]
        gate = group_nmse(g_hat, g_fp, labels)
        gelu = group_nmse(a_hat, a_fp, labels)
        gate_only = group_nmse(a_hat*u_fp, h_fp, labels)
        full = group_nmse(a_hat*u_hat, h_fp, labels)
        lo, hi = mixed.effective_bounds(params["shift"])
        c = dict(layer=layer,candidate_j=j,sx_base_fixed=gate_spec["sx"],
                 s10_incumbent=gate_spec["sout"],s10_candidate=s10,
                 selected_s_act=inner["s_act"],selected_percentile=inner["retained_percentile"],
                 clipping_fraction=inner["clipping_fraction"],lut_saturation_rate=inner["saturation_rate"],
                 gate_nmse_prefill=gate["prefill"],gate_nmse_decode=gate["decode"],gate_nmse_balanced=balanced(gate.values()),
                 gelu_nmse_prefill=gelu["prefill"],gelu_nmse_decode=gelu["decode"],gelu_nmse_balanced=balanced(gelu.values()),
                 standalone_gelu_total_nmse=inner["total_nmse_balanced"],
                 up_nmse_prefill=up_nmse["prefill"],up_nmse_decode=up_nmse["decode"],up_nmse_balanced=balanced(up_nmse.values()),
                 fusion_gate_only_nmse_prefill=gate_only["prefill"],fusion_gate_only_nmse_decode=gate_only["decode"],fusion_gate_only_nmse_balanced=balanced(gate_only.values()),
                 fusion_full_nmse_prefill=full["prefill"],fusion_full_nmse_decode=full["decode"],fusion_full_nmse_balanced=balanced(full.values()),
                 fusion_full_worst_group=max(v for v in full.values() if v is not None),
                 gate_input_clip_rate=gate_details["input_clip_rate"],gate_output_saturation_rate=gate_details["output_saturation_rate"],
                 feasible_k_min=lo,feasible_k_max=hi,observed_k_min=int(gate_details["k"].min()),observed_k_max=int(gate_details["k"].max()),
                 effective_shift_min=gate_details["effective_shift_min"],effective_shift_max=gate_details["effective_shift_max"],
                 boundary_candidate=j in (-4,4),params=params,lut=inner["lut"])
        candidates.append(c)
    selected, incumbent = select_outer(candidates, gate_spec["sout"])
    # Verify the inner policy regenerates the established incumbent table/scale.
    if not np.isclose(incumbent["selected_s_act"],incumbent_lut["s_act"],rtol=0,atol=1e-15):
        raise RuntimeError(f"incumbent standalone s_act mismatch at layer {layer}")
    np.testing.assert_array_equal(incumbent["lut"],incumbent_lut["lut"])
    return selected, incumbent, candidates


def compact_candidate(candidate):
    return {k:v for k,v in candidate.items() if k not in ("params","lut")}


def classify_layer(old, new):
    change=(old["fusion_full_nmse_balanced"]-new["fusion_full_nmse_balanced"])/old["fusion_full_nmse_balanced"]
    return "BENEFICIAL" if change>=.01 else ("REGRESSION" if change < -1e-12 else "NEUTRAL")


def evaluate_sequence(source, specs, fusion_specs, fusion_luts, example, stored):
    fp_score, fusion_score = response.Scores(), response.Scores()
    fp_cache=fusion_cache=None
    for t,target in enumerate(example["targets"][:24]):
        ids=example["prompt_ids"] if t==0 else [example["targets"][t-1]]
        fp_logits,_,fp_cache=cal.run_fp(source,ids,fp_cache,layers=False)
        fusion_logits,_,fusion_cache=mixed.run_quant(source,fusion_specs,ids,fusion_cache,fusion_luts)
        if fp_cache is fusion_cache:raise RuntimeError("KV cache alias")
        fp_score.add(source.torch,fp_logits,fp_logits,[target])
        fusion_score.add(source.torch,fusion_logits,fp_logits,[target])
    fresh_fp=fp_score.raw()
    prior_fp=stored["FP_FULL"]
    for key in ("nll","n"):
        if not np.isclose(fresh_fp[key],prior_fp[key],rtol=0,atol=1e-9):raise RuntimeError("stored FP mismatch")
    return {"FP_FULL":fresh_fp,"CURRENT_MIXED_BASELINE":stored["MIXED_LINEAR_GELU_LUT"],
            "FUSION_AWARE_GATE_S10":fusion_score.raw()}


def e2e_classification(old,new):
    better=sum((new[k]<old[k]) for k in ("kl","nmse","ppl")) + (new["top1"]>old["top1"])
    worse=sum((new[k]>old[k]) for k in ("kl","nmse","ppl")) + (new["top1"]<old["top1"])
    if better>=3 and worse<=1:return "IMPROVEMENT"
    if worse>=3 and better<=1:return "REGRESSION"
    if all(abs(new[k]-old[k])<=1e-6 for k in ("kl","nmse","ppl","top1")):return "NO_MEANINGFUL_CHANGE"
    return "TRADEOFF"


def main():
    PROGRESS.write_text("");progress("RUN_START")
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);LUT_DIR.mkdir(parents=True,exist_ok=True);QB_DIR.mkdir(parents=True,exist_ok=True)
    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    if branch!="lut_cali":raise RuntimeError("wrong branch")
    if subprocess.run(["git","merge-base","--is-ancestor",BASELINE_COMMIT,"HEAD"]).returncode:raise RuntimeError("baseline commit absent")
    progress("GIT_CHECK_DONE")
    baseline_paths=[BASELINE/"linear_parameters.json",BASELINE/"gelu_selection.csv",BASELINE/"capture_manifest.json",BASELINE/"e2e_dataset_manifest.json"]
    baseline_hashes={str(p.relative_to(ROOT)):sha(p) for p in baseline_paths}
    baseline_manifest={"baseline_commit":BASELINE_COMMIT,"artifact_hashes":baseline_hashes,"status":"READ_ONLY_BASELINE"}
    base.write_json(OUT/"baseline_manifest.json",baseline_manifest);progress("BASELINE_VERIFY_DONE")

    progress("UNIT_TEST_START")
    tests=[sys.executable,"-m","unittest","tests.test_fusion_aware_gate_s10_pilot","tests.test_mixed_linear_gelu_calibration"]
    with (OUT/"unit_test.log").open("w") as log:
        code=subprocess.run(tests,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPYCACHEPREFIX="/tmp/fusion-aware-gate",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",KMP_USE_SHM="0")).returncode
    if code:progress("UNIT_TEST_FAILED");raise RuntimeError("tests failed")
    progress("UNIT_TEST_DONE")

    source,model_manifest,_,_,_,_=final.load_source_and_policies()
    historical_manifest=json.loads((mixed.ART/"manifest.json").read_text());historical=mixed.load_historical(source,historical_manifest)
    baseline_specs=load_baseline_specs(source,historical);baseline_luts=load_baseline_luts(baseline_specs)
    if len([n for n in baseline_specs if n.endswith("gate_proj")])!=18 or len([n for n in baseline_specs if n.endswith("up_proj")])!=18:raise RuntimeError("Gate/Up inventory")
    capture_manifest=json.loads((BASELINE/"capture_manifest.json").read_text());capture_npz=BASELINE/"work/capture.npz";capture_meta=BASELINE/"work/capture.json"
    progress("CAPTURE_REUSE_CHECK")
    if not capture_npz.exists() or not capture_meta.exists():raise RuntimeError("baseline compatible capture unavailable")
    meta=json.loads(capture_meta.read_text())
    if sha(capture_npz)!=meta["sha256"]:raise RuntimeError("capture checksum mismatch")
    with np.load(capture_npz) as z:captured={k:z[k] for k in z.files}
    progress("CAPTURE_REUSED")

    inventory=mixed.expected_inventory();selected_specs=dict(baseline_specs);selected_luts=dict(baseline_luts)
    candidate_rows=[];selection_rows=[];qparam_hashes=[];lut_hashes=[]
    progress("FUSION_SEARCH_START")
    for layer in range(18):
        gate_name=f"model.layers.{layer}.mlp.gate_proj";up_name=f"model.layers.{layer}.mlp.up_proj"
        gi=inventory.index(gate_name)
        # gate_proj and up_proj consume the same MLP input tensor. Their
        # independently seeded reservoirs may retain different positions, so
        # use the gate reservoir as the single paired X for both branches.
        x=np.concatenate([captured[f"op{gi:04d}_{g}"] for g in GROUPS]);labels=np.concatenate([[g]*len(captured[f"op{gi:04d}_{g}"]) for g in GROUPS])
        selected,old,candidates=evaluate_layer(layer,x,labels,baseline_specs[gate_name],baseline_specs[up_name],baseline_luts[layer])
        candidate_rows.extend(compact_candidate(c) for c in candidates)
        params=selected["params"];selected_specs[gate_name]=dict(baseline_specs[gate_name],sout=selected["s10_candidate"],params=params)
        selected_luts[layer]={"s10":selected["s10_candidate"],"s_act":selected["selected_s_act"],"lut":selected["lut"]}
        relative=(old["fusion_full_nmse_balanced"]-selected["fusion_full_nmse_balanced"])/old["fusion_full_nmse_balanced"]
        row={"layer":layer,"s10_old":old["s10_candidate"],"s10_new":selected["s10_candidate"],"log2_ratio":math.log2(selected["s10_candidate"]/old["s10_candidate"]),
             "s_act_old":old["selected_s_act"],"s_act_new":selected["selected_s_act"],"gate_nmse_old":old["gate_nmse_balanced"],"gate_nmse_new":selected["gate_nmse_balanced"],
             "gelu_nmse_old":old["gelu_nmse_balanced"],"gelu_nmse_new":selected["gelu_nmse_balanced"],"fusion_gate_only_nmse_old":old["fusion_gate_only_nmse_balanced"],
             "fusion_gate_only_nmse_new":selected["fusion_gate_only_nmse_balanced"],"fusion_full_nmse_old":old["fusion_full_nmse_balanced"],
             "fusion_full_nmse_new":selected["fusion_full_nmse_balanced"],"relative_fusion_full_improvement":relative,"selected_j":selected["candidate_j"],
             "boundary_win":selected["candidate_j"] in (-4,4),"classification":classify_layer(old,selected)}
        selection_rows.append(row)
        packed=Profile().pack(params["multiplier"],params["shift"]).astype("<u4")
        qb=QB_DIR/f"layer_{layer:02d}_gate_qb.bin";qb.write_bytes(packed.tobytes());qhash=sha(qb);qparam_hashes.append(qhash)
        lut_path=LUT_DIR/f"layer_{layer:02d}_gelu.bin";lut_path.write_bytes(selected["lut"].tobytes());lhash=sha(lut_path);lut_hashes.append(lhash)
        base.write_json(LUT_DIR/f"layer_{layer:02d}_gelu.json",{"layer":layer,"s10_incumbent":old["s10_candidate"],"s10_selected":selected["s10_candidate"],
                        "selected_j":selected["candidate_j"],"s_act_selected":selected["selected_s_act"],"retained_percentile":selected["selected_percentile"],
                        "clipping_fraction":selected["clipping_fraction"],"saturation_rate":selected["lut_saturation_rate"],"LUT_SHA256":lhash})
        progress(f"LAYER {layer+1}/18 DONE")
    write_csv(OUT/"candidate_summary.csv",candidate_rows);write_csv(OUT/"layer_selection.csv",selection_rows)
    gate_meta=[]
    for layer,row_sel in enumerate(selection_rows):
        name=f"model.layers.{layer}.mlp.gate_proj";spec=selected_specs[name];params=spec["params"];lo,hi=mixed.effective_bounds(params["shift"])
        gate_meta.append({"layer":layer,"module":name,"sx_base_fixed":spec["sx"],"s10_incumbent":row_sel["s10_old"],"s10_selected":spec["sout"],"selected_j":row_sel["selected_j"],
                          "feasible_k_min":lo,"feasible_k_max":hi,"M_min":int(params["multiplier"].min()),"M_max":int(params["multiplier"].max()),
                          "S_min":int(params["shift"].min()),"S_max":int(params["shift"].max()),"qparam_binary_path":str((QB_DIR/f"layer_{layer:02d}_gate_qb.bin").relative_to(ROOT)),
                          "qparam_SHA256":qparam_hashes[layer],"LUT_path":str((LUT_DIR/f"layer_{layer:02d}_gelu.bin").relative_to(ROOT)),"LUT_SHA256":lut_hashes[layer],
                          "s_act":selected_luts[layer]["s_act"],"percentile":next(c["selected_percentile"] for c in candidate_rows if c["layer"]==layer and c["candidate_j"]==row_sel["selected_j"])})
    base.write_json(OUT/"gate_parameters.json",{"status":"SMALL_BATCH_FUSION_AWARE_PILOT","layers":gate_meta})
    progress("FUSION_SEARCH_DONE");progress("QPARAM_EXPORT_DONE");progress("LUT_EXPORT_DONE")

    e2e_manifest=json.loads((BASELINE/"e2e_dataset_manifest.json").read_text());base.write_json(OUT/"e2e_dataset_manifest.json",e2e_manifest)
    examples=mixed.e2e_examples(source);run_id=digest({"baseline":baseline_hashes,"selection":selection_rows,"data":e2e_manifest})
    progress("E2E_START sequences=8 targets=192");records=[]
    for number,example in enumerate(examples):
        prior_path=BASELINE/"work"/f"e2e_{number:02d}.json";prior=json.loads(prior_path.read_text())["value"]
        path=WORK/f"e2e_{number:02d}.json";key=digest({"run":run_id,"example":mixed.identity(example)})
        cached=json.loads(path.read_text()) if path.exists() else None
        if cached and cached.get("identity")!=key:raise RuntimeError("checkpoint mismatch")
        if cached:value=cached["value"]
        else:
            value=evaluate_sequence(source,baseline_specs,selected_specs,selected_luts,example,prior);base.write_json(path,{"identity":key,"value":value})
        records.append(value);progress(f"SEQUENCE {number+1}/8 DONE")
    metrics={mode:response.merged([r[mode] for r in records]) for mode in MODES}
    old,new=metrics[MODES[1]],metrics[MODES[2]]
    deltas={k:{"absolute":new[k]-old[k],"relative":((new[k]-old[k])/old[k] if old[k] else None)} for k in ("kl","nmse","ppl")}
    deltas["top1"]={"absolute_pp":100*(new["top1"]-old["top1"])}
    classification=e2e_classification(old,new)
    e2e={"modes":metrics,"baseline_to_fusion_aware":deltas,"classification":classification,"e2e_used_for_selection":False}
    base.write_json(OUT/"e2e_summary.json",e2e);progress("E2E_DONE")

    changed=sum(r["selected_j"]!=0 for r in selection_rows);boundaries=sum(r["boundary_win"] for r in selection_rows)
    before_after={str(p.relative_to(ROOT)):sha(p) for p in baseline_paths}
    if before_after!=baseline_hashes:raise RuntimeError("baseline artifact mutated")
    verification={"branch":branch,"source_HEAD":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"baseline_commit_present":True,
                  "model_SHA256":historical_manifest["gguf"]["sha256"],"GGUF_SHA256":historical_manifest["gguf"]["sha256"],"baseline_artifact_hashes":baseline_hashes,
                  "target_Gate_op_count":18,"fixed_Up_op_count":18,"Gate_sx_changed":False,"ordinary_params_changed":False,"Up_params_changed":False,
                  "Gate_s10_search_performed":True,"candidate_count_per_layer":9,"s_act_inner_objective":"standalone_GELU_total_NMSE",
                  "s10_outer_objective":"GeGLU_product_NMSE","e2e_used_for_selection":False,"GPALU_product_calibration_performed":False,
                  "large_batch_performed":False,"clustering_performed":False,"number_layers_changed":changed,"number_boundary_winners":boundaries,
                  "all_qparam_hashes":qparam_hashes,"all_LUT_hashes":lut_hashes,"observed_effective_shift_global_min":min(c["effective_shift_min"] for c in candidate_rows),
                  "observed_effective_shift_global_max":max(c["effective_shift_max"] for c in candidate_rows),"test_command":tests,"test_return_code":code,"elapsed_sec":time.monotonic()-START}
    base.write_json(OUT/"verification.json",verification)

    def avg(field,which):return float(np.mean([r[f"{field}_{which}"] for r in selection_rows]))
    beneficial=sum(r["classification"]=="BENEFICIAL" for r in selection_rows)
    recommendation="PROCEED_TO_LARGE_BATCH" if beneficial>=4 and classification!="REGRESSION" else ("NO_CLEAR_FUSION_AWARE_BENEFIT" if beneficial==0 else "INVESTIGATE_BEFORE_LARGE_BATCH")
    lines=["# Fusion-aware Gate s10 small-batch pilot","","## 1. Scope","Only Gate s10 was searched. This is a small-batch policy pilot, not a final calibration.",
           "## 2. Frozen baseline","All ordinary GEMM parameters, Up parameters, Gate sX_base, weights, and the absmax row selector were unchanged.",
           "## 3. Nested calibration policy","Outer s10 selection minimized balanced GeGLU product NMSE. For every fixed s10, inner s_act selection independently minimized standalone GELU total NMSE. Each candidate s10 received newly generated M/S and a newly calibrated standalone GELU LUT.",
           "## 4. Search grid","Nine candidates per layer: incumbent s10 × 2^(j/8), j=-4..4. No expansion was performed.",
           "## 5. Per-layer results","| Layer | j | s10 old | s10 new | Fusion NMSE old | Fusion NMSE new | Improvement | Class | Boundary |","|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in selection_rows:lines.append(f"| {r['layer']} | {r['selected_j']} | {r['s10_old']:.9g} | {r['s10_new']:.9g} | {r['fusion_full_nmse_old']:.9g} | {r['fusion_full_nmse_new']:.9g} | {100*r['relative_fusion_full_improvement']:.4f}% | {r['classification']} | {r['boundary_win']} |")
    lines += ["## 6. Aggregate local results",f"Changed layers: {changed}/18; beneficial: {beneficial}; boundary winners: {boundaries}. Mean Gate NMSE old/new: {avg('gate_nmse','old'):.9g}/{avg('gate_nmse','new'):.9g}. Mean GELU NMSE old/new: {avg('gelu_nmse','old'):.9g}/{avg('gelu_nmse','new'):.9g}. Mean fusion gate-only old/new: {avg('fusion_gate_only_nmse','old'):.9g}/{avg('fusion_gate_only_nmse','new'):.9g}. Mean full fusion old/new: {avg('fusion_full_nmse','old'):.9g}/{avg('fusion_full_nmse','new'):.9g}.",
              "## 7. Gate-local vs fusion-product tradeoff","Gate-local NMSE was diagnostic and could worsen when full GeGLU product NMSE improved; the outer selection did not reject that intended tradeoff.",
              "## 8. Small E2E comparison","| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode,value in metrics.items():lines.append("| "+mode+" | "+" | ".join(f"{value[k]:.9g}" for k in ("nll","ppl","kl","nmse","mse","mae","cosine","flattened_cosine","top1","in5","overlap"))+" |")
    lines += [f"E2E classification: **{classification}**. E2E was NOT used for parameter selection.","## 9. Boundary winners",f"{boundaries} layers selected j=±4; no outward search was run.",
              "## 10. Decision for large-batch follow-up",f"**{recommendation}**","## 11. Limitations","GPALU product requantization was NOT modeled. GeGLU multiplication used floating-point multiplication of reconstructed branch values. Ordinary GEMM and Up parameters were unchanged; Gate sX_base was unchanged. The calibration and validation populations are deliberately small."]
    (OUT/"report.md").write_text("\n\n".join(lines).replace("|\n\n|","|\n|")+"\n")
    progress("TEST_DONE");progress("VERIFY_DONE");progress("ARTIFACT_WRITE_DONE");progress("RUN_COMPLETE")


if __name__=="__main__":
    try:main()
    except BaseException:progress("RUN_FAILED");raise
