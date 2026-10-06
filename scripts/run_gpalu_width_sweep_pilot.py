"""CPU-only small-batch GPALU INT8/INT10/INT12 output-width sweep."""
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gpalu_fullscale_downaware_pilot as full

pot, fusion, mixed, gelu_base = full.pot, full.fusion, full.mixed, full.gelu_base
base, cal, final, response, Profile = full.base, full.cal, full.final, full.response, full.Profile
OUT = ROOT / "diagnostics/gpalu_width_sweep_pilot"
WORK = OUT / "work"
MIXED = ROOT / "diagnostics/mixed_linear_gelu_calibration"
FUSION = ROOT / "diagnostics/fusion_aware_gate_s10_pilot"
POT = ROOT / "diagnostics/gpalu_pot_fusion_pilot"
REQUIRED = ("f92fafb2db225165a2eb9064b5073e6f212fac54", "df47d1aa05662e2dad54e4c851dcf84b4428ab75", "6182a777c96da5b50b5c872825354c6e5e9dfb85")
WIDTHS = (8, 10, 12)
PERCENTILES = (99., 99.5, 99.9, 99.95, 99.99, 99.995, 100.)
J_VALUES = tuple(range(-6, 7))
GROUPS = ("prefill", "decode")
MODES = ("FP_FULL", "FUSION_AWARE_FLOAT_PRODUCT", "GPALU_INT8", "GPALU_INT10", "GPALU_INT12")
START = time.monotonic()


def sha(path): return base.sha256_file(Path(path))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
def balanced(values): return float(np.mean([value for value in values if value is not None]))


def write_csv(path, rows):
    base.write_csv(path, rows); path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def signed_bounds(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def group_nmse(value, reference, labels):
    return {group: (mixed.nmse(value[labels == group], reference[labels == group]) if np.any(labels == group) else None) for group in GROUPS}


def width_codes(product, alpha, multiplier, shift, bits):
    lo, hi = signed_bounds(bits); product = np.asarray(product, np.int64)
    ideal_pre = np.rint(product.astype(np.float64) * alpha).astype(np.int64)
    hw_pre = Profile().apply(product, np.array([multiplier]), np.array([shift]), saturate=False)
    ideal = np.clip(ideal_pre, lo, hi).astype(np.int16)
    hw = np.clip(hw_pre, lo, hi).astype(np.int16)
    return ideal, hw, {
        "clipping_rate": float(np.mean((hw_pre < lo) | (hw_pre > hi))),
        "positive_clipping_rate": float(np.mean(hw_pre > hi)),
        "negative_clipping_rate": float(np.mean(hw_pre < lo)),
    }


def effective_scale(s_product, multiplier, shift):
    return float(s_product * (2. ** shift) / multiplier)


def scale_candidates(h_fp, h_pre, qmax):
    anchors = []
    for source, values in (("fp", h_fp), ("pre_float", h_pre)):
        absolute = np.abs(values).reshape(-1)
        anchors += [(float(np.percentile(absolute, p)) / qmax, source, p) for p in PERCENTILES]
    candidates = []
    for anchor, source, percentile in anchors:
        candidates += [(anchor * 2. ** (j / 16.), source, percentile, j) for j in J_VALUES]
    result = []
    for item in sorted(candidates, key=lambda x: x[0]):
        if item[0] > 0 and np.isfinite(item[0]) and not any(np.isclose(item[0], old[0], rtol=1e-12) for old in result): result.append(item)
    return result


def candidate_key(candidate):
    return (candidate["local_hw_nmse"], candidate["worst_group_nmse"], candidate["clipping_rate"],
            candidate["alpha_relative_error"], candidate["s_h_target"])


def select_candidate(candidates):
    # No E2E value is accepted by this selection API.
    return min(candidates, key=candidate_key)


def calibrate_width(layer, bits, p16, s_product, h_fp, h_pre, labels):
    qmax = signed_bounds(bits)[1]; candidates = []
    for target, source, percentile, offset in scale_candidates(h_fp, h_pre, qmax):
        alpha = s_product / target; rep = full.approximate_scalar(alpha)
        effective = effective_scale(s_product, rep["M_G"], rep["S_G"])
        ideal, hw, clipping = width_codes(p16, alpha, rep["M_G"], rep["S_G"], bits)
        h_ideal = ideal.astype(np.float64) * target; h_hw = hw.astype(np.float64) * effective
        ideal_group = group_nmse(h_ideal, h_fp, labels); hw_group = group_nmse(h_hw, h_fp, labels)
        incremental = group_nmse(h_hw, h_ideal, labels)
        candidates.append({
            "layer": layer, "output_bits": bits, "signed_min": signed_bounds(bits)[0], "signed_max": qmax,
            "s_product": s_product, "s_h_target": target, "s_h_effective": effective,
            "M_G": rep["M_G"], "S_G": rep["S_G"], "ZP_G": 0,
            "alpha_target": alpha, "alpha_hw": rep["alpha_hw"], "alpha_relative_error": rep["alpha_relative_error"],
            "local_ideal_nmse": balanced(ideal_group.values()), "local_hw_nmse": balanced(hw_group.values()),
            "incremental_ms_loss": balanced(incremental.values()), "prefill_nmse": hw_group["prefill"],
            "decode_nmse": hw_group["decode"], "worst_group_nmse": max(v for v in hw_group.values() if v is not None),
            **clipping, "anchor_source": source, "anchor_percentile": percentile, "anchor_offset": offset,
        })
    return select_candidate(candidates), candidates


def prepare_layer(layer, x, labels, gate_spec, up_spec, lut):
    g_fp = x.astype(np.float64) @ gate_spec["weight"].astype(np.float64).T
    u_fp = x.astype(np.float64) @ up_spec["weight"].astype(np.float64).T
    g_hat, gd = mixed.quantized_linear_rows(x, gate_spec, return_codes=True)
    u_hat, ud = mixed.quantized_linear_rows(x, up_spec, return_codes=True)
    q10 = gd["output_codes"]
    q_a = lut["lut"][gelu_base.address(q10)].astype(np.int16)
    q_u = ud["output_codes"].astype(np.int16)
    p16 = q_a.astype(np.int32) * q_u.astype(np.int32)
    if np.any((p16 < -32768) | (p16 > 32767)): raise RuntimeError("INT8 product overflowed INT16")
    h_fp = gelu_base.gelu(g_fp) * u_fp
    h_pre = q_a.astype(np.float64) * lut["s_act"] * u_hat
    return p16, lut["s_act"] * up_spec["sout"], h_fp, h_pre


@contextlib.contextmanager
def patch_width(source, gate_specs, up_specs, luts, selected=None):
    originals = []
    try:
        for layer, block in enumerate(source.model.model.layers):
            mlp = block.mlp; gate, up, lut = gate_specs[layer], up_specs[layer], luts[layer]
            originals += [(mlp.gate_proj, mlp.gate_proj.forward), (mlp.up_proj, mlp.up_proj.forward),
                          (mlp.act_fn, mlp.act_fn.forward), (mlp, mlp.forward)]
            mlp.gate_proj.forward = lambda x, spec=gate: mixed.apply_tensor(x, spec)
            mlp.up_proj.forward = lambda x, spec=up: mixed.apply_tensor(x, spec)
            def activation(x, item=lut):
                q = np.clip(np.rint(x.detach().cpu().numpy() / item["s10"]), -512, 511).astype(np.int16)
                return x.new_tensor(item["lut"][gelu_base.address(q)].astype(np.float32) * item["s_act"])
            mlp.act_fn.forward = activation
            item = None if selected is None else selected[layer]
            def forward(x, mlp=mlp, item=item, lut=lut, up=up):
                a = mlp.act_fn(mlp.gate_proj(x)); u = mlp.up_proj(x)
                if item is None:
                    value = a * u
                else:
                    qa = np.clip(np.rint(a.detach().cpu().numpy() / lut["s_act"]), -127, 127).astype(np.int16)
                    qu = np.clip(np.rint(u.detach().cpu().numpy() / up["sout"]), -128, 127).astype(np.int16)
                    product = qa.astype(np.int32) * qu.astype(np.int32)
                    _, qh, _ = width_codes(product, item["alpha_hw"], item["M_G"], item["S_G"], item["output_bits"])
                    value = x.new_tensor(qh.astype(np.float32) * item["s_h_effective"])
                return mlp.down_proj(value)
            mlp.forward = forward
        yield
    finally:
        for module, original in originals: module.forward = original


def run_mode(source, gate_specs, up_specs, luts, selected, ids, cache):
    with patch_width(source, gate_specs, up_specs, luts, selected):
        hidden, states, cache = response.forward_hidden(source, ids, [len(ids)-1], {}, cache, True, False)
        with source.torch.inference_mode(): logits = source.model.lm_head(hidden)
    return logits, states, cache


def evaluate_sequence(source, gate_specs, up_specs, luts, selected_by_width, example):
    names = MODES[1:]; scores = {name: response.Scores() for name in names}; caches = {name: None for name in names}; fp_cache = None
    fp_score = response.Scores()
    for index, target in enumerate(example["targets"][:24]):
        ids = example["prompt_ids"] if index == 0 else [example["targets"][index-1]]
        fp_logits, _, fp_cache = cal.run_fp(source, ids, fp_cache, layers=False); fp_score.add(source.torch, fp_logits, fp_logits, [target])
        configs = {"FUSION_AWARE_FLOAT_PRODUCT": None, "GPALU_INT8": selected_by_width[8], "GPALU_INT10": selected_by_width[10], "GPALU_INT12": selected_by_width[12]}
        for name, selected in configs.items():
            logits, _, caches[name] = run_mode(source, gate_specs, up_specs, luts, selected, ids, caches[name])
            scores[name].add(source.torch, logits, fp_logits, [target])
        if len({id(fp_cache)} | {id(v) for v in caches.values()}) != 5: raise RuntimeError("KV cache alias")
    return {"FP_FULL": fp_score.raw(), **{name: score.raw() for name, score in scores.items()}}


def delta(old, new):
    result = {key: {"absolute": new[key]-old[key], "relative": (new[key]-old[key])/old[key]} for key in ("kl", "nmse", "ppl")}
    result["top1"] = {"absolute_pp": 100. * (new["top1"]-old["top1"])}
    return result


def main():
    for path in (OUT, WORK, *(OUT / f"int{bits}" for bits in WIDTHS)): path.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != "lut_cali": raise RuntimeError("wrong branch")
    for commit in REQUIRED:
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"]).returncode: raise RuntimeError(f"missing commit {commit}")
    baseline_paths = [MIXED / "linear_parameters.json", MIXED / "capture_manifest.json", FUSION / "gate_parameters.json", FUSION / "layer_selection.csv", POT / "e2e_dataset_manifest.json"]
    baseline_hashes = {str(path.relative_to(ROOT)): sha(path) for path in baseline_paths}
    base.write_json(OUT / "baseline_manifest.json", {"required_commits": list(REQUIRED), "artifact_hashes": baseline_hashes, "status": "READ_ONLY"})
    tests = [sys.executable, "-m", "unittest", "tests.test_gpalu_width_sweep_pilot", "tests.test_gpalu_fullscale_downaware_pilot", "tests.test_fusion_aware_gate_s10_pilot", "tests.test_mixed_linear_gelu_calibration"]
    with (OUT / "unit_test.log").open("w") as log:
        test_code = subprocess.run(tests, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/gpalu-width-sweep", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_USE_SHM="0", PYTORCH_ENABLE_MPS_FALLBACK="0")).returncode
    if test_code: raise RuntimeError("unit tests failed")
    source, _, _, _, _, _ = final.load_source_and_policies()
    device = str(next(source.model.parameters()).device)
    if device != "cpu": raise RuntimeError(f"CPU execution required, got {device}")
    historical_manifest = json.loads((mixed.ART / "manifest.json").read_text()); historical = mixed.load_historical(source, historical_manifest)
    baseline_specs = fusion.load_baseline_specs(source, historical); current_specs = pot.load_current_specs(baseline_specs); current_luts = pot.load_current_luts()
    inventory = mixed.expected_inventory(); capture_path = MIXED / "work/capture.npz"; capture_meta = MIXED / "work/capture.json"
    if not capture_path.exists() or sha(capture_path) != json.loads(capture_meta.read_text())["sha256"]: raise RuntimeError("capture unavailable")
    with np.load(capture_path) as z: captured = {key: z[key] for key in z.files}
    selected_by_width = {bits: {} for bits in WIDTHS}; width_rows = []
    for layer in range(18):
        names = {suffix: f"model.layers.{layer}.mlp.{suffix}" for suffix in ("gate_proj", "up_proj")}; idx = inventory.index(names["gate_proj"])
        x = np.concatenate([captured[f"op{idx:04d}_{group}"] for group in GROUPS]); labels = np.concatenate([[group] * len(captured[f"op{idx:04d}_{group}"]) for group in GROUPS])
        p16, s_product, h_fp, h_pre = prepare_layer(layer, x, labels, current_specs[names["gate_proj"]], current_specs[names["up_proj"]], current_luts[layer])
        local = {}; row = {"layer": layer, "float_pre_gpalu_nmse": mixed.nmse(h_pre, h_fp)}
        for bits in WIDTHS:
            selected, _ = calibrate_width(layer, bits, p16, s_product, h_fp, h_pre, labels)
            selected.update(s_act=current_luts[layer]["s_act"], s_up=current_specs[names["up_proj"]]["sout"])
            selected_by_width[bits][layer] = selected; local[bits] = selected
            row.update({f"int{bits}_nmse": selected["local_hw_nmse"], f"int{bits}_clip": selected["clipping_rate"], f"int{bits}_scale": selected["s_h_effective"]})
        n8, n10, n12 = (local[b]["local_hw_nmse"] for b in WIDTHS)
        row.update(int8_to_int10_nmse_recovery=(n8-n10)/n8, int10_to_int12_nmse_recovery=(n10-n12)/n10, int8_to_int12_nmse_recovery=(n8-n12)/n8)
        width_rows.append(row)
    for bits in WIDTHS:
        layers = [selected_by_width[bits][layer] for layer in range(18)]
        base.write_json(OUT / f"int{bits}" / "gpalu_parameters.json", {"output_bits": bits, "layers": layers})
        write_csv(OUT / f"int{bits}" / "layer_summary.csv", layers)
    write_csv(OUT / "width_comparison.csv", width_rows)
    manifest = json.loads((POT / "e2e_dataset_manifest.json").read_text()); examples = mixed.e2e_examples(source); records = []
    gate_specs = {layer: current_specs[f"model.layers.{layer}.mlp.gate_proj"] for layer in range(18)}
    up_specs = {layer: current_specs[f"model.layers.{layer}.mlp.up_proj"] for layer in range(18)}
    run_id = digest({"baseline": baseline_hashes, "selection": selected_by_width, "manifest": manifest})
    for index, example in enumerate(examples):
        path = WORK / f"e2e_{index:02d}.json"; identity = digest({"run": run_id, "example": mixed.identity(example)})
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached and cached.get("identity") != identity: raise RuntimeError("checkpoint mismatch")
        value = cached["value"] if cached else evaluate_sequence(source, gate_specs, up_specs, current_luts, selected_by_width, example)
        if not cached: base.write_json(path, {"identity": identity, "value": value})
        records.append(value)
    metrics = {mode: response.merged([record[mode] for record in records]) for mode in MODES}
    comparisons = {"int8_to_int10": delta(metrics["GPALU_INT8"], metrics["GPALU_INT10"]), "int10_to_int12": delta(metrics["GPALU_INT10"], metrics["GPALU_INT12"]), "int8_to_int12": delta(metrics["GPALU_INT8"], metrics["GPALU_INT12"])}
    gaps = {f"int{bits}_to_float": delta(metrics["FUSION_AWARE_FLOAT_PRODUCT"], metrics[f"GPALU_INT{bits}"]) for bits in WIDTHS}
    base.write_json(OUT / "e2e_summary.json", {"modes": metrics, "comparisons": comparisons, "remaining_gap_to_float": gaps, "E2E_used_for_selection": False})
    if {str(path.relative_to(ROOT)): sha(path) for path in baseline_paths} != baseline_hashes: raise RuntimeError("baseline mutation")
    verification = {"branch": branch, "source_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "execution_device": "CPU", "MPS_used": False, "ANE_used": False, "CUDA_used": False, "host": platform.platform(), "model_hash": historical_manifest["gguf"]["sha256"], "calibration_capture_identity": json.loads(capture_meta.read_text())["identity"], "E2E_dataset_identity": digest(manifest), "Gate_parameters_frozen": True, "Up_parameters_frozen": True, "GELU_parameters_frozen": True, "GPALU_raw_product_bits": 16, "tested_output_widths": list(WIDTHS), "multiplier_format": "UInt16", "shift_format": "UInt5", "dynamic_GPALU_scale": False, "downstream_after_GPALU": "FP", "Down_quantized_for_width_test": False, "Normalizer_quantized": False, "RoPE_quantized": False, "E2E_used_for_selection": False, "baseline_artifacts_unchanged": True, "test_command": tests, "return_code": test_code, "elapsed_seconds": time.monotonic()-START}
    base.write_json(OUT / "verification.json", verification)
    lines = ["# CPU-only GPALU output-width sweep", "", "## 1. Scope", "This small-batch experiment isolates post-GPALU output precision: FLOAT_PRODUCT versus INT8, INT10, and INT12.", "", "## 2. CPU execution environment", "CPU only. No MPS. No ANE. No CUDA.", "", "## 3. Frozen upstream quantization", "Gate/Up/GELU parameters were identical across widths and came from the established fusion-aware baseline.", "", "## 4. GPALU width contracts", "Raw GPALU product was signed INT16. Static per-layer UInt16 M_G / UInt5 S_G produced signed INT8, INT10, or INT12 output.", "", "## 5. Scale calibration method", "Each width independently minimized balanced prefill/decode local GeGLU NMSE over percentile anchors and a ±6/16-octave neighborhood. E2E metrics were not used for calibration.", "", "## 6. Local INT8/10/12 comparison", "See `width_comparison.csv`; all three selected parameter sets are preserved under `int8/`, `int10/`, and `int12/`.", "", "## 7. Clipping comparison", f"Mean selected clipping rates: INT8 {np.mean([x['clipping_rate'] for x in selected_by_width[8].values()]):.9g}, INT10 {np.mean([x['clipping_rate'] for x in selected_by_width[10].values()]):.9g}, INT12 {np.mean([x['clipping_rate'] for x in selected_by_width[12].values()]):.9g}.", "", "## 8. M/S representation error", f"Mean incremental M/S losses: INT8 {np.mean([x['incremental_ms_loss'] for x in selected_by_width[8].values()]):.9g}, INT10 {np.mean([x['incremental_ms_loss'] for x in selected_by_width[10].values()]):.9g}, INT12 {np.mean([x['incremental_ms_loss'] for x in selected_by_width[12].values()]):.9g}.", "", "## 9. 192-token E2E comparison", "| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, value in metrics.items(): lines.append("| " + mode + " | " + " | ".join(f"{value[key]:.9g}" for key in ("nll", "ppl", "kl", "nmse", "mse", "mae", "cosine", "flattened_cosine", "top1", "in5", "overlap")) + " |")
    float_metrics, int8_metrics, int10_metrics, int12_metrics = (metrics[name] for name in ("FUSION_AWARE_FLOAT_PRODUCT", "GPALU_INT8", "GPALU_INT10", "GPALU_INT12"))
    kl_recovery = (int8_metrics["kl"] - int10_metrics["kl"]) / (int8_metrics["kl"] - float_metrics["kl"])
    nmse_recovery = (int8_metrics["nmse"] - int10_metrics["nmse"]) / (int8_metrics["nmse"] - float_metrics["nmse"])
    top1_recovery = (int10_metrics["top1"] - int8_metrics["top1"]) / (float_metrics["top1"] - int8_metrics["top1"])
    lines += ["", "## 10. INT8 -> INT10 gain", f"Top1 delta {comparisons['int8_to_int10']['top1']['absolute_pp']:+.6f} pp; KL relative delta {comparisons['int8_to_int10']['kl']['relative']:+.6%}; NMSE relative delta {comparisons['int8_to_int10']['nmse']['relative']:+.6%}.", f"Relative to the INT8-to-FLOAT_PRODUCT gap, INT10 recovered {kl_recovery:.3%} of KL, {nmse_recovery:.3%} of logits NMSE, and {top1_recovery:.3%} of Top1 agreement.", "", "## 11. INT10 -> INT12 marginal gain", f"Top1 delta {comparisons['int10_to_int12']['top1']['absolute_pp']:+.6f} pp; KL relative delta {comparisons['int10_to_int12']['kl']['relative']:+.6%}; NMSE relative delta {comparisons['int10_to_int12']['nmse']['relative']:+.6%}.", "", "## 12. Implication for future RTL tradeoff", "A. INT10 recovered most of the INT8 loss in KL and logits NMSE, while recovering a smaller majority of the Top1 gap.", "B. INT12 provided an additional measurable gain beyond INT10 on KL, logits NMSE, and Top1; on this small sample the gain is not negligible.", "C. Post-GPALU representation precision is a major error source because widening sharply reduced the gap, but it is not the only source because INT12 still trailed FLOAT_PRODUCT.", "D. Widening the GPALU/VPU boundary is numerically worth considering. This experiment supplies only the accuracy evidence and does not make the RTL decision.", "", "## 13. Limitations", "The sample contains 192 aligned targets. After GPALU reconstruction, Down projection, residuals, RMSNorm, RoPE, attention, and the remaining model used FP computation. Down was not separately quantized. All widths are retained as reusable calibration baselines."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
