"""Small-batch hardware-aware GeGLU calibration with static GPALU POT shifts."""
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
import run_fusion_aware_gate_s10_pilot as fusion

mixed, gelu_base = fusion.mixed, fusion.gelu_base
base, cal, final, response, Profile = fusion.base, fusion.cal, fusion.final, fusion.response, fusion.Profile
OUT = ROOT / "diagnostics/gpalu_pot_fusion_pilot"
WORK = OUT / "work"
LUT_DIR = OUT / "gelu_tables"
QB_DIR = OUT / "gate_qparams"
MIXED = ROOT / "diagnostics/mixed_linear_gelu_calibration"
FUSION = ROOT / "diagnostics/fusion_aware_gate_s10_pilot"
PROGRESS = ROOT / "gpalu_pot_fusion_calibration_progress.txt"
REQUIRED_COMMITS = (
    "2d77cf52907d8b1f910e26a1952c69248090190f",
    "f92fafb2db225165a2eb9064b5073e6f212fac54",
)
J_VALUES = tuple(range(-4, 5))
KG_VALUES = tuple(range(8))
GROUPS = ("prefill", "decode")
MODES = (
    "FP_FULL",
    "CURRENT_FUSION_AWARE_FLOAT_PRODUCT",
    "CURRENT_FUSION_AWARE_STATIC_KG",
    "JOINT_POST_GPALU_AWARE",
)
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as stream:
        stream.write(message + "\n")
        stream.flush()


def sha(path):
    return base.sha256_file(Path(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_csv(path, rows):
    base.write_csv(path, rows)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def balanced(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values))


def group_metric(value, reference, labels):
    result = {}
    for group in GROUPS:
        mask = labels == group
        result[group] = mixed.nmse(value[mask], reference[mask]) if mask.any() else None
    return result


def signed_rne_right_shift(values, shift):
    """Divide signed integers by 2**shift with round-to-nearest-even."""
    values = np.asarray(values, np.int64)
    if shift not in KG_VALUES:
        raise ValueError("kG must be in 0..7")
    if shift == 0:
        return values.copy()
    magnitude = np.abs(values)
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = quotient + increment.astype(np.int64)
    return np.where(values < 0, -rounded, rounded)


def gpalu_codes(q_a, q_u, kg):
    q_a = np.asarray(q_a, np.int16)
    q_u = np.asarray(q_u, np.int16)
    product64 = q_a.astype(np.int64) * q_u.astype(np.int64)
    overflow = int(((product64 < -32768) | (product64 > 32767)).sum())
    if overflow:
        raise RuntimeError("signed INT16 raw product overflow")
    product = product64.astype(np.int16)
    preclip = signed_rne_right_shift(product, kg)
    codes = np.clip(preclip, -128, 127).astype(np.int8)
    details = {
        "raw_product_min": int(product.min()),
        "raw_product_max": int(product.max()),
        "raw_product_absmax": int(np.abs(product.astype(np.int32)).max()),
        "raw_product_mean_abs": float(np.mean(np.abs(product.astype(np.int32)))),
        "raw_product_p99_abs": float(np.percentile(np.abs(product.astype(np.int32)), 99)),
        "raw_product_p99_9_abs": float(np.percentile(np.abs(product.astype(np.int32)), 99.9)),
        "raw_product_p99_99_abs": float(np.percentile(np.abs(product.astype(np.int32)), 99.99)),
        "preclip_min": int(preclip.min()),
        "preclip_max": int(preclip.max()),
        "clip_count": int(((preclip < -128) | (preclip > 127)).sum()),
        "positive_clip_count": int((preclip > 127).sum()),
        "negative_clip_count": int((preclip < -128).sum()),
        "raw_product_overflow_count": overflow,
    }
    details["clip_rate"] = details["clip_count"] / codes.size
    details["positive_clip_rate"] = details["positive_clip_count"] / codes.size
    details["negative_clip_rate"] = details["negative_clip_count"] / codes.size
    return codes, details


def output_scale(s_act, s8_up, kg):
    return float(s_act * s8_up * (2 ** kg))


def candidate_scales(current, original):
    rows = []
    for j in J_VALUES:
        rows.append({"s10_j": j, "s10": current * 2.0 ** (j / 8.0), "source": "centered_grid"})
    if not any(item["s10"] == original for item in rows):
        rows.append({"s10_j": "original", "s10": original, "source": "original_mixed"})
    if len({item["s10"] for item in rows}) != len(rows):
        raise RuntimeError("duplicate s10 candidate")
    if not any(item["s10_j"] == 0 and item["s10"] == current for item in rows):
        raise RuntimeError("current fusion-aware center missing")
    return rows


def kg_key(row):
    return (
        row["post_gpalu_nmse_balanced"],
        row["post_gpalu_nmse_worst"],
        row["clip_rate"],
        row["gpalu_incremental_nmse_balanced"],
        row["kG"],
    )


def select_best_kg(rows):
    if {row["kG"] for row in rows} != set(KG_VALUES):
        raise RuntimeError("incomplete kG sweep")
    return min(rows, key=kg_key)


def s10_key(row, current):
    return (
        row["post_gpalu_nmse_balanced"],
        row["post_gpalu_nmse_worst"],
        row["clip_rate"],
        row["pre_gpalu_nmse_balanced"],
        row["gelu_nmse_balanced"],
        row["gate_nmse_balanced"],
        abs(math.log2(row["s10"] / current)),
        row["s10"],
    )


def select_joint(rows, current):
    current_row = next(row for row in rows if row["s10_j"] == 0)
    selected = min(rows, key=lambda row: s10_key(row, current))
    if selected["post_gpalu_nmse_balanced"] > current_row["post_gpalu_nmse_balanced"] + 1e-12:
        raise RuntimeError("joint search regressed against current candidate")
    return selected, current_row


def load_current_specs(baseline_specs):
    metadata = json.loads((FUSION / "gate_parameters.json").read_text())["layers"]
    if [item["layer"] for item in metadata] != list(range(18)):
        raise RuntimeError("fusion Gate inventory mismatch")
    result = dict(baseline_specs)
    for item in metadata:
        name = item["module"]
        old = baseline_specs[name]
        s10 = float(item["s10_selected"])
        params = Profile().approximate(old["sx"] * old["sw"] / s10)
        packed = Profile().pack(params["multiplier"], params["shift"]).astype("<u4")
        path = ROOT / item["qparam_binary_path"]
        if packed.tobytes() != path.read_bytes() or sha(path) != item["qparam_SHA256"]:
            raise RuntimeError("fusion Gate qparam mismatch")
        result[name] = dict(old, sout=s10, params=params)
    return result


def load_current_luts():
    metadata = json.loads((FUSION / "gate_parameters.json").read_text())["layers"]
    result = {}
    for item in metadata:
        path = ROOT / item["LUT_path"]
        lut = np.frombuffer(path.read_bytes(), dtype=np.int8).copy()
        if len(lut) != 1024 or sha(path) != item["LUT_SHA256"]:
            raise RuntimeError("fusion LUT mismatch")
        result[item["layer"]] = {"s10": item["s10_selected"], "s_act": item["s_act"], "lut": lut}
    return result


def evaluate_layer(layer, x, labels, gate_base, gate_current, up_spec, original_lut, current_lut):
    g_fp = x.astype(np.float64) @ gate_base["weight"].astype(np.float64).T
    u_fp = x.astype(np.float64) @ up_spec["weight"].astype(np.float64).T
    u_hat, up_details = mixed.quantized_linear_rows(x, up_spec, return_codes=True)
    q_u = up_details["output_codes"].astype(np.int16)
    a_fp = gelu_base.gelu(g_fp)
    h_fp = a_fp * u_fp
    up_group = group_metric(u_hat, u_fp, labels)
    all_rows, best_per_s10, payloads = [], [], {}
    original_s10, current_s10 = gate_base["sout"], gate_current["sout"]
    for item in candidate_scales(current_s10, original_s10):
        s10 = item["s10"]
        params = Profile().approximate(gate_base["sx"] * gate_base["sw"] / s10)
        if np.any(params["status"] != "ok"):
            raise RuntimeError(f"invalid Gate profile layer={layer} s10={s10}")
        spec = dict(gate_base, sout=s10, params=params)
        g_hat, gate_details = mixed.quantized_linear_rows(x, spec, return_codes=True)
        q10 = gate_details["output_codes"]
        inner, _ = fusion.inner_sact(g_fp, q10, labels, s10)
        if s10 == original_s10:
            np.testing.assert_array_equal(inner["lut"], original_lut["lut"])
        if s10 == current_s10:
            np.testing.assert_array_equal(inner["lut"], current_lut["lut"])
        q_a = inner["lut"][gelu_base.address(q10)].astype(np.int16)
        a_hat = q_a.astype(np.float64) * inner["s_act"]
        h_pre = a_hat * u_hat
        gate_group = group_metric(g_hat, g_fp, labels)
        gelu_group = group_metric(a_hat, a_fp, labels)
        pre_group = group_metric(h_pre, h_fp, labels)
        rows = []
        for kg in KG_VALUES:
            q_h, details = gpalu_codes(q_a, q_u, kg)
            s_h = output_scale(inner["s_act"], up_spec["sout"], kg)
            h_hat = q_h.astype(np.float64) * s_h
            post = group_metric(h_hat, h_fp, labels)
            incremental = group_metric(h_hat, h_pre, labels)
            row = {
                "layer": layer, "s10_j": item["s10_j"], "s10_source": item["source"],
                "s10": s10, "s_act": inner["s_act"], "gelu_percentile": inner["retained_percentile"],
                "kG": kg, "s_product": inner["s_act"] * up_spec["sout"], "s_h": s_h,
                "gate_nmse_prefill": gate_group["prefill"], "gate_nmse_decode": gate_group["decode"],
                "gate_nmse_balanced": balanced(gate_group.values()),
                "gelu_nmse_prefill": gelu_group["prefill"], "gelu_nmse_decode": gelu_group["decode"],
                "gelu_nmse_balanced": balanced(gelu_group.values()), "up_nmse_balanced": balanced(up_group.values()),
                "pre_gpalu_nmse_prefill": pre_group["prefill"], "pre_gpalu_nmse_decode": pre_group["decode"],
                "pre_gpalu_nmse_balanced": balanced(pre_group.values()),
                "post_gpalu_nmse_prefill": post["prefill"], "post_gpalu_nmse_decode": post["decode"],
                "post_gpalu_nmse_balanced": balanced(post.values()),
                "post_gpalu_nmse_worst": max(v for v in post.values() if v is not None),
                "gpalu_incremental_nmse_balanced": balanced(incremental.values()),
                **{key: details[key] for key in (
                    "raw_product_min", "raw_product_max", "raw_product_absmax", "raw_product_mean_abs",
                    "raw_product_p99_abs", "raw_product_p99_9_abs", "raw_product_p99_99_abs",
                    "preclip_min", "preclip_max", "clip_rate", "positive_clip_rate", "negative_clip_rate",
                )},
                "output_clip_count": details["clip_count"],
                "positive_clip_count": details["positive_clip_count"],
                "negative_clip_count": details["negative_clip_count"],
                "output_element_count": int(q_h.size),
                "selected_kG_for_s10": False, "selected_final": False,
            }
            rows.append(row)
        best = select_best_kg(rows)
        best["selected_kG_for_s10"] = True
        all_rows.extend(rows)
        best_per_s10.append(best)
        payloads[s10] = {"params": params, "lut": inner["lut"], "s_act": inner["s_act"],
                         "percentile": inner["retained_percentile"], "gate_details": gate_details}
    selected, current = select_joint(best_per_s10, current_s10)
    selected["selected_final"] = True
    original = next(row for row in best_per_s10 if row["s10"] == original_s10)
    return selected, current, original, all_rows, payloads[selected["s10"]]


@contextlib.contextmanager
def patch_gpalu(source, specs, luts, kg_by_layer):
    originals = []
    with mixed.patch_mixed(source, specs, luts):
        try:
            for layer, block in enumerate(source.model.model.layers):
                mlp = block.mlp
                originals.append((mlp, mlp.forward))
                item, kg = luts[layer], int(kg_by_layer[layer])
                up_scale = specs[f"model.layers.{layer}.mlp.up_proj"]["sout"]

                def forward(x, mlp=mlp, item=item, kg=kg, up_scale=up_scale):
                    a = mlp.act_fn(mlp.gate_proj(x))
                    u = mlp.up_proj(x)
                    q_a = np.clip(np.rint(a.detach().cpu().numpy() / item["s_act"]), -127, 127).astype(np.int16)
                    q_u = np.clip(np.rint(u.detach().cpu().numpy() / up_scale), -128, 127).astype(np.int16)
                    q_h, _ = gpalu_codes(q_a, q_u, kg)
                    value = q_h.astype(np.float32) * output_scale(item["s_act"], up_scale, kg)
                    return mlp.down_proj(x.new_tensor(value))

                mlp.forward = forward
            yield
        finally:
            for module, original in originals:
                module.forward = original


def run_gpalu(source, specs, luts, kg_by_layer, ids, cache):
    with patch_gpalu(source, specs, luts, kg_by_layer):
        hidden, states, cache = response.forward_hidden(source, ids, [len(ids)-1], {}, cache, True, False)
        with source.torch.inference_mode():
            logits = source.model.lm_head(hidden)
    return logits, states, cache


def evaluate_sequence(source, current_specs, current_luts, current_kg, joint_specs, joint_luts, joint_kg, example):
    scores = {mode: response.Scores() for mode in MODES}
    caches = {mode: None for mode in MODES}
    for index, target in enumerate(example["targets"][:24]):
        ids = example["prompt_ids"] if index == 0 else [example["targets"][index-1]]
        logits = {}
        logits[MODES[0]], _, caches[MODES[0]] = cal.run_fp(source, ids, caches[MODES[0]], layers=False)
        logits[MODES[1]], _, caches[MODES[1]] = mixed.run_quant(source, current_specs, ids, caches[MODES[1]], current_luts)
        logits[MODES[2]], _, caches[MODES[2]] = run_gpalu(source, current_specs, current_luts, current_kg, ids, caches[MODES[2]])
        logits[MODES[3]], _, caches[MODES[3]] = run_gpalu(source, joint_specs, joint_luts, joint_kg, ids, caches[MODES[3]])
        if len({id(value) for value in caches.values()}) != len(MODES):
            raise RuntimeError("KV cache alias")
        for mode in MODES:
            scores[mode].add(source.torch, logits[mode], logits[MODES[0]], [target])
    return {mode: score.raw() for mode, score in scores.items()}


def delta(old, new):
    result = {key: {"absolute": new[key]-old[key], "relative": (new[key]-old[key])/old[key]}
              for key in ("kl", "nmse", "ppl")}
    result["top1"] = {"absolute_pp": 100*(new["top1"]-old["top1"])}
    return result


def classify_e2e(float_mode, static_mode, joint_mode):
    primary = ("kl", "nmse")
    recovery = sum(joint_mode[key] < static_mode[key] for key in primary) + (joint_mode["top1"] > static_mode["top1"])
    residual = sum(joint_mode[key] > float_mode[key] for key in primary) + (joint_mode["top1"] < float_mode["top1"])
    if recovery >= 2 and residual <= 1:
        return "PROCEED_TO_LARGE_BATCH"
    if residual == 3 and joint_mode["top1"] < static_mode["top1"]:
        return "STATIC_KG_NOT_SUPPORTED_BY_PILOT"
    return "STATIC_KG_PROMISING_BUT_NEEDS_REFINEMENT"


def main():
    PROGRESS.write_text("")
    progress("RUN_START")
    for path in (OUT, WORK, LUT_DIR, QB_DIR):
        path.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != "lut_cali":
        raise RuntimeError("wrong branch")
    for commit in REQUIRED_COMMITS:
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"]).returncode:
            raise RuntimeError(f"required commit absent: {commit}")
    progress("GIT_CHECK_DONE")

    baseline_paths = [
        MIXED/"linear_parameters.json", MIXED/"gelu_selection.csv", MIXED/"capture_manifest.json",
        MIXED/"e2e_dataset_manifest.json", FUSION/"gate_parameters.json", FUSION/"layer_selection.csv",
        FUSION/"e2e_dataset_manifest.json",
    ]
    baseline_hashes = {str(path.relative_to(ROOT)): sha(path) for path in baseline_paths}
    base.write_json(OUT/"baseline_manifest.json", {
        "required_commits": list(REQUIRED_COMMITS), "artifact_hashes": baseline_hashes,
        "search_center": "fusion_aware_gate_s10_pilot", "status": "READ_ONLY_BASELINES",
    })
    progress("BASELINE_VERIFY_DONE")

    progress("UNIT_TEST_START")
    tests = [sys.executable, "-m", "unittest", "tests.test_gpalu_pot_fusion_pilot", "tests.test_fusion_aware_gate_s10_pilot", "tests.test_mixed_linear_gelu_calibration"]
    with (OUT/"unit_test.log").open("w") as log:
        code = subprocess.run(tests, stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/gpalu-pot-fusion", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_USE_SHM="0")).returncode
    if code:
        progress("UNIT_TEST_FAILED")
        raise RuntimeError("unit tests failed")
    progress("UNIT_TEST_DONE")

    source, model_manifest, _, _, _, _ = final.load_source_and_policies()
    historical_manifest = json.loads((mixed.ART/"manifest.json").read_text())
    historical = mixed.load_historical(source, historical_manifest)
    baseline_specs = fusion.load_baseline_specs(source, historical)
    baseline_luts = fusion.load_baseline_luts(baseline_specs)
    current_specs = load_current_specs(baseline_specs)
    current_luts = load_current_luts()

    capture_npz, capture_meta = MIXED/"work/capture.npz", MIXED/"work/capture.json"
    if not capture_npz.exists() or not capture_meta.exists() or sha(capture_npz) != json.loads(capture_meta.read_text())["sha256"]:
        raise RuntimeError("compatible baseline capture unavailable")
    with np.load(capture_npz) as archive:
        captured = {key: archive[key] for key in archive.files}
    progress("CAPTURE_REUSE_DONE")

    inventory = mixed.expected_inventory()
    joint_specs, joint_luts = dict(current_specs), dict(current_luts)
    candidate_rows, selections, payloads = [], [], {}
    current_kg, joint_kg = {}, {}
    qparam_hashes, lut_hashes = [], []
    overflow_count = 0
    progress("SEARCH_START")
    for layer in range(18):
        gate_name = f"model.layers.{layer}.mlp.gate_proj"
        up_name = f"model.layers.{layer}.mlp.up_proj"
        gate_index = inventory.index(gate_name)
        x = np.concatenate([captured[f"op{gate_index:04d}_{group}"] for group in GROUPS])
        labels = np.concatenate([[group]*len(captured[f"op{gate_index:04d}_{group}"]) for group in GROUPS])
        selected, current, original, rows, payload = evaluate_layer(
            layer, x, labels, baseline_specs[gate_name], current_specs[gate_name], baseline_specs[up_name], baseline_luts[layer], current_luts[layer]
        )
        candidate_rows.extend(rows)
        payloads[layer] = payload
        current_kg[layer], joint_kg[layer] = int(current["kG"]), int(selected["kG"])
        joint_specs[gate_name] = dict(baseline_specs[gate_name], sout=selected["s10"], params=payload["params"])
        joint_luts[layer] = {"s10": selected["s10"], "s_act": payload["s_act"], "lut": payload["lut"]}
        boundary = selected["s10_j"] in (-4, 4)
        selections.append({
            "layer": layer, "s10_original_mixed": original["s10"], "s10_previous": current["s10"],
            "s10_selected": selected["s10"], "selected_j": selected["s10_j"], "boundary_win": boundary,
            "s_act_selected": selected["s_act"], "kG_current_fixed_s10": current["kG"], "kG_selected": selected["kG"],
            "original_mixed_pre_gpalu_nmse": original["pre_gpalu_nmse_balanced"],
            "current_float_pre_gpalu_nmse": current["pre_gpalu_nmse_balanced"],
            "current_static_post_gpalu_nmse": current["post_gpalu_nmse_balanced"],
            "joint_post_gpalu_nmse": selected["post_gpalu_nmse_balanced"],
            "joint_pre_gpalu_nmse": selected["pre_gpalu_nmse_balanced"],
            "gpalu_incremental_nmse": selected["gpalu_incremental_nmse_balanced"],
            "clip_rate": selected["clip_rate"], "positive_clip_rate": selected["positive_clip_rate"],
            "negative_clip_rate": selected["negative_clip_rate"],
        })
        packed = Profile().pack(payload["params"]["multiplier"], payload["params"]["shift"]).astype("<u4")
        qb_path = QB_DIR/f"layer_{layer:02d}_gate_qb.bin"
        qb_path.write_bytes(packed.tobytes())
        qparam_hashes.append(sha(qb_path))
        lut_path = LUT_DIR/f"layer_{layer:02d}_gelu.bin"
        lut_path.write_bytes(payload["lut"].tobytes())
        lut_hashes.append(sha(lut_path))
        base.write_json(LUT_DIR/f"layer_{layer:02d}_gelu.json", {
            "layer": layer, "s10_previous": current["s10"], "s10_selected": selected["s10"],
            "selected_j": selected["s10_j"], "s_act_selected": payload["s_act"],
            "retained_percentile": payload["percentile"], "kG_selected": selected["kG"], "LUT_SHA256": lut_hashes[-1],
        })
        progress(f"LAYER {layer+1}/18 DONE")

    write_csv(OUT/"candidate_summary.csv", candidate_rows)
    write_csv(OUT/"layer_selection.csv", selections)
    progress("SEARCH_DONE")

    oracle_rows = []
    for kg in KG_VALUES:
        values = []
        for selected in selections:
            rows = [row for row in candidate_rows if row["layer"] == selected["layer"] and row["s10"] == selected["s10_selected"] and row["kG"] == kg]
            values.append(rows[0]["post_gpalu_nmse_balanced"])
        oracle_rows.append({"global_kG": kg, "mean_balanced_post_gpalu_nmse": float(np.mean(values)), "worst_layer_nmse": float(np.max(values))})
    write_csv(OUT/"global_kg_oracle.csv", oracle_rows)
    progress("KG_ORACLE_DONE")

    kg_rows, down_rows, gate_meta, gpalu_meta = [], [], [], []
    for selected in selections:
        layer = selected["layer"]
        gate_name, up_name, down_name = (f"model.layers.{layer}.mlp.{suffix}" for suffix in ("gate_proj", "up_proj", "down_proj"))
        candidate = next(row for row in candidate_rows if row["layer"] == layer and row["selected_final"])
        spec, params = joint_specs[gate_name], joint_specs[gate_name]["params"]
        down_sx = baseline_specs[down_name]["sx"]
        ratio = candidate["s_h"] / down_sx
        kg_rows.append({"layer": layer, "kG": candidate["kG"], "clip_rate": candidate["clip_rate"], "post_gpalu_nmse": candidate["post_gpalu_nmse_balanced"]})
        down_rows.append({"layer": layer, "gpalu_output_scale": candidate["s_h"], "sX_down_base": down_sx,
                          "output_to_down_base_ratio": ratio, "output_to_down_base_log2_ratio": math.log2(ratio)})
        gate_meta.append({
            "layer": layer, "module": gate_name, "sx_base_fixed": spec["sx"], "s10_previous": selected["s10_previous"],
            "s10_selected": spec["sout"], "selected_j": selected["selected_j"],
            "M_min": int(params["multiplier"].min()), "M_max": int(params["multiplier"].max()),
            "S_min": int(params["shift"].min()), "S_max": int(params["shift"].max()),
            "qparam_binary_path": str((QB_DIR/f"layer_{layer:02d}_gate_qb.bin").relative_to(ROOT)), "qparam_SHA256": qparam_hashes[layer],
            "LUT_path": str((LUT_DIR/f"layer_{layer:02d}_gelu.bin").relative_to(ROOT)), "LUT_SHA256": lut_hashes[layer],
        })
        gpalu_meta.append({
            "layer": layer, "gate_module": gate_name, "up_module": up_name,
            "s10_previous": selected["s10_previous"], "s10_selected": candidate["s10"],
            "s_act_selected": candidate["s_act"], "s8_up_fixed": baseline_specs[up_name]["sout"], "kG_selected": candidate["kG"],
            "product_scale": candidate["s_product"], "output_scale": candidate["s_h"],
            "raw_product_absmax": candidate["raw_product_absmax"], "preclip_absmax": max(abs(candidate["preclip_min"]), abs(candidate["preclip_max"])),
            "clip_rate": candidate["clip_rate"], "positive_clip_rate": candidate["positive_clip_rate"], "negative_clip_rate": candidate["negative_clip_rate"],
            "pre_gpalu_nmse": candidate["pre_gpalu_nmse_balanced"], "post_gpalu_nmse": candidate["post_gpalu_nmse_balanced"],
            "gpalu_incremental_nmse": candidate["gpalu_incremental_nmse_balanced"], "sX_down_base": down_sx,
            "output_to_down_base_ratio": ratio, "output_to_down_base_log2_ratio": math.log2(ratio),
        })
    write_csv(OUT/"kg_summary.csv", kg_rows)
    write_csv(OUT/"down_scale_alignment.csv", down_rows)
    base.write_json(OUT/"gate_parameters.json", {"status": "SMALL_BATCH_POST_GPALU_AWARE", "layers": gate_meta})
    base.write_json(OUT/"gpalu_parameters.json", {"contract": "INT8xINT8_INT16_RNE_POT_TO_INT8", "layers": gpalu_meta})
    progress("DOWN_SCALE_DIAGNOSTIC_DONE")

    manifest = json.loads((FUSION/"e2e_dataset_manifest.json").read_text())
    base.write_json(OUT/"e2e_dataset_manifest.json", manifest)
    examples = mixed.e2e_examples(source)
    run_id = digest({"baseline": baseline_hashes, "selected": selections, "dataset": manifest})
    progress("E2E_START")
    records = []
    for number, example in enumerate(examples):
        checkpoint = WORK/f"e2e_{number:02d}.json"
        identity = digest({"run": run_id, "example": mixed.identity(example)})
        cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
        if cached and cached.get("identity") != identity:
            raise RuntimeError("E2E checkpoint mismatch")
        if cached:
            value = cached["value"]
        else:
            value = evaluate_sequence(source, current_specs, current_luts, current_kg, joint_specs, joint_luts, joint_kg, example)
            base.write_json(checkpoint, {"identity": identity, "value": value})
        records.append(value)
        progress(f"SEQUENCE {number+1}/8 DONE")
    metrics = {mode: response.merged([record[mode] for record in records]) for mode in MODES}
    previous = json.loads((FUSION/"e2e_summary.json").read_text())["modes"]["FUSION_AWARE_GATE_S10"]
    for key in ("nll", "ppl", "kl", "nmse", "mse", "mae", "cosine", "flattened_cosine", "top1", "in5", "overlap"):
        if not np.isclose(metrics[MODES[1]][key], previous[key], rtol=0, atol=1e-9):
            raise RuntimeError(f"current fusion-aware E2E reproduction mismatch: {key}")
    comparisons = {
        "float_to_current_static": delta(metrics[MODES[1]], metrics[MODES[2]]),
        "current_static_to_joint": delta(metrics[MODES[2]], metrics[MODES[3]]),
        "float_to_joint": delta(metrics[MODES[1]], metrics[MODES[3]]),
    }
    decision = classify_e2e(metrics[MODES[1]], metrics[MODES[2]], metrics[MODES[3]])
    base.write_json(OUT/"e2e_summary.json", {"modes": metrics, "comparisons": comparisons, "decision": decision, "e2e_used_for_selection": False})
    progress("E2E_DONE")

    before_after = {str(path.relative_to(ROOT)): sha(path) for path in baseline_paths}
    if before_after != baseline_hashes:
        raise RuntimeError("baseline artifact mutated")
    histogram = {str(kg): sum(value == kg for value in joint_kg.values()) for kg in KG_VALUES}
    selected_candidates = [next(row for row in candidate_rows if row["layer"] == layer and row["selected_final"]) for layer in range(18)]
    total_clips = sum(row["output_clip_count"] for row in selected_candidates)
    total_elements = sum(row["output_element_count"] for row in selected_candidates)
    verification = {
        "branch": branch, "source_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "required_baseline_commits_present": True, "model_SHA256": historical_manifest["gguf"]["sha256"], "GGUF_SHA256": historical_manifest["gguf"]["sha256"],
        "calibration_population_identity": sha(MIXED/"capture_manifest.json"), "Gate_op_count": 18, "Up_op_count": 18,
        "ordinary_params_changed": False, "Up_params_changed": False, "Gate_sx_changed": False,
        "s10_search_performed": True, "kG_search_performed": True, "kG_candidate_set": list(KG_VALUES),
        "kG_dynamic": False, "kG_per_layer": True, "raw_product_dtype": "signed_INT16", "raw_product_overflow_count": overflow_count,
        "post_GPALU_output_dtype": "signed_INT8", "selected_kG_min": min(joint_kg.values()), "selected_kG_max": max(joint_kg.values()),
        "selected_kG_mean": float(np.mean(list(joint_kg.values()))), "selected_kG_histogram": histogram,
        "total_clip_count": total_clips, "aggregate_clip_rate": total_clips/total_elements,
        "worst_layer_clip_rate": float(np.max([row["clip_rate"] for row in selected_candidates])),
        "number_of_s10_boundary_winners": sum(row["boundary_win"] for row in selections), "E2E_used_for_selection": False,
        "large_batch_performed": False, "Down_recalibrated": False, "baseline_artifact_hashes": baseline_hashes,
        "all_qparam_hashes": qparam_hashes, "all_LUT_hashes": lut_hashes,
        "test_command": tests, "test_return_code": code, "elapsed_sec": time.monotonic()-START,
    }
    base.write_json(OUT/"verification.json", verification)

    distribution = "tightly clustered" if max(joint_kg.values())-min(joint_kg.values()) <= 1 else ("moderately distributed" if max(joint_kg.values())-min(joint_kg.values()) <= 3 else "widely distributed")
    lines = [
        "# Static GPALU POT fusion-aware small-batch pilot", "",
        "## 1. Scope", "A small-batch calibration of Gate s10, dependent standalone GELU LUTs, and static per-layer GPALU kG.",
        "## 2. Hardware path modeled", "Signed INT8 GELU × signed INT8 Up produced a full signed INT16 raw product. No operand was pre-shifted. Signed RNE right shift by static kG and signed INT8 saturation produced the GeGLU output.",
        "## 3. Frozen parameters", "All ordinary GEMMs, Up parameters, Gate sX, weights, and the runtime absmax row selector remained fixed.",
        "## 4. Nested calibration policy", "For each Gate s10, standalone GELU calibration selected s_act without observing product error. kG then minimized post-GPALU NMSE, and s10 was finally selected by post-GPALU fidelity. E2E was NOT used for parameter selection.",
        "## 5. s10 search", "Nine candidates centered on the previous fusion-aware s10 (j=-4..4), plus the original mixed baseline when distinct. No outward expansion was run.",
        "## 6. kG search", "Every s10 candidate evaluated static kG in {0,1,2,3,4,5,6,7}. Clipping was measured rather than automatically rejected.",
        "## 7. Per-layer selected parameters", "| Layer | j | s10 | s_act | kG | pre NMSE | post NMSE | clip |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selections:
        lines.append(f"| {row['layer']} | {row['selected_j']} | {row['s10_selected']:.9g} | {row['s_act_selected']:.9g} | {row['kG_selected']} | {row['joint_pre_gpalu_nmse']:.9g} | {row['joint_post_gpalu_nmse']:.9g} | {row['clip_rate']:.9g} |")
    lines += [
        "## 8. GPALU clipping analysis", f"Mean selected clip rate: {np.mean([r['clip_rate'] for r in selections]):.9g}; worst layer: {np.max([r['clip_rate'] for r in selections]):.9g}. Raw INT16 overflow count: 0.",
        "## 9. Pre-GPALU vs post-GPALU local error", f"Mean current float pre-GPALU NMSE: {np.mean([r['current_float_pre_gpalu_nmse'] for r in selections]):.9g}; current-s10 static-kG post NMSE: {np.mean([r['current_static_post_gpalu_nmse'] for r in selections]):.9g}; joint post NMSE: {np.mean([r['joint_post_gpalu_nmse'] for r in selections]):.9g}.",
        "## 10. kG distribution", f"Histogram: {histogram}. Mean/min/max: {np.mean(list(joint_kg.values())):.3f}/{min(joint_kg.values())}/{max(joint_kg.values())}; {distribution}.",
        "## 11. Global shared-kG oracle", "The global shared-kG sweep is recorded in `global_kg_oracle.csv` and was diagnostic only.",
        "## 12. Down-proj scale alignment diagnostic", "Selected GPALU output scales and ratios to fixed Down sX_base are recorded in `down_scale_alignment.csv`. Down projection was NOT recalibrated.",
        "## 13. Small E2E comparison", "| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, value in metrics.items():
        lines.append("| "+mode+" | "+" | ".join(f"{value[key]:.9g}" for key in ("nll","ppl","kl","nmse","mse","mae","cosine","flattened_cosine","top1","in5","overlap"))+" |")
    lines += ["### E2E deltas", "| Comparison | KL rel. | NMSE rel. | PPL rel. | Top1 pp |", "|---|---:|---:|---:|---:|"]
    for name, values in comparisons.items():
        lines.append(f"| {name} | {values['kl']['relative']:.6%} | {values['nmse']['relative']:.6%} | {values['ppl']['relative']:.6%} | {values['top1']['absolute_pp']:.6f} |")
    lines += [
        "## 14. Hardware interpretation", "kG was static per layer. No runtime product absmax, general GPALU M/S requantizer, or independent post-GPALU scale was modeled. Output scale followed s_h = s_act*s8_up*2^kG.",
        "## 15. Decision for large-batch calibration", f"**{decision}**",
        "## 16. Limitations", "This is a small-batch pilot. GPALU uses a full signed INT16 product and static UInt3-compatible kG. Down projection was not recalibrated and no extra requantization stage was added.",
    ]
    (OUT/"report.md").write_text("\n\n".join(lines).replace("|\n\n|", "|\n|")+"\n")
    progress("TEST_DONE")
    progress("VERIFY_DONE")
    progress("ARTIFACT_WRITE_DONE")
    progress("RUN_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        progress("RUN_FAILED")
        raise
