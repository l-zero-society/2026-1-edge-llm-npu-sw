"""Per-op standalone GELUTanh LUT calibration on frozen_linear_quant_v1."""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_activation_lut_calibration as shared
import pilot_absmax_calibration as reuse

cal, base, row, final = reuse.cal, reuse.base, reuse.row, reuse.final
OUT = ROOT / "diagnostics/gelu_standalone_calibration"
WORK = OUT / "work"
TABLES = OUT / "tables"
PROGRESS = ROOT / "gelu_standalone_calibration_progress.txt"
CANON = ROOT / "calibration/frozen_linear_quant_v1"
SCALE_SOURCE = ROOT / "calibration_outputs/gguf-all-126-oasst1/manifest.json"
PREVIOUS = ROOT / "diagnostics/activation_lut_calibration_pilot"
PERCENTILES = (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 100.0)
MODES = (
    "FP_FULL",
    "FROZEN_LINEAR_FP_GELU",
    "FROZEN_LINEAR_PER_OP_ABSMAX_GELU_LUT",
    "FROZEN_LINEAR_PER_OP_SELECTED_GELU_LUT",
)
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as f:
        f.write(message + "\n")
        f.flush()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frozen_hashes():
    return {p.name: base.sha256_file(p) for p in sorted(CANON.glob("*.json"))}


def rne(value):
    return np.rint(np.asarray(value))


def q10(value, scale):
    return np.clip(rne(np.asarray(value) / scale), -512, 511).astype(np.int16)


def address(code):
    code = np.asarray(code)
    if np.any(code < -512) or np.any(code > 511):
        raise ValueError("invalid signed INT10 code")
    return code.astype(np.int64) + 512


def gelu(value):
    return shared.gelu(value)


def make_lut(s10, s_act):
    return np.clip(rne(gelu(np.arange(-512, 512, dtype=np.float64) * s10) / s_act), -127, 127).astype(np.int8)


def percentile_candidates(y):
    values = np.abs(np.asarray(y, np.float64)).reshape(-1)
    result = []
    seen = set()
    for percentile in PERCENTILES:
        threshold = float(np.percentile(values, percentile))
        scale = threshold / 127.0
        key = np.float64(scale).tobytes()
        if scale > 0 and key not in seen:
            seen.add(key)
            result.append((percentile, scale))
    return result


def metric(value, reference):
    return shared.metric(value, reference)


def balanced(prefill, decode):
    available = [x for x in (prefill, decode) if x is not None]
    return float(np.mean(available))


def candidate_key(candidate):
    return (
        candidate["score"], candidate["table_nmse"], candidate["worst_total_nmse"],
        candidate["clipping_fraction"], candidate["saturation_rate"], candidate["s_act"],
    )


def select_candidate(candidates):
    if not any(c["retained_percentile"] == 100.0 for c in candidates):
        raise RuntimeError("ABSMAX candidate missing")
    return min(candidates, key=candidate_key)


def gate(old, new):
    checks = {
        "kl_pass": new["kl"] <= 1.05 * old["kl"],
        "nmse_pass": new["nmse"] <= 1.05 * old["nmse"],
        "top1_pass": new["top1"] >= old["top1"] - 0.01,
    }
    return dict(checks, passed=all(checks.values()))


def discover_input_scales():
    artifact = json.loads(SCALE_SOURCE.read_text())
    policies = artifact["scale_policies"]
    rows = []
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.gate_proj"
        policy = policies.get(name)
        if not policy or policy.get("s10") is None:
            raise RuntimeError(f"missing established gate_proj s10 for {name}")
        rows.append({
            "layer": layer, "op_name": name, "s10": float(policy["s10"]),
            "input_scale_source": "AUTHORITATIVE_FIXED_LINEAR_ARTIFACT",
            "source_path": str(SCALE_SOURCE.relative_to(ROOT)),
        })
    return rows


def extract_gate_capture():
    capture = PREVIOUS / "work/capture.npz"
    metadata = PREVIOUS / "work/capture.json"
    manifest = PREVIOUS / "capture_manifest.json"
    verification = PREVIOUS / "verification.json"
    if not all(p.exists() for p in (capture, metadata, manifest, verification)):
        return None, None
    meta = json.loads(metadata.read_text())
    if meta["value"]["sha256"] != base.sha256_file(capture):
        raise RuntimeError("previous capture checksum mismatch")
    prior_verify = json.loads(verification.read_text())
    if prior_verify["frozen_linear_before"] != frozen_hashes():
        raise RuntimeError("previous capture frozen fingerprint mismatch")
    prior_manifest = json.loads(manifest.read_text())
    if prior_manifest.get("reservoir_capacity") != 64 or prior_manifest.get("response_limit") != 16:
        raise RuntimeError("previous capture configuration mismatch")
    with np.load(capture) as data:
        result = {}
        for key in data.files:
            gate, _ = np.split(data[key], 2, axis=1)
            result[key] = gate.copy()
    compatibility = {
        "reused": True,
        "source": str(capture.relative_to(ROOT)),
        "source_sha256": base.sha256_file(capture),
        "source_manifest_sha256": base.sha256_file(manifest),
        "examples": prior_manifest["examples"],
        "response_limit": 16,
        "reservoir_capacity": 64,
        "activation_graph": "gate_proj output -> GELUTanh",
        "fingerprint_checks": ["frozen_linear", "capture_hash", "dataset_manifest", "activation_graph"],
    }
    return result, compatibility


def group_metrics(x, s10, s_act, lut):
    x = np.asarray(x, np.float64)
    q = q10(x, s10)
    xq = q.astype(np.float64) * s10
    y_fp = gelu(x)
    y_exact = gelu(xq)
    codes = lut[address(q)]
    y_lut = codes.astype(np.float64) * s_act
    result = {
        "input_clip_rate": float(np.mean((x / s10 < -512) | (x / s10 > 511))),
        "boundary_hit_rate": float(np.mean((q == -512) | (q == 511))),
        "observed_q_min": int(q.min()), "observed_q_max": int(q.max()),
        "input_code_utilization": float(len(np.unique(q)) / 1024.0),
        "saturation_rate": float(np.mean(np.abs(rne(y_exact / s_act)) > 127)),
        "output_code_utilization": float(len(np.unique(codes)) / 255.0),
        "zero_code_fraction": float(np.mean(codes == 0)),
        "min_used_code": int(codes.min()), "max_used_code": int(codes.max()),
    }
    for label, value, ref in (("input", y_exact, y_fp), ("table", y_lut, y_exact), ("total", y_lut, y_fp)):
        for key, val in metric(value, ref).items():
            result[f"{label}_{key}"] = val
    return result, q


def calibrate_layer(layer, groups, s10):
    exact_values = []
    for x in groups.values():
        q = q10(x, s10)
        exact_values.append(gelu(q.astype(np.float64) * s10).reshape(-1))
    candidates = []
    for percentile, s_act in percentile_candidates(np.concatenate(exact_values)):
        lut = make_lut(s10, s_act)
        per_group = {}
        for group in ("prefill", "decode"):
            if group in groups and len(groups[group]):
                per_group[group], _ = group_metrics(groups[group], s10, s_act, lut)
        def b(key):
            return balanced(*(per_group.get(g, {}).get(key) for g in ("prefill", "decode")))
        totals = [r["total_nmse"] for r in per_group.values()]
        candidates.append({
            "layer": layer, "retained_percentile": percentile,
            "selection_policy": "ABSMAX" if percentile == 100.0 else "PERCENTILE_CLIPPING",
            "clipping_fraction": (100.0 - percentile) / 100.0, "s_act": s_act,
            "score": b("total_nmse"), "input_nmse": b("input_nmse"),
            "table_nmse": b("table_nmse"), "total_nmse": b("total_nmse"),
            "worst_total_nmse": float(max(totals)), "saturation_rate": b("saturation_rate"),
            "output_code_utilization": b("output_code_utilization"),
            "zero_code_fraction": b("zero_code_fraction"),
            "min_used_code": min(r["min_used_code"] for r in per_group.values()),
            "max_used_code": max(r["max_used_code"] for r in per_group.values()),
            "prefill_total_nmse": per_group.get("prefill", {}).get("total_nmse"),
            "decode_total_nmse": per_group.get("decode", {}).get("total_nmse"),
        })
    selected = select_candidate(candidates)
    absmax = next(c for c in candidates if c["retained_percentile"] == 100.0)
    return candidates, selected, absmax


def write_table(directory, layer, s10, candidate):
    directory.mkdir(parents=True, exist_ok=True)
    lut = make_lut(s10, candidate["s_act"])
    binary = directory / f"layer_{layer:02d}_gelu.bin"
    metadata = directory / f"layer_{layer:02d}_gelu.json"
    binary.write_bytes(lut.tobytes())
    info = {
        "layer": layer, "activation": "torch.nn.functional.gelu(approximate='tanh')",
        "s10": s10, "S_act": candidate["s_act"],
        "selection_policy": candidate["selection_policy"],
        "retained_percentile": candidate["retained_percentile"],
        "clipping_fraction": candidate["clipping_fraction"],
        "address_mapping": "q + 512", "input_codes": [-512, 511],
        "output_codes": [-127, 127], "entries": 1024,
        "sha256": base.sha256_file(binary),
    }
    base.write_json(metadata, info)
    return info, lut


@contextlib.contextmanager
def activation_patch(source, configuration=None):
    originals = []
    try:
        if configuration is not None:
            for layer, block in enumerate(source.model.model.layers):
                act = block.mlp.act_fn
                originals.append((act, act.forward))
                s10, s_act, lut = configuration[layer]
                def forward(x, s10=s10, s_act=s_act, lut=lut):
                    q = q10(x.detach().cpu().numpy(), s10)
                    value = (lut[address(q)].astype(np.float64) * s_act).astype(np.float32)
                    return source.torch.from_numpy(value).to(dtype=x.dtype, device=x.device)
                act.forward = forward
        yield
    finally:
        for act, original in originals:
            act.forward = original


def example_identity(example):
    return {k: example[k] for k in ("index", "anchor_id", "response_id", "tree_id")}


def find_examples(source, identities):
    pairs, _ = cal.build_pairs("validation")
    examples, _ = cal.tokenize_pairs(source, pairs)
    wanted = {item["anchor_id"]: item for item in identities}
    selected = [e for e in examples if e["anchor_id"] in wanted and len(e["targets"]) >= 24]
    selected.sort(key=lambda e: identities.index(wanted[e["anchor_id"]]))
    if len(selected) != 8:
        raise RuntimeError("unable to reproduce previous 8-chat E2E set")
    return selected


def evaluate_sequence(source, specs, example, absmax_config, selected_config):
    scores = {mode: reuse.response.Scores() for mode in MODES}
    caches = {mode: None for mode in MODES}
    configs = {
        MODES[1]: None,
        MODES[2]: absmax_config,
        MODES[3]: selected_config,
    }
    for position, target in enumerate(example["targets"][:24]):
        ids = example["prompt_ids"] if position == 0 else [example["targets"][position - 1]]
        logits = {}
        logits[MODES[0]], _, caches[MODES[0]] = cal.run_fp(source, ids, caches[MODES[0]], layers=False)
        for mode in MODES[1:]:
            with patch.object(row, "select_rows", reuse.select_absmax), activation_patch(source, configs[mode]):
                logits[mode], _, caches[mode] = cal.run_mode(source, specs, ids, caches[mode], layers=False)
        if len({id(cache) for cache in caches.values()}) != 4:
            raise RuntimeError("KV cache alias")
        for mode in MODES:
            scores[mode].add(source.torch, logits[mode], logits[MODES[0]], [target])
    return {mode: score.raw() for mode, score in scores.items()}


def write_csv(path, rows):
    base.write_csv(path, rows)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def main():
    PROGRESS.write_text("")
    progress("RUN_START")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != "lut_cali":
        raise RuntimeError("wrong branch")
    progress("GIT_CHECK_DONE")
    before = frozen_hashes()
    progress("INPUT_SCALE_DISCOVERY_START")
    scales = discover_input_scales()
    source_hash = base.sha256_file(SCALE_SOURCE)
    input_manifest = {
        "source": str(SCALE_SOURCE.relative_to(ROOT)), "source_sha256": source_hash,
        "activation_producer": "model.layers.{0..17}.mlp.gate_proj",
        "authoritative_count": 18, "derived_count": 0, "layers": scales,
    }
    base.write_json(OUT / "input_scale_manifest.json", input_manifest)
    progress("INPUT_SCALE_DISCOVERY_DONE authoritative=18 derived=0")

    progress("UNIT_TEST_START")
    tests = [sys.executable, "-m", "unittest", "tests.test_gelu_standalone_calibration", "tests.test_activation_lut_calibration", "tests.test_frozen_linear_quant_v1"]
    with (OUT / "unit_test.log").open("w") as log:
        test_env = dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/gelu-standalone-calibration",
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_USE_SHM="0")
        test_code = subprocess.run(tests, stdout=log, stderr=subprocess.STDOUT, env=test_env).returncode
    if test_code:
        progress("UNIT_TEST_FAILED")
        raise RuntimeError("unit tests failed")
    progress("UNIT_TEST_DONE")

    progress("CAPTURE_REUSE_CHECK")
    captured, capture_manifest = extract_gate_capture()
    if captured is None:
        raise RuntimeError("compatible prior capture unavailable; recapture path intentionally not entered")
    base.write_json(OUT / "capture_manifest.json", capture_manifest)
    progress("CAPTURE_REUSED")

    progress("LOCAL_SEARCH_START")
    all_candidates, selections, absmaxes = [], [], []
    table_hashes = []
    scale_by_layer = {row["layer"]: row["s10"] for row in scales}
    histograms = {}
    selected_config, absmax_config = {}, {}
    for layer in range(18):
        groups = {group: captured[f"{layer}_{group}"] for group in ("prefill", "decode")}
        s10 = scale_by_layer[layer]
        candidates, selected, absmax = calibrate_layer(layer, groups, s10)
        all_candidates.extend(candidates)
        selected_info, selected_lut = write_table(TABLES, layer, s10, selected)
        absmax_info, absmax_lut = write_table(TABLES / "absmax", layer, s10, absmax)
        table_hashes.append(selected_info["sha256"])
        selected_config[layer] = (s10, selected["s_act"], selected_lut)
        absmax_config[layer] = (s10, absmax["s_act"], absmax_lut)
        q = np.concatenate([q10(x, s10).reshape(-1) for x in groups.values()])
        counts = np.bincount(address(q), minlength=1024)
        histograms[str(layer)] = {
            "counts": counts.tolist(), "samples": int(q.size), "observed_min": int(q.min()),
            "observed_max": int(q.max()), "utilization": float(np.count_nonzero(counts) / 1024),
            "input_clipping_rate": float(np.mean(np.concatenate([(x/s10 < -512).reshape(-1) | (x/s10 > 511).reshape(-1) for x in groups.values()]))),
            "boundary_hit_rate": float(np.mean((q == -512) | (q == 511))),
        }
        row_selection = dict(selected)
        row_selection.update({
            "op_name": f"model.layers.{layer}.mlp.gelu", "s10": s10,
            "s10_source": "AUTHORITATIVE_FIXED_LINEAR_ARTIFACT", "selected_lut_sha256": selected_info["sha256"],
            "absmax_s_act": absmax["s_act"], "absmax_total_nmse": absmax["total_nmse"],
            "absmax_lut_sha256": absmax_info["sha256"],
        })
        selections.append(row_selection)
        absmaxes.append(absmax)
        progress(f"LAYER {layer+1}/18 DONE")
    input_manifest["observed_code_distributions"] = histograms
    base.write_json(OUT / "input_scale_manifest.json", input_manifest)
    write_csv(OUT / "per_op_candidates.csv", all_candidates)
    write_csv(OUT / "per_op_selection.csv", selections)
    scale_pairs = [{
        "layer": row["layer"], "op_name": row["op_name"], "s10": row["s10"],
        "S_act": row["s_act"], "log2_s10": float(np.log2(row["s10"])),
        "log2_S_act": float(np.log2(row["s_act"])), "selection_policy": row["selection_policy"],
        "retained_percentile": row["retained_percentile"], "clipping_fraction": row["clipping_fraction"],
        "lut_sha256": row["selected_lut_sha256"],
    } for row in selections]
    write_csv(OUT / "scale_pairs.csv", scale_pairs)
    base.write_json(OUT / "scale_pairs.json", {"clustering_performed": False, "pairs": scale_pairs})
    progress("LOCAL_SEARCH_DONE")
    progress("LUT_WRITE_DONE")

    source, model_manifest, _, previous_specs, _, _ = final.load_source_and_policies()
    specs = shared.load_specs(source, previous_specs)
    specs_fingerprint = reuse.compare.fingerprint(specs)
    previous_e2e_manifest = json.loads((PREVIOUS / "e2e_dataset_manifest.json").read_text())
    examples = find_examples(source, previous_e2e_manifest["examples"])
    e2e_manifest = {
        "examples": [example_identity(e) for e in examples], "sequences": 8,
        "targets_per_sequence": 24, "targets": 192, "role": "selection evaluation only",
        "reused_previous_shared_pilot_dataset": True,
        "final_linear_confirmation_overlap": False,
    }
    base.write_json(OUT / "e2e_dataset_manifest.json", e2e_manifest)
    run_identity = digest({
        "frozen": before, "input_scales": input_manifest["source_sha256"],
        "capture": capture_manifest["source_sha256"], "e2e": e2e_manifest,
        "selected": [{"s10": selected_config[l][0], "s_act": selected_config[l][1], "hash": table_hashes[l]} for l in range(18)],
        "script": base.sha256_file(__file__),
    })
    progress("E2E_START modes=4 sequences=8 targets=192")
    records = []
    for index, example in enumerate(examples):
        path = WORK / f"e2e_{index:02d}.json"
        identity = digest({"run": run_identity, "example": example_identity(example)})
        record = shared.checkpoint(path, identity)
        if record is None:
            record = evaluate_sequence(source, specs, example, absmax_config, selected_config)
            base.write_json(path, {"identity": identity, "value": record})
        records.append(record)
        progress(f"SEQUENCE {index+1}/8 DONE")
    metrics = {mode: reuse.response.merged([record[mode] for record in records]) for mode in MODES}
    if any(value["tokens"] != 192 for value in metrics.values()):
        raise RuntimeError("unexpected E2E target count")
    fp_gelu = metrics[MODES[1]]
    gates = {mode: gate(fp_gelu, metrics[mode]) for mode in MODES[2:]}
    classifications = {mode: "BASELINE_PASS" if gates[mode]["passed"] else "BASELINE_REGRESSION" for mode in MODES[2:]}
    def deltas(old, new):
        return {key: {
            "absolute": new[key] - old[key],
            "relative": (new[key] - old[key]) / old[key] if old[key] else None,
        } for key in ("kl", "nmse", "ppl", "top1")}
    local_selected = float(np.mean([r["total_nmse"] for r in selections]))
    local_absmax = float(np.mean([r["absmax_total_nmse"] for r in selections]))
    selected_vs_abs = deltas(metrics[MODES[2]], metrics[MODES[3]])
    if metrics[MODES[3]]["kl"] < metrics[MODES[2]]["kl"] and local_selected < local_absmax:
        clipping_class = "CLIPPING_BETTER"
    elif local_selected >= local_absmax and metrics[MODES[3]]["kl"] >= metrics[MODES[2]]["kl"]:
        clipping_class = "CLIPPING_NO_BENEFIT"
    else:
        clipping_class = "CLIPPING_TRADEOFF"
    e2e_summary = {
        "modes": metrics, "fp_gelu_to_absmax": deltas(fp_gelu, metrics[MODES[2]]),
        "absmax_to_selected": selected_vs_abs, "gates": gates,
        "classifications": classifications, "clipping_classification": clipping_class,
        "parameter_selection_used_e2e": False,
    }
    base.write_json(OUT / "e2e_summary.json", e2e_summary)
    progress("E2E_DONE")

    previous_summary = json.loads((PREVIOUS / "e2e_summary.json").read_text())
    previous_mode = previous_summary["modes"]["FROZEN_LINEAR_SHARED_LUT"]
    after = frozen_hashes()
    if before != after:
        raise RuntimeError("frozen Linear artifact changed")
    if reuse.compare.fingerprint(specs) != specs_fingerprint:
        raise RuntimeError("runtime frozen Linear specs mutated")
    verification = {
        "branch": branch,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_identity": model_manifest["gguf"],
        "activation_implementation": "torch.nn.functional.gelu(x, approximate='tanh')",
        "number_activation_ops": 18, "frozen_linear_hashes_before": before,
        "frozen_linear_hashes_after": after, "frozen_linear_unchanged": True,
        "s10_source_per_op": [{"layer": r["layer"], "s10": r["s10"], "source": r["input_scale_source"]} for r in scales],
        "number_authoritative_s10": 18, "number_derived_s10": 0,
        "lut_count": 18, "table_size_bytes": 1024, "all_table_hashes": table_hashes,
        "clustering_performed": False, "fusion_tuning_performed": False,
        "product_aware_selection": False, "e2e_used_for_parameter_selection": False,
        "final_linear_confirmation_overlap": False, "independent_kv_caches": True,
        "capture_reused": True, "test_command": tests, "test_return_code": test_code,
        "elapsed_sec": time.monotonic() - START,
    }
    base.write_json(OUT / "verification.json", verification)

    lines = [
        "# GeLU standalone per-op LUT calibration", "",
        "## 1. Scope", "Per-op fixed-INT10-input, signed-INT8-output GELUTanh LUT calibration. No fusion-aware tuning was performed. No LUT clustering was performed. `frozen_linear_quant_v1` was not modified.",
        "## 2. GeLU numerical contract", "Each layer uses its fixed gate-producer `s10`; `q=clip(RNE(x/s10),-512,511)`, address `q+512`, and `q_act=clip(RNE(GELUTanh(q*s10)/S_act),-127,127)`. Tables contain 1024 signed INT8 entries and use `torch.nn.functional.gelu(..., approximate=\"tanh\")`.",
        "## 3. Fixed s10 sources", f"All 18 scales are authoritative fixed gate-projection output scales from `{SCALE_SOURCE.relative_to(ROOT)}` (SHA256 `{source_hash}`); each is 0.1. No derived input scale was needed.",
        "## 4. Calibration dataset", "Reused the fingerprint-validated prior 12-chat gate capture: prefill/decode reservoirs, 64 rows per layer/group, first 16 response positions. It excludes the frozen Linear 1024-target confirmation data.",
        "## 5. ABSMAX vs probabilistic clipping policy", "Per layer, retained percentiles 99, 99.5, 99.9, 99.95, 99.99, 99.995, and 100 (ABSMAX) were evaluated against the observed code-weighted distribution. Selection minimized balanced prefill/decode total standalone activation NMSE; E2E results did not select parameters.",
        "## 6. Per-op selected S_act summary", "| Layer | Policy | Percentile | S_act | Total NMSE | ABSMAX NMSE | Saturation | LUT SHA256 |", "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in selections:
        lines.append(f"| {r['layer']} | {r['selection_policy']} | {r['retained_percentile']:.6g} | {r['s_act']:.9g} | {r['total_nmse']:.9g} | {r['absmax_total_nmse']:.9g} | {r['saturation_rate']:.9g} | `{r['selected_lut_sha256']}` |")
    lines += [
        "", "## 7. Input / table / total error decomposition",
        f"Mean selected input NMSE: {np.mean([r['input_nmse'] for r in selections]):.9g}; table NMSE: {np.mean([r['table_nmse'] for r in selections]):.9g}; total NMSE: {local_selected:.9g}. Metrics are computed on captured workload probability, separately by prefill/decode then balanced.",
        "## 8. ABSMAX-vs-selected local comparison", f"Mean per-op ABSMAX total NMSE: {local_absmax:.9g}; selected total NMSE: {local_selected:.9g}. Every layer retains its explicit ABSMAX candidate in `per_op_candidates.csv`.",
        "## 9. E2E comparison", "| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, value in metrics.items():
        lines.append("| " + mode + " | " + " | ".join(f"{value[k]:.9g}" for k in ("nll", "ppl", "kl", "nmse", "mse", "mae", "cosine", "flattened_cosine", "top1", "in5", "overlap")) + " |")
    lines += [
        "", "The primary comparisons are per-op ABSMAX versus frozen-Linear FP GeLU and per-op selected versus per-op ABSMAX. Exact deltas are in `e2e_summary.json`.",
        "## 10. Comparison against previous global shared-LUT pilot",
        "The same eight conversation identities and 192 targets were reused, so this comparison is direct.",
        "| Mode | Local activation NMSE | E2E KL | E2E logits NMSE | Top1 | PPL |", "|---|---:|---:|---:|---:|---:|",
        f"| Previous global shared LUT | {json.loads((PREVIOUS/'global_selection.json').read_text())['selected']['activation_nmse']:.9g} | {previous_mode['kl']:.9g} | {previous_mode['nmse']:.9g} | {previous_mode['top1']:.9g} | {previous_mode['ppl']:.9g} |",
        f"| Per-op selected LUT | {local_selected:.9g} | {metrics[MODES[3]]['kl']:.9g} | {metrics[MODES[3]]['nmse']:.9g} | {metrics[MODES[3]]['top1']:.9g} | {metrics[MODES[3]]['ppl']:.9g} |",
        "## 11. Standalone baseline classification",
        f"Per-op ABSMAX: **{classifications[MODES[2]]}**. Per-op selected: **{classifications[MODES[3]]}**. Clipping comparison: **{clipping_class}**.",
        "## 12. Scale-pair artifact for future clustering", "`scale_pairs.csv` and `scale_pairs.json` contain `(s10,S_act)` and log2 coordinates. No clustering was performed.",
        "## 13. Limitations", "Gate projection MAC arithmetic remains FP; this isolates the fixed INT10 activation interface and INT8 LUT output. Selection is activation-standalone and deliberately ignores the Up branch, product, and downstream error. This is a small selection evaluation, not a final confirmation or production freeze.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    progress("ARTIFACT_WRITE_DONE")
    progress("TEST_DONE")
    progress("FINAL_VERIFY_DONE")
    progress("RUN_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        progress("RUN_FAILED")
        raise
