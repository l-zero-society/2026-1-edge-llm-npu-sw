#!/usr/bin/env python3
"""Larger-data stability run for the fixed joint absmax calibration policy."""
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
import absmax_joint_calibration as joint
import pilot_absmax_calibration as pilot

final, base, row = pilot.final, pilot.base, pilot.row
OUT = ROOT / "diagnostics/absmax_joint_stability"
WORK = OUT / "work"
PROGRESS = ROOT / "stability_calibration_progress.txt"
CAL_POSITIONS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
VAL_POSITIONS = [0, 3, 6, 9, 12, 15, 18, 21, 2, 8, 14, 20]
CAL_RESPONSE_LIMIT = 32
VAL_TARGETS = 32
RESERVOIR_CAPACITY = 64
SEED = 20261002
SX_JS = [-2, -1, 0, 1, 2]
S10_IS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
MODES = ("FP", "old_absmax", "pilot_joint_absmax", "stability_joint_absmax")
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as handle:
        handle.write(message + "\n")
        handle.flush()


def configuration(before, pilot_hash):
    return dict(calibration_positions=CAL_POSITIONS, calibration_response_limit=CAL_RESPONSE_LIMIT,
                reservoir_capacity=RESERVOIR_CAPACITY, seed=SEED, sx_js=SX_JS,
                s10_is=S10_IS, validation_positions=VAL_POSITIONS,
                validation_targets=VAL_TARGETS, frozen=before, pilot_sha256=pilot_hash,
                selector="compare_k_selectors_e2e.select_absmax",
                calibration_policy="absmax_joint_calibration.calibrate_layer_joint")


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_matches(path, fingerprint):
    return path.exists() and json.loads(path.read_text()).get("configuration_fingerprint") == fingerprint


def direction(value):
    return 0 if value == 0 else (1 if value > 0 else -1)


def stability_row(pilot_row, current_row):
    psx, ssx = int(pilot_row["sx_j"]), int(current_row["sx_j"])
    ps10, ss10 = int(pilot_row["s10_i"]), int(current_row["s10_i"])
    pbx = str(pilot_row["sx_boundary_hit"]).lower() == "true"
    pbo = str(pilot_row["s10_boundary_hit"]).lower() == "true"
    sbx, sbo = bool(current_row["sx_boundary_hit"]), bool(current_row["s10_boundary_hit"])
    return dict(layer=int(current_row["layer"]), pilot_sx_j=psx, stability_sx_j=ssx,
        exact_sx_match=psx == ssx, pilot_s10_i=ps10, stability_s10_i=ss10,
        exact_s10_match=ps10 == ss10, exact_pair_match=(psx, ps10) == (ssx, ss10),
        sx_direction_match=direction(psx) == direction(ssx),
        s10_direction_match=direction(ps10) == direction(ss10),
        pilot_sx_boundary_hit=pbx, stability_sx_boundary_hit=sbx,
        pilot_s10_boundary_hit=pbo, stability_s10_boundary_hit=sbo,
        sx_boundary_persisted=pbx and sbx, s10_boundary_persisted=pbo and sbo,
        pilot_local_change=float(pilot_row["relative_local_nmse_change"]),
        stability_local_change=float(current_row["relative_local_nmse_change"]))


def stability_summary(rows):
    return dict(exact_sx_match_count=sum(r["exact_sx_match"] for r in rows),
        exact_s10_match_count=sum(r["exact_s10_match"] for r in rows),
        exact_pair_match_count=sum(r["exact_pair_match"] for r in rows),
        sx_direction_match_count=sum(r["sx_direction_match"] for r in rows),
        s10_direction_match_count=sum(r["s10_direction_match"] for r in rows),
        persistent_sx_boundary_count=sum(r["sx_boundary_persisted"] for r in rows),
        persistent_s10_boundary_count=sum(r["s10_boundary_persisted"] for r in rows))


def load_pilot_specs(frozen):
    stored = json.loads((pilot.ROOT / "diagnostics/absmax_joint_calibration_pilot/proposed_parameters.json").read_text())
    if set(stored) != set(frozen):
        raise RuntimeError("pilot proposed module set mismatch")
    specs = {}
    for name, old in frozen.items():
        item = stored[name]
        params = joint.Profile().approximate(float(item["sx_base"]) *
            np.asarray(old["params"]["ratio"]) * old["s10"] / old["sx"] / float(item["s10"]))
        np.testing.assert_array_equal(params["multiplier"], np.asarray(item["multiplier"]))
        np.testing.assert_array_equal(params["shift"], np.asarray(item["shift"]))
        specs[name] = dict(old, sx=float(item["sx_base"]), s10=float(item["s10"]), params=params)
        np.testing.assert_array_equal(specs[name]["wq"], old["wq"])
    return specs


def capture(source, specs, examples):
    reservoirs = {layer: {g: pilot.cal.Reservoir(RESERVOIR_CAPACITY, SEED)
                    for g in pilot.cal.GROUPS} for layer in range(18)}
    with patch.object(row, "select_rows", pilot.select_absmax):
        for number, example in enumerate(examples):
            if len(example["targets"]) < CAL_RESPONSE_LIMIT:
                raise RuntimeError(f"calibration response too short at fixed position {CAL_POSITIONS[number]}")
            context = {}
            with pilot.cal.patch_down(source, specs, reservoirs, "prefill", example["index"], 0):
                _, _, cache = pilot.response.forward_hidden(source, example["prompt_ids"],
                    [len(example["prompt_ids"]) - 1], context, use_cache=True, layers=False)
            for t in range(1, CAL_RESPONSE_LIMIT):
                context = {}
                absolute = len(example["prompt_ids"]) + t - 1
                with pilot.cal.patch_down(source, specs, reservoirs, "decode", example["index"], absolute):
                    _, _, cache = pilot.response.forward_hidden(source,
                        [example["targets"][t - 1]], [0], context, cache, True, False)
            progress(f"CAPTURE {number + 1}/16 DONE")
    return {layer: {g: reservoirs[layer][g].arrays()[0] for g in pilot.cal.GROUPS}
            for layer in range(18)}


def save_capture(path, captured, fingerprint):
    arrays = {f"layer_{layer}_{group}": captured[layer][group]
              for layer in range(18) for group in pilot.cal.GROUPS}
    np.savez_compressed(path, **arrays)
    base.write_json(WORK / "capture.json", dict(configuration_fingerprint=fingerprint,
                    arrays=sorted(arrays), complete=True))


def load_capture(path):
    with np.load(path) as data:
        return {layer: {group: data[f"layer_{layer}_{group}"].copy()
                for group in pilot.cal.GROUPS} for layer in range(18)}


def serialize_layer(layer, result):
    selected, meta = result["selected"], result["metadata"]
    return dict(layer=layer, old_sx=result["old"]["sx"], selected_sx=selected["sx"],
        sx_j=selected["sx_j"], old_s10=result["old"]["s10"], selected_s10=selected["s10"],
        s10_i=selected["s10_i"], **meta, prefill_nmse=selected["prefill_nmse"],
        decode_nmse=selected["decode_nmse"], worst_nmse=selected["worst"],
        clip_rate=selected["clip_rate"], effective_shift_min=selected["effective_shift_min"],
        effective_shift_max=selected["effective_shift_max"],
        multiplier=selected["multiplier"].tolist(), shift=selected["shift"].tolist())


def spec_from_detail(old, detail):
    sx, s10 = float(detail["selected_sx"]), float(detail["selected_s10"])
    sw = np.asarray(old["params"]["ratio"]) * old["s10"] / old["sx"]
    params = joint.Profile().approximate(sx * sw / s10)
    np.testing.assert_array_equal(params["multiplier"], detail["multiplier"])
    np.testing.assert_array_equal(params["shift"], detail["shift"])
    return dict(old, sx=sx, s10=s10, params=params)


def execute(source, specs, mode, ids, cache):
    if mode == "FP":
        return final.cached.run_fp(source, ids, cache, layers=False)
    with patch.object(row, "select_rows", pilot.select_absmax):
        return final.cached.run_mode(source, specs[mode], ids, cache, layers=False)


def validate_conversation(source, specs, example, position):
    if len(example["targets"]) < VAL_TARGETS:
        raise RuntimeError(f"validation response too short at fixed position {position}")
    scores = {mode: pilot.response.Scores() for mode in MODES}
    caches, logits = {}, {}
    for mode in MODES:
        logits[mode], _, caches[mode] = execute(source, specs, mode, example["prompt_ids"], None)
    pilot.independent([caches[m] for m in MODES[:3]])
    if len({id(caches[m]) for m in MODES}) != 4:
        raise RuntimeError("validation caches are not independent")
    for t, target in enumerate(example["targets"][:VAL_TARGETS]):
        if t:
            for mode in MODES:
                logits[mode], _, caches[mode] = execute(source, specs, mode,
                    [example["targets"][t - 1]], caches[mode])
        if len({id(caches[m]) for m in MODES}) != 4:
            raise RuntimeError("validation caches are not independent")
        for mode in MODES:
            scores[mode].add(source.torch, logits[mode], logits["FP"], [target])
    return dict(index=position, source_index=example["index"], targets=VAL_TARGETS,
                scores={m: s.raw() for m, s in scores.items()})


def relative_deltas(old, new):
    return dict(kl_relative=(new["kl"] - old["kl"]) / old["kl"],
                nmse_relative=(new["nmse"] - old["nmse"]) / old["nmse"],
                ppl_relative=(new["ppl"] - old["ppl"]) / old["ppl"],
                top1_pp=100 * (new["top1"] - old["top1"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(""); progress("RUN_START")
    progress("UNIT_TEST_START")
    test_command = [sys.executable, "-m", "unittest", "tests.test_absmax_joint_stability",
        "tests.test_absmax_joint_calibration", "tests.test_pilot_absmax_calibration",
        "tests.test_k_selector_fidelity_384", "tests.test_k_selector_compare",
        "tests.test_finalize_downproj_ptq"]
    env = dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/absmax-joint-stability")
    with (OUT / "unit_test.log").open("w") as log:
        tested = subprocess.run(test_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if tested.returncode:
        progress("UNIT_TEST_FAILED"); raise RuntimeError("unit tests failed")
    progress("UNIT_TEST_DONE")
    source, _, _, previous, _, _ = final.load_source_and_policies()
    frozen, _ = final.restore_final(source, previous)
    expected = {f"model.layers.{i}.mlp.down_proj" for i in range(18)}
    if set(frozen) != expected: raise RuntimeError("unexpected quantized modules")
    decisions = final.OUT / "decisions.json"
    protected = [final.OUT, pilot.OUT, ROOT / "diagnostics/absmax_joint_calibration_pilot"]
    def frozen_fp():
        return dict(decisions_sha256=base.sha256_file(decisions),
                    parameters_sha256=pilot.compare.fingerprint(frozen),
                    artifacts_sha256=pilot.fingerprint_tree(protected))
    before = frozen_fp()
    pilot_parameters = ROOT / "diagnostics/absmax_joint_calibration_pilot/proposed_parameters.json"
    pilot_hash = base.sha256_file(pilot_parameters)
    config = configuration(before, pilot_hash); config_hash = config_fingerprint(config)
    existing = WORK / "configuration.json"
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise RuntimeError("checkpoint configuration/fingerprint mismatch")
    base.write_json(existing, config)
    cal_examples, cal_skipped = pilot.select_examples(source, "calibration", CAL_POSITIONS)
    if len(cal_examples) != 16: raise RuntimeError("expected 16 fixed calibration chats")
    capture_path = WORK / "capture.npz"
    if checkpoint_matches(WORK / "capture.json", config_hash) and capture_path.exists():
        captured = load_capture(capture_path)
        progress("CAPTURE_DONE checkpoint=reused")
    else:
        progress("CAPTURE_START")
        try: captured = capture(source, frozen, cal_examples)
        except BaseException:
            progress("CAPTURE_FAILED"); raise
        save_capture(capture_path, captured, config_hash); progress("CAPTURE_DONE")
    progress("JOINT_CALIBRATION_START")
    proposed, details = {}, []
    try:
        for layer in range(18):
            name = f"model.layers.{layer}.mlp.down_proj"; checkpoint = WORK / f"layer_{layer:02d}.json"
            if checkpoint_matches(checkpoint, config_hash):
                payload = json.loads(checkpoint.read_text()); detail = payload["detail"]
            else:
                groups = {g: captured[layer][g] for g in pilot.cal.GROUPS}
                result = joint.calibrate_layer_joint(source.weight(name), frozen[name], groups,
                                                      sx_js=SX_JS, s10_is=S10_IS)
                detail = serialize_layer(layer, result)
                if detail["selected_score"] > detail["old_score"] + joint.LOCAL_REGRESSION_TOL:
                    raise RuntimeError(f"local regression at layer {layer}")
                base.write_json(checkpoint, dict(configuration_fingerprint=config_hash, detail=detail))
            proposed[name] = spec_from_detail(frozen[name], detail); details.append(detail)
            progress(f"LAYER {layer}/18 DONE")
    except BaseException:
        progress("CALIBRATION_FAILED"); raise
    progress("JOINT_CALIBRATION_DONE")
    del captured
    base.write_csv(OUT / "calibration_layers.csv", [{k:v for k,v in d.items() if k not in ("multiplier","shift")} for d in details])
    changes = [d["relative_local_nmse_change"] for d in details]
    calibration_summary = dict(changed_sx_layers=sum(d["changed_sx"] for d in details),
        changed_s10_layers=sum(d["changed_s10"] for d in details),
        sx_boundary_hits=sum(d["sx_boundary_hit"] for d in details),
        s10_boundary_hits=sum(d["s10_boundary_hit"] for d in details),
        boundary_hit_layers=sum(d["sx_boundary_hit"] or d["s10_boundary_hit"] for d in details),
        mean_relative_local_nmse_change=float(np.mean(changes)),
        median_relative_local_nmse_change=float(np.median(changes)),
        best_layer_improvement=-min(changes), worst_layer_change=max(changes),
        total_feasible_candidates=sum(d["feasible_candidate_count"] for d in details),
        total_rejected_candidates=sum(d["rejected_candidate_count"] for d in details),
        total_unique_acc=sum(d["unique_acc_count"] for d in details))
    base.write_json(OUT / "calibration_summary.json", calibration_summary)
    with (ROOT / "diagnostics/absmax_joint_calibration_pilot/calibration_layers.csv").open(newline="") as handle:
        pilot_rows = {int(r["layer"]): r for r in csv.DictReader(handle)}
    stable_rows = [stability_row(pilot_rows[i], details[i]) for i in range(18)]
    stable_summary = stability_summary(stable_rows)
    base.write_csv(OUT / "parameter_stability.csv", stable_rows)
    base.write_json(OUT / "parameter_stability_summary.json", stable_summary)
    base.write_json(OUT / "proposed_parameters.json", {name: dict(sx_base=spec["sx"],
        s10=spec["s10"], multiplier=spec["params"]["multiplier"].tolist(),
        shift=spec["params"]["shift"].tolist()) for name,spec in proposed.items()})
    progress("STABILITY_ANALYSIS_DONE")
    pilot_specs = load_pilot_specs(frozen)
    specs = dict(old_absmax=frozen, pilot_joint_absmax=pilot_specs,
                 stability_joint_absmax=proposed)
    validation, val_skipped = pilot.select_examples(source, "validation", VAL_POSITIONS)
    if len(validation) != 12: raise RuntimeError("expected 12 fixed validation chats")
    progress("E2E_VALIDATION_START")
    records=[]; last=time.monotonic()
    try:
        for number,(position,example) in enumerate(zip(VAL_POSITIONS, validation), 1):
            checkpoint=WORK/f"validation_{number:02d}.json"
            if checkpoint_matches(checkpoint, config_hash):
                record=json.loads(checkpoint.read_text())["record"]
            else:
                record=validate_conversation(source,specs,example,position)
                base.write_json(checkpoint,dict(configuration_fingerprint=config_hash,record=record))
            records.append(record); now=time.monotonic()
            progress(f"E2E_BATCH {number}/12 DONE targets=32 total_targets={number*32}/384 "
                     f"batch_sec={now-last:.3f} elapsed_sec={now-START:.3f}"); last=now
    except BaseException:
        progress("E2E_FAILED"); raise
    modes=[dict(mode=m,**pilot.response.merged([r["scores"][m] for r in records])) for m in MODES]
    if any(m["tokens"] != 384 for m in modes): raise RuntimeError("expected 384 aligned targets")
    per=[]
    for record in records:
        item=dict(index=record["index"],source_index=record["source_index"],targets=32)
        for mode in MODES:
            metrics=pilot.response.merged([record["scores"][mode]])
            item.update({f"{mode}_{k}":metrics[k] for k in ("nll","kl","nmse","top1")})
        per.append(item)
    old,pilot_mode,stable=modes[1:]
    deltas=dict(old_to_stability=relative_deltas(old,stable),
                pilot_to_stability=relative_deltas(pilot_mode,stable))
    gate=joint.e2e_gate(old,stable)
    summary=dict(modes=modes,deltas=deltas,gate=gate,
                 classification="STABILITY_PASS" if gate["pass"] else "STABILITY_REGRESSION")
    pilot.fidelity.finite_tree(summary)
    base.write_json(OUT/"validation_summary.json",summary);base.write_csv(OUT/"validation_per_conversation.csv",per)
    after=frozen_fp()
    if before != after: raise RuntimeError("frozen artifacts changed")
    verification=dict(frozen_before=before,frozen_after=after,frozen_unchanged=True,
        quantized_modules=sorted(expected),calibration_positions=CAL_POSITIONS,
        calibration_response_limit=CAL_RESPONSE_LIMIT,reservoir_capacity=RESERVOIR_CAPACITY,
        seed=SEED,sx_js=SX_JS,s10_is=S10_IS,validation_positions=VAL_POSITIONS,
        validation_targets=384,selector_identity=joint.select_absmax is pilot.select_absmax,
        calibration_policy_identity="absmax_joint_calibration.calibrate_layer_joint",
        local_no_regression=all(d["selected_score"]<=d["old_score"]+joint.LOCAL_REGRESSION_TOL for d in details),
        effective_shifts_feasible=all(0<=d["effective_shift_min"]<=d["effective_shift_max"]<=31 for d in details),
        independent_validation_caches=True,test_command=test_command,test_return_code=tested.returncode,
        pilot_artifact_fingerprint=pilot_hash,
        stability_parameter_fingerprint=pilot.compare.fingerprint(proposed),
        configuration_fingerprint=config_hash,calibration_skipped=cal_skipped,
        validation_skipped=val_skipped,elapsed_sec=time.monotonic()-START)
    base.write_json(OUT/"verification.json",verification)
    lines=["# Joint absmax calibration stability", "",
        f"Calibration: 16 fixed chats, response limit 32, reservoir 64/group, seed {SEED}; 45 fixed candidates/layer.",
        "All layers satisfy selected_score <= old_score + 1e-12; no boundary expansion.", "",
        "## Parameter stability", "", json.dumps(stable_summary,indent=2), "",
        "## 384-target E2E", "",
        "| Mode | NLL | PPL | KL | Logits NMSE | Cosine | Top1 | FP-top1 in top5 | Top5 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in modes: lines.append("| "+m["mode"]+" | "+" | ".join(f"{m[k]:.9g}" for k in
        ("nll","ppl","kl","nmse","cosine","top1","in5","overlap"))+" |")
    lines += ["", "Old to stability: `"+json.dumps(deltas["old_to_stability"])+"`",
              "Pilot joint to stability: `"+json.dumps(deltas["pilot_to_stability"])+"`",
              "", "E2E gate: `"+json.dumps(gate)+"`", "Classification: **"+summary["classification"]+"**.",
              "This is a descriptive fixed-scope stability experiment on 12 conversations; it does not establish statistical significance or optimality."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    progress("ARTIFACT_WRITE_DONE");progress("VERIFY_DONE");progress("RUN_COMPLETE")


if __name__ == "__main__":
    try: main()
    except BaseException:
        progress("RUN_FAILED")
        raise
