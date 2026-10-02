#!/usr/bin/env python3
"""Fixed small joint pilot. Outputs are isolated; no sequential search is used."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import absmax_joint_calibration as joint
import pilot_absmax_calibration as pilot

ROOT = pilot.ROOT
OUT = ROOT / "diagnostics/absmax_joint_calibration_pilot"
PROGRESS = ROOT / "calibration_progress.txt"
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as handle:
        handle.write(message + "\n")
        handle.flush()


class BatchLog:
    """Forward existing helper logs while flushing per-conversation progress."""
    def __init__(self, stream):
        self.stream, self.pending = stream, ""
        self.last = time.monotonic()

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            parts = line.split()
            if len(parts) == 3 and parts[0] in ("CAPTURE", "VALIDATE"):
                number, count = map(int, parts[1:])
                now = time.monotonic()
                if parts[0] == "CAPTURE":
                    progress(f"CAPTURE {number}/{count}")
                else:
                    progress(f"E2E_BATCH {number}/{count} DONE targets=24 "
                             f"total_targets={number*24}/144 batch_sec={now-self.last:.3f} "
                             f"elapsed_sec={now-START:.3f}")
                self.last = now
        return len(text)

    def flush(self):
        self.stream.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "validation_summary.json").exists():
        raise RuntimeError("joint pilot results already exist")
    PROGRESS.write_text("")
    progress("RUN_START")
    progress("UNIT_TEST_START")
    command = [sys.executable, "-m", "unittest", "tests.test_absmax_joint_calibration",
               "tests.test_pilot_absmax_calibration", "tests.test_k_selector_fidelity_384",
               "tests.test_k_selector_compare", "tests.test_finalize_downproj_ptq"]
    env = dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/absmax-joint-calibration")
    with (OUT / "unit_test.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        progress("UNIT_TEST_FAILED")
        raise RuntimeError("unit tests failed; calibration was not started")
    progress("UNIT_TEST_DONE")
    final, base = pilot.final, pilot.base
    source, _, _, previous, _, _ = final.load_source_and_policies()
    frozen, _ = final.restore_final(source, previous)
    expected = {f"model.layers.{i}.mlp.down_proj" for i in range(18)}
    if set(frozen) != expected:
        raise RuntimeError("unexpected quantized modules")
    protected = [final.OUT, pilot.OUT, pilot.compare.OUT, pilot.fidelity.OUT,
                 ROOT / "diagnostics/k_selector_fidelity_384"]
    def fingerprint():
        return dict(decisions_sha256=base.sha256_file(final.OUT / "decisions.json"),
                    parameters_sha256=pilot.compare.fingerprint(frozen),
                    artifacts_sha256=pilot.fingerprint_tree(protected))
    before = fingerprint()
    examples, skipped = pilot.select_examples(source, "calibration", pilot.CAL_POSITIONS)
    if len(examples) != 6:
        raise RuntimeError("expected six calibration chats")
    progress("CAPTURE_START")
    with contextlib.redirect_stdout(BatchLog(sys.stdout)):
        capture = pilot.capture_absmax(source, frozen, examples)
    progress("JOINT_CALIBRATION_START")
    proposed, details = {}, []
    for layer in range(18):
        name = f"model.layers.{layer}.mlp.down_proj"
        old = frozen[name]
        groups = {g: capture[layer][g][0] for g in pilot.cal.GROUPS}
        result = joint.calibrate_layer_joint(source.weight(name), old, groups,
                    sx_js=pilot.SX_JS, s10_is=pilot.S10_IS)
        selected, meta = result["selected"], result["metadata"]
        if selected["score"] > result["old"]["score"] + joint.LOCAL_REGRESSION_TOL:
            raise RuntimeError(f"local regression at layer {layer}")
        proposed[name] = joint.proposed_spec(old, selected)
        details.append(dict(layer=layer, old_sx=old["sx"], selected_sx=selected["sx"],
            sx_j=selected["sx_j"], old_s10=old["s10"], selected_s10=selected["s10"],
            s10_i=selected["s10_i"], **meta,
            prefill_nmse=selected["prefill_nmse"], decode_nmse=selected["decode_nmse"],
            worst_nmse=selected["worst"], clip_rate=selected["clip_rate"],
            effective_shift_min=selected["effective_shift_min"],
            effective_shift_max=selected["effective_shift_max"]))
        progress(f"LAYER {layer}/18 DONE")
    if set(proposed) != expected:
        raise RuntimeError("unexpected proposed modules")
    changes = [d["relative_local_nmse_change"] for d in details]
    aggregate = dict(changed_sx_layers=sum(d["changed_sx"] for d in details),
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
    base.write_csv(OUT / "calibration_layers.csv", details)
    base.write_json(OUT / "calibration_summary.json", aggregate)
    base.write_json(OUT / "proposed_parameters.json", {name: dict(
        sx_base=s["sx"], s10=s["s10"], multiplier=s["params"]["multiplier"].tolist(),
        shift=s["params"]["shift"].tolist()) for name, s in proposed.items()})
    del capture
    validation, val_skipped = pilot.select_examples(source, "validation", pilot.VAL_POSITIONS)
    if len(validation) != 6:
        raise RuntimeError("expected six validation chats")
    progress("E2E_VALIDATION_START")
    with contextlib.redirect_stdout(BatchLog(sys.stdout)):
        records = pilot.validate(source, frozen, proposed, validation)
    modes, per = [], []
    for mode in pilot.MODES:
        metrics = pilot.response.merged([r["scores"][mode] for r in records])
        if metrics["tokens"] != 144:
            raise RuntimeError("expected exactly 144 aligned targets")
        modes.append(dict(mode="joint_calibrated_absmax" if mode == "calibrated_absmax" else mode, **metrics))
    for r in records:
        item = dict(index=r["index"], source_index=r["source_index"], targets=r["targets"])
        for mode in pilot.MODES:
            label = "joint_calibrated_absmax" if mode == "calibrated_absmax" else mode
            metrics = pilot.response.merged([r["scores"][mode]])
            item.update({f"{label}_{k}": metrics[k] for k in ("nll", "kl", "nmse", "top1")})
        per.append(item)
    old, new = modes[1:]
    delta = {f"{k}_relative_percent": 100*(new[k]-old[k])/old[k] for k in ("kl", "nmse", "ppl")}
    delta["top1_pp"] = 100*(new["top1"]-old["top1"])
    gate = joint.e2e_gate(old, new)
    strong = gate["pass"] and (delta["kl_relative_percent"] <= -3 or
                delta["nmse_relative_percent"] <= -3 or delta["top1_pp"] >= 1)
    after = fingerprint()
    if before != after:
        raise RuntimeError("frozen state changed")
    previous_summary = json.loads((pilot.OUT / "validation_summary.json").read_text())
    summary = dict(modes=modes, deltas=delta, gate=gate,
        classification="JOINT_PASS" if gate["pass"] else "JOINT_REGRESSION",
        descriptive_label="STRONG_IMPROVEMENT" if strong else ("SAFE_NEUTRAL" if gate["pass"] else "REGRESSION"),
        previous_sequential_stored=previous_summary)
    pilot.fidelity.finite_tree(summary)
    base.write_json(OUT / "validation_summary.json", summary)
    base.write_csv(OUT / "validation_per_conversation.csv", per)
    verification = dict(before=before, after=after, frozen_unchanged=True,
        quantized_modules=sorted(expected), calibration_positions=pilot.CAL_POSITIONS,
        validation_positions=pilot.VAL_POSITIONS, calibration_response_limit=16,
        reservoir_capacity=32, seed=pilot.SEED, validation_targets=144,
        sx_js=pilot.SX_JS, s10_is=pilot.S10_IS, capture_selector="select_absmax",
        calibration_policy="absmax_joint_calibration.calibrate_layer_joint",
        selector_identity=joint.select_absmax is pilot.select_absmax,
        local_no_regression=True, independent_validation_caches=True,
        effective_shifts_feasible=all(0 <= d["effective_shift_min"] <= d["effective_shift_max"] <= 31 for d in details),
        calibration_skipped=skipped, validation_skipped=val_skipped,
        test_command=command, tests_returncode=result.returncode if hasattr(result, "returncode") else 0,
        command=sys.argv, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        elapsed_sec=time.monotonic()-START)
    base.write_json(OUT / "verification.json", verification)
    lines = ["# Joint absmax calibration pilot", "", "Calibration-only joint output-domain selection; fixed small subsets and ranges.",
        "", "| Mode | NLL | PPL | KL | Logits NMSE | Cosine | Top1 | In top5 | Overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in modes + [dict(m, mode="previous sequential (stored): " + m["mode"])
                     for m in previous_summary["modes"] if m["mode"] == "calibrated_absmax"]:
        lines.append("| " + m["mode"] + " | " + " | ".join(str(m[k]) for k in
                     ("nll", "ppl", "kl", "nmse", "cosine", "top1", "in5", "overlap")) + " |")
    lines += ["", json.dumps(aggregate, indent=2), "", json.dumps(delta, indent=2), "",
              summary["classification"] + ": " + summary["descriptive_label"],
              "All 18 layers satisfy the local no-regression runtime invariant. E2E does not select parameters.",
              "Previous sequential metrics are stored reference only. No automatic expansion or further calibration."]
    (OUT / "report.md").write_text("\n".join(lines)+"\n")
    progress("ARTIFACT_WRITE_DONE")
    progress("RUN_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        progress("RUN_FAILED")
        raise
