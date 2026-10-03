#!/usr/bin/env python3
"""Confirmation-only evaluation of the frozen L13/L16/L17 absmax policy."""
import csv
import gzip
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
import absmax_joint_calibration as joint
import calibrate_cached_decode_ptq as cal
import run_absmax_cross_layer_calibration as cross

pilot, final, base, row = cross.pilot, cross.final, cross.base, cross.row
OUT = ROOT / "diagnostics/frozen_cross_layer_confirmation"
WORK = OUT / "work"
PROGRESS = ROOT / "frozen_cross_layer_confirmation_progress.txt"
SELECTED_PATH = ROOT / "diagnostics/absmax_cross_layer_calibration/selected_configuration.json"
PREVIOUS_DIRS = [ROOT / "diagnostics" / name for name in (
    "final_downproj_ptq", "absmax_calibration_pilot",
    "absmax_joint_calibration_pilot", "absmax_joint_stability",
    "absmax_cross_layer_calibration")]
CHANGED_LAYERS = {13, 16, 17}
SEED = 20261004
TARGETS = 32
BOOTSTRAP_REPLICATES = 2000
START = time.monotonic()


def progress(message):
    with PROGRESS.open("a") as handle:
        handle.write(message + "\n")
        handle.flush()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_keys(record):
    result = set()
    for key in ("anchor_id", "response_id", "tree_id"):
        if record.get(key):
            result.add(f"{key}:{record[key]}")
    return result


def stable_id(record):
    values = [str(record.get(k, "")) for k in ("tree_id", "anchor_id", "response_id")]
    if any(values):
        return "oasst:" + ":".join(values)
    return str(record["stable_id"])


def is_previously_used(record, exclusion_keys):
    """Stable IDs are global; split-relative source_index is deliberately ignored."""
    return bool(stable_keys(record) & set(exclusion_keys))


def deterministic_unseen_order(examples, seed=SEED):
    return sorted(examples, key=lambda e: (hashlib.sha256(
        f"{seed}:{stable_id(e)}".encode()).digest(), stable_id(e)))


def local_dataset_policy(examples):
    if len(examples) >= 32:
        return examples[:32], False
    if len(examples) >= 24:
        return examples, False
    return [], True


def wikitext_windows(token_ids, count=32, prompt=128, targets=32):
    width = prompt + targets
    if len(token_ids) < count * width:
        raise ValueError("WikiText-2 test does not contain enough tokens")
    result = []
    for index in range(count):
        start = index * width
        ids = list(map(int, token_ids[start:start + width]))
        fingerprint = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()
        result.append(dict(index=index, source_index=index,
            stable_id=f"wikitext2:test:{start}:{start+width}:{fingerprint}",
            prompt_ids=ids[:prompt], targets=ids[prompt:], window_start=start,
            window_end=start + width, token_sha256=fingerprint,
            source_partition="wikitext2_test"))
    return result


def confirmation_gate(old, new):
    return joint.e2e_gate(old, new)


def strict_improvement(old, new):
    return new["kl"] < old["kl"] and new["nmse"] <= old["nmse"] and new["top1"] >= old["top1"]


def checkpoint_valid(payload, keys):
    return payload.get("checkpoint_keys") == keys


def bootstrap(records, replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("kl_relative", "nmse_relative", "top1_pp", "ppl_relative")}
    favorable = dict(kl=0, nmse=0, top1=0)
    n = len(records)
    for _ in range(replicates):
        picked = rng.integers(0, n, n)
        old = pilot.response.merged([records[i]["scores"]["old_absmax"] for i in picked])
        new = pilot.response.merged([records[i]["scores"]["frozen_cross_layer_absmax"] for i in picked])
        values = cross.stability.relative_deltas(old, new)
        for key in samples:
            samples[key].append(float(values[key]))
        favorable["kl"] += new["kl"] < old["kl"]
        favorable["nmse"] += new["nmse"] < old["nmse"]
        favorable["top1"] += new["top1"] >= old["top1"]
    intervals = {key: dict(mean=float(np.mean(value)),
        low=float(np.percentile(value, 2.5)), high=float(np.percentile(value, 97.5)))
        for key, value in samples.items()}
    return dict(seed=seed, replicates=replicates, unit="sequence",
        intervals=intervals, favorable_fraction={k: v / replicates for k, v in favorable.items()})


def protected_tree_fingerprint(paths):
    h = hashlib.sha256()
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "work" not in path.relative_to(root).parts:
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def frozen_fingerprint(frozen, selected_sha):
    return dict(decisions_sha256=base.sha256_file(final.OUT / "decisions.json"),
        baseline_parameters_sha256=pilot.compare.fingerprint(frozen),
        selected_configuration_sha256=selected_sha,
        protected_artifacts_sha256=protected_tree_fingerprint(PREVIOUS_DIRS))


def load_selected_specs(frozen):
    document = json.loads(SELECTED_PATH.read_text())
    expected = {f"model.layers.{i}.mlp.down_proj" for i in range(18)}
    if document.get("changed_layers") != 3 or set(document.get("layers", {})) != expected:
        raise RuntimeError("selected configuration module/count invariant failed")
    specs = {}
    changed = set()
    for name in sorted(expected):
        old, item = frozen[name], document["layers"][name]
        layer = int(item["layer"])
        params = cross.params_for(old, float(item["sx"]), float(item["s10"]),
                                  item["multiplier"], item["shift"])
        spec = dict(old, sx=float(item["sx"]), s10=float(item["s10"]), params=params)
        np.testing.assert_array_equal(spec["wq"], old["wq"])
        differs = (spec["sx"] != old["sx"] or spec["s10"] != old["s10"] or
                   not np.array_equal(spec["params"]["multiplier"], old["params"]["multiplier"]) or
                   not np.array_equal(spec["params"]["shift"], old["params"]["shift"]))
        if differs:
            changed.add(layer)
        elif item.get("source") != "old":
            raise RuntimeError(f"unchanged layer {layer} is not marked old")
        specs[name] = spec
    if changed != CHANGED_LAYERS:
        raise RuntimeError(f"selected changed layers differ: {sorted(changed)}")
    return specs, document


def recursively_collect_stable(value, output):
    if isinstance(value, dict):
        output.update(stable_keys(value))
        for child in value.values():
            recursively_collect_stable(child, output)
    elif isinstance(value, list):
        for child in value:
            recursively_collect_stable(child, output)


def audit_previous_usage(calibration_pairs, validation_pairs):
    keys = set()
    entries = []
    def add(pair, partition, reason):
        keys.update(stable_keys(pair))
        entries.append(dict(stable_id=stable_id(pair), source_partition=partition,
            source_index=pair.get("index"), anchor_id=pair.get("anchor_id"),
            response_id=pair.get("response_id"), tree_id=pair.get("tree_id"), reason=reason))
    # The deployment PTQ protocol used the first 32 recoverable calibration chats
    # and all 24 recoverable validation chats. Later experiments are subsets thereof.
    for pair in calibration_pairs[:32]:
        add(pair, "calibration", "final_downproj_ptq calibration capture")
    for pair in validation_pairs:
        add(pair, "validation", "final_downproj_ptq/evaluation")
    artifact_keys = set()
    for directory in PREVIOUS_DIRS:
        for path in directory.rglob("*.json"):
            if "work" in path.relative_to(directory).parts or "progress" in path.relative_to(directory).parts:
                continue
            try:
                recursively_collect_stable(json.loads(path.read_text()), artifact_keys)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    keys.update(artifact_keys)
    return dict(stable_keys=sorted(keys), entries=entries,
        artifact_stable_keys=sorted(artifact_keys), audited_directories=[str(p.relative_to(ROOT)) for p in PREVIOUS_DIRS])


def discover_local_examples(source, audit):
    all_examples = []
    skipped = []
    for partition in ("calibration", "validation"):
        pairs, pair_skipped = cal.build_pairs(partition)
        examples, token_skipped = cal.tokenize_pairs(source, pairs)
        for example in examples:
            example["source_partition"] = partition
            example["source_index"] = example["index"]
        all_examples.extend(examples)
        skipped.extend([dict(source_partition=partition, **x) for x in pair_skipped + token_skipped])
    exclusion = set(audit["stable_keys"])
    fresh = [e for e in all_examples if not is_previously_used(e, exclusion) and len(e["targets"]) >= TARGETS]
    return all_examples, deterministic_unseen_order(fresh), skipped


def load_wikitext_fallback(source):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    token_ids = source.tokenizer(text, add_special_tokens=False)["input_ids"]
    return wikitext_windows(token_ids)


def observed_selector(stats):
    def select(x, sx, shift):
        q, k, info = pilot.select_absmax(x, sx, shift)
        effective = np.asarray(shift, np.int64)[None, :] - np.asarray(k, np.int64)[:, None]
        lo, hi = int(effective.min()), int(effective.max())
        if lo < 0 or hi > 31:
            raise RuntimeError("observed effective shift outside [0,31]")
        stats["min"] = lo if stats["min"] is None else min(stats["min"], lo)
        stats["max"] = hi if stats["max"] is None else max(stats["max"], hi)
        return q, k, info
    return select


def evaluate_sequence(source, specs, example, selector_stats):
    modes = {"FP": None, "old_absmax": specs[0], "frozen_cross_layer_absmax": specs[1]}
    scores = {mode: pilot.response.Scores() for mode in modes}
    caches, logits = {}, {}
    for mode, policy in modes.items():
        if mode == "FP":
            logits[mode], _, caches[mode] = final.cached.run_fp(source, example["prompt_ids"], None, layers=False)
        else:
            with patch.object(row, "select_rows", observed_selector(selector_stats)):
                logits[mode], _, caches[mode] = final.cached.run_mode(source, policy, example["prompt_ids"], None, layers=False)
    if len({id(cache) for cache in caches.values()}) != 3:
        raise RuntimeError("KV caches are not independent")
    for t, target in enumerate(example["targets"][:TARGETS]):
        if t:
            token = [example["targets"][t - 1]]
            logits["FP"], _, caches["FP"] = final.cached.run_fp(source, token, caches["FP"], layers=False)
            for mode in ("old_absmax", "frozen_cross_layer_absmax"):
                with patch.object(row, "select_rows", observed_selector(selector_stats)):
                    logits[mode], _, caches[mode] = final.cached.run_mode(source, modes[mode], token, caches[mode], layers=False)
        for mode in modes:
            scores[mode].add(source.torch, logits[mode], logits["FP"], [target])
    return dict(stable_id=stable_id(example), source_partition=example["source_partition"],
        source_index=example.get("source_index"), targets=TARGETS,
        scores={mode: score.raw() for mode, score in scores.items()})


def metrics_table(records):
    return [dict(mode=mode, **pilot.response.merged([r["scores"][mode] for r in records]))
            for mode in ("FP", "old_absmax", "frozen_cross_layer_absmax")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("")
    progress("RUN_START")
    progress("UNIT_TEST_START")
    tests = [sys.executable, "-m", "unittest",
        "tests.test_frozen_cross_layer_confirmation",
        "tests.test_absmax_cross_layer_calibration", "tests.test_absmax_joint_stability",
        "tests.test_absmax_joint_calibration", "tests.test_k_selector_compare",
        "tests.test_finalize_downproj_ptq"]
    env = dict(os.environ, PYTHONPYCACHEPREFIX="/tmp/frozen-cross-layer-confirmation")
    with (OUT / "unit_test.log").open("w") as log:
        tested = subprocess.run(tests, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if tested.returncode:
        progress("UNIT_TEST_FAILED")
        raise RuntimeError("unit tests failed")
    progress("UNIT_TEST_DONE")
    progress("CONFIG_FREEZE_START")
    source, _, _, previous, _, _ = final.load_source_and_policies()
    frozen, _ = final.restore_final(source, previous)
    expected = {f"model.layers.{i}.mlp.down_proj" for i in range(18)}
    if set(frozen) != expected:
        raise RuntimeError("unexpected quantized module set")
    selected_sha = base.sha256_file(SELECTED_PATH)
    selected_specs, selected_document = load_selected_specs(frozen)
    selected_parameters_sha = pilot.compare.fingerprint(selected_specs)
    before = frozen_fingerprint(frozen, selected_sha)
    progress("CONFIG_FREEZE_DONE changed_layers=3")

    progress("USAGE_AUDIT_START")
    calibration_pairs, calibration_skipped = cal.build_pairs("calibration")
    validation_pairs, validation_skipped = cal.build_pairs("validation")
    audit = audit_previous_usage(calibration_pairs, validation_pairs)
    base.write_json(OUT / "previous_usage_manifest.json", dict(**audit,
        calibration_pair_count=len(calibration_pairs), validation_pair_count=len(validation_pairs),
        calibration_skipped=calibration_skipped, validation_skipped=validation_skipped))
    progress("USAGE_AUDIT_DONE")

    progress("DATASET_DISCOVERY_START")
    all_examples, fresh, discovery_skipped = discover_local_examples(source, audit)
    chosen, fallback = local_dataset_policy(fresh)
    pool_manifest = dict(total_local_examples=len(all_examples), previously_used_examples=sum(
        is_previously_used(e, audit["stable_keys"]) for e in all_examples),
        fresh_eligible_examples=len(fresh), skipped=discovery_skipped,
        fresh=[dict(stable_id=stable_id(e), source_partition=e["source_partition"],
            source_index=e["source_index"], anchor_id=e.get("anchor_id"), response_id=e.get("response_id"),
            tree_id=e.get("tree_id"), response_targets=len(e["targets"])) for e in fresh])
    base.write_json(OUT / "unseen_pool_manifest.json", pool_manifest)
    progress(f"DATASET_DISCOVERY_DONE fresh_local={len(fresh)}")
    if fallback:
        progress("DATASET_FALLBACK_START")
        try:
            chosen = load_wikitext_fallback(source)
        except Exception as exc:
            progress("CONFIRMATION_DATASET_UNAVAILABLE")
            raise RuntimeError("WikiText-2 fallback unavailable") from exc
        dataset_type = "wikitext2_test_fallback"
    else:
        dataset_type = "local_unseen"
    dataset_entries = [dict(stable_id=stable_id(e), source_partition=e["source_partition"],
        source_index=e.get("source_index"), anchor_id=e.get("anchor_id"), response_id=e.get("response_id"),
        tree_id=e.get("tree_id"), targets=TARGETS,
        **({k: e[k] for k in ("window_start", "window_end", "token_sha256") if k in e})) for e in chosen]
    dataset_manifest = dict(seed=SEED, dataset_type=dataset_type, sequences=len(chosen),
        targets=len(chosen) * TARGETS, examples=dataset_entries)
    dataset_manifest["fingerprint"] = digest(dataset_manifest)
    base.write_json(OUT / "confirmation_dataset_manifest.json", dataset_manifest)
    progress(f"DATASET_SELECTED type={dataset_type} sequences={len(chosen)} targets={len(chosen)*TARGETS}")

    selector_stats = {"min": None, "max": None}
    records = []
    progress("CONFIRMATION_START")
    last = time.monotonic()
    for number, example in enumerate(chosen, 1):
        identity = stable_id(example)
        keys = {mode: digest(dict(baseline=before["baseline_parameters_sha256"],
            selected=selected_parameters_sha, dataset=dataset_manifest["fingerprint"],
            sequence=identity, mode=mode)) for mode in ("FP", "old_absmax", "frozen_cross_layer_absmax")}
        checkpoint = WORK / f"sequence_{number:03d}.json"
        if checkpoint.exists() and checkpoint_valid(json.loads(checkpoint.read_text()), keys):
            record = json.loads(checkpoint.read_text())["record"]
        else:
            record = evaluate_sequence(source, (frozen, selected_specs), example, selector_stats)
            base.write_json(checkpoint, dict(checkpoint_keys=keys, record=record))
        records.append(record)
        now = time.monotonic()
        progress(f"SEQUENCE {number}/{len(chosen)} DONE targets=32 total_targets={number*32} "
                 f"batch_sec={now-last:.3f} elapsed_sec={now-START:.3f}")
        last = now
    aggregate = metrics_table(records)
    expected_targets = len(chosen) * TARGETS
    if any(row_["tokens"] != expected_targets for row_ in aggregate):
        raise RuntimeError("aligned target count mismatch")
    old, new = aggregate[1], aggregate[2]
    deltas = cross.stability.relative_deltas(old, new)
    gate = confirmation_gate(old, new)
    strict = strict_improvement(old, new)

    per_sequence = []
    for record in records:
        item = dict(stable_id=record["stable_id"], source_partition=record["source_partition"],
                    source_index=record.get("source_index"), targets=record["targets"])
        for mode in ("old_absmax", "frozen_cross_layer_absmax"):
            metric = pilot.response.merged([record["scores"][mode]])
            item.update({f"{mode}_{key}": metric[key] for key in ("kl", "nmse", "top1", "nll")})
        per_sequence.append(item)
    base.write_csv(OUT / "confirmation_per_sequence.csv", per_sequence)
    progress("BOOTSTRAP_START")
    boot = bootstrap(records)
    base.write_json(OUT / "bootstrap_summary.json", boot)
    progress("BOOTSTRAP_DONE")

    summary = dict(modes=aggregate, deltas=deltas, gate=gate,
        classification="CONFIRMATION_PASS" if gate["pass"] else "CONFIRMATION_REGRESSION",
        relation="STRICT_IMPROVEMENT" if strict else "TRADEOFF")
    base.write_json(OUT / "confirmation_summary.json", summary)
    after = frozen_fingerprint(frozen, base.sha256_file(SELECTED_PATH))
    if before != after or base.sha256_file(SELECTED_PATH) != selected_sha:
        raise RuntimeError("frozen/protected artifacts changed")
    if pilot.compare.fingerprint(selected_specs) != selected_parameters_sha:
        raise RuntimeError("selected parameters mutated")
    overlap = any(is_previously_used(e, audit["stable_keys"]) for e in chosen if dataset_type == "local_unseen")
    verification = dict(old_baseline_fingerprint=before["baseline_parameters_sha256"],
        selected_configuration_fingerprint=selected_parameters_sha,
        selected_configuration_sha256=selected_sha, changed_layers=sorted(CHANGED_LAYERS),
        frozen_before=before, frozen_after=after, frozen_unchanged=True,
        quantized_modules=sorted(expected), selector_identity=pilot.select_absmax is cross.pilot.select_absmax,
        effective_shifts_feasible=(selector_stats["min"] is None or
            0 <= selector_stats["min"] <= selector_stats["max"] <= 31),
        observed_effective_shift_min=selector_stats["min"], observed_effective_shift_max=selector_stats["max"],
        independent_kv_caches=True, dataset_type=dataset_type,
        dataset_manifest_fingerprint=dataset_manifest["fingerprint"], total_sequences=len(chosen),
        total_aligned_targets=expected_targets, previous_data_overlap=overlap,
        search_performed=False, calibration_performed=False, parameter_changes_during_run=False,
        bootstrap_seed=SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        test_command=tests, test_return_code=tested.returncode, elapsed_sec=time.monotonic()-START)
    if overlap or not verification["effective_shifts_feasible"]:
        raise RuntimeError("confirmation verification failed")
    base.write_json(OUT / "verification.json", verification)
    consistent = sum(r["frozen_cross_layer_absmax_kl"] < r["old_absmax_kl"] for r in per_sequence)
    lines = ["# Frozen cross-layer absmax confirmation", "",
        "This was confirmation only. Layers 13, 16, and 17 were frozen before dataset inspection; no parameter was tuned on confirmation data.",
        f"Dataset: `{dataset_type}`; {len(chosen)} sequences and {expected_targets} aligned targets. " +
        ("The fallback is intentionally out-of-domain." if dataset_type != "local_unseen" else "All local examples were excluded by stable identity if previously used."), "",
        "| Mode | NLL | PPL | KL | Logits NMSE | MSE | MAE | Cosine | Flattened cosine | Top1 | FP top1 in Q top5 | Top5 overlap | FP energy | Quant energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for metric in aggregate:
        lines.append("| " + metric["mode"] + " | " + " | ".join(f"{metric[k]:.9g}" for k in
            ("nll", "ppl", "kl", "nmse", "mse", "mae", "cosine", "flattened_cosine",
             "top1", "in5", "overlap", "fp_energy", "quant_energy")) + " |")
    lines += ["", f"Old to frozen-cross deltas: `{json.dumps(deltas)}`.",
        f"Safety gate: `{json.dumps(gate)}` — **{summary['classification']}**.",
        f"Strict comparison: **{summary['relation']}**.",
        f"Bootstrap (sequence unit, {BOOTSTRAP_REPLICATES} replicates): `{json.dumps(boot)}`.",
        f"Cross-layer KL was lower for {consistent}/{len(chosen)} sequences.",
        "No statistical significance or optimality is claimed; parameters were not changed after evaluation."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    progress("ARTIFACT_WRITE_DONE")
    progress("FINAL_VERIFY_DONE")
    progress("RUN_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        progress("RUN_FAILED")
        raise
