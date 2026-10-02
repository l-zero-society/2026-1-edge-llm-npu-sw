"""Reusable output-domain joint search; no model loading, execution driver or I/O.

Callers supply calibration rows only. Caches belong to one search by default;
their keys include both activation codes and weights for safe explicit reuse.
"""
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_k_selectors_e2e import Profile, base, select_absmax

LOCAL_REGRESSION_TOL = 1e-12
GROUPS = ("prefill", "decode")


def _fingerprint(array):
    a = np.ascontiguousarray(array)
    return (a.shape, a.dtype.str, hashlib.sha256(a.tobytes()).digest())


def output_scores(out, ref, labels):
    """Pooled NMSE within groups, equal weight across available groups.

    A zero-energy group has zero NMSE only for exact reconstruction, otherwise
    infinite NMSE (the candidate is rejected rather than hiding its error).
    """
    scores = {}
    for group in GROUPS:
        mask = labels == group
        if mask.any():
            energy = float(np.square(ref[mask]).sum())
            error = float(np.square(out[mask] - ref[mask]).sum())
            scores[group] = error / energy if energy else (0.0 if error == 0 else math.inf)
    return dict(prefill_nmse=scores.get("prefill"), decode_nmse=scores.get("decode"),
                score=float(np.mean(list(scores.values()))), worst=max(scores.values()))


def evaluate_joint_candidate(x, labels, ref, wq, sw, sx, s10, *,
                             sx_j, s10_i, acc_cache):
    """Evaluate one pair using its own static shifts and exact absmax selector.

    Invalid numerical candidates return feasible=False and a rejection_reason.
    Malformed input arrays raise ValueError. No input/spec is mutated.
    """
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    labels, wq, sw = np.asarray(labels), np.asarray(wq), np.asarray(sw)
    if (x.ndim != 2 or not all(x.shape) or wq.ndim != 2 or
            wq.shape[1] != x.shape[1] or not wq.shape[0] or
            ref.shape != (len(x), len(wq)) or sw.shape != (len(wq),) or
            labels.shape != (len(x),) or not np.isin(labels, GROUPS).all()):
        raise ValueError("incompatible calibration arrays or group labels")
    if not all(np.isfinite(a).all() for a in (x, ref, sw)) or np.any(sw <= 0):
        raise ValueError("calibration arrays/scales must be finite, sw positive")
    c = dict(sx=float(sx), s10=float(s10), sx_j=sx_j, s10_i=s10_i,
             feasible=False, score=None, worst=None, prefill_nmse=None,
             decode_nmse=None, clip_rate=None, effective_shift_min=None,
             effective_shift_max=None, multiplier=None, shift=None, params=None,
             is_old_pair=False, sx_boundary_hit=False, s10_boundary_hit=False)

    def reject(reason):
        c["rejection_reason"] = reason
        return c

    if not all(math.isfinite(v) and v > 0 for v in (sx, s10)):
        return reject("nonpositive or nonfinite scale")
    profile = Profile()
    params = profile.approximate(sx * sw / s10)
    c.update(params=params, multiplier=params["multiplier"], shift=params["shift"])
    if np.any(np.asarray(params["status"]) != "ok"):
        return reject("invalid multiplier/shift approximation")
    shifts = profile.effective(params["shift"])
    if max(-8, int(shifts.max()) - 31) > min(7, int(shifts.min())):
        return reject("empty feasible k range")
    q, ks, _ = select_absmax(x, sx, params["shift"])
    effective = shifts[None, :] - ks[:, None]
    c.update(effective_shift_min=int(effective.min()),
             effective_shift_max=int(effective.max()))
    if effective.min() < 0 or effective.max() > 31:
        return reject("effective shift outside [0,31]")
    # k affects requantization, not ACC. Identical codes share the integer GEMM.
    key = (_fingerprint(q), _fingerprint(wq))
    if key not in acc_cache:
        acc_cache[key] = base.integer_dot(q, wq, "fp64-exact")
    acc = acc_cache[key]
    out = np.empty(acc.shape, np.float64)
    clips = 0
    for k in np.unique(ks):
        mask = ks == k
        raw = profile.apply(acc[mask], params["multiplier"],
                            params["shift"] - int(k), saturate=False)
        clips += int(((raw < -512) | (raw > 511)).sum())
        out[mask] = np.clip(raw, -512, 511) * s10
    c.update(output_scores(out, ref, labels), clip_rate=clips / out.size)
    if not math.isfinite(c["score"]):
        return reject("nonfinite output NMSE")
    c["feasible"] = True
    return c


def candidate_key(c, old_sx, old_s10):
    return (c["score"], c["worst"], c["clip_rate"],
            abs(math.log2(c["sx"] / old_sx)), abs(math.log2(c["s10"] / old_s10)),
            abs(c["sx_j"]), abs(c["s10_i"]), c["sx_j"], c["s10_i"])


def search_joint_pair(x, labels, ref, wq, sw, old_sx, old_s10, *, sx_js, s10_is):
    """Search caller-specified Cartesian grid plus incumbent; never expand it."""
    if not all(math.isfinite(v) and v > 0 for v in (old_sx, old_s10)):
        raise ValueError("old scales must be finite and positive")
    js, indices = sorted(set(sx_js)), sorted(set(s10_is))
    if any(not isinstance(v, (int, np.integer)) for v in js + indices):
        raise ValueError("grid offsets must be integers")
    pairs = [(0, 0)] + [(j, i) for j in js for i in indices]
    seen, cache, candidates = set(), {}, []
    for j, i in pairs:
        sx, s10 = old_sx * 2.0 ** (j / 4), old_s10 * 2.0 ** (i / 8)
        if (sx, s10) in seen:
            continue
        seen.add((sx, s10))
        c = evaluate_joint_candidate(x, labels, ref, wq, sw, sx, s10,
                                     sx_j=j, s10_i=i, acc_cache=cache)
        c.update(is_old_pair=(sx == old_sx and s10 == old_s10),
                 sx_boundary_hit=bool(js and j in (js[0], js[-1])),
                 s10_boundary_hit=bool(indices and i in (indices[0], indices[-1])))
        candidates.append(c)
    old = candidates[0]
    if not old["feasible"]:
        raise ValueError("old pair is infeasible: " + old["rejection_reason"])
    feasible = [c for c in candidates if c["feasible"]]
    selected = min(feasible, key=lambda c: candidate_key(c, old_sx, old_s10))
    if selected["score"] > old["score"] + LOCAL_REGRESSION_TOL:
        raise RuntimeError(
            "joint calibration selected a locally regressive candidate: "
            f"selected={selected['score']!r}, old={old['score']!r}, "
            f"tolerance={LOCAL_REGRESSION_TOL!r}")
    relative = ((selected["score"] - old["score"]) / old["score"]
                if old["score"] else 0.0)
    metadata = dict(old_score=old["score"], selected_score=selected["score"],
                    relative_local_nmse_change=relative,
                    changed_sx=selected["sx"] != old_sx,
                    changed_s10=selected["s10"] != old_s10,
                    sx_boundary_hit=selected["sx_boundary_hit"],
                    s10_boundary_hit=selected["s10_boundary_hit"],
                    feasible_candidate_count=len(feasible),
                    rejected_candidate_count=len(candidates) - len(feasible),
                    unique_acc_count=len(cache))
    return dict(selected=selected, old=old, candidates=candidates, metadata=metadata)


def calibrate_layer_joint(weight, old_spec, groups, *, sx_js, s10_is):
    """Calibrate supplied prefill/decode rows; caller owns split provenance."""
    expected_wq, sw = base.quantize_weight(weight)
    np.testing.assert_array_equal(expected_wq, old_spec["wq"])
    available = [g for g in GROUPS if len(groups.get(g, []))]
    if not available:
        raise ValueError("no calibration rows")
    x = np.concatenate([groups[g] for g in available]).astype(np.float64)
    labels = np.concatenate([[g] * len(groups[g]) for g in available])
    ref = x.astype(np.float64) @ weight.astype(np.float64).T
    return search_joint_pair(x, labels, ref, old_spec["wq"], sw,
                             old_spec["sx"], old_spec["s10"],
                             sx_js=sx_js, s10_is=s10_is)


def proposed_spec(old_spec, selected):
    if not selected["feasible"]:
        raise ValueError("cannot propose an infeasible candidate")
    proposed = dict(old_spec, sx=selected["sx"], s10=selected["s10"],
                    params=selected["params"])
    np.testing.assert_array_equal(proposed["wq"], old_spec["wq"])
    return proposed


def e2e_gate(old_metrics, proposed_metrics):
    """Pure safety gate; metrics use keys kl, nmse, top1 (fractions)."""
    for metrics in (old_metrics, proposed_metrics):
        if (not all(math.isfinite(metrics[k]) for k in ("kl", "nmse", "top1"))
                or min(metrics["kl"], metrics["nmse"]) < 0
                or not 0 <= metrics["top1"] <= 1):
            raise ValueError("invalid E2E metrics")
    checks = dict(kl_pass=proposed_metrics["kl"] <= 1.05 * old_metrics["kl"],
                  nmse_pass=proposed_metrics["nmse"] <= 1.05 * old_metrics["nmse"],
                  top1_pass=proposed_metrics["top1"] >= old_metrics["top1"] - 0.01)
    return {"pass": all(checks.values()), **checks}
