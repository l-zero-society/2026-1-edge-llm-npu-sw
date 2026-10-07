"""Autonomous CPU-only INT10 VPU Norm/Softmax/RoPE precision pilot."""
from __future__ import annotations

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
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gpalu_width_sweep_pilot as width

base, cal, final, response = width.base, width.cal, width.final, width.response
mixed, sweep = width.mixed, width
OUT = ROOT / "diagnostics/vpu10_e2e_pilot"
WORK = OUT / "work"
RTL = ROOT.parent / "2026-1-edge-llm-npu-rtl"
GPALU = ROOT / "diagnostics/gpalu_width_sweep_pilot"
MODES = (
    "FP_FULL", "GPALU10_FP_VPU2", "GPALU10_NORM10_COEFF16",
    "GPALU10_NORM10_COEFF10", "GPALU10_ROPE10",
    "VPU10_FULL_COEFF16", "VPU10_FULL_COEFF10",
)
RTL_FILES = (
    "src/main/scala/npu/top/VPU.scala", "src/main/scala/npu/core/NormUnit.scala",
    "src/main/scala/npu/core/Rope.scala", "src/test/scala/NormUnit_Test.scala",
    "src/test/scala/NormUnit_Distributed_Test.scala", "src/test/scala/Rope_Test.scala",
)
START = time.monotonic()
STAGE = "INITIALIZING"


def sha(path): return base.sha256_file(Path(path))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
def signed_bounds(bits=10): return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def status(state="RUNNING", **extra):
    value = {"state": state, "stage": STAGE, "elapsed_seconds": time.monotonic() - START, **extra}
    atomic_json(OUT / "status.json", value)


def log(message):
    print(message, flush=True)


def accumulator_bits(in_bits=10, vector_size=4096):
    return max(32, 2 * in_bits + math.ceil(math.log2(vector_size)) + 3)


def normalized_index(value, acc_bits=35, index_bits=10):
    value = int(value) & ((1 << acc_bits) - 1)
    if value == 0: return 0, 0
    exponent = value.bit_length() - 1
    shifted = (value << ((acc_bits - 1) - exponent)) & ((1 << acc_bits) - 1)
    return (shifted >> (acc_bits - index_bits)) & ((1 << index_bits) - 1), exponent


def delta_to_index(a, b, index_bits=10):
    return min(abs(int(a) - int(b)), (1 << index_bits) - 1)


def round_even(values): return np.rint(np.asarray(values, np.float64)).astype(np.int64)


def norm_tables(frac_bits):
    max_code = (1 << (frac_bits + 2)) - 1 if frac_bits == 8 else 65535
    indices = np.arange(1024, dtype=np.float64)
    mantissa = np.maximum(indices, 512.) / 512.
    rsqrt = np.clip(round_even((1. / np.sqrt(mantissa)) * (1 << frac_bits)), 0, max_code)
    recip = np.clip(round_even((1. / mantissa) * (1 << frac_bits)), 0, max_code)
    exp = np.clip(round_even(np.exp(-indices) * (1 << frac_bits)), 0, max_code)
    return {"rsqrt": rsqrt, "recip": recip, "exp": exp, "frac_bits": frac_bits}


def adjust_rsqrt(raw, exponent, frac_bits):
    raw = int(raw)
    if exponent & 1:
        raw = (raw * round((1. / math.sqrt(2.)) * (1 << frac_bits))) >> frac_bits
    return raw >> (exponent >> 1)


def adjust_recip(raw, signed_exponent):
    return int(raw) >> signed_exponent if signed_exponent >= 0 else int(raw) << (-signed_exponent)


def rope_tables():
    angle = 2. * np.pi * np.arange(1024) / 1024.
    return (np.clip(round_even(np.cos(angle) * (1 << 14)), -32768, 32767).astype(np.int16),
            np.clip(round_even(np.sin(angle) * (1 << 14)), -32768, 32767).astype(np.int16))


def q016_angle_index(position, turns_per_token):
    frequency = int(np.rint(float(turns_per_token) * 65536.)) & 0xffff
    return ((int(position) * frequency) & 0xffff) >> 6


def quantize10(value, scale):
    raw = round_even(np.asarray(value, np.float64) / float(scale))
    code = np.clip(raw, -512, 511).astype(np.int16)
    return code, float(np.mean((raw < -512) | (raw > 511)))


def rms_hw_numpy(x, input_scale, output_scale, weight, eps, tables):
    q, input_clip = quantize10(x, input_scale)
    frac = tables["frac_bits"]
    out = np.empty_like(np.asarray(x, np.float64)); approx_errors = []
    for row_i, row in enumerate(q.reshape(-1, q.shape[-1])):
        stat = int(np.rint(np.mean(row.astype(np.int64) ** 2) + eps / (input_scale ** 2)))
        idx, exponent = normalized_index(stat)
        scale_code = adjust_rsqrt(tables["rsqrt"][idx], exponent, frac)
        scale_real = scale_code / float(1 << frac)
        exact = 1. / math.sqrt(max(stat, 1))
        approx_errors.append((scale_real - exact) ** 2)
        normalized = row.astype(np.float64) * scale_real
        out.reshape(-1, q.shape[-1])[row_i] = normalized * (1. + np.asarray(weight, np.float64))
    out_code, output_clip = quantize10(out, output_scale)
    return out_code.astype(np.float64) * output_scale, {"input_clipping_rate": input_clip,
        "output_clipping_rate": output_clip, "scale_lut_mse": float(np.mean(approx_errors)),
        "code_utilization": int(np.unique(out_code).size)}


def softmax_hw_numpy(scores, score_scale, tables):
    scores = np.asarray(scores, np.float64)
    finite = np.isfinite(scores)
    safe = np.where(finite, scores, -1e30)
    q, clip = quantize10(safe, score_scale)
    q = np.where(finite, q, -512).astype(np.int16)
    frac = tables["frac_bits"]
    output = np.empty_like(scores); exp_error = []; recip_error = []
    for i, row in enumerate(q.reshape(-1, q.shape[-1])):
        mask = finite.reshape(-1, finite.shape[-1])[i]
        max_q = int(row[mask].max()) if np.any(mask) else 0
        delta = np.minimum(max_q - row.astype(np.int32), 1023)
        exp_code = np.where(mask, np.clip(round_even(np.exp(-delta * score_scale) * (1 << frac)), 0,
                                         (1 << (frac + 2)) - 1), 0).astype(np.int64)
        exp_exact = np.where(mask, np.exp(-delta * score_scale), 0.)
        exp_error.append(float(np.mean((exp_code / (1 << frac) - exp_exact) ** 2)))
        total = int(exp_code.sum())
        idx, exponent = normalized_index(total)
        recip_code = adjust_recip(tables["recip"][idx], exponent - frac)
        recip_real = recip_code / float(1 << frac)
        exact_recip = 1. / max(total / float(1 << frac), 1e-30)
        recip_error.append((recip_real - exact_recip) ** 2)
        prob = exp_code.astype(np.float64) / (1 << frac) * recip_real
        prob_code = np.clip(round_even(prob * 512.), 0, 511)
        reconstructed = prob_code / 512.
        denominator = reconstructed.sum()
        output.reshape(-1, row.size)[i] = reconstructed / denominator if denominator else reconstructed
    return output, {"input_clipping_rate": clip, "exp_lut_mse": float(np.mean(exp_error)),
                    "reciprocal_lut_mse": float(np.mean(recip_error)),
                    "output_saturation_rate": float(np.mean(output >= 511 / 512.))}


def rope_hw_numpy(x, cos, sin, scale, cos_lut=None, sin_lut=None, coefficient_lut=True):
    q, clip = quantize10(x, scale); half = q.shape[-1] // 2
    angle = np.mod(np.arctan2(np.asarray(sin)[..., :half], np.asarray(cos)[..., :half]) / (2 * np.pi), 1.)
    phase = round_even(angle * 65536.) & 0xffff; idx = (phase >> 6).astype(np.int64)
    if coefficient_lut:
        c, s = np.asarray(cos_lut)[idx].astype(np.int64), np.asarray(sin_lut)[idx].astype(np.int64)
    else:
        c, s = round_even(np.cos(2*np.pi*idx/1024.) * (1 << 14)), round_even(np.sin(2*np.pi*idx/1024.) * (1 << 14))
    first, second = q[..., :half].astype(np.int64), q[..., half:].astype(np.int64)
    out = np.concatenate(((first*c - second*s) >> 14, (second*c + first*s) >> 14), axis=-1)
    clipped = np.clip(out, -512, 511).astype(np.int16)
    return clipped.astype(np.float64) * scale, {"input_clipping_rate": clip,
        "output_saturation_rate": float(np.mean((out < -512) | (out > 511))),
        "angle_index_utilization": int(np.unique(idx).size)}


class Bounded:
    def __init__(self, rows=64): self.rows, self.parts = rows, []
    def add(self, x):
        x = np.asarray(x, np.float32).reshape(-1, x.shape[-1])
        have = sum(len(v) for v in self.parts)
        if have < self.rows: self.parts.append(x[:self.rows-have].copy())
    def array(self): return np.concatenate(self.parts) if self.parts else np.empty((0, 1), np.float32)


def module_names(source):
    names = [name for name, module in source.model.named_modules() if type(module).__name__ == "GemmaRMSNorm"]
    if len(names) != 37: raise RuntimeError(f"expected 37 GemmaRMSNorm modules, got {len(names)}")
    return names


def collect_calibration(source):
    from transformers.models.gemma import modeling_gemma
    norm_names = module_names(source)
    norm_in = {name: Bounded() for name in norm_names}; norm_out = {name: Bounded() for name in norm_names}
    rope_q = {i: Bounded() for i in range(18)}; rope_k = {i: Bounded() for i in range(18)}
    score_values = {i: [] for i in range(18)}; handles = []
    for name in norm_names:
        module = dict(source.model.named_modules())[name]
        handles.append(module.register_forward_pre_hook(lambda m, a, name=name: norm_in[name].add(a[0].detach().cpu().numpy())))
        handles.append(module.register_forward_hook(lambda m, a, o, name=name: norm_out[name].add(o.detach().cpu().numpy())))
    for layer, block in enumerate(source.model.model.layers):
        handles.append(block.self_attn.q_proj.register_forward_hook(lambda m,a,o,layer=layer: rope_q[layer].add(o.detach().cpu().numpy())))
        handles.append(block.self_attn.k_proj.register_forward_hook(lambda m,a,o,layer=layer: rope_k[layer].add(o.detach().cpu().numpy())))
    original_attention = modeling_gemma.eager_attention_forward
    def collecting_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        repeated = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
        weights = (source.torch.matmul(query, repeated.transpose(2,3)) * scaling).detach().cpu().float().numpy()
        flat = weights[np.isfinite(weights)].reshape(-1)
        if sum(len(x) for x in score_values[module.layer_idx]) < 65536:
            score_values[module.layer_idx].append(flat[:max(0,65536-sum(len(x) for x in score_values[module.layer_idx]))])
        return original_attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    modeling_gemma.eager_attention_forward = collecting_attention
    try:
        for example in mixed.calibration_examples(source):
            _, _, cache = cal.run_fp(source, example["prompt_ids"], None, layers=False)
            for t in range(1, min(mixed.CAL_LIMIT, len(example["targets"]))):
                _, _, cache = cal.run_fp(source, [example["targets"][t-1]], cache, layers=False)
    finally:
        modeling_gemma.eager_attention_forward = original_attention
        for handle in handles: handle.remove()
    norm_specs = {}
    modules = dict(source.model.named_modules())
    for name in norm_names:
        x, y = norm_in[name].array(), norm_out[name].array()
        norm_specs[name] = {"input_scale": max(float(np.max(np.abs(x))) / 511., 1e-12),
                            "output_scale": max(float(np.max(np.abs(y))) / 511., 1e-12),
                            "eps": float(modules[name].eps)}
    rope_specs = {}
    for layer in range(18):
        q, k = rope_q[layer].array(), rope_k[layer].array()
        rope_specs[layer] = {"q_scale": max(float(np.max(np.abs(q))) / 511., 1e-12),
                             "k_scale": max(float(np.max(np.abs(k))) / 511., 1e-12),
                             "score_scale": max(float(np.max(np.abs(np.concatenate(score_values[layer])))) / 511., 1e-12)}
    return norm_specs, rope_specs, {"calibration_examples": [mixed.identity(e) for e in mixed.calibration_examples(source)],
        "norm_modules": len(norm_specs), "response_limit": mixed.CAL_LIMIT, "bounded_rows": 64}


def local_diagnostics(source, norm_specs, rope_specs, tables16, tables10, cos_lut, sin_lut):
    # Deterministic synthetic probes exercise each calibrated contract without another model pass.
    rng = np.random.default_rng(20261007); modules = dict(source.model.named_modules())
    norm_rows = []
    for index, (name, spec) in enumerate(norm_specs.items()):
        x = rng.normal(0., spec["input_scale"] * 80., (4, 2048))
        weight = modules[name].weight.detach().cpu().float().numpy()
        fp = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + spec["eps"]) * (1 + weight)
        a, da = rms_hw_numpy(x, spec["input_scale"], spec["output_scale"], weight, spec["eps"], tables16)
        b, db = rms_hw_numpy(x, spec["input_scale"], spec["output_scale"], weight, spec["eps"], tables10)
        norm_rows.append({"module":name,"index":index,"coeff16_nmse":mixed.nmse(a,fp),"coeff10_nmse":mixed.nmse(b,fp),
                          "coeff16_clip":da["output_clipping_rate"],"coeff10_clip":db["output_clipping_rate"],
                          "input_scale":spec["input_scale"],"output_scale":spec["output_scale"],
                          "coeff16_scale_lut_mse":da["scale_lut_mse"],"coeff10_scale_lut_mse":db["scale_lut_mse"]})
    soft_rows=[]; rope_rows=[]
    for layer,spec in rope_specs.items():
        scores=rng.normal(0.,spec["score_scale"]*40.,(8,64)); fp=np.exp(scores-scores.max(-1,keepdims=True));fp/=fp.sum(-1,keepdims=True)
        p16,d16=softmax_hw_numpy(scores,spec["score_scale"],tables16);p10,d10=softmax_hw_numpy(scores,spec["score_scale"],tables10)
        soft_rows.append({"layer":layer,"coeff16_nmse":mixed.nmse(p16,fp),"coeff10_nmse":mixed.nmse(p10,fp),
                          "coeff16_mae":float(np.mean(np.abs(p16-fp))),"coeff10_mae":float(np.mean(np.abs(p10-fp))),
                          "coeff16_exp_lut_mse":d16["exp_lut_mse"],"coeff10_exp_lut_mse":d10["exp_lut_mse"],
                          "coeff16_recip_lut_mse":d16["reciprocal_lut_mse"],"coeff10_recip_lut_mse":d10["reciprocal_lut_mse"]})
        x=rng.normal(0.,spec["q_scale"]*80.,(2,4,8,256));pos=np.arange(8)[None,:,None];inv=np.exp(-np.arange(128)*math.log(10000.)/128.)
        ang=pos*inv[None,None,:]; c=np.concatenate((np.cos(ang),np.cos(ang)),-1)[:,None];s=np.concatenate((np.sin(ang),np.sin(ang)),-1)[:,None]
        fp_rope=np.concatenate((x[...,:128]*c[...,:128]-x[...,128:]*s[...,:128],x[...,128:]*c[...,:128]+x[...,:128]*s[...,:128]),-1)
        act,_=rope_hw_numpy(x,c,s,spec["q_scale"],cos_lut,sin_lut,False);hw,dh=rope_hw_numpy(x,c,s,spec["q_scale"],cos_lut,sin_lut,True)
        rope_rows.append({"layer":layer,"activation_index_nmse":mixed.nmse(act,fp_rope),"q214_nmse":mixed.nmse(hw,fp_rope),
                          "coefficient_increment_nmse":mixed.nmse(hw,act),"saturation_rate":dh["output_saturation_rate"],"scale":spec["q_scale"]})
    return norm_rows,soft_rows,rope_rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@contextlib.contextmanager
def hardware_patch(source, gate_specs, up_specs, luts, gpalu, norm_specs, rope_specs, norm_coeff=None, rope=False):
    from transformers.models.gemma import modeling_gemma
    originals=[]; original_apply=modeling_gemma.apply_rotary_pos_emb; original_attention=modeling_gemma.eager_attention_forward
    tables = norm_tables(norm_coeff) if norm_coeff else None; cos_lut,sin_lut=rope_tables(); modules=dict(source.model.named_modules())
    state={"layer":0}
    with sweep.patch_width(source,gate_specs,up_specs,luts,gpalu):
        try:
            if norm_coeff:
                for name,spec in norm_specs.items():
                    module=modules[name];originals.append((module,module.forward))
                    def forward(x,module=module,spec=spec):
                        y,_=rms_hw_numpy(x.detach().cpu().float().numpy(),spec["input_scale"],spec["output_scale"],module.weight.detach().cpu().float().numpy(),module.eps,tables)
                        return x.new_tensor(y)
                    module.forward=forward
            for layer,block in enumerate(source.model.model.layers):
                module=block.self_attn;originals.append((module,module.forward));old=module.forward
                def wrapped(*args,old=old,layer=layer,**kwargs): state["layer"]=layer;return old(*args,**kwargs)
                module.forward=wrapped
            if rope:
                def apply(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
                    layer=state["layer"];spec=rope_specs[layer];c=cos.unsqueeze(unsqueeze_dim);s=sin.unsqueeze(unsqueeze_dim)
                    qh,_=rope_hw_numpy(q.detach().cpu().float().numpy(),c.detach().cpu().float().numpy(),s.detach().cpu().float().numpy(),spec["q_scale"],cos_lut,sin_lut,True)
                    kh,_=rope_hw_numpy(k.detach().cpu().float().numpy(),c.detach().cpu().float().numpy(),s.detach().cpu().float().numpy(),spec["k_scale"],cos_lut,sin_lut,True)
                    return q.new_tensor(qh),k.new_tensor(kh)
                modeling_gemma.apply_rotary_pos_emb=apply
            if norm_coeff:
                def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
                    key_states=modeling_gemma.repeat_kv(key,module.num_key_value_groups);value_states=modeling_gemma.repeat_kv(value,module.num_key_value_groups)
                    scores=source.torch.matmul(query,key_states.transpose(2,3))*scaling
                    if attention_mask is not None:scores=scores+attention_mask[:,:,:,:key_states.shape[-2]]
                    probs,_=softmax_hw_numpy(scores.detach().cpu().float().numpy(),rope_specs[module.layer_idx]["score_scale"],tables)
                    probs=source.torch.nn.functional.dropout(scores.new_tensor(probs),p=dropout,training=module.training)
                    return source.torch.matmul(probs,value_states).transpose(1,2).contiguous(),probs
                modeling_gemma.eager_attention_forward=attention
            yield
        finally:
            modeling_gemma.apply_rotary_pos_emb=original_apply;modeling_gemma.eager_attention_forward=original_attention
            for module,old in reversed(originals):module.forward=old


def run_mode(source, gate_specs, up_specs, luts, gpalu, norm_specs, rope_specs, mode, ids, cache):
    coeff = 14 if mode in ("GPALU10_NORM10_COEFF16","VPU10_FULL_COEFF16") else 8 if mode in ("GPALU10_NORM10_COEFF10","VPU10_FULL_COEFF10") else None
    rope = mode in ("GPALU10_ROPE10","VPU10_FULL_COEFF16","VPU10_FULL_COEFF10")
    with hardware_patch(source,gate_specs,up_specs,luts,gpalu,norm_specs,rope_specs,coeff,rope):
        hidden,_,cache=response.forward_hidden(source,ids,[len(ids)-1],{},cache,True,False)
        with source.torch.inference_mode():logits=source.model.lm_head(hidden)
    return logits,cache


def evaluate_example(source, gate_specs, up_specs, luts, gpalu, norm_specs, rope_specs, example):
    scores={mode:response.Scores() for mode in MODES};caches={mode:None for mode in MODES}
    for t,target in enumerate(example["targets"][:24]):
        ids=example["prompt_ids"] if t==0 else [example["targets"][t-1]]
        fp,_,caches["FP_FULL"]=cal.run_fp(source,ids,caches["FP_FULL"],layers=False);scores["FP_FULL"].add(source.torch,fp,fp,[target])
        for mode in MODES[1:]:
            logits,caches[mode]=run_mode(source,gate_specs,up_specs,luts,gpalu,norm_specs,rope_specs,mode,ids,caches[mode])
            scores[mode].add(source.torch,logits,fp,[target])
        if len({id(v) for v in caches.values()}) != len(MODES):raise RuntimeError("KV cache alias")
    return {mode:scores[mode].raw() for mode in MODES}


def delta(old,new):
    return {"kl_relative":(new["kl"]-old["kl"])/old["kl"],"nmse_relative":(new["nmse"]-old["nmse"])/old["nmse"],
            "ppl_relative":(new["ppl"]-old["ppl"])/old["ppl"],"top1_pp":100*(new["top1"]-old["top1"])}


def artifact_paths():
    paths=["scripts/run_vpu10_e2e_pilot.py","scripts/launch_vpu10_e2e_pilot.sh","tests/test_vpu10_e2e_pilot.py"]
    paths += [f"diagnostics/vpu10_e2e_pilot/{name}" for name in (
        "baseline_manifest.json","norm10_coeff16_parameters.json","norm10_coeff10_parameters.json","rope10_parameters.json",
        "norm_local_summary.csv","softmax_local_summary.csv","rope_local_summary.csv","lut_error_decomposition.json",
        "e2e_dataset_manifest.json","e2e_summary.json","verification.json","report.md","run.log","unit_test.log")]
    return paths


def main():
    global STAGE
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    (OUT/"DONE").unlink(missing_ok=True);(OUT/"FAILED").unlink(missing_ok=True);status()
    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    if branch!="lut_cali":raise RuntimeError("SW branch must be lut_cali")
    if subprocess.run(["git","merge-base","--is-ancestor","b7daf5ad1fa7bde89e92bbf6d66d692d5ad77108","HEAD"]).returncode:raise RuntimeError("required GPALU baseline commit missing")
    rtl_branch=subprocess.check_output(["git","-C",str(RTL),"branch","--show-current"],text=True).strip()
    if rtl_branch!="feat/tpu-rocc-driver":raise RuntimeError("wrong RTL branch")
    rtl_hashes={name:sha(RTL/name) for name in RTL_FILES}
    baseline_paths=[GPALU/"int10/gpalu_parameters.json",GPALU/"e2e_summary.json",GPALU/"baseline_manifest.json"]
    baseline_hashes={str(p.relative_to(ROOT)):sha(p) for p in baseline_paths}
    base.write_json(OUT/"baseline_manifest.json",{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"gpalu_hashes":baseline_hashes,"rtl_branch":rtl_branch,"rtl_hashes":rtl_hashes})
    STAGE="UNIT_TESTS";status();log("UNIT_TEST_START")
    tests=[sys.executable,"-m","unittest","tests.test_vpu10_e2e_pilot","tests.test_gpalu_width_sweep_pilot"]
    with (OUT/"unit_test.log").open("w") as f:rc=subprocess.run(tests,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPYCACHEPREFIX="/tmp/vpu10-tests")).returncode
    if rc:raise RuntimeError("unit tests failed")
    log("UNIT_TEST_DONE")
    STAGE="MODEL_LOAD";status();source,_,_,_,_,_=final.load_source_and_policies()
    if str(next(source.model.parameters()).device)!="cpu":raise RuntimeError("CPU required")
    historical_manifest=json.loads((mixed.ART/"manifest.json").read_text());historical=mixed.load_historical(source,historical_manifest)
    baseline_specs=width.fusion.load_baseline_specs(source,historical);current_specs=width.pot.load_current_specs(baseline_specs);luts=width.pot.load_current_luts()
    gpalu_rows=json.loads((GPALU/"int10/gpalu_parameters.json").read_text())["layers"]
    gpalu={int(x["layer"]):x for x in gpalu_rows}
    gate={i:current_specs[f"model.layers.{i}.mlp.gate_proj"] for i in range(18)};up={i:current_specs[f"model.layers.{i}.mlp.up_proj"] for i in range(18)}
    STAGE="CALIBRATION_CAPTURE";status();log("CAPTURE_START")
    capture_file=WORK/"calibration.json"
    identity=digest({"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"baseline":baseline_hashes,"rtl":rtl_hashes})
    if capture_file.exists():
        capture=json.loads(capture_file.read_text())
        if capture.get("identity")!=identity:raise RuntimeError("calibration checkpoint mismatch")
        norm_specs={k:v for k,v in capture["norm_specs"].items()};rope_specs={int(k):v for k,v in capture["rope_specs"].items()};capture_manifest=capture["manifest"]
    else:
        norm_specs,rope_specs,capture_manifest=collect_calibration(source);atomic_json(capture_file,{"identity":identity,"norm_specs":norm_specs,"rope_specs":rope_specs,"manifest":capture_manifest})
    log("CAPTURE_READY")
    tables16,tables10=norm_tables(14),norm_tables(8);cos_lut,sin_lut=rope_tables()
    base.write_json(OUT/"norm10_coeff16_parameters.json",{"activation_bits":10,"index_bits":10,"coefficient_format":"Q2.14","physical_container_bits":16,"specs":norm_specs})
    base.write_json(OUT/"norm10_coeff10_parameters.json",{"activation_bits":10,"index_bits":10,"coefficient_format":"Q2.8","physical_container_bits":16,"effective_bits":10,"specs":norm_specs})
    base.write_json(OUT/"rope10_parameters.json",{"activation_bits":10,"angle_index_bits":10,"trig_bits":16,"trig_frac_bits":14,"specs":rope_specs,"cos_sha256":hashlib.sha256(cos_lut.tobytes()).hexdigest(),"sin_sha256":hashlib.sha256(sin_lut.tobytes()).hexdigest()})
    STAGE="LOCAL_DIAGNOSTICS";status();norm_rows,soft_rows,rope_rows=local_diagnostics(source,norm_specs,rope_specs,tables16,tables10,cos_lut,sin_lut)
    write_csv(OUT/"norm_local_summary.csv",norm_rows);write_csv(OUT/"softmax_local_summary.csv",soft_rows);write_csv(OUT/"rope_local_summary.csv",rope_rows)
    base.write_json(OUT/"lut_error_decomposition.json",{"normalizer":{"activation_plus_index_coeff16_mean_nmse":float(np.mean([x["coeff16_nmse"] for x in norm_rows])),"activation_plus_index_coeff10_mean_nmse":float(np.mean([x["coeff10_nmse"] for x in norm_rows]))},"softmax":{"coeff16_mean_nmse":float(np.mean([x["coeff16_nmse"] for x in soft_rows])),"coeff10_mean_nmse":float(np.mean([x["coeff10_nmse"] for x in soft_rows]))},"rope":{"activation_index_mean_nmse":float(np.mean([x["activation_index_nmse"] for x in rope_rows])),"q214_mean_nmse":float(np.mean([x["q214_nmse"] for x in rope_rows])),"coefficient_increment_mean_nmse":float(np.mean([x["coefficient_increment_nmse"] for x in rope_rows]))}})
    STAGE="E2E";status();log("E2E_START")
    examples=mixed.e2e_examples(source);manifest=json.loads((GPALU/"../gpalu_pot_fusion_pilot/e2e_dataset_manifest.json").resolve().read_text()) if False else json.loads((ROOT/"diagnostics/gpalu_pot_fusion_pilot/e2e_dataset_manifest.json").read_text())
    base.write_json(OUT/"e2e_dataset_manifest.json",manifest);run_id=digest({"identity":identity,"norm":norm_specs,"rope":rope_specs,"data":manifest})
    records=[]
    for i,example in enumerate(examples):
        path=WORK/f"e2e_{i:02d}.json";checkpoint=json.loads(path.read_text()) if path.exists() else None;key=digest({"run":run_id,"example":mixed.identity(example)})
        if checkpoint and checkpoint.get("identity")!=key:raise RuntimeError("E2E checkpoint mismatch")
        value=checkpoint["value"] if checkpoint else evaluate_example(source,gate,up,luts,gpalu,norm_specs,rope_specs,example)
        if not checkpoint:atomic_json(path,{"identity":key,"value":value})
        records.append(value);log(f"E2E_MODE_BATCH_{i+1}_DONE")
    metrics={mode:response.merged([r[mode] for r in records]) for mode in MODES}
    prior=json.loads((GPALU/"e2e_summary.json").read_text())["modes"]["GPALU_INT10"]
    reproduced=metrics["GPALU10_FP_VPU2"]
    if any(abs(reproduced[k]-prior[k])>1e-8 for k in ("nll","kl","nmse","top1")):raise RuntimeError("GPALU INT10 baseline reproduction mismatch")
    pairs=(("GPALU10_FP_VPU2","GPALU10_NORM10_COEFF16"),("GPALU10_NORM10_COEFF16","GPALU10_NORM10_COEFF10"),("GPALU10_FP_VPU2","GPALU10_ROPE10"),("GPALU10_FP_VPU2","VPU10_FULL_COEFF16"),("VPU10_FULL_COEFF16","VPU10_FULL_COEFF10"),("GPALU10_FP_VPU2","VPU10_FULL_COEFF10"))
    comparisons={f"{a}_to_{b}":delta(metrics[a],metrics[b]) for a,b in pairs}
    base.write_json(OUT/"e2e_summary.json",{"modes":metrics,"comparisons":comparisons,"baseline_reproduced":True,"E2E_used_for_selection":False})
    STAGE="REPORT";status();verification={"SW_branch":branch,"SW_source_HEAD":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"RTL_repo":str(RTL),"RTL_branch":rtl_branch,"RTL_files":rtl_hashes,"execution_device":"CPU","MPS_used":False,"ANE_used":False,"CUDA_used":False,"E2E_dataset_identity":digest(manifest),"GPALU_output_bits":10,"Norm_input_bits":10,"Norm_output_bits":10,"Norm_index_bits":10,"Norm_coeff16_format":"Q2.14","Norm_coeff10_effective_format":"Q2.8","Norm_physical_container_bits":16,"Norm_accumulator_bits":accumulator_bits(),"RoPE_activation_input_bits":10,"RoPE_activation_output_bits":10,"RoPE_angle_index_bits":10,"RoPE_trig_bits":16,"RoPE_trig_frac_bits":14,"RMSNorm_hardware_like":True,"Softmax_hardware_like":True,"RoPE_hardware_like":True,"E2E_used_for_selection":False,"test_return_code":rc,"elapsed_seconds":time.monotonic()-START,"baseline_artifacts_unchanged":{str(p.relative_to(ROOT)):sha(p) for p in baseline_paths}==baseline_hashes}
    base.write_json(OUT/"verification.json",verification)
    lines=["# CPU-only INT10 VPU path E2E precision pilot","","## 1. Scope","GPALU INT10 plus hardware-like INT10 RMSNorm, Softmax, and RoPE on the established 192-token set.","","## 2. RTL contract inspected",* [f"- `{name}` ({rtl_hashes[name]})" for name in RTL_FILES],"","## 3. CPU execution environment","CPU only; MPS, ANE, CUDA, Core ML, and MLX accelerators were not used.","","## 4. INT10 activation path","GPALU, Normalizer inputs/outputs, and RoPE inputs/outputs use signed INT10. Static operation scales are recorded in the parameter artifacts.","","## 5. Normalizer 10-bit model",f"The accumulator width is {accumulator_bits()} bits. Indexing follows `normalizedIndex()` and `deltaToIndex()` with 1024 entries. Effective 10-bit coefficients remain packed in a wider software container; the 128-bit RTL write interface was not redesigned.","","## 6. Norm index10 / coeff16 vs coeff10","See `norm_local_summary.csv` and the E2E table.","","## 7. Softmax Exp/Scale LUT results","See `softmax_local_summary.csv`.","","## 8. RoPE index10 / coeff16 results","RoPE uses a 10-bit angle index and signed 16-bit Q2.14 sine/cosine coefficients; see `rope_local_summary.csv`.","","## 9. Local error decomposition","See `lut_error_decomposition.json`.","","## 10. 192-token E2E results","| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode,value in metrics.items():lines.append("| "+mode+" | "+" | ".join(f"{value[k]:.9g}" for k in ("nll","ppl","kl","nmse","mse","mae","cosine","flattened_cosine","top1","in5","overlap"))+" |")
    primary=metrics["VPU10_FULL_COEFF10"];base_mode=metrics["GPALU10_FP_VPU2"]
    lines += ["","## 11. Accuracy loss from GPALU-only INT10 baseline",json.dumps(delta(base_mode,primary),sort_keys=True),"","## 12. Whether VPU10 remains above the desired ~82-83% region",f"Primary VPU10 Top1 is {100*primary['top1']:.4f}%. The 82-83% region was interpretive only and was not an optimization target.","","## 13. Hardware implications","The numerical result indicates whether further RTL investigation is supported; it is not an RTL implementation decision.","","## 14. Limitations","This is a 192-token CPU emulation. GEMMs other than the established Gate/Up/GELU/GPALU path remained FP. Static scales came only from the established small calibration population."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    required=[OUT/name for name in ("baseline_manifest.json","norm10_coeff16_parameters.json","norm10_coeff10_parameters.json","rope10_parameters.json","norm_local_summary.csv","softmax_local_summary.csv","rope_local_summary.csv","lut_error_decomposition.json","e2e_dataset_manifest.json","e2e_summary.json","verification.json","report.md","run.log","unit_test.log")]
    if not all(p.exists() for p in required):raise RuntimeError("required artifact missing")
    if not verification["baseline_artifacts_unchanged"]:raise RuntimeError("baseline artifact mutated")
    STAGE="GIT";status();log("REPORT_WRITTEN")
    subprocess.run(["git","add","--",*artifact_paths()],cwd=ROOT,check=True)
    staged=subprocess.check_output(["git","diff","--cached","--name-only"],cwd=ROOT,text=True).splitlines()
    if any("/work/" in name or name.endswith((".npz",".gguf")) for name in staged):raise RuntimeError("forbidden staged artifact")
    subprocess.run(["git","diff","--cached","--check"],cwd=ROOT,check=True)
    subprocess.run(["git","commit","-m","test: evaluate INT10 VPU norm and RoPE precision"],cwd=ROOT,check=True)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    subprocess.run(["git","push","origin","lut_cali"],cwd=ROOT,check=True)
    STAGE="COMPLETE";status("DONE",commit=commit,report="diagnostics/vpu10_e2e_pilot/report.md")
    (OUT/"DONE").write_text(f"COMPLETE\ncommit={commit}\nreport=diagnostics/vpu10_e2e_pilot/report.md\n")
    log("COMPLETE")


def entrypoint():
    global STAGE
    try: main()
    except BaseException as exc:
        traceback.print_exc();OUT.mkdir(parents=True,exist_ok=True);(OUT/"DONE").unlink(missing_ok=True)
        atomic_json(OUT/"status.json",{"state":"FAILED","stage":STAGE,"exception_type":type(exc).__name__,"error":str(exc),"elapsed_seconds":time.monotonic()-START})
        (OUT/"FAILED").write_text(f"{type(exc).__name__}: {exc}\n");raise


if __name__ == "__main__": entrypoint()
