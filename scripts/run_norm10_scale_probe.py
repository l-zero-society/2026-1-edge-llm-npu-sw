"""Small CPU-only RMSNorm scale-contract probe; scientific completion precedes Git.

No calibration search: existing s_in/s_out and GPALU10 parameters remain frozen.
The fused path is a hypothetical software contract, not a production RTL change.
"""
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_norm10_debug_pilot as debug

old, w = debug.old, debug.w
base, response, mixed = debug.base, debug.response, debug.mixed
OUT = ROOT / 'diagnostics/norm10_scale_probe'
WORK = OUT / 'work'
START = time.monotonic()
STAGE = 'UNIT_TESTS'
LOCAL_MODES = ('FP_REFERENCE', 'INPUT_QUANT_ORACLE', 'OLD_EMULATOR', 'RTL_LITERAL',
               'POST_GAIN_DIAGNOSTIC', 'FUSED_IDEAL_COEFFICIENT', 'FUSED_REQUANT_ORACLE')
E2E_MODES = ('FP_FULL', 'GPALU10_FP_NORM', 'GPALU10_OLD_NORM',
             'GPALU10_RTL_LITERAL_NORM', 'GPALU10_POST_GAIN_NORM', 'GPALU10_FUSED_REQUANT_NORM')
E2E_MAP = dict(zip(E2E_MODES[2:], ('OLD_EMULATOR', 'RTL_LITERAL', 'POST_GAIN_DIAGNOSTIC', 'FUSED_REQUANT_ORACLE')))
TARGET_MODULES = ('model.layers.0.input_layernorm', 'model.layers.0.post_attention_layernorm',
                  'model.layers.9.input_layernorm', 'model.layers.17.post_attention_layernorm', 'model.norm')
SOURCE_FILES = ('scripts/run_norm10_scale_probe.py', 'scripts/launch_norm10_scale_probe.sh', 'tests/test_norm10_scale_probe.py')
RESULT_FILES = ('accumulator_verification.json', 'norm_local_comparison.csv', 'output_code_distribution.csv',
                'scaling_trace.json', 'e2e_small_summary.json', 'verification.json', 'report.md', 'unit_test.log',
                'debugging_history.md', 'foreground_run.log')
PUBLISH_FILES = SOURCE_FILES + tuple('diagnostics/norm10_scale_probe/' + p for p in RESULT_FILES)


def write_json(name, value):
    old.atomic_json(OUT / name, value)


def set_stage(name, state='RUNNING', **extra):
    global STAGE
    STAGE = name
    write_json('status.json', dict(state=state, stage=name, elapsed_seconds=time.monotonic()-START, **extra))
    print(name, flush=True)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return old.sha(path)


def sat10(values):
    if not np.isfinite(np.asarray(values,dtype=np.float64)).all():
        raise FloatingPointError('nonfinite output codes')
    return np.clip(values, -512, 511).astype(np.int16)


def quantize_input10(value, scale):
    value=np.asarray(value,np.float64)
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid input scale')
    if not np.isfinite(value).all():raise FloatingPointError('nonfinite activation')
    with np.errstate(over='ignore'):
        raw=np.rint(value/scale)
    # A finite activation/positive finite scale may exceed float64 quotient range.
    # Its saturation sign remains exact, and no narrow integer cast precedes clip.
    clipping=float(np.mean((raw < -512)|(raw > 511)))
    return sat10(np.clip(raw,-512,511)),clipping


def merge_scores(records):
    """Reuse metric aggregation, with explicit representation of exp(NLL) overflow.

    NLL/log-PPL remain finite scientific data even when PPL exceeds float64.
    Historical response.merged stays unchanged for reproducibility.
    """
    n=sum(r['n'] for r in records)
    if not n:raise ValueError('empty score records')
    result=dict(tokens=n,**base.merge_metrics([r['metric'] for r in records]))
    result.update({k:sum(r[k] for r in records)/n for k in ('nll','kl','top1','in5','overlap')})
    if not all(math.isfinite(v) for v in result.values()):raise FloatingPointError('nonfinite score aggregate')
    result['log_ppl']=result['nll']
    max_log=math.log(sys.float_info.max)
    result['ppl']=math.exp(result['nll']) if result['nll']<=max_log else None
    result['ppl_status']='finite' if result['ppl'] is not None else 'exceeds_float64_range'
    return result


def rounded_shift(value, shift):
    """Arbitrary precision signed RNE, including negative ties."""
    value, shift = int(value), int(shift)
    if shift < 0:
        return value << -shift
    sign = -1 if value < 0 else 1
    quotient, remainder = divmod(abs(value), 1 << shift)
    half = (1 << shift) >> 1
    if shift and (remainder > half or (remainder == half and quotient & 1)):
        quotient += 1
    return sign * quotient


def statistic(q, eps_q=0):
    """Python integer reference plus every RTL intermediate capacity check."""
    row = [int(a) for a in q]
    if not row or any(a < -512 or a > 511 for a in row):
        raise ValueError('expected a nonempty signed INT10 vector')
    n = len(row); acc_bits = old.accumulator_bits(10, 4096)
    inv = debug.inv_n(n); squares = [a*a for a in row]
    total = sum(squares); product = total * inv
    mean = product >> 20; stat_full = mean + int(eps_q)
    # square: SInt10*SInt10 = SInt20, then asUInt/pad to accBits.
    # mean: UInt35 * UInt24 = UInt59, shift20 then low35 retained.
    checks = dict(square_signed20=max(squares) < (1 << 19),
                  square_sum_uint35=total < (1 << acc_bits),
                  inverse_uint24=inv < (1 << 24),
                  mean_product_uint59=product < (1 << (acc_bits + 24)),
                  shifted_mean_uint35=mean < (1 << acc_bits),
                  epsilon_uint35=0 <= int(eps_q) < (1 << acc_bits),
                  statistic_uint35=0 <= stat_full < (1 << acc_bits))
    stat_rtl = stat_full & ((1 << acc_bits)-1)
    index, exponent = old.normalized_index(stat_rtl)
    raw = int(debug.rsqrt_table()[index])
    odd_constant = int(math.floor((1/math.sqrt(2)) * (1 << 14) + .5))
    odd_product = raw * odd_constant
    odd_corrected = (odd_product >> 14) if exponent & 1 else raw
    corrected = odd_corrected >> (exponent >> 1)
    phase_products = [a*corrected for a in row]
    checks.update(raw_scale_uint16=raw < (1 << 16),
                  odd_product_uint72=odd_product < (1 << 72),
                  corrected_uint36=0 <= corrected < (1 << 36),
                  phase2_product_sint72=all(-(1 << 71) <= a < (1 << 71) for a in phase_products))
    return dict(N=n, maximum_square=max(squares), square_sum=total,
                required_square_sum_bits=max(1, total.bit_length()), accumulator_bits=acc_bits,
                invN_Q20=inv, mean_product=product, mean_product_required_bits=max(1, product.bit_length()),
                mean_square=mean, exact_integer_mean=total // n,
                mean_error_vs_integer_mean=mean-total//n, mean_error_vs_rational_mean=mean-total/n,
                epsilon_q=int(eps_q), statistic=stat_full, normalized_index=index, exponent=exponent,
                raw_lut=raw, odd_corrected_lut=odd_corrected, corrected_lut=corrected,
                phase2_product_min=min(phase_products), phase2_product_max=max(phase_products),
                intermediate_checks=checks, overflow=not all(checks.values()))


def static_gain(gamma, sout):
    """Static per-channel gamma/s_out, sign + UInt16 magnitude and UInt5 shift.

    No model metrics enter approximation. Inaccurate/unrepresentable factors are
    recorded; they are never replaced with floating point in the integer mode.
    """
    gamma = np.asarray(gamma, np.float64)
    if not math.isfinite(sout) or sout <= 0 or not np.isfinite(gamma).all():
        raise ValueError('invalid static output gain')
    factors = abs(gamma) / float(sout)
    if not np.isfinite(factors).all():raise ValueError('unrepresentable static output gain')
    shifts = np.arange(32, dtype=np.int64)
    candidates = np.rint(factors[:, None] * np.exp2(shifts))
    valid = (candidates >= 1) & (candidates <= 65535)
    approximate = candidates / np.exp2(shifts)
    errors = np.where(valid, abs(approximate-factors[:, None]), np.inf)
    selected = np.argmin(errors, axis=1)
    representable = np.any(valid, axis=1) | (factors == 0)
    m = candidates[np.arange(len(gamma)), selected]
    m = np.where(representable & (factors > 0), m, 0).astype(np.int64)
    s = selected.astype(np.int64)
    sign = np.sign(gamma).astype(np.int64)
    actual = m / np.exp2(s)
    relative = np.divide(abs(actual-factors), factors, out=np.zeros_like(factors), where=factors > 0)
    return dict(multiplier=m, shift=s, sign=sign, relative_error=relative,
                representable=representable, inaccurate=relative > 1e-3)


def quantized(value, sout):
    if not math.isfinite(sout) or sout<=0:raise ValueError('invalid output scale')
    raw = np.rint(np.asarray(value, np.float64) / sout)
    return sat10(raw), float(np.mean((raw < -512) | (raw > 511)))


def mode_row(x, spec, gamma, mode, gain=None, trace=False):
    """One observed row, explicit code domain for every path."""
    x = np.asarray(x, np.float64); gamma = np.asarray(gamma, np.float64)
    si, so, eps = spec['input_scale'], spec['output_scale'], spec['eps']
    if not np.isfinite(x).all():
        raise FloatingPointError('nonfinite propagated RMSNorm input')
    q, input_clip = quantize_input10(x, si)
    info = statistic(q, debug.epsilon_q(eps, si))
    actual_q = q.astype(np.float64) * si
    true_factor = si / math.sqrt(float(np.mean(actual_q**2)) + eps)
    # Retain the mantissa-LUT value as a rational coefficient through exponent
    # normalization. No right-shift to an integer normalized value before gain.
    denom_shift = 14 + (info['exponent'] >> 1)
    numerator = info['odd_corrected_lut']
    factor = numerator / (2. ** denom_shift)
    literal_code, _, _ = debug.phase2(q, info['corrected_lut'])
    clip = 0.; code = None
    if mode == 'FP_REFERENCE':
        y = debug.reference(x, gamma-1., eps)
    elif mode == 'INPUT_QUANT_ORACLE':
        ideal = actual_q / math.sqrt(float(np.mean(actual_q**2)) + eps) * gamma
        code, clip = quantized(ideal, so); y = code.astype(np.float64) * so
    elif mode == 'OLD_EMULATOR':
        y, details = old.rms_hw_numpy(x[None], si, so, gamma-1., eps, old.norm_tables(14))
        y = y[0]; clip = details['output_clipping_rate']; code = sat10(np.rint(y/so))
    elif mode == 'RTL_LITERAL':
        code = literal_code
        # Exactly the requested comparison adapter, NOT the RTL core.
        y = code.astype(np.float64) * so * gamma
    elif mode == 'POST_GAIN_DIAGNOSTIC':
        # Undo missing 1/s_out only after the literal core has discarded fractions.
        code, clip = quantized(literal_code.astype(np.float64)*gamma, so)
        y = code.astype(np.float64)*so
    elif mode == 'FUSED_IDEAL_COEFFICIENT':
        code, clip = quantized(q.astype(np.float64)*factor*gamma, so)
        y = code.astype(np.float64)*so
    elif mode == 'FUSED_REQUANT_ORACLE':
        gain = static_gain(gamma, so) if gain is None else gain
        # Python arbitrary-precision products: q*rsqrt_mantissa*M, then one RNE.
        raw = [rounded_shift(int(a)*numerator*int(m)*int(sign), denom_shift+int(s))
               for a, m, s, sign in zip(q, gain['multiplier'], gain['shift'], gain['sign'])]
        code = np.array([min(511, max(-512, a)) for a in raw], np.int16)
        clip = sum(a < -512 or a > 511 for a in raw)/len(raw)
        y = code.astype(np.float64)*so
    else:
        raise ValueError(mode)
    measured_factor = factor
    if mode in ('FP_REFERENCE', 'INPUT_QUANT_ORACLE'):
        measured_factor = true_factor
    elif mode == 'RTL_LITERAL':
        measured_factor = info['corrected_lut'] / (1 << 14) * so
    elif mode == 'POST_GAIN_DIAGNOSTIC':
        measured_factor = info['corrected_lut'] / (1 << 14)
    elif mode == 'OLD_EMULATOR':
        old_stat = int(np.rint(np.mean(q.astype(np.int64)**2) + eps/(si*si)))
        old_idx, old_exp = old.normalized_index(old_stat)
        measured_factor = old.adjust_rsqrt(old.norm_tables(14)['rsqrt'][old_idx], old_exp, 14) / (1 << 14)
    scale_error = abs(measured_factor-true_factor)/max(true_factor, 1e-300)
    details = dict(input_clip=input_clip, output_clip=clip, scale_factor_error=scale_error,
                   overflow=info['overflow'], code=code)
    if trace:
        gain = static_gain(gamma, so) if gain is None else gain
        channels = sorted(set([0, 1, len(q)//2, len(q)-1, int(np.argmax(abs(x)))]))
        details['trace'] = dict(input_scale=si, output_scale=so, statistic=info,
            fused_rsqrt_numerator=numerator, fused_rsqrt_denominator_shift=denom_shift,
            channels=[dict(channel=i, input_integer=int(q[i]), gamma=float(gamma[i]),
                           literal_result=int(literal_code[i]),
                           ideal_output_code=int(quantized(q.astype(np.float64)*true_factor*gamma, so)[0][i]),
                           fused_pretruncation_product=int(q[i])*numerator*int(gain['multiplier'][i])*int(gain['sign'][i]),
                           M=int(gain['multiplier'][i]), S=int(gain['shift'][i]),
                           total_combined_shift=denom_shift+int(gain['shift'][i])) for i in channels])
    return y, details


def metrics(value, ref):
    value, ref = np.asarray(value, np.float64), np.asarray(ref, np.float64)
    err = value-ref
    energy = float(np.square(ref).sum())
    rms_ref, rms_value = float(np.sqrt(np.mean(ref**2))), float(np.sqrt(np.mean(value**2)))
    return dict(nmse=float(np.square(err).sum())/max(energy, 1e-300),
                mse=float(np.mean(err**2)), mae=float(np.mean(abs(err))), cosine=debug.cosine(value, ref),
                fp_output_rms=rms_ref, reconstructed_output_rms=rms_value,
                output_rms_ratio=rms_value/max(rms_ref, 1e-300))


def accumulator_checks(samples, exp):
    rng = np.random.default_rng(20261102)
    rows = []
    for n in (256, 512, 1024, 2048, 4096):
        for name, q in (('negative_boundary', [-512]*n), ('positive_boundary', [511]*n),
                        ('alternating', [-512, 511]*(n//2)), ('zero', [0]*n),
                        ('random', rng.integers(-512, 512, n).tolist())):
            rows.append(dict(identity=f'{name}_{n}', **statistic(q)))
    for name, group_samples in samples.items():
        spec = exp.specs[name]
        for group, values in group_samples.items():
            for i, x in enumerate(values):
                q, _ = quantize_input10(x, spec['input_scale'])
                rows.append(dict(identity=f'{name}:{group}:{i}', **statistic(q, debug.epsilon_q(spec['eps'], spec['input_scale']))))
    result = dict(accumulator_bits=35, tests=rows, overflow_detected=any(r['overflow'] for r in rows),
                  maximum_square_sum=max(r['square_sum'] for r in rows),
                  maximum_required_square_sum_bits=max(r['required_square_sum_bits'] for r in rows),
                  chisel_widths=dict(square_signed=20, accumulator_unsigned=35, invN_unsigned=24,
                                    mean_multiply_unsigned=59, statistic_add_carry=36,
                                    stored_statistic_unsigned=35, raw_scale_unsigned=16,
                                    scale_register_unsigned=36, odd_product_unsigned=72,
                                    phase2_x_signed=35, phase2_scale_zext_signed=37,
                                    phase2_product_signed=72),
                  semantics='Unsigned sums/statistic retain low accBits after +&; full multiply width before >>; Python int oracle checks before any masking.')
    write_json('accumulator_verification.json', result)
    return result


def capture_once(exp, calibration, identity):
    """Reuse this probe's fingerprinted capture or capture the existing four chats once.

    Older debug NPZ lacks a content hash and model/weight/config fingerprint, so
    filename alone cannot authorize its reuse.
    """
    npz, meta = WORK/'capture.npz', WORK/'capture.json'
    if npz.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved['identity'] == identity and saved['sha256'] == sha(npz):
            with np.load(npz, allow_pickle=False) as arrays:
                samples = {name: {g: arrays[f'{i}_{g}'].copy() for g in ('prefill', 'decode')}
                           for i, name in enumerate(TARGET_MODULES)}
            return samples, dict(reused=True, **saved)
    for example in calibration:
        exp.label='scale_probe_calibration'; exp.conversation=example['index']; cache=None
        # Existing actual activation hooks and original FP Norm predecessor path.
        with exp.patch(capture=True, gpalu=False):
            for t in range(min(mixed.CAL_LIMIT, len(example['targets']))):
                exp.group='prefill' if t == 0 else 'decode'
                exp.offset=0 if t == 0 else len(example['prompt_ids'])+t-1
                _, _, cache=exp.forward(example['prompt_ids'] if t == 0 else [example['targets'][t-1]], cache)
    samples = {}
    for name in TARGET_MODULES:
        samples[name] = {}
        for group in ('prefill', 'decode'):
            values = exp.samples.get(('scale_probe_calibration', name, group), [])
            if not values:
                raise RuntimeError(f'no actual {group} sample for {name}')
            samples[name][group]=np.stack([x for x, _ in values])
    np.savez_compressed(npz, **{f'{i}_{g}': samples[n][g] for i,n in enumerate(TARGET_MODULES) for g in ('prefill','decode')})
    saved=dict(identity=identity, sha256=sha(npz), examples=[mixed.identity(e) for e in calibration],
               rows_per_module_group=mixed.RESERVOIR, response_limit=mixed.CAL_LIMIT,
               previous_debug_capture_reuse='not trusted: missing model/config/content-hash binding')
    write_json('work/capture.json', saved)
    return samples, dict(reused=False, **saved)


def local_analysis(exp, samples):
    rows, distributions, traces, decomposition = [], [], [], []
    for name, groups in samples.items():
        spec=exp.specs[name]; gamma=1+exp.weights[name]; gain=static_gain(gamma,spec['output_scale'])
        for group,x in groups.items():
            reference=np.stack([debug.reference(a,gamma-1,spec['eps']) for a in x])
            outputs={}
            for mode in LOCAL_MODES:
                values=[]; details=[]
                for a in x:
                    y,d=mode_row(a,spec,gamma,mode,gain,trace=(mode=='FUSED_REQUANT_ORACLE' and len(values)==0))
                    values.append(y); details.append(d)
                value=np.stack(values);outputs[mode]=value
                row=dict(module=name,group=group,mode=mode,rows=len(x), **metrics(value,reference),
                         input_clipping_rate=float(np.mean([d['input_clip'] for d in details])),
                         output_clipping_rate=float(np.mean([d['output_clip'] for d in details])),
                         mean_relative_scale_error=float(np.mean([d['scale_factor_error'] for d in details])),
                         max_relative_scale_error=max(d['scale_factor_error'] for d in details),
                         static_gain_mean_relative_error=float(gain['relative_error'].mean()),
                         static_gain_max_relative_error=float(gain['relative_error'].max()),
                         unrepresentable_static_gain_channels=int(np.count_nonzero(~gain['representable'])),
                         inaccurate_static_gain_channels=int(np.count_nonzero(gain['inaccurate'])))
                rows.append(row)
                codes=[d['code'] for d in details if d['code'] is not None]
                if codes:
                    codes=np.stack(codes)
                    distributions.append(dict(module=name,group=group,mode=mode,zero_percent=100*float(np.mean(codes==0)),
                        distinct_codes=int(np.unique(codes).size),code_p1=float(np.percentile(codes,1)),code_p50=float(np.percentile(codes,50)),
                        code_p99=float(np.percentile(codes,99)),max_magnitude=int(np.max(abs(codes.astype(np.int64)))),
                        code_domain='literal normalized core before FP gamma adapter' if mode=='RTL_LITERAL' else 'final INT10 output'))
                if mode=='FUSED_REQUANT_ORACLE':
                    trace=details[0]['trace'];trace.update(module=name,group=group,row_id=0,fp_reference_rms=float(np.sqrt(np.mean(reference[0]**2))),
                        fused_output_rms=float(np.sqrt(np.mean(value[0]**2))))
                    literal,_=mode_row(x[0],spec,gamma,'RTL_LITERAL',gain)
                    trace['rtl_output_rms']=float(np.sqrt(np.mean(literal**2)));traces.append(trace)
            for label,a,b in (('input_output_INT10','INPUT_QUANT_ORACLE','FP_REFERENCE'),
                              ('LUT_and_fixed_point','FUSED_REQUANT_ORACLE','INPUT_QUANT_ORACLE'),
                              ('missing_gain_early_truncation','RTL_LITERAL','FUSED_REQUANT_ORACLE'),
                              ('post_gain_vs_literal','POST_GAIN_DIAGNOSTIC','RTL_LITERAL'),
                              ('old_vs_fused','OLD_EMULATOR','FUSED_REQUANT_ORACLE'),
                              ('integer_MS_vs_ideal_fused','FUSED_REQUANT_ORACLE','FUSED_IDEAL_COEFFICIENT')):
                decomposition.append(dict(module=name,group=group,comparison=label,reference=b,tested=a,**metrics(outputs[a],outputs[b])))
    write_csv('norm_local_comparison.csv',rows);write_csv('output_code_distribution.csv',distributions)
    write_json('scaling_trace.json',dict(vectors=traces,decomposition=decomposition,
        static_MS='Per-channel gamma/s_out gain M:UInt16, S:UInt5, sign; combine q*rsqrt_mantissa*M before final RNE. Total shift adds exponent/14 fractional bits; requires wider hypothetical fused hardware.',
        ideal_and_integer_fused_separate=True,rtl_literal_gamma_adapter_not_part_of_RTL=True))
    return rows,decomposition


@contextlib.contextmanager
def rms_patch(exp, mode, gains):
    originals=[]
    try:
        for name in exp.names:
            module=exp.modules[name];originals.append((module,module.forward))
            def forward(value, name=name):
                x=value.detach().cpu().float().numpy();shape=x.shape
                gamma=1+exp.weights[name];spec=exp.specs[name]
                out=np.stack([mode_row(row,spec,gamma,mode,gains[name])[0] for row in x.reshape(-1,shape[-1])])
                if not np.isfinite(out).all(): raise FloatingPointError(f'nonfinite RMSNorm output: {name}/{mode}')
                return value.new_tensor(out.reshape(shape))
            module.forward=forward
        yield
    finally:
        for module,original in originals:module.forward=original


def stable_ids(example):
    return {str(example[k]) for k in ('anchor_id','response_id','tree_id') if example.get(k) is not None}


def small_e2e(exp, calibration, run_identity):
    used=set().union(*(stable_ids(e) for e in calibration))
    # Fixed first three in the existing ordered set; never replace based on metrics.
    examples=mixed.e2e_examples(exp.source)[:3]
    if len(examples)!=3 or any(len(e['targets'])<8 for e in examples):raise RuntimeError('24-target subset unavailable')
    if any(stable_ids(e) & used for e in examples):raise RuntimeError('calibration/E2E identity overlap')
    gains={n:static_gain(1+exp.weights[n],exp.specs[n]['output_scale']) for n in exp.names}
    records=[]
    result=dict(targets=24,sequences=3,targets_per_sequence=8,
                examples=[mixed.identity(e) for e in examples],
                dataset_identity=old.digest([dict(identity=mixed.identity(e),prompt=e['prompt_ids'],targets=e['targets'][:8]) for e in examples]),
                calibration_overlap=False,Softmax='FP',RoPE='FP',modes={},completed_targets=0,error=None)
    write_json('e2e_small_summary.json',result)
    try:
        for i,example in enumerate(examples):
            checkpoint=WORK/f'e2e_{i:02d}.json'
            key=old.digest(dict(run=run_identity,data=result['dataset_identity'],sequence=i))
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text())
                if saved.get('identity') != key: raise RuntimeError('incompatible E2E checkpoint')
                records.append(saved['value']);continue
            scores={m:response.Scores() for m in E2E_MODES};caches={m:None for m in E2E_MODES}
            for t,target in enumerate(example['targets'][:8]):
                ids=example['prompt_ids'] if t==0 else [example['targets'][t-1]]
                fp,_,caches['FP_FULL']=debug.cal.run_fp(exp.source,ids,caches['FP_FULL'],layers=False)
                scores['FP_FULL'].add(exp.source.torch,fp,fp,[target])
                for mode in E2E_MODES[1:]:
                    with w.patch_width(exp.source,exp.gate,exp.up,exp.luts,exp.gpalu):
                        with (rms_patch(exp,E2E_MAP[mode],gains) if mode in E2E_MAP else contextlib.nullcontext()):
                            logits,_,caches[mode]=exp.forward(ids,caches[mode])
                    if not exp.source.torch.isfinite(logits).all():raise FloatingPointError(f'nonfinite logits {mode}')
                    scores[mode].add(exp.source.torch,logits,fp,[target])
                if len(set(map(id,caches.values())))!=6:raise RuntimeError('shared KV cache')
            value={m:scores[m].raw() for m in E2E_MODES}
            write_json(f'work/e2e_{i:02d}.json',dict(identity=key,value=value));records.append(value)
        result['modes']={m:merge_scores([r[m] for r in records]) for m in E2E_MODES}
        if any(m['tokens']!=24 for m in result['modes'].values()):
            raise FloatingPointError('invalid final E2E metric/count')
        result['completed_targets']=24
    except Exception as exc:
        result['error']=dict(type=type(exc).__name__,message=str(exc))
        result['completed_targets']=8*len(records)
        if records:result['partial_modes']={m:merge_scores([r[m] for r in records]) for m in E2E_MODES}
        traceback.print_exc()
    write_json('e2e_small_summary.json',result)
    return result


def diagnose(local, accumulator, e2e):
    means={mode:float(np.mean([r['nmse'] for r in local if r['mode']==mode])) for mode in LOCAL_MODES}
    amplitudes={mode:float(np.mean([r['output_rms_ratio'] for r in local if r['mode']==mode])) for mode in LOCAL_MODES}
    tests=dict(local_fused_reduces_literal_error_50pct=means['FUSED_REQUANT_ORACLE'] < .5*means['RTL_LITERAL'],
               literal_amplitude_below_half_oracle=amplitudes['RTL_LITERAL'] < .5*amplitudes['INPUT_QUANT_ORACLE'],
               post_gain_error_more_than_twice_fused=means['POST_GAIN_DIAGNOSTIC'] > 2*means['FUSED_REQUANT_ORACLE'],
               input_oracle_nmse_above_5pct=means['INPUT_QUANT_ORACLE']>.05)
    em=e2e.get('modes',{})
    tests['small_e2e_fused_improves_literal']=bool(em and em['GPALU10_FUSED_REQUANT_NORM']['nmse'] < .5*em['GPALU10_RTL_LITERAL_NORM']['nmse']
        and em['GPALU10_FUSED_REQUANT_NORM']['kl'] < em['GPALU10_RTL_LITERAL_NORM']['kl'])
    supported=[]
    if accumulator['overflow_detected']:supported.append('ACCUMULATOR_ERROR')
    if tests['literal_amplitude_below_half_oracle'] and tests['local_fused_reduces_literal_error_50pct'] and tests['small_e2e_fused_improves_literal']:
        supported.append('OUTPUT_SCALE_CONTRACT_CONFIRMED')
    # Early truncation can coexist with the missing gain: report both, not force
    # a unique mechanism. Amplitude restoration is measured, not presumed.
    restored=.5 <= amplitudes['POST_GAIN_DIAGNOSTIC'] <= 1.5
    if restored and tests['post_gain_error_more_than_twice_fused']:supported.append('EARLY_TRUNCATION_IS_DOMINANT')
    if tests['input_oracle_nmse_above_5pct']:supported.append('INT10_PRECISION_LIMIT_SUSPECTED')
    primary=supported[0] if len(supported)==1 else 'MIXED_OR_INCONCLUSIVE'
    return dict(primary=primary,supported_cases=supported,descriptive_checks=tests,
                mean_local_nmse=means,mean_output_rms_ratio=amplitudes,
                thresholds='Descriptive rules fixed in source before execution; no scale/parameter acceptance depends on E2E.')


def write_report(local, decomposition, accumulator, e2e, diagnosis):
    means=diagnosis['mean_local_nmse'];amp=diagnosis['mean_output_rms_ratio']
    lines=['# INT10 RMSNorm output-scale probe','',
        '## 1. Exact hypothesis','',
        'A missing output gain and early phase-2 integer truncation may explain the Norm collapse. Accumulator overflow is tested independently. All scales and GPALU parameters are frozen.',
        '', '## 2. Why the prior debug runner failed','',
        'The old runner rejected unrelated staged files at INITIALIZING. This probe never requires an empty index. Scientific DONE is written before isolated-worktree publication; a Git failure only creates PUSH_FAILED.',
        '', '## 3. Accumulator width verification','',
        f"Overflow detected: {accumulator['overflow_detected']}. Maximum tested square sum {accumulator['maximum_square_sum']}; required bits {accumulator['maximum_required_square_sum_bits']}; configured accumulator 35 bits. Full stage capacities and observed products are in accumulator_verification.json.",
        '', '## 4. RMSNorm integer scaling derivation','',
        '`q=RNE(x/s_in)`, `mean=(sum(q²)*round(2^20/N))>>20`, `epsilon_q=RNE(eps/s_in²)`. The literal rsqrt Q2.14 coefficient is exponent-corrected with truncating shifts. RTL phase 2 is arithmetic `q*scale>>14`, then INT10 saturation.',
        'Correct model output codes need `RNE(q*rsqrt(stat)*gamma/s_out)`. The literal comparison uses `q_literal*s_out*gamma`; gamma is a software adapter outside RTL. POST_GAIN uses the already-truncated literal value, gamma, then RNE/s_out.',
        'Fused ideal retains the exponent-normalized Q2.14 mantissa as a rational. Integer fused uses static per-channel gamma/s_out with sign, UInt16 M, UInt5 S. A wide product q*mantissa*M receives one signed RNE by the combined exponent/fraction/gain shift. This is hypothetical fused hardware; it is not the current core. Unsupported/inaccurate gain factors are recorded, never silently replaced with float.',
        '', '## 5. Old emulator vs literal RTL','',
        'OLD_EMULATOR is reused unmodified. It keeps fractional normalized values, applies gamma, and quantizes with s_out; literal RTL truncates earlier. Previous combined Norm experiments also replaced Softmax, so their 6–8% Top1 is not an isolated RMSNorm precision result.',
        '', '## 6. Local real-activation measurements','',
        'Only actual rows from the existing calibration population are used. Per-module prefill/decode metrics are in norm_local_comparison.csv. Below are unweighted means of per-module/group NMSE, not pooled-energy metrics.',
        '', '| Mode | Mean local NMSE | Mean output RMS / FP RMS |','|---|---:|---:|']
    for mode in LOCAL_MODES:lines.append(f'| {mode} | {means[mode]:.9g} | {amp[mode]:.9g} |')
    lines += ['', '## 7. Output-code utilization','',
        'output_code_distribution.csv records zero fraction, distinct codes, p1/p50/p99, and maximum magnitude. Literal codes are normalized core codes before the FP gamma adapter; other codes are final INT10 output codes.',
        '', '## 8. Missing-gain vs early-truncation decomposition','',
        'scaling_trace.json contains pairwise measured errors: input/output quantization, LUT/fixed point, literal-to-fused, post-gain-to-literal, old-to-fused, and integer M/S-to-ideal fused. Input/output oracle includes final output quantization, so it is not a pure input-only error. Gamma and clipping are included in these paths; this is not a separate gamma-storage hardware experiment.',
        '', '## 9. Small 24-token E2E results','',
        'Three fixed conversations × eight targets from the existing E2E set, disjoint from calibration by anchor/response/tree IDs. Every mode owns its cache; Softmax and RoPE stay FP. These 24-token percentages must not be compared directly with the old 192-token 85.42% Top1.']
    if e2e['error']:lines += ['', f"E2E encountered {e2e['error']}; local scientific artifacts are preserved. Completed {e2e['completed_targets']} targets."]
    else:
        keys=('nll','ppl','kl','nmse','mae','cosine','top1','in5','overlap')
        lines += ['', '| Mode | '+' | '.join(keys)+' |','|---|'+'---:|'*len(keys)]
        for mode,m in e2e['modes'].items():lines.append('| '+mode+' | '+' | '.join(f'{m[k]:.9g}' if m[k] is not None else 'above float64 range (see log_ppl)' for k in keys)+' |')
    lines += ['', '## 10. Primary diagnosis','',diagnosis['primary'],
        'Supported descriptive cases: '+', '.join(diagnosis['supported_cases'])+'.',
        json.dumps(diagnosis['descriptive_checks'],sort_keys=True),
        '', '## 11. What this establishes','',
        f"Q1. Overflow detected in tested boundary/actual vectors: {accumulator['overflow_detected']}.",
        f"Q2. Literal output amplitude ratio is {amp['RTL_LITERAL']:.9g}, compared with {amp['INPUT_QUANT_ORACLE']:.9g} for the proper INT10 oracle; local NMSE is {means['RTL_LITERAL']:.9g}.",
        f"Q3. Fused gain before requantization gives local NMSE {means['FUSED_REQUANT_ORACLE']:.9g}; ideal fused {means['FUSED_IDEAL_COEFFICIENT']:.9g}. E2E evidence is limited to the table above.",
        f"Q4. Post-gain local NMSE {means['POST_GAIN_DIAGNOSTIC']:.9g}, RMS ratio {amp['POST_GAIN_DIAGNOSTIC']:.9g}, versus fused NMSE {means['FUSED_REQUANT_ORACLE']:.9g}; this quantifies what cannot be recovered after truncation.",
        f"Q5. The INT10 input/output oracle NMSE {means['INPUT_QUANT_ORACLE']:.9g} is evaluated separately from the literal implementation. The measured diagnosis is {diagnosis['primary']}; the former combined Norm/Softmax result alone cannot identify the cause.",
        '', '## 12. What remains unverified','',
        'This is a tiny CPU numerical probe, not RTL co-simulation or synthesis. Fused coefficient banking, gamma storage, combined shifts/intermediate widths, compiler integration, and wider unseen validation remain unverified. Adding one gain is not guaranteed to fix production RTL. No E2E metrics selected coefficients or scales; no follow-up experiment is run automatically.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def scientific_done(required, tests_ok):
    return bool(tests_ok and all(Path(p).is_file() for p in required))


def write_debugging_history(e2e,accumulator):
    text='''# Norm10 scale probe debugging history

## Original recorded failure

The requested scale-probe directory initially contained only `pid` and an empty
`run.log`; it had no status, FAILED, capture or numeric traceback. PID 98026 was
not alive. Therefore no scale-probe-specific original overflow can be asserted.

The preserved historical `diagnostics/norm10_debug_pilot/run.log`, lines 10–21,
does record `OverflowError: math range error` at
`scripts/evaluate_response_quant.py:223`, function `merged`, from `math.exp(nll)`.
Its callers were `run_norm10_debug_pilot.py:178` (`measure`) and line 238
(`sweeps`), in CUMULATIVE_SWEEP. This is derived-PPL float64 range exhaustion,
not evidence of accumulator overflow. Exact original NLL/logits were not saved
at the failing point; they cannot be recovered from the traceback. The code
uses torch float64 log-softmax and Python float aggregation; the diagnostic had
two target predictors. The actual failing logits shape and numeric NLL are
unknown, rather than guessed from the architecture.

A deterministic minimal score record with shape (1,8) float64 operands and
NLL=1000 reproduces the original aggregation OverflowError. Float64 exp has
maximum finite log input approximately 709.782712893384. This reproducer is
independent of model loading and hardware arithmetic.

## Debug iteration 1: derived metric and integer-boundary fixes

Regression tests were written first. The original response.merged demonstrably
raises on the finite NLL=1000 record. The new merge_scores adapter reuses the
existing accumulators and preserves NLL as log_ppl; PPL is null with an explicit
`exceeds_float64_range` label when mathematically outside float64. No NLL or
logit is clipped, and a nonfinite primary metric still fails explicitly.
The historical scoring implementation is unchanged.

Audit also found that the inherited input quantizer casts to np.int64 before
INT10 saturation: finite input +1e30 exceeds the intermediate's representation.
The exact cast behavior is platform dependent; on this Mac it clips at INT64_MAX
and the final INT10 sign happens to remain correct. The regression demonstrates
the out-of-range intermediate, not an observed wrong-sign model activation.
The probe's new input
quantizer rounds and saturates before integer conversion. Nonfinite activations
are rejected explicitly; finite quotient overflow saturates with its exact sign.
OLD_EMULATOR remains unmodified as requested. Static output gains validate
positive finite s_out before division. Signed arbitrary-precision integer
products, RNE, RTL shifts and all hardware widths are retained.

## Execution and scope

The runner's os.setsid call was removed. The foreground command directly owns
the interpreter and is awaited. An identified old launchd job,
gui/501/com.lzero.norm10-debug-pilot (then PID 33980, command
run_norm10_debug_pilot.py), was booted out to stop its automatic restarts and
concurrent historical-log mutation. No unrelated process was terminated.

Existing staged/unstaged changes and historical artifacts are preserved.
Publication uses a detached Git worktree only after scientific DONE. The worktree
is a Git checkout, not a detached experiment process.

## Remaining limitations

The original failing NLL and logits were not logged. Historical combined Norm
modes also changed Softmax. This probe keeps Softmax and RoPE FP, uses 24 aligned
targets, and does not establish production RTL integration or broad accuracy.
Null PPL is a representational limitation with finite log_ppl, not fabricated
finite precision. Hardware findings, if any, remain in accumulator_verification.
'''
    text+=f"\nObserved hardware overflow: {accumulator['overflow_detected']}.\n"
    text+=f"Final small E2E completed targets: {e2e['completed_targets']}; error: {e2e['error']}.\n"
    (OUT/'debugging_history.md').write_text(text)


def mark_experiment_done():
    required=[OUT/p for p in RESULT_FILES]
    if not scientific_done(required,True):raise RuntimeError('scientific artifacts incomplete')
    (OUT/'DONE').write_text('EXPERIMENT_COMPLETE\nreport=diagnostics/norm10_scale_probe/report.md\n')
    set_stage('REPORT','EXPERIMENT_DONE',report='diagnostics/norm10_scale_probe/report.md',publication='pending')


def publish(source_head, index_before):
    """Separate linked worktree/index; never stage in ROOT and never force push."""
    set_stage('PUBLICATION','EXPERIMENT_DONE',publication='pending')
    temporary=Path(tempfile.mkdtemp(prefix='norm10-scale-publish-'))
    checkout=temporary/'checkout';added=False
    try:
        remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
        allowed=('https://github.com/l-zero-society/2026-1-edge-llm-npu-sw.git',
                 'git@github.com:l-zero-society/2026-1-edge-llm-npu-sw.git')
        if remote not in allowed:raise RuntimeError('unexpected publication remote')
        subprocess.run(['git','worktree','add','--detach',str(checkout),source_head],cwd=ROOT,check=True)
        added=True
        for name in PUBLISH_FILES:
            destination=checkout/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination)
        subprocess.run(['git','add','--',*PUBLISH_FILES],cwd=checkout,check=True)
        subprocess.run(['git','commit','-m','test: diagnose INT10 RMSNorm output scale'],cwd=checkout,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
        branch='norm10-scale-probe-results'
        for attempt in range(20):
            candidate=branch if attempt==0 else branch+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+str(attempt)
            exists=subprocess.run(['git','ls-remote','--exit-code','--heads','origin',candidate],cwd=checkout,capture_output=True,text=True)
            if exists.returncode==2:
                branch=candidate;break
            if exists.returncode!=0:raise RuntimeError('remote branch lookup failed: '+exists.stderr.strip())
        else:raise RuntimeError('cannot find nonconflicting publication branch')
        # Normal push only: a concurrent conflicting branch creation is rejected.
        subprocess.run(['git','push','origin',f'HEAD:refs/heads/{branch}'],cwd=checkout,check=True)
        index_after=index_fingerprint()
        metadata=dict(branch=branch,commit=commit,url=f'https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/tree/{branch}',
                      active_index_unchanged=(index_after==index_before),
                      publication_used_user_index=False)
        write_json('publication.json',metadata)
        (OUT/'PUBLISHED').write_text(json.dumps(metadata,indent=2)+'\n')
        set_stage('PUBLICATION','EXPERIMENT_DONE',publication='PUBLISHED',**metadata)
    except Exception as exc:
        traceback.print_exc();(OUT/'PUSH_FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        set_stage('PUBLICATION','EXPERIMENT_DONE',publication='PUSH_FAILED',error=str(exc))
    finally:
        if added:
            # Only this runner-created linked worktree; never the user's checkout.
            removed=subprocess.run(['git','worktree','remove',str(checkout)],cwd=ROOT,capture_output=True,text=True)
            if removed.returncode:print('Publication worktree cleanup warning: '+removed.stderr,flush=True)
        if not checkout.exists():temporary.rmdir()


def index_fingerprint():
    path=subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=ROOT,text=True).strip()
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    return sha(path) if path.exists() else None


def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
    # Identity and index are recorded, never required to be clean.
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='lut_cali':raise RuntimeError('expected branch lut_cali')
    source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    index_before=index_fingerprint()
    set_stage('UNIT_TESTS')
    cmd=[sys.executable,'-m','unittest','tests.test_norm10_scale_probe','tests.test_norm10_debug_pilot']
    with (OUT/'unit_test.log').open('w') as f:
        test_rc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    if test_rc:raise RuntimeError('unit tests failed')
    protected=[old.GPALU/'int10/gpalu_parameters.json',old.GPALU/'baseline_manifest.json',
               old.OUT/'norm10_coeff16_parameters.json',ROOT/'scripts/run_norm10_debug_pilot.py',
               ROOT/'scripts/run_vpu10_e2e_pilot.py',ROOT/'scripts/run_gpalu_width_sweep_pilot.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    sources={p:sha(ROOT/p) for p in SOURCE_FILES}
    rtl_files=('src/main/scala/npu/core/NormUnit.scala','src/test/scala/NormUnit_Test.scala')
    rtl_hashes={p:sha(old.RTL/p) for p in rtl_files}
    set_stage('MODEL_LOAD');exp=debug.load()
    if str(next(exp.source.model.parameters()).device)!='cpu':raise RuntimeError('CPU-only experiment')
    calibration=mixed.calibration_examples(exp.source)
    capture_identity=old.digest(dict(baseline=before,capture_config=dict(version=1,predecessor='FP',rows=mixed.RESERVOIR,response_limit=mixed.CAL_LIMIT),rtl=rtl_hashes,
            model=exp.source.metadata if hasattr(exp.source,'metadata') else 'recovered model loader',
            weights={n:hashlib.sha256(exp.weights[n].tobytes()).hexdigest() for n in exp.names},
            examples=[dict(identity=mixed.identity(e),prompt=e['prompt_ids'],targets=e['targets'][:mixed.CAL_LIMIT]) for e in calibration]))
    set_stage('CAPTURE');samples,capture_meta=capture_once(exp,calibration,capture_identity)
    set_stage('ACCUMULATOR');accumulator=accumulator_checks(samples,exp)
    set_stage('LOCAL_SCALE');local,decomposition=local_analysis(exp,samples)
    set_stage('E2E_SMALL');e2e=small_e2e(exp,calibration,old.digest(dict(capture=capture_identity,sources=sources)))
    set_stage('REPORT');diagnosis=diagnose(local,accumulator,e2e)
    after={str(p.relative_to(ROOT)):sha(p) for p in protected}
    if before!=after:raise RuntimeError('historical source/parameters changed during run')
    if {p:sha(ROOT/p) for p in SOURCE_FILES}!=sources:raise RuntimeError('probe source changed during run')
    verification=dict(branch=branch,source_HEAD=source_head,execution_device='CPU',MPS_used=False,CUDA_used=False,ANE_used=False,
        baseline_before=before,baseline_after=after,baseline_unchanged=True,rtl_hashes=rtl_hashes,
        calibration_capture=capture_meta,activation_modules=list(TARGET_MODULES),all_e2e_norm_modules=exp.names,
        fixed_scale_source=str(old.OUT/'norm10_coeff16_parameters.json'),scale_search_performed=False,
        active_index_before=index_before,active_index_after_science=index_fingerprint(),
        E2E_used_for_selection=False,calibration_evaluation_overlap=False,independent_KV_caches=True,
        e2e_complete=e2e['error'] is None,e2e_error=e2e['error'],e2e_dataset_identity=e2e['dataset_identity'],
        full_192_token_baseline_not_compared_to_24_token_percentage=True,
        tests_command=cmd,test_return_code=test_rc,diagnosis=diagnosis,elapsed_seconds=time.monotonic()-START)
    write_json('verification.json',verification);write_report(local,decomposition,accumulator,e2e,diagnosis)
    write_debugging_history(e2e,accumulator)
    mark_experiment_done()
    publish(source_head,index_before)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        traceback.print_exc();OUT.mkdir(parents=True,exist_ok=True)
        if (OUT/'DONE').exists():
            (OUT/'PUSH_FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
            set_stage('PUBLICATION','EXPERIMENT_DONE',publication='PUSH_FAILED',error=str(exc))
        else:
            (OUT/'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
            write_json('status.json',dict(state='FAILED',stage=STAGE,exception_type=type(exc).__name__,error=str(exc)))
        sys.exit(1)
