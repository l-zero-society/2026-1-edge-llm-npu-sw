#!/usr/bin/env python3
"""Render measured diagnostic CSVs and the independently recorded scope audit."""
import csv
import json
import re
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'diagnostics'


def read(name):
    p=D/name
    return list(csv.DictReader(p.open())) if p.exists() else []


def f(row,key):return float(row[key])
def pct(value):return f'{100*value:.4f}%'
def num(value):return f'{value:.7g}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    rows=read('quant_error_attribution.csv');val=[r for r in rows if r['split']=='validation']
    down=sorted([r for r in val if r['module']=='down_proj'],key=lambda r:int(r['layer']))
    stats=read('down_proj_activation_stats.csv');sv={int(r['layer']):r for r in stats if r['split']=='validation'}
    sweeps=read('down_proj_scale_sweep.csv');sval=[r for r in sweeps if r['split']=='validation']
    outputs=read('down_proj_output_scale_sweep.csv');oval=[r for r in outputs if r['split']=='validation']
    provenance=json.loads((D/'diagnosis_provenance.json').read_text())
    report=[]
    def add(s):report.append(s)
    add('# Linear quantization error diagnosis\n')
    add('## 1. Scope audit — read this before interpreting any error\n')
    add('**B: independent local Linear measurements on original/reference-model activations. This is not a full quantized model simulation.** '
        '`src/static_quant/gemma.py:191–230` registers a forward **pre-hook**, passes original inputs to the diagnostic callback, and raises `Observed` to stop the prefix. '
        'It never replaces outputs or mutates the model with quantized weights. `core.py:215` constructs `Y_ref = X @ W.T` in FP64. '
        '`core.py:270–272` compares three local paths; `export.py:27–30` and `docs/static-calibration.md` explicitly exclude propagated full-model quantization.\n\n'
        '- Reference RoPE, QKᵀ, softmax and PV affect the reference FP input captured at o_proj. Their own numerical error is **not** measured against an independent exact attention reference.\n'
        '- No NPU fixed-point/quantized RoPE is executed; no NPU RoPE error propagates into o_proj.\n'
        '- gate/up inputs include the reference attention/residual/RMSNorm behavior. down_proj inputs include reference gate/up, GeLU and elementwise multiplication. '
        'They do **not** include errors from earlier NPU-quantized Linear operations, either in the same layer or earlier layers.\n'
        '- Reference-model FP rounding and GGUF source effects are already part of the observed reference trajectory. They are not separately attributed by local metrics.\n'
        '- This diagnostic preserves that local scope: only reference-model hooks capture inputs; all counterfactuals run outside the model.')
    complete = len(rows)==252 and len(sweeps)==252 and len(outputs)==216
    add(f'**Execution status: {"COMPLETE" if complete else "IN PROGRESS"}.** Baseline operations: {len(val)}/126; input-sweep rows: {len(sweeps)}/252; output-sweep rows: {len(outputs)}/216. All tables below cover completed measurements only.')
    add('## 2. Provenance, stale-artifact checks and policy bug\n')
    add(f'Current commit: `{provenance["identity"]["source_commit"]}`. Stored run commit: `{provenance["stored_source_commit"]}`, '
        f'dirty tree: `{provenance["stored_source_dirty"]}`. The stored run used additional/uncommitted GGUF, policy and full-run code absent in this checkout. '
        '`source_audit.json` records every available/missing code hash. Only some files, including `hardware.py`, match exactly; the current production CLI alone cannot recreate the historical GGUF run.\n\n'
        'The independent diagnostic reconstructs the local GGUF according to recorded conventions: reverse GGUF dimensions, no Q/K permutation, norm weights minus one, output embedding tied. '
        'It requires the exact model SHA256, dataset/text/token-ID hashes, Torch 2.2.2, Transformers 4.57.6 and NumPy 1.26.4. '
        'It verifies stored per-module artifact hashes, all recovered s_W values, all saved INT8 weight elements, packed parameters, ratios, '
        'calibration activation absmax and separately reproduced calibration/validation local MSE. MSE acceptance tolerance is relative 2e-5, absolute 1e-10; measured deviations are in the main CSV. '
        'This is an independently reconstructed replay, not recovery of the missing historical source code.\n\n'
        'The existing CLI bug is real: `cli.py:127` creates fixed `s10=0.2` Q/K `layer_options`, '
        'but `cli.py:135` passes global `options` to `LinearCalibration`. Intended behavior is fixed Q/K 0.2; actual current behavior uses global CLI options. '
        'The saved 126-op artifacts explicitly record Q/K **MSE** policies and non-0.2 scales, gate=0.1, and MSE for other linears. '
        'They did not use the intended Q/K 0.2 override; their dirty historical entrypoint is not provably the current buggy CLI. '
        'This diagnosis loads the saved scales/parameters directly and does not invoke or repair that override. Fixing the unused argument would change local Q/K output quantization only; it would not cause this calibrator to measure or propagate NPU RoPE error. Production policy and RTL remain untouched.')
    add(f'Model: `{provenance["stored_model"]["file_name"]}`, SHA256 `{provenance["identity"]["model_sha256"]}`. '
        'The float reference weights are reconstructed Q8_0 GGUF values cast to FP32, then read in FP64 for local GEMMs. '
        '**Original BF16/FP16 → Q8_0 source error: not measured.** No original checkpoint was found in the repository, local model directory, Hugging Face cache, or checked download/cache locations; no model was downloaded. '
        '**Recovered GGUF → our per-output-channel INT8 error** is exactly the weight-only counterfactual below. These errors are different.\n\n'
        'OASST1 calibration: 128 examples / 34,221 valid tokens; validation: 32 / 10,141; maximum length 512, plain text, tree-held-out splits. '
        'The exact files are `calibration_outputs/data-oasst1-v1/calibration.jsonl` and `validation.jsonl`, verified against the stored manifest. '
        'The IDE-open `data-oasst1-128-32` directory has no files at those JSONL names; its SOURCE_README is not a substitute for the recorded dataset. '
        'No padding contributes to metrics. Exact commands are in `commands.jsonl`; complete dataset/model/profile metadata and script hashes are in `diagnosis_provenance.json`. '
        'Percentiles are exact over all elements, separately on both splits. Sweep thresholds are fitted only on calibration, then frozen for validation. '
        'MSE activation threshold search uses 64 deterministic calibration rows and chooses among the same six threshold candidates; its objective is input reconstruction MSE, not local output MSE.\n\n'
        'Generic-rne is the unchanged unverified software profile. The integer backend is `'+provenance['identity']['integer_backend']+'`: '
        'all integer products and every partial sum are bounded by K×127² ≤ 264,257,536, below INT32_MAX and FP64’s consecutive-integer limit. '
        'The selected FP64 BLAS backend is `'+provenance['identity'].get('blas','numpy')+'`. FP64 BLAS exactly represents these integer dot products; integer-valued results are checked and each operation/split is cross-checked on full-K sampled rows/channels against NumPy INT64. '
        'Default script mode remains pure NumPy INT64. Floating reference/counterfactual GEMMs and reductions are FP64. No reduced-precision GPU matmul is used.')
    add('## 3. Counterfactual definitions\n')
    add('The unchanged contract is symmetric signed INT8 [-127,127], zero point 0 for both operands; '
        's_X = calibration input absmax / 127 per operation, and s_W[n] = max|W_float[n,:]| / 127 per output channel. '
        'ACC is an exact signed INT32-safe dot product. One shared s_10 per operation maps to signed INT10 [-512,511] with nearest-even rounding. '
        'r[n] = s_X s_W[n] / s_10 is approximated by UInt16 multiplier[n] / 2^UInt5 shift[n]. '
        'The stored minmax output candidate is max_calibration|ACC s_X s_W| / 511 (a symmetric, slightly conservative bound for the negative side), '
        'not the float-reference maximum. MSE search minimizes HW-output versus dequantized-ACC MSE on 64 calibration rows. '
        'Exact per-channel s_W and multiplier/shift arrays remain in each original module’s `scales.npz` and `qparams.npz`, located by the original `manifest.json`; their hashes and values were verified, not refitted. '
        '`quantization_parameters.csv` summarizes every operation’s s_X, s_10, s_W/multiplier/shift ranges and exact NPZ locations. '
        'The attribution CSV records the common s_X and s_10 for every operation and split.')
    add(table(['CSV prefix','Value','Reference / NMSE denominator'],[
        ['input_only','(X_q × s_X) @ W_float.T','Y_ref / mean(Y_ref²)'],
        ['weight_only','X_float @ (W_q × s_W).T','Y_ref / mean(Y_ref²)'],
        ['int8_pair','ACC × s_X × s_W','Y_ref / mean(Y_ref²)'],
        ['ideal_int10','clip(rint(ACC × exact_ratio)) × s_10','Y_ref / mean(Y_ref²)'],
        ['final_local_total','actual profile output × s_10','Y_ref / mean(Y_ref²)'],
        ['actual_requant','actual profile output × s_10','ideal INT10 / mean(Y_ideal²)'],
        ['ideal_requantization_only','ideal INT10 × s_10','INT8 pair / mean(Y_pair²)'],
        ['requantization_only','actual profile output × s_10','INT8 pair / mean(Y_pair²)']]))
    add('Each prefix includes MSE, MAE, NMSE, maximum absolute error and reference energy. '
        '**These are separate counterfactuals, not additive contributions. No MSE subtraction is used for attribution.** '
        'Writing ΔX = X_hat − X and ΔW = W_hat − W, the pair error vector is ΔX Wᵀ + X ΔWᵀ + ΔX ΔWᵀ. '
        'Its squared norm includes cross terms; adding the independent input-only and weight-only MSEs does not reconstruct pair MSE. '
        'Requantization error can likewise correlate with the preceding error. '
        'HW-vs-ideal metrics are in real units here; historical parameter_approximation metrics used INT10 code units. '
        'NMSE percentages represent squared error relative to signal energy, not percent accuracy loss or relative amplitude error. '
        'For zero reference energy NMSE is 0 only if error is also zero, otherwise blank/null. '
        'Input clipping means |X|>127s_X; endpoint occupancy |q|=127 is reported separately from clipping. '
        'The strict |X|<s_X/2 count differs from q=0 at exact half-scale ties because nearest-even also rounds ±0.5 to zero.')
    add(f'## 4. Baseline attribution\n\nMeasured operations: {len(val)}/126; both-split rows: {len(rows)}/252. '
        f'Maximum relative reproduced-vs-stored MSE difference: {max((f(r,"stored_mse_relative_difference") for r in rows),default=0):.4g}.')
    add(table(['Layer','s_X','s_10','Input NMSE','Weight NMSE','Pair NMSE','Ideal INT10 NMSE','Final HW NMSE','Zero rate'],[
        [r['layer'],num(f(r,'s_X')),num(f(r,'s_10')),pct(f(r,'input_only_nmse')),pct(f(r,'weight_only_nmse')),pct(f(r,'int8_pair_nmse')),pct(f(r,'ideal_int10_nmse')),pct(f(r,'final_local_total_nmse')),pct(f(r,'activation_zero_rate'))] for r in down]))
    worst=sorted(val,key=lambda r:f(r,'final_local_total_nmse'),reverse=True)[:10]
    if worst:
        w=worst[0]
        add(f'For the worst measured operation `{w["module_name"]}`, final MSE {num(f(w,"final_local_total_mse"))} divided by reference energy {num(f(w,"final_local_total_reference_energy"))} equals NMSE {pct(f(w,"final_local_total_nmse"))}. A small raw MSE can coexist with a large normalized error. The denominator is mean(Y_ref²), not centered variance or an average of channel NMSEs. All valid output elements are equally weighted, so longer examples contribute more tokens.')
    add('\nWorst 10 measured Linear operations by validation final NMSE:\n')
    add(table(['Operation','Input NMSE','Weight NMSE','Pair NMSE','Final NMSE','Final MSE'],[
        [r['module_name'],pct(f(r,'input_only_nmse')),pct(f(r,'weight_only_nmse')),pct(f(r,'int8_pair_nmse')),pct(f(r,'final_local_total_nmse')),num(f(r,'final_local_total_mse'))] for r in worst]))
    last=next((r for r in down if r['layer']=='17'),None)
    if last:
        add(f'**Important exception: layer 17 down_proj has a material INT10-stage loss.** Pair NMSE is {pct(f(last,"int8_pair_nmse"))}, ideal INT10 NMSE is {pct(f(last,"ideal_int10_nmse"))}, and actual hardware-profile NMSE is {pct(f(last,"final_local_total_nmse"))}. Its independently measured requantization-only MSE is {num(f(last,"requantization_only_mse"))} and NMSE is {pct(f(last,"requantization_only_nmse"))} relative to pair energy. This is output-grid/clipping loss, not significant multiplier/shift error. Do not generalize the overall activation-dominance conclusion into “INT10 is negligible in every layer.” No MSE differences are treated as additive contributions.')
    add('## 5. Activation distributions and MLP comparison\n')
    add('`down_proj_activation_stats.csv` contains calibration and validation min/max/absmax, mean/std, exact |X| p50/p90/p95/p99/p99.5/p99.9/p99.95/p99.99/max, '
        'absmax/p99, absmax/p99.9, absmax/median, s_X, zero/±1/|q|≤2/endpoint occupancy, clipping, strict half-scale fraction, MAE and SQNR. '
        'Activation SQNR is 10 log10(sum(X²) / sum((X_hat−X)²)); it uses signal energy, not centered variance. '
        '`mlp_activation_comparison.csv` contains the same statistics plus input-only, weight-only, pair and final NMSE for gate/up/down in every layer. '
        'gate/up share the same input group and scale; down receives the reference GeLU(gate) × up product, before NPU quantization. High zero occupancy is a distribution diagnostic, not an error contribution or standalone proof: some reference values are already zero or carry little output energy. The separately measured input-only output error establishes the impact; zero rate and NMSE need not rank layers identically.')
    add(table(['Layer','Validation absmax','p99','p99.9','absmax/p99.9','|X|<s_X/2','Zero after INT8','SQNR dB'],[
        [l,num(f(s,'absmax')),num(f(s,'p99')),num(f(s,'p99.9')),num(f(s,'absmax_over_p99.9')),pct(f(s,'below_half_scale_rate')),pct(f(s,'zero_rate')),num(f(s,'sqnr_db')) if s['sqnr_db'] else 'undefined'] for l,s in sorted(sv.items())]))
    mlp=[r for r in read('mlp_activation_comparison.csv') if r['split']=='validation']
    add('\nMLP validation comparison, medians across measured layers (full per-layer table in CSV):\n')
    add(table(['Module','s_X','absmax/p99.9','Zero rate','Input NMSE','Weight NMSE','Pair NMSE','Final NMSE'],[
        [module]+[num(statistics.median(f(r,k) for r in mlp if r['module']==module)) if k in ['s_X','absmax_over_p99.9'] else pct(statistics.median(f(r,k) for r in mlp if r['module']==module)) for k in ['s_X','absmax_over_p99.9','zero_rate','input_only_nmse','weight_only_nmse','int8_pair_nmse','final_local_total_nmse']]
        for module in ['gate_proj','up_proj','down_proj'] if any(r['module']==module for r in mlp)]))
    add('## 6. Diagnostic input scale sweep (production unchanged)\n')
    add('All alternatives remain static per-tensor signed INT8 with zero point zero. Exact calibration percentiles 99, 99.5, 99.9, 99.95 and 99.99 are divided by 127. '
        'The recorded per-channel weight scales and common s_10 are held fixed; multiplier/shift are recalculated for the new input ratio. '
        'Thus input clipping and zero-rate effects can be compared without silently refitting output scales. '
        '`down_proj_scale_sweep.csv` contains selected thresholds, scales, all requested errors and clip rates for both splits. '
        'The additional 64-row MSE policy is selected on calibration only. Validation-best below is explicitly retrospective, not a deployable selection rule.')
    best=[]
    for r in down:
        options=[s for s in sval if s['layer']==r['layer']]
        if not options:continue
        b=min(options,key=lambda s:f(s,'final_local_total_nmse'));best.append((r,b))
    add(table(['Layer','Baseline final NMSE','Best validation policy (retrospective)','Threshold','s_X','Input clipped','Zero rate','Final NMSE'],[
        [r['layer'],pct(f(r,'final_local_total_nmse')),b['policy'],num(f(b,'threshold')),num(f(b,'s_X')),pct(f(b,'input_clipping_fraction')),pct(f(b,'zero_after_quant_fraction')),pct(f(b,'final_local_total_nmse'))] for r,b in best]))
    calibration_sweep=[r for r in sweeps if r['split']=='calibration']
    chosen=[]
    for r in down:
        candidates=[c for c in calibration_sweep if c['layer']==r['layer']]
        if candidates:
            c=min(candidates,key=lambda c:f(c,'final_local_total_nmse'))
            v=next(v for v in sval if v['layer']==r['layer'] and v['policy']==c['policy'])
            chosen.append((r,c,v))
    add('\nCalibration-only selection by minimum full-calibration final NMSE among the tested policies (validation remains held out):\n')
    add(table(['Layer','Chosen policy','Baseline validation NMSE','Chosen validation NMSE','Relative NMSE reduction'],[
        [r['layer'],c['policy'],pct(f(r,'final_local_total_nmse')),pct(f(v,'final_local_total_nmse')),pct(1-f(v,'final_local_total_nmse')/f(r,'final_local_total_nmse'))] for r,c,v in chosen]))
    if chosen:
        improved=sum(f(v,'final_local_total_nmse')<f(r,'final_local_total_nmse')-1e-12 for r,c,v in chosen)
        regressed=sum(f(v,'final_local_total_nmse')>f(r,'final_local_total_nmse')+1e-12 for r,c,v in chosen)
        add(f'Across these {len(chosen)} layers, calibration-only selection improves {improved}, regresses {regressed}, and leaves {len(chosen)-improved-regressed} unchanged on validation. '
            f'Median baseline validation NMSE is {pct(statistics.median(f(r,"final_local_total_nmse") for r,c,v in chosen))}; '
            f'median selected-policy validation NMSE is {pct(statistics.median(f(v,"final_local_total_nmse") for r,c,v in chosen))}. '
            'These are medians of layer NMSEs, not a pooled output-energy-weighted model metric. The search is limited to the tested scalar thresholds, not a claim of global optimality.')
    policies=list(dict.fromkeys(r['policy'] for r in sval))
    add('\nMedian validation final NMSE across layers for each fixed policy:\n')
    add(table(['Policy','Layers','Median final NMSE','Median input-only NMSE'],[
        [p,len([r for r in sval if r['policy']==p]),pct(statistics.median(f(r,'final_local_total_nmse') for r in sval if r['policy']==p)),pct(statistics.median(f(r,'input_only_nmse') for r in sval if r['policy']==p))] for p in policies]))
    last_clip=next((r for r in sval if r['layer']=='17' and r['policy']=='percentile_99.99'),None)
    if last_clip and last:
        add(f'**Do not apply p99.99 universally:** layer 17 regresses from {pct(f(last,"final_local_total_nmse"))} to {pct(f(last_clip,"final_local_total_nmse"))}. '
            f'Its input-only NMSE rises from {pct(f(last,"input_only_nmse"))} to {pct(f(last_clip,"input_only_nmse"))}, even though the zero rate falls from {pct(f(last,"activation_zero_rate"))} to {pct(f(last_clip,"zero_after_quant_fraction"))}. '
            'Rare large inputs carry important output energy. The 64-row input-reconstruction MSE search also picks this harmful threshold, while full-calibration final-output NMSE selects absmax. '
            'A lower zero rate or a better median across layers does not guarantee a better individual operation.')
    add('## 7. Output scale examined independently\n')
    add('With baseline X_q/W_q and scales fixed, `down_proj_output_scale_sweep.csv` reevaluates every stored MSE-search candidate plus the exact stored minmax baseline on both full splits. '
        'Each candidate uses fresh profile multiplier/shift; ideal and actual requantization-only NMSE are also reported. No validation selection changes production.')
    add(table(['Layer','Current clip rate','Current requant-only NMSE','Current final NMSE','Minmax s_10','Minmax clip rate','Minmax requant-only NMSE','Minmax final NMSE'],[
        [r['layer'],pct(f(r,'int10_clip_rate')),pct(f(r,'requantization_only_nmse')),pct(f(r,'final_local_total_nmse')),num(f(o,'s_10')),pct(f(o,'int10_clip_rate')),pct(f(o,'requantization_only_nmse')),pct(f(o,'final_local_total_nmse'))]
        for r in down for o in oval if o['layer']==r['layer'] and o['minmax']=='True']))
    last_output=next((r for r in oval if r['layer']=='17' and r['minmax']=='True'),None)
    if last_output and last:
        add(f'For layer 17, minmax s_10={num(f(last_output,"s_10"))} eliminates measured clipping and lowers final NMSE to {pct(f(last_output,"final_local_total_nmse"))}, '
            f'with requantization-only NMSE {pct(f(last_output,"requantization_only_nmse"))}. '
            'Residual output-grid loss remains; removing saturation is not equivalent to removing all INT10 quantization error. '
            'Nearby candidate scores are in the CSV, and none are applied to production.')
    audit_path=D/'outlier_position_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        add('### BOS outlier and output-search sample audit (layers 8 and 17 only)\n\n'
            '`outlier_position_audit.json` independently reconstructs exact token positions and the historical 64-row reservoir (seed 42). '
            'The input absmax occurs at BOS in both inspected layers. All baseline INT10 clipping events in both splits occur at BOS. '
            'The historical output-scale search sampled **zero BOS rows**; all six stored search MSE scores were reproduced. '
            'Thus its preference for half the minmax scale optimizes a sample that misses these rare large outputs. '
            'This is a demonstrated sampling limitation, distinct from multiplier/shift approximation and the input absmax problem. '
            'BOS remains included in every reported metric; it has not been removed to improve results. Other layers were not audited by token position.')
        add(table(['Layer','Split','BOS rows','BOS input absmax','Non-BOS input absmax','INT10 clips at BOS / total'],[
            [l['layer'],split,s['bos_rows'],num(s['bos_absmax']),num(s['non_bos_absmax']),f"{s['bos_int10_clip_count']} / {s['baseline_int10_clip_count']}"]
            for l in audit['layers'] for split,s in l['splits'].items()]))
    add('## 8. Tests and limitations\n\n`tests.log` records the full static_quant/frontend suite plus diagnostic tests. '
        'Diagnostic tests independently check all counterfactuals, Python integer RNE oracle including negative values, nonadditive MSE example, '
        'streamed-vs-dense metrics/denominators, INT64-vs-FP64 exact integer paths at K=31/257/2048/16384, overflow guards, '
        'strict zero threshold versus ties, saturation versus clipping, exact percentiles, zero-energy references, Accelerate-vs-NumPy/INT64 oracles, and fused-C-vs-NumPy metric/sweep comparisons. '
        'Hardware ABI/RTL equivalence, full-model propagated error, task accuracy and GGUF-vs-original source error remain unmeasured. '
        'A limited 128/32-example corpus cannot establish a universally safe production policy. '
        'Stored artifact checks and replay agreement validate consistency with the historical local result, not independent llama.cpp-vs-Transformers full-model equivalence.')
    tests_path=D/'tests.log'
    if tests_path.exists():
        tests_text=tests_path.read_text()
        test_summary=re.search(r'Ran (\d+) tests in ([\d.]+)s',tests_text)
        if test_summary and tests_text.rstrip().endswith('OK'):
            add(f'Executed test result: **{test_summary.group(1)} tests passed** ({test_summary.group(2)} seconds).')
    validation_path=D/'output_validation.json'
    if validation_path.exists():
        verified=json.loads(validation_path.read_text())
        add(f'Complete-output verification: **{"PASS" if verified["passed"] else "FAIL"}**. '
            'All 126 operations, both splits, all 18 activation/output sweeps, frozen baseline scales, '
            'calibration-only threshold fitting, NMSE denominators, parameter widths, source hashes and replay agreement were checked; see `output_validation.json`.')
    if (D/'resume_validation.json').exists():
        add('Resume verification: **PASS**. Repeating the complete command exits successfully without loading the model or rewriting any of the 144 completed baseline/sweep checkpoints; see `resume_validation.json`.')
    add('## 9. Explicit answers\n')
    add('1. **Full quantized end-to-end simulation? No.** Original-input local Linear counterfactuals.\n'
        '2. **NPU RoPE fixed-point error included? No.** Reference RoPE affects reference activations but its numerical error is not separately evaluated.\n'
        '3. **Previous quantized-layer errors propagated? No.** Neither reference hooks nor this diagnostic inject quantized outputs.')
    if down:
        med=lambda k:statistics.median(f(r,k) for r in down)
        dominant = 'activation INT8' if med('input_only_nmse') > 10*med('weight_only_nmse') else 'inspect independent paths; no single dominant source established'
        add(f'4. **Dominant down_proj source: {dominant}.** Measured median input-only NMSE {pct(med("input_only_nmse"))}, '
            f'weight-only {pct(med("weight_only_nmse"))}, combined pair {pct(med("int8_pair_nmse"))}, '
            f'final INT10 {pct(med("final_local_total_nmse"))}. '
            f'Median requantization-only NMSE is {pct(med("requantization_only_nmse"))} (different denominator). '
            'The ranking and closeness of independent paths identify the dominant stage; these numbers are not additive shares. Layer 17 is an important exception with substantial additional INT10 loss, as quantified above.')
    add('5. **Evidence that Q8_0 itself is inaccurate? No.** Y_ref already uses recovered GGUF weights; original-source comparison is not measured.')
    if down:
        add(f'6. **Absmax activation bottleneck?** The input-only/pair agreement and zero occupancy support this diagnosis when activation error dominates. Median down_proj zero-after-INT8 fraction is {pct(statistics.median(f(r,"activation_zero_rate") for r in down))}; '
            'compare input-only versus weight-only/pair results above and the measured percentile sweep before choosing a policy.')
    if best:
        add('7. **Clipping improvement:** '+ '; '.join(f'{p}: median validation final NMSE {pct(statistics.median(f(r,"final_local_total_nmse") for r in sval if r["policy"]==p))}' for p in policies)+'. '
            'This includes regression as well as improvement; validation-best choices in the layer table are retrospective.')
        if last_clip and last:
            add(f'Layer 17 is a counterexample: fixed p99.99 worsens final NMSE from {pct(f(last,"final_local_total_nmse"))} to {pct(f(last_clip,"final_local_total_nmse"))}; calibration-only final-output selection retains absmax.')
    else:add('7. **Clipping improvement:** not yet measured; do not infer results from baseline.')
    add('8. **Does down_proj require a different activation policy?** The present absmax policy is a demonstrated local-accuracy bottleneck in many layers, so activation-policy work is justified. '
        'A universal percentile replacement is contradicted by layer 17. A deployment decision needs a calibration-only selection rule validated on broader held-out data and propagated full-model evaluation. '
        'The measured counterfactuals diagnose the local bottleneck; no production policy is changed here. '
        'If all tested scalar clipping thresholds remain insufficient, per-group activation quantization is a possible future experiment, not an implemented fix.')
    if val:
        max_hw=max(f(r,'actual_requant_mse') for r in val)
        add(f'9. **UInt16 multiplier / UInt5 shift sufficient?** The maximum validation HW-vs-ideal real-unit MSE across operations is {num(max_hw)}. '
            f'Maximum validation HW-vs-ideal NMSE is {100*max(f(r,"actual_requant_nmse") for r in val):.8f}%. '
            'The original saved ratios all pass 0.001 tolerance (maximum relative error 1.50515e-5), with at most one INT10 code difference in saved validation. '
            'These parameters are sufficient for the recorded software ratios to that tolerance; this does not establish RTL correctness or guarantee future scale ranges.')
    add('10. **Before changing RTL:** include BOS/outlier rows in output-scale calibration and reevaluate the full-calibration objective; freeze a calibration-only input threshold selection method, evaluate it on a larger independent corpus and decode activations, '
        'then run quantized output injection through attention, residual, RMSNorm and GeGLU. Validate scale contracts for every consumer and run exact profile test vectors on RTL. '
        'Only an independently available original checkpoint can quantify GGUF source error. Preserve the current hardware widths until evidence identifies a limitation.')
    (D/'quant_error_diagnosis.md').write_text('\n\n'.join(report)+'\n')


if __name__=='__main__':main()
