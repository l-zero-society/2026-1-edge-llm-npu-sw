#!/usr/bin/env python3
"""Summarize measured BOS-aware calibration without using validation to fit scales."""
import csv
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'diagnostics';OUT=D/'bos_aware'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(v):return 'undefined' if v is None else f'{100*v:.4f}%'
def rate(v):return 'undefined' if v is None else f'{100*v:.7g}%'
def num(v):return f'{v:.7g}'


def main():
    data=[json.loads(p.read_text()) for p in sorted((OUT/'results').glob('*.json'))]
    down=sorted([d for d in data if d['module']=='down_proj'],key=lambda d:d['layer'])
    if not down:return
    def result(d,tag,split='validation'):return next(r for r in d['evaluation'] if r['policy']==tag and r['split']==split)
    def values(tag,key='all_final_local_total_nmse',split='validation'):return [result(d,tag,split)[key] for d in down]
    def med(tag,key='all_final_local_total_nmse',split='validation'):return statistics.median(values(tag,key,split))
    p=json.loads((OUT/'provenance.json').read_text());report=[];add=report.append
    add('# BOS-aware offline static calibration\n')
    add('## Scope and numerical contract\n\nThis remains **local Linear calibration on original/reference-model prefill inputs**, not end-to-end quantized propagation. NPU RoPE error, task accuracy and decode behavior are not measured. '
        'X/W INT8 [-127,127], per-output-channel s_W, INT32 accumulation, one static s_X and s_10 per operation, INT10 [-512,511], RNE, UInt16 multiplier, UInt5 shift and zero point 0 are unchanged. No RTL, LUT binary or QB format changes. '
        'W_float is recovered Q8_0 GGUF; BF16→GGUF source error is **not measured**. No model was downloaded.')
    add(f'**Status: {"COMPLETE (18 down_proj)" if len(down)==18 else "IN PROGRESS"}.** {len(down)}/18 down_proj completed; {len(data)} total operations. '
        f'Current base commit `{p["commit"]}`; exact modified-source hashes, model/data/token fingerprints, versions and configuration are in `bos_aware/provenance.json`. '
        'Historical outputs are preserved. Each operation is accepted only after reproducing both historical split MSEs and verifying recovered per-channel scales and every original INT8 weight. '
        'Same 128 calibration / 32 validation examples, 34,221 / 10,141 valid tokens, length limit 512, seed 42 and local GGUF. Full execution commands are in `bos_aware/commands.md`.')
    add('## Sampling and joint objective\n\nPosition zero/BOS, positions 1..8, and general positions have independent deterministic priority reservoirs. '
        'Default requested quotas are 16 BOS and 16 early rows, with the remainder general; capacity is redistributed when a category is short. Total retained rows never exceeds the budget. '
        'BOS inclusion is an invariant when present and enabled. Position groups are **calibration metadata only**. '
        'The objective weights each selected row by `N_group/(N_total*n_selected_group)`. This restores population proportions rather than giving BOS 16/64 of the objective. '
        'The real data has one BOS per sequence (128 calibration BOS rows), whose population weight is 128/34221, about 0.374%.')
    add('Input candidates are absmax, p99, p99.5, p99.9, p99.95 and p99.99, fitted on calibration only. '
        'For each, the existing quantizers, exact integer reference and hardware profile evaluate the final dequantized output against the same `X_float @ W_float.T`. '
        'The six common output candidates use the original full-calibration absmax INT8-pair minmax baseline times {1,.95,.9,.8,.7,.5}. '
        'The selected pair minimizes population-weighted final-output MSE on the selected rows. All choices are written before validation evaluation. '
        'The common grid is bounded; it does not claim a continuous or globally optimal scale search. '
        'Q/K keep fixed s_10=0.2; gate keeps fixed s_10=0.1 for the existing GeLU LUT. These operations search s_X only.')
    add('**Estimator distinction:** this real experiment uses exact full-calibration activation percentiles over disk-cached inputs. '
        'The generic streaming CLI uses its configurable scalar reservoir (default 65,536 elements) to estimate candidate percentiles. '
        'Both use the same sampler/search implementation, but the resulting thresholds need not match. p99.99 has only about 6.6 tail observations in that default scalar reservoir; do not claim it is equivalent to the exact experiment. '
        'Candidate CSV MSEs are population-weighted **search estimates**; full calibration/validation metrics are separate.')
    add('**Controlled comparisons:** `historical` is the old absmax input / uniform 64-row requantization-only output search. '
        '`uniform_64` uses the new joint/output-domain objective and exact candidates with the historical uniform priority sequence, isolating representation from the other changes. '
        '`stratified_64_absmax` holds old absmax s_X and selects s_10 using the new stratified final-output objective. '
        'The 64/128/256 stratified experiments keep the same fixed BOS/early quotas (16/16), increasing general coverage. '
        'The main table below uses 64 rows, the unchanged default; other budgets are independently frozen alternatives, not validation-selected per-operation choices.')
    add('## All 18 down_proj: old versus stratified 64-row joint search\n\nAll errors below are validation NMSE. Full precision and calibration rows are in `down_proj_old_vs_new.csv`; BOS/non-BOS MSE as well as NMSE are in `bos_aware_evaluation.csv`.')
    add(table(['Layer','Old s_X','New s_X','Old s_10','New s_10','Method','Threshold','BOS rows','Old zero','New zero','Old Q10 clip','New Q10 clip','Old NMSE','New NMSE','BOS NMSE','Non-BOS NMSE'],[
        [d['layer'],num(o['s_X']),num(n['s_X']),num(o['s_10']),num(n['s_10']),n['method'],num(n['threshold']),d['searches']['stratified_64']['sampling']['bos_rows'],pct(o['activation_zero_rate']),pct(n['activation_zero_rate']),rate(o['int10_clip_rate']),rate(n['int10_clip_rate']),pct(o['all_final_local_total_nmse']),pct(n['all_final_local_total_nmse']),pct(n['bos_final_local_total_nmse']),pct(n['non_bos_final_local_total_nmse'])]
        for d in down for o,n in [(result(d,'historical'),result(d,'stratified_64'))]]))
    add('## Search row sensitivity and cost\n')
    tags=['historical','uniform_64','stratified_64','stratified_128','stratified_256']
    add(table(['Policy','Calibration median NMSE','Validation median NMSE','Validation max NMSE','Search seconds (sum)','Input thresholds changed vs 64','Output scales changed vs 64'],[
        [t,pct(med(t,split='calibration')),pct(med(t)),pct(max(values(t))),
         'historical' if t=='historical' else num(sum(d['searches'][t]['seconds'] for d in down)),
         '-' if t in ('historical','uniform_64') else sum(result(d,t)['s_X']!=result(d,'stratified_64')['s_X'] for d in down),
         '-' if t in ('historical','uniform_64') else sum(result(d,t)['s_10']!=result(d,'stratified_64')['s_10'] for d in down)] for t in tags]))
    add('Full per-layer sensitivity is in `search_row_sensitivity.csv`, including both splits, threshold/scale stability and actual search time. '
        'For K=16384 FP32, final 64/128/256-row arrays use 4/8/16 MiB; three bounded reservoirs retain at most 12/24/48 MiB, plus metadata and merge temporaries. '
        'Search timing includes representative-row collection and candidate evaluation, but excludes shared model capture, exact-percentile construction and full-data evaluation. '
        'Full-data evaluation is shared across duplicate scale pairs and input scales; its runtime is not falsely assigned separately to each budget. '
        'All-operation capture uses ~67 GiB temporary disk and the full reference model; the exact down_proj percentile temporary is ~2.2 GiB. None of this changes runtime accelerator memory.')
    add(f'Total completed per-operation experiment time is {sum(d["seconds"] for d in down)/60:.2f} minutes, including exact percentiles, four search configurations and both full-data splits with counterfactual attribution, excluding the shared reference capture. '
        'This is diagnostic cost, not the cost of fitting just one production policy. Reported search timings use the checked Accelerate backend; portable NumPy INT64 search can have different runtime.')
    add('## BOS, ordinary tokens and output-only controls\n')
    add(table(['Layer','Policy','Cal total NMSE','Val BOS NMSE','Val non-BOS NMSE','Val total NMSE','Val BOS Q10 clip'],[
        [d['layer'],t,pct(result(d,t,'calibration')['all_final_local_total_nmse']),pct(result(d,t)['bos_final_local_total_nmse']),pct(result(d,t)['non_bos_final_local_total_nmse']),pct(result(d,t)['all_final_local_total_nmse']),pct(result(d,t)['bos_int10_clip_rate'])]
        for d in down if d['layer'] in (8,17) for t in ['historical','stratified_64_absmax','uniform_64','stratified_64','stratified_128','stratified_256']]))
    regressions=[d['layer'] for d in down if result(d,'stratified_64','calibration')['all_final_local_total_mse']>result(d,'historical','calibration')['all_final_local_total_mse']]
    add(f'Full-calibration total-MSE regressions of stratified_64 versus historical: {regressions or "none"}. '
        'BOS/non-BOS NMSE denominators are their own reference energies; their percentages cannot be added. '
        'Population-weighted MSE prevents deliberate BOS-only optimization, but finite sampled objectives can still misrank close candidates. '
        'Guaranteed representation does not enforce zero saturation: an MSE-optimal grid candidate may trade rare clipping against output-grid precision.')
    eighth=next((d for d in down if d['layer']==8),None)
    if eighth:
        old,new=result(eighth,'historical'),result(eighth,'stratified_64')
        add(f'**Coverage is not a BOS accuracy guarantee.** Layer 8 BOS validation NMSE changes from {pct(old["bos_final_local_total_nmse"])} to {pct(new["bos_final_local_total_nmse"])}; '
            f'non-BOS changes from {pct(old["non_bos_final_local_total_nmse"])} to {pct(new["non_bos_final_local_total_nmse"])}. '
            f'The selected policy has BOS input-only NMSE {pct(new["bos_input_only_nmse"])} and BOS INT8-pair NMSE {pct(new["bos_int8_pair_nmse"])}. '
            'Thus eliminating INT10 saturation does not eliminate BOS damage from activation clipping. The population-weighted total-MSE objective knowingly makes this trade-off with BOS included. '
            'Do not describe the result as universally safe for rare tokens. If BOS fidelity has a separate deployment requirement, define a calibration-only acceptance constraint or broaden the threshold grid before promotion; do not infer that requirement from validation.')
    last=next((d for d in down if d['layer']==17),None)
    if last:
        old,new=result(last,'historical'),result(last,'stratified_64')
        add(f'**Layer 17 has the opposite trade-off:** BOS NMSE improves from {pct(old["bos_final_local_total_nmse"])} to {pct(new["bos_final_local_total_nmse"])}, while non-BOS rises from {pct(old["non_bos_final_local_total_nmse"])} to {pct(new["non_bos_final_local_total_nmse"])}. '
            f'The unchanged absmax s_X maps {pct(new["activation_zero_rate"])} of activation elements to zero. Non-BOS input-only NMSE is {pct(new["non_bos_input_only_nmse"])}; this is already a serious baseline limitation. '
            'The selected larger s_10 improves total calibration MSE, so it is not rejected by the requested total-MSE objective. Nevertheless it cannot be presented as uniformly good ordinary-token accuracy. '
            'At 128/256 rows, the selected 0.95×minmax output scale has small nonzero BOS saturation (0.048828125% of BOS output elements); guaranteed inclusion does not require choosing the no-clipping minmax endpoint.')
    add('## Remaining error and hardware parameters\n')
    add('The following group comparison makes regressions visible even when total NMSE improves. All values are validation NMSE; full calibration group metrics are in `bos_aware_evaluation.csv`.')
    add(table(['Layer','Old BOS','New BOS','Old non-BOS','New non-BOS'],[
        [d['layer'],pct(o['bos_final_local_total_nmse']),pct(n['bos_final_local_total_nmse']),pct(o['non_bos_final_local_total_nmse']),pct(n['non_bos_final_local_total_nmse'])]
        for d in down for o,n in [(result(d,'historical'),result(d,'stratified_64'))]]))
    for group in ['bos','non_bos']:
        worsened=[d['layer'] for d in down if result(d,'stratified_64')[group+'_final_local_total_mse'] > result(d,'historical')[group+'_final_local_total_mse']]
        add(f'{group} validation MSE increases versus historical in layers: {worsened or "none"}. These observations are reported after parameters were frozen; they did not select any scale.')
    add(table(['Layer','Input-only NMSE','Weight-only NMSE','INT8 pair NMSE','Final NMSE','HW vs ideal INT10 NMSE'],[
        [d['layer']]+[pct(result(d,'stratified_64')['all_'+k+'_nmse']) for k in ['input_only','weight_only','int8_pair','final_local_total']]+[rate(result(d,'stratified_64')['all_actual_requant_nmse'])] for d in down]))
    add('These are independently evaluated counterfactuals, not additive MSE shares. '
        'The real evaluator reuses exact bounded integer dots (K*127² < INT32_MAX), with independent NumPy INT64 oracle checks. '
        'No FP32 integer approximation is used. Generic production search uses the existing NumPy INT64 accumulator. '
        'Multiplier/shift generation and packing code are unchanged; candidate tables record any unrepresentable channels. '
        'RTL equivalence, scale handling by every downstream consumer, propagated model error and task accuracy remain separate validation requirements.')
    add('## Q/K bug and integration\n\nThe current checkout did construct fixed Q/K layer_options but passed global options to LinearCalibration. '
        'The CLI now calls operation_options and actually passes its result. A regression test invokes the CLI and verifies exported Q/K=0.2 and gate=0.1, including fixed candidate tables. '
        'Historical Q/K artifacts used MSE/non-0.2 scales; they did not use the intended fixed override. Their missing dirty GGUF entrypoint prevents claiming the current CLI was the exact producer. '
        'The new mechanism is generic; same-input groups share statistics but can select different s_X. Do not reuse an INT8 input cache across operations with different scales. No position-dependent runtime interface is introduced.')
    if len(data)==len(down):
        add('The new real-model numerical experiments cover all 18 down_proj operations. The optional full 126-operation rerun was not performed: complete down_proj attribution already requires substantial full-corpus evaluation, and the measured rare-token regressions need an explicit acceptance policy before broad promotion. '
            'Q/K and gate integration is covered by the CLI regression test, not claimed as a new 126-operation real-model result. Reference inputs for all 126 operations were captured and the resumable runner supports `--layers all --modules all`.')
    add('## Tests and reproducibility\n\nThe full unittest run passed **42 tests, 0 failures** (including 10 new tests). See `bos_aware/tests.log`, `bos_aware/provenance.json`, selected-policy checkpoints and full execution logs. '
        'Tests cover BOS invariants, quotas and redistribution, deterministic streaming, population weighting, masks/position propagation, '
        'dense output-MSE oracles, minimum-candidate selection, validation independence, LUT constraints, actual CLI Q/K option use, '
        'zero/nonfinite cases, unchanged parameter generation, full-data group metrics and existing static_quant/frontend/export tests. '
        'No synthetic data substitutes for these real-model measurements. The old diagnostic reports remain historical snapshots; their old source hashes intentionally differ after this policy change.')
    verification=OUT/'verification.json'
    if verification.exists():
        verified=json.loads(verification.read_text())
        add(f'Artifact verification passed for {verified["total_operations"]} operations: source fingerprints, frozen calibration-only selection, exact candidate argmin, both dataset splits, BOS counts, finite metrics and protected numerical code/LUT bytes. '
            f'Maximum relative historical-MSE replay difference is {verified["historical_mse_max_relative_difference"]:.3g}; maximum multiplier/shift relative ratio error across evaluated policies is {verified["maximum_ratio_relative_error"]:.3g}.')
    exported=OUT/'calibrated/manifest.json'
    if exported.exists():
        exported_metadata=json.loads(exported.read_text())
        add(f'Exported and reloaded {len(exported_metadata["modules"])} frozen stratified-64 policies through the existing exporter into `bos_aware/calibrated/`. '
            'NPZ scale/parameter arrays, mapping, software vectors and search sidecar are available. The existing generic-rne profile remains unverified for RTL; its existing export guard prevents default binary publication. No RTL-ready binary claim is made.')
    resumed=OUT/'resume_verification.json'
    if resumed.exists():
        resume=json.loads(resumed.read_text())
        add(f'No-op resume passed: all {resume["unchanged_checkpoint_files"]} frozen selection/result checkpoints retained their modification times. '
            'The task-specific file inventory and SHA256 hashes are in `bos_aware/files.json`; previous diagnostics were preserved.')
    add('## Explicit answers\n')
    specials=[d for d in down if d['layer'] in (8,17)]
    add('1. **Does BOS inclusion remove layer 8/17 clipping?** '+ '; '.join(f'layer {d["layer"]}: old {rate(result(d,"historical")["int10_clip_rate"])}, output-only {rate(result(d,"stratified_64_absmax")["int10_clip_rate"])}, joint {rate(result(d,"stratified_64")["int10_clip_rate"])}' for d in specials)+'. Inclusion guarantees evaluation, not a zero-clipping constraint.')
    add(f'2. **Stratified versus uniform 64?** With the same new joint objective, median validation NMSE is {pct(med("stratified_64"))} versus {pct(med("uniform_64"))}; maxima are {pct(max(values("stratified_64")))} versus {pct(max(values("uniform_64")))}. '
        'Compare both total and BOS/ordinary metrics above; historical-to-new improvement also changes input thresholds and objective, so it is not all attributable to sampling.')
    add('3. **64, 128 or 256 sufficient?** '+ '; '.join(f'{t}: median {pct(med(t))}, maximum {pct(max(values(t)))}' for t in ['stratified_64','stratified_128','stratified_256'])+'. '
        '64 is an effective baseline but not converged for every operation. Recommend an explicit 256-row quality preset for further qualification: layer 14 changes to p99.99, lowering full-calibration MSE by 7.3189% and validation NMSE from 10.8110% to 10.0109%; 128 misses this. '
        'Compared with 64, 256 changes one input threshold and five output scales. Search costs 91.82 versus 65.63 seconds across 18 operations (+39.9%, +26.19 seconds with this backend), with reservoir upper bounds of 48 versus 12 MiB per down_proj. '
        'Two layers have tiny full-calibration MSE regressions versus 64 (layers 1/3, +0.03165%/+0.02868%); larger samples are not a universal monotonic guarantee. The CLI default remains 64, and the main table/export remains stratified_64. No validation-selected per-layer mixing is used.')
    add('4. **Selected thresholds:** '+', '.join(f'{d["layer"]}: {result(d,"stratified_64")["method"]}' for d in down)+'. Exact thresholds are in the table/CSV.')
    methods={result(d,'stratified_64')['method'] for d in down}
    add(f'5. **One fixed percentile for all?** Selected methods are {sorted(methods)}; the measured per-operation choices must be preserved.\n\n'
        '6. **Per-operation selection necessary?** Yes: output-domain error and rare-token energy differ by operation; a universal histogram percentile is not established.')
    worst=max(down,key=lambda d:result(d,'stratified_64')['all_final_local_total_nmse'])
    add(f'7. **New down_proj median/max:** {pct(med("stratified_64"))} / {pct(max(values("stratified_64")))} validation NMSE at 64 rows.\n\n'
        f'8. **Worst remaining layer:** {worst["layer"]}.')
    add('9. **Remaining attribution:** median input-only '+pct(med('stratified_64','all_input_only_nmse'))+', weight-only '+pct(med('stratified_64','all_weight_only_nmse'))+', pair '+pct(med('stratified_64','all_int8_pair_nmse'))+', final '+pct(med('stratified_64'))+'. Inspect the layer table for output-stage exceptions; these are not additive contributions.')
    add('10. **More complex hardware required?** These experiments test static software policy only and do not establish a need for more hardware complexity. If scalar clipping is insufficient, per-group activation quantization is a future controlled experiment, not part of this change.\n\n'
        '11. **Keep static per-tensor s_X interface?** Yes, every measured choice is one static s_X per operation; positions are used only offline.\n\n'
        '12. **Keep UInt16/UInt5?** Yes for the measured representable candidates; generation and QB format are unchanged. This does not substitute for RTL equivalence testing.')
    add('13. **Production calibration recommendation:** use calibration-only, population-weighted representative sampling with guaranteed position-zero coverage and output-domain per-operation joint search; preserve fixed consumer scales. '
        'Use an explicit 256-row quality preset for the next qualification run based on the measured layer-14 stability benefit; retain the current 64-row CLI default pending acceptance, and validate the generic streaming percentile estimator separately from exact offline percentiles. '
        'Check full-calibration total and ordinary-token errors before promoting a policy; do not deploy a policy merely because BOS improves. '
        'BOS representation fixes a sampling defect, but total MSE can still sacrifice BOS fidelity (layers 7/8) or ordinary-token quality (layer 17). Do not automatically promote these clipping policies as production defaults. Define calibration-only group-error acceptance criteria and expand candidates between p99.99 and absmax before promotion; those constraints are not silently added to this experiment. '
        'Validate frozen scales on a broader independent corpus, decode inputs and end-to-end quantized propagation before making model-quality claims.')
    (D/'bos_aware_calibration_report.md').write_text('\n\n'.join(report)+'\n')


if __name__=='__main__':main()
