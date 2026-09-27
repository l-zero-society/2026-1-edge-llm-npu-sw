# BOS-aware offline static calibration


## Scope and numerical contract

This remains **local Linear calibration on original/reference-model prefill inputs**, not end-to-end quantized propagation. NPU RoPE error, task accuracy and decode behavior are not measured. X/W INT8 [-127,127], per-output-channel s_W, INT32 accumulation, one static s_X and s_10 per operation, INT10 [-512,511], RNE, UInt16 multiplier, UInt5 shift and zero point 0 are unchanged. No RTL, LUT binary or QB format changes. W_float is recovered Q8_0 GGUF; BF16→GGUF source error is **not measured**. No model was downloaded.

**Status: COMPLETE (18 down_proj).** 18/18 down_proj completed; 18 total operations. Current base commit `f31f9e6e52a65362acd473f27ed0bfc05097335f`; exact modified-source hashes, model/data/token fingerprints, versions and configuration are in `bos_aware/provenance.json`. Historical outputs are preserved. Each operation is accepted only after reproducing both historical split MSEs and verifying recovered per-channel scales and every original INT8 weight. Same 128 calibration / 32 validation examples, 34,221 / 10,141 valid tokens, length limit 512, seed 42 and local GGUF. Full execution commands are in `bos_aware/commands.md`.

## Sampling and joint objective

Position zero/BOS, positions 1..8, and general positions have independent deterministic priority reservoirs. Default requested quotas are 16 BOS and 16 early rows, with the remainder general; capacity is redistributed when a category is short. Total retained rows never exceeds the budget. BOS inclusion is an invariant when present and enabled. Position groups are **calibration metadata only**. The objective weights each selected row by `N_group/(N_total*n_selected_group)`. This restores population proportions rather than giving BOS 16/64 of the objective. The real data has one BOS per sequence (128 calibration BOS rows), whose population weight is 128/34221, about 0.374%.

Input candidates are absmax, p99, p99.5, p99.9, p99.95 and p99.99, fitted on calibration only. For each, the existing quantizers, exact integer reference and hardware profile evaluate the final dequantized output against the same `X_float @ W_float.T`. The six common output candidates use the original full-calibration absmax INT8-pair minmax baseline times {1,.95,.9,.8,.7,.5}. The selected pair minimizes population-weighted final-output MSE on the selected rows. All choices are written before validation evaluation. The common grid is bounded; it does not claim a continuous or globally optimal scale search. Q/K keep fixed s_10=0.2; gate keeps fixed s_10=0.1 for the existing GeLU LUT. These operations search s_X only.

**Estimator distinction:** this real experiment uses exact full-calibration activation percentiles over disk-cached inputs. The generic streaming CLI uses its configurable scalar reservoir (default 65,536 elements) to estimate candidate percentiles. Both use the same sampler/search implementation, but the resulting thresholds need not match. p99.99 has only about 6.6 tail observations in that default scalar reservoir; do not claim it is equivalent to the exact experiment. Candidate CSV MSEs are population-weighted **search estimates**; full calibration/validation metrics are separate.

**Controlled comparisons:** `historical` is the old absmax input / uniform 64-row requantization-only output search. `uniform_64` uses the new joint/output-domain objective and exact candidates with the historical uniform priority sequence, isolating representation from the other changes. `stratified_64_absmax` holds old absmax s_X and selects s_10 using the new stratified final-output objective. The 64/128/256 stratified experiments keep the same fixed BOS/early quotas (16/16), increasing general coverage. The main table below uses 64 rows, the unchanged default; other budgets are independently frozen alternatives, not validation-selected per-operation choices.

## All 18 down_proj: old versus stratified 64-row joint search

All errors below are validation NMSE. Full precision and calibration rows are in `down_proj_old_vs_new.csv`; BOS/non-BOS MSE as well as NMSE are in `bos_aware_evaluation.csv`.

| Layer | Old s_X | New s_X | Old s_10 | New s_10 | Method | Threshold | BOS rows | Old zero | New zero | Old Q10 clip | New Q10 clip | Old NMSE | New NMSE | BOS NMSE | Non-BOS NMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.870994 | 0.3282365 | 0.02822045 | 0.03950863 | p99.99 | 41.68604 | 16 | 88.2034% | 68.0600% | 0.0006981637% | 3.851938e-05% | 3.9183% | 3.5461% | 5.7043% | 3.5457% |
| 1 | 0.653449 | 0.1038245 | 0.009368304 | 0.01311563 | p99.99 | 13.18572 | 16 | 89.2054% | 49.1548% | 0.0007944621% | 3.370445e-05% | 3.9941% | 1.8136% | 8.9784% | 1.8133% |
| 2 | 0.8677332 | 0.08881361 | 0.007314705 | 0.007314705 | p99.99 | 11.27933 | 16 | 94.2588% | 44.9304% | 0.0002070417% | 0.002653022% | 16.5793% | 4.5372% | 5.9316% | 4.5370% |
| 3 | 0.6794692 | 0.05072093 | 0.005114523 | 0.00584517 | p99.99 | 6.441558 | 16 | 96.0109% | 25.3967% | 3.851938e-05% | 5.296414e-05% | 34.6998% | 12.9315% | 1.9196% | 12.9350% |
| 4 | 0.549462 | 0.0382234 | 0.003328987 | 0.003328987 | p99.99 | 4.854371 | 16 | 96.0436% | 23.2091% | 0.0003755639% | 0.00122299% | 39.7763% | 6.1201% | 1.0404% | 6.1239% |
| 5 | 0.3957401 | 0.03359187 | 0.005974751 | 0.008364652 | p99.99 | 4.266167 | 16 | 93.9216% | 22.7101% | 0.0003851938% | 3.370445e-05% | 29.0266% | 5.5108% | 0.9343% | 5.5151% |
| 6 | 0.3423329 | 0.03251184 | 0.005328721 | 0.005328721 | p99.99 | 4.129004 | 16 | 92.2672% | 24.2816% | 0.0006403846% | 0.0001925969% | 26.8561% | 3.8064% | 0.6211% | 3.8113% |
| 7 | 0.3126632 | 0.02816695 | 0.0087618 | 0.0087618 | p99.99 | 3.577202 | 16 | 90.8609% | 21.1724% | 0.0004622325% | 0% | 25.9400% | 3.3030% | 62.1468% | 2.2927% |
| 8 | 0.5708727 | 0.02452638 | 0.01383242 | 0.01383242 | p99.99 | 3.11485 | 16 | 97.3488% | 18.7408% | 0.0004622325% | 0% | 48.1294% | 6.0467% | 80.0887% | 2.0891% |
| 9 | 0.1579513 | 0.02629306 | 0.005881323 | 0.004200945 | p99.99 | 3.339219 | 16 | 69.7141% | 19.8590% | 3.851938e-05% | 0.0005922354% | 6.8476% | 1.6964% | 0.6468% | 1.6978% |
| 10 | 0.1985953 | 0.02806822 | 0.004704841 | 0.004704841 | p99.99 | 3.564663 | 16 | 76.7960% | 20.4059% | 9.148352e-05% | 0.0002359312% | 9.7901% | 1.7986% | 0.5146% | 1.8011% |
| 11 | 0.2596352 | 0.04009638 | 0.007084301 | 0.007084301 | p99.99 | 5.09224 | 16 | 80.8256% | 24.8093% | 0.0001444477% | 0.0001540775% | 9.3082% | 1.9727% | 0.8864% | 1.9737% |
| 12 | 0.3489435 | 0.04774678 | 0.007809064 | 0.005577903 | p99.99 | 6.063841 | 16 | 88.0028% | 28.3755% | 1.925969e-05% | 0.0003129699% | 11.7295% | 5.0008% | 1.4399% | 5.0034% |
| 13 | 0.588631 | 0.07070487 | 0.009247759 | 0.009247759 | p99.99 | 8.979519 | 16 | 92.4466% | 35.7105% | 8.185368e-05% | 0.0005633459% | 15.1084% | 5.0403% | 1.2829% | 5.0435% |
| 14 | 0.8091608 | 0.8091608 | 0.007030724 | 0.007030724 | absmax | 102.7634 | 16 | 95.8303% | 95.8303% | 9.629844e-05% | 9.629844e-05% | 10.8110% | 10.8110% | 25.2510% | 10.7996% |
| 15 | 0.7310743 | 0.123967 | 0.018482 | 0.01155125 | p99.99 | 15.74381 | 16 | 94.1106% | 52.7494% | 1.444477e-05% | 0.0006500145% | 8.1544% | 5.0946% | 1.2696% | 5.1072% |
| 16 | 1.144403 | 0.1346317 | 0.01814517 | 0.0326613 | p99.99 | 17.09823 | 16 | 93.7367% | 46.8609% | 0.002282273% | 0.000245561% | 11.8090% | 4.4656% | 0.6783% | 4.5161% |
| 17 | 18.48648 | 18.48648 | 0.4180036 | 0.8360071 | absmax | 2347.783 | 16 | 99.9140% | 99.9140% | 0.001078543% | 0% | 20.0998% | 15.0232% | 0.5495% | 77.4380% |

## Search row sensitivity and cost


| Policy | Calibration median NMSE | Validation median NMSE | Validation max NMSE | Search seconds (sum) | Input thresholds changed vs 64 | Output scales changed vs 64 |
| --- | --- | --- | --- | --- | --- | --- |
| historical | 13.4364% | 13.4587% | 48.1294% | historical | - | - |
| uniform_64 | 4.6565% | 4.7695% | 45.6690% | 65.2865 | - | - |
| stratified_64 | 4.6456% | 4.7690% | 15.0232% | 65.63085 | 0 | 0 |
| stratified_128 | 4.6437% | 4.7675% | 14.9760% | 70.33529 | 0 | 3 |
| stratified_256 | 4.6437% | 4.7675% | 14.9760% | 91.81753 | 1 | 5 |

Full per-layer sensitivity is in `search_row_sensitivity.csv`, including both splits, threshold/scale stability and actual search time. For K=16384 FP32, final 64/128/256-row arrays use 4/8/16 MiB; three bounded reservoirs retain at most 12/24/48 MiB, plus metadata and merge temporaries. Search timing includes representative-row collection and candidate evaluation, but excludes shared model capture, exact-percentile construction and full-data evaluation. Full-data evaluation is shared across duplicate scale pairs and input scales; its runtime is not falsely assigned separately to each budget. All-operation capture uses ~67 GiB temporary disk and the full reference model; the exact down_proj percentile temporary is ~2.2 GiB. None of this changes runtime accelerator memory.

Total completed per-operation experiment time is 49.73 minutes, including exact percentiles, four search configurations and both full-data splits with counterfactual attribution, excluding the shared reference capture. This is diagnostic cost, not the cost of fitting just one production policy. Reported search timings use the checked Accelerate backend; portable NumPy INT64 search can have different runtime.

## BOS, ordinary tokens and output-only controls


| Layer | Policy | Cal total NMSE | Val BOS NMSE | Val non-BOS NMSE | Val total NMSE | Val BOS Q10 clip |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | historical | 48.2995% | 17.1658% | 49.7844% | 48.1294% | 0.1465% |
| 8 | stratified_64_absmax | 47.6734% | 1.9485% | 50.0828% | 47.6405% | 0.0488% |
| 8 | uniform_64 | 6.6594% | 80.0887% | 2.0891% | 6.0467% | 0.0000% |
| 8 | stratified_64 | 6.6594% | 80.0887% | 2.0891% | 6.0467% | 0.0000% |
| 8 | stratified_128 | 6.6594% | 80.0887% | 2.0891% | 6.0467% | 0.0000% |
| 8 | stratified_256 | 6.6594% | 80.0887% | 2.0891% | 6.0467% | 0.0000% |
| 17 | historical | 18.8296% | 8.2555% | 71.1757% | 20.0998% | 0.3418% |
| 17 | stratified_64_absmax | 13.4789% | 0.5495% | 77.4380% | 15.0232% | 0.0000% |
| 17 | uniform_64 | 46.5440% | 54.1897% | 8.9253% | 45.6690% | 0.0000% |
| 17 | stratified_64 | 13.4789% | 0.5495% | 77.4380% | 15.0232% | 0.0000% |
| 17 | stratified_128 | 13.4468% | 0.6498% | 76.7542% | 14.9760% | 0.0488% |
| 17 | stratified_256 | 13.4468% | 0.6498% | 76.7542% | 14.9760% | 0.0488% |

Full-calibration total-MSE regressions of stratified_64 versus historical: none. BOS/non-BOS NMSE denominators are their own reference energies; their percentages cannot be added. Population-weighted MSE prevents deliberate BOS-only optimization, but finite sampled objectives can still misrank close candidates. Guaranteed representation does not enforce zero saturation: an MSE-optimal grid candidate may trade rare clipping against output-grid precision.

**Coverage is not a BOS accuracy guarantee.** Layer 8 BOS validation NMSE changes from 17.1658% to 80.0887%; non-BOS changes from 49.7844% to 2.0891%. The selected policy has BOS input-only NMSE 80.0523% and BOS INT8-pair NMSE 80.0500%. Thus eliminating INT10 saturation does not eliminate BOS damage from activation clipping. The population-weighted total-MSE objective knowingly makes this trade-off with BOS included. Do not describe the result as universally safe for rare tokens. If BOS fidelity has a separate deployment requirement, define a calibration-only acceptance constraint or broaden the threshold grid before promotion; do not infer that requirement from validation.

**Layer 17 has the opposite trade-off:** BOS NMSE improves from 8.2555% to 0.5495%, while non-BOS rises from 71.1757% to 77.4380%. The unchanged absmax s_X maps 99.9140% of activation elements to zero. Non-BOS input-only NMSE is 67.8154%; this is already a serious baseline limitation. The selected larger s_10 improves total calibration MSE, so it is not rejected by the requested total-MSE objective. Nevertheless it cannot be presented as uniformly good ordinary-token accuracy. At 128/256 rows, the selected 0.95×minmax output scale has small nonzero BOS saturation (0.048828125% of BOS output elements); guaranteed inclusion does not require choosing the no-clipping minmax endpoint.

## Remaining error and hardware parameters


The following group comparison makes regressions visible even when total NMSE improves. All values are validation NMSE; full calibration group metrics are in `bos_aware_evaluation.csv`.

| Layer | Old BOS | New BOS | Old non-BOS | New non-BOS |
| --- | --- | --- | --- | --- |
| 0 | 60.9846% | 5.7043% | 3.9094% | 3.5457% |
| 1 | 54.4506% | 8.9784% | 3.9923% | 1.8133% |
| 2 | 53.4446% | 5.9316% | 16.5745% | 4.5370% |
| 3 | 69.1329% | 1.9196% | 34.6889% | 12.9350% |
| 4 | 40.5186% | 1.0404% | 39.7757% | 6.1239% |
| 5 | 48.3740% | 0.9343% | 29.0082% | 5.5151% |
| 6 | 47.0153% | 0.6211% | 26.8248% | 3.8113% |
| 7 | 16.3431% | 62.1468% | 26.1048% | 2.2927% |
| 8 | 17.1658% | 80.0887% | 49.7844% | 2.0891% |
| 9 | 14.0999% | 0.6468% | 6.8380% | 1.6978% |
| 10 | 12.6367% | 0.5146% | 9.7845% | 1.8011% |
| 11 | 17.6851% | 0.8864% | 9.2999% | 1.9737% |
| 12 | 31.5771% | 1.4399% | 11.7154% | 5.0034% |
| 13 | 29.5119% | 1.2829% | 15.0959% | 5.0435% |
| 14 | 25.2510% | 25.2510% | 10.7996% | 10.7996% |
| 15 | 5.7186% | 1.2696% | 8.1625% | 5.1072% |
| 16 | 4.7788% | 0.6783% | 11.9028% | 4.5161% |
| 17 | 8.2555% | 0.5495% | 71.1757% | 77.4380% |

bos validation MSE increases versus historical in layers: [7, 8]. These observations are reported after parameters were frozen; they did not select any scale.

non_bos validation MSE increases versus historical in layers: [17]. These observations are reported after parameters were frozen; they did not select any scale.

| Layer | Input-only NMSE | Weight-only NMSE | INT8 pair NMSE | Final NMSE | HW vs ideal INT10 NMSE |
| --- | --- | --- | --- | --- | --- |
| 0 | 3.5325% | 0.0098% | 3.5404% | 3.5461% | 1.200368e-05% |
| 1 | 1.7925% | 0.0175% | 1.8100% | 1.8136% | 8.795022e-06% |
| 2 | 4.5173% | 0.0134% | 4.5294% | 4.5372% | 7.310058e-06% |
| 3 | 12.9155% | 0.0136% | 12.9269% | 12.9315% | 1.229638e-05% |
| 4 | 6.0983% | 0.0128% | 6.1091% | 6.1201% | 9.042696e-06% |
| 5 | 5.4682% | 0.0142% | 5.4807% | 5.5108% | 2.864847e-05% |
| 6 | 3.7734% | 0.0161% | 3.7881% | 3.8064% | 1.685129e-05% |
| 7 | 3.2422% | 0.0151% | 3.2555% | 3.3030% | 3.060046e-05% |
| 8 | 5.9046% | 0.0148% | 5.9179% | 6.0467% | 4.786141e-05% |
| 9 | 1.6637% | 0.0155% | 1.6782% | 1.6964% | 1.409765e-05% |
| 10 | 1.7708% | 0.0150% | 1.7853% | 1.7986% | 1.709764e-05% |
| 11 | 1.9361% | 0.0154% | 1.9511% | 1.9727% | 1.728435e-05% |
| 12 | 4.9752% | 0.0146% | 4.9889% | 5.0008% | 1.006119e-05% |
| 13 | 5.0066% | 0.0160% | 5.0203% | 5.0403% | 1.677546e-05% |
| 14 | 10.7964% | 0.0128% | 10.8071% | 10.8110% | 7.267463e-06% |
| 15 | 5.0709% | 0.0134% | 5.0823% | 5.0946% | 1.420021e-05% |
| 16 | 4.4208% | 0.0140% | 4.4350% | 4.4656% | 2.002046e-05% |
| 17 | 13.1922% | 0.0108% | 13.1995% | 15.0232% | 0% |

These are independently evaluated counterfactuals, not additive MSE shares. The real evaluator reuses exact bounded integer dots (K*127² < INT32_MAX), with independent NumPy INT64 oracle checks. No FP32 integer approximation is used. Generic production search uses the existing NumPy INT64 accumulator. Multiplier/shift generation and packing code are unchanged; candidate tables record any unrepresentable channels. RTL equivalence, scale handling by every downstream consumer, propagated model error and task accuracy remain separate validation requirements.

## Q/K bug and integration

The current checkout did construct fixed Q/K layer_options but passed global options to LinearCalibration. The CLI now calls operation_options and actually passes its result. A regression test invokes the CLI and verifies exported Q/K=0.2 and gate=0.1, including fixed candidate tables. Historical Q/K artifacts used MSE/non-0.2 scales; they did not use the intended fixed override. Their missing dirty GGUF entrypoint prevents claiming the current CLI was the exact producer. The new mechanism is generic; same-input groups share statistics but can select different s_X. Do not reuse an INT8 input cache across operations with different scales. No position-dependent runtime interface is introduced.

The new real-model numerical experiments cover all 18 down_proj operations. The optional full 126-operation rerun was not performed: complete down_proj attribution already requires substantial full-corpus evaluation, and the measured rare-token regressions need an explicit acceptance policy before broad promotion. Q/K and gate integration is covered by the CLI regression test, not claimed as a new 126-operation real-model result. Reference inputs for all 126 operations were captured and the resumable runner supports `--layers all --modules all`.

## Tests and reproducibility

The full unittest run passed **42 tests, 0 failures** (including 10 new tests). See `bos_aware/tests.log`, `bos_aware/provenance.json`, selected-policy checkpoints and full execution logs. Tests cover BOS invariants, quotas and redistribution, deterministic streaming, population weighting, masks/position propagation, dense output-MSE oracles, minimum-candidate selection, validation independence, LUT constraints, actual CLI Q/K option use, zero/nonfinite cases, unchanged parameter generation, full-data group metrics and existing static_quant/frontend/export tests. No synthetic data substitutes for these real-model measurements. The old diagnostic reports remain historical snapshots; their old source hashes intentionally differ after this policy change.

Artifact verification passed for 18 operations: source fingerprints, frozen calibration-only selection, exact candidate argmin, both dataset splits, BOS counts, finite metrics and protected numerical code/LUT bytes. Maximum relative historical-MSE replay difference is 0; maximum multiplier/shift relative ratio error across evaluated policies is 1.51e-05.

Exported and reloaded 18 frozen stratified-64 policies through the existing exporter into `bos_aware/calibrated/`. NPZ scale/parameter arrays, mapping, software vectors and search sidecar are available. The existing generic-rne profile remains unverified for RTL; its existing export guard prevents default binary publication. No RTL-ready binary claim is made.

No-op resume passed: all 36 frozen selection/result checkpoints retained their modification times. The task-specific file inventory and SHA256 hashes are in `bos_aware/files.json`; previous diagnostics were preserved.

## Explicit answers


1. **Does BOS inclusion remove layer 8/17 clipping?** layer 8: old 0.0004622325%, output-only 0.0001540775%, joint 0%; layer 17: old 0.001078543%, output-only 0%, joint 0%. Inclusion guarantees evaluation, not a zero-clipping constraint.

2. **Stratified versus uniform 64?** With the same new joint objective, median validation NMSE is 4.7690% versus 4.7695%; maxima are 15.0232% versus 45.6690%. Compare both total and BOS/ordinary metrics above; historical-to-new improvement also changes input thresholds and objective, so it is not all attributable to sampling.

3. **64, 128 or 256 sufficient?** stratified_64: median 4.7690%, maximum 15.0232%; stratified_128: median 4.7675%, maximum 14.9760%; stratified_256: median 4.7675%, maximum 14.9760%. 64 is an effective baseline but not converged for every operation. Recommend an explicit 256-row quality preset for further qualification: layer 14 changes to p99.99, lowering full-calibration MSE by 7.3189% and validation NMSE from 10.8110% to 10.0109%; 128 misses this. Compared with 64, 256 changes one input threshold and five output scales. Search costs 91.82 versus 65.63 seconds across 18 operations (+39.9%, +26.19 seconds with this backend), with reservoir upper bounds of 48 versus 12 MiB per down_proj. Two layers have tiny full-calibration MSE regressions versus 64 (layers 1/3, +0.03165%/+0.02868%); larger samples are not a universal monotonic guarantee. The CLI default remains 64, and the main table/export remains stratified_64. No validation-selected per-layer mixing is used.

4. **Selected thresholds:** 0: p99.99, 1: p99.99, 2: p99.99, 3: p99.99, 4: p99.99, 5: p99.99, 6: p99.99, 7: p99.99, 8: p99.99, 9: p99.99, 10: p99.99, 11: p99.99, 12: p99.99, 13: p99.99, 14: absmax, 15: p99.99, 16: p99.99, 17: absmax. Exact thresholds are in the table/CSV.

5. **One fixed percentile for all?** Selected methods are ['absmax', 'p99.99']; the measured per-operation choices must be preserved.

6. **Per-operation selection necessary?** Yes: output-domain error and rare-token energy differ by operation; a universal histogram percentile is not established.

7. **New down_proj median/max:** 4.7690% / 15.0232% validation NMSE at 64 rows.

8. **Worst remaining layer:** 17.

9. **Remaining attribution:** median input-only 4.7463%, weight-only 0.0144%, pair 4.7592%, final 4.7690%. Inspect the layer table for output-stage exceptions; these are not additive contributions.

10. **More complex hardware required?** These experiments test static software policy only and do not establish a need for more hardware complexity. If scalar clipping is insufficient, per-group activation quantization is a future controlled experiment, not part of this change.

11. **Keep static per-tensor s_X interface?** Yes, every measured choice is one static s_X per operation; positions are used only offline.

12. **Keep UInt16/UInt5?** Yes for the measured representable candidates; generation and QB format are unchanged. This does not substitute for RTL equivalence testing.

13. **Production calibration recommendation:** use calibration-only, population-weighted representative sampling with guaranteed position-zero coverage and output-domain per-operation joint search; preserve fixed consumer scales. Use an explicit 256-row quality preset for the next qualification run based on the measured layer-14 stability benefit; retain the current 64-row CLI default pending acceptance, and validate the generic streaming percentile estimator separately from exact offline percentiles. Check full-calibration total and ordinary-token errors before promoting a policy; do not deploy a policy merely because BOS improves. BOS representation fixes a sampling defect, but total MSE can still sacrifice BOS fidelity (layers 7/8) or ordinary-token quality (layer 17). Do not automatically promote these clipping policies as production defaults. Define calibration-only group-error acceptance criteria and expand candidates between p99.99 and absmax before promotion; those constraints are not silently added to this experiment. Validate frozen scales on a broader independent corpus, decode inputs and end-to-end quantized propagation before making model-quality claims.
