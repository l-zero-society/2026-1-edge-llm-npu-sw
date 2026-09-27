# Linear quantization error diagnosis


## 1. Scope audit — read this before interpreting any error


**B: independent local Linear measurements on original/reference-model activations. This is not a full quantized model simulation.** `src/static_quant/gemma.py:191–230` registers a forward **pre-hook**, passes original inputs to the diagnostic callback, and raises `Observed` to stop the prefix. It never replaces outputs or mutates the model with quantized weights. `core.py:215` constructs `Y_ref = X @ W.T` in FP64. `core.py:270–272` compares three local paths; `export.py:27–30` and `docs/static-calibration.md` explicitly exclude propagated full-model quantization.

- Reference RoPE, QKᵀ, softmax and PV affect the reference FP input captured at o_proj. Their own numerical error is **not** measured against an independent exact attention reference.
- No NPU fixed-point/quantized RoPE is executed; no NPU RoPE error propagates into o_proj.
- gate/up inputs include the reference attention/residual/RMSNorm behavior. down_proj inputs include reference gate/up, GeLU and elementwise multiplication. They do **not** include errors from earlier NPU-quantized Linear operations, either in the same layer or earlier layers.
- Reference-model FP rounding and GGUF source effects are already part of the observed reference trajectory. They are not separately attributed by local metrics.
- This diagnostic preserves that local scope: only reference-model hooks capture inputs; all counterfactuals run outside the model.

**Execution status: COMPLETE.** Baseline operations: 126/126; input-sweep rows: 252/252; output-sweep rows: 216/216. All tables below cover completed measurements only.

## 2. Provenance, stale-artifact checks and policy bug


Current commit: `f31f9e6e52a65362acd473f27ed0bfc05097335f`. Stored run commit: `a4a2a1ac26f3291aba1ca1933275598280afc060`, dirty tree: `True`. The stored run used additional/uncommitted GGUF, policy and full-run code absent in this checkout. `source_audit.json` records every available/missing code hash. Only some files, including `hardware.py`, match exactly; the current production CLI alone cannot recreate the historical GGUF run.

The independent diagnostic reconstructs the local GGUF according to recorded conventions: reverse GGUF dimensions, no Q/K permutation, norm weights minus one, output embedding tied. It requires the exact model SHA256, dataset/text/token-ID hashes, Torch 2.2.2, Transformers 4.57.6 and NumPy 1.26.4. It verifies stored per-module artifact hashes, all recovered s_W values, all saved INT8 weight elements, packed parameters, ratios, calibration activation absmax and separately reproduced calibration/validation local MSE. MSE acceptance tolerance is relative 2e-5, absolute 1e-10; measured deviations are in the main CSV. This is an independently reconstructed replay, not recovery of the missing historical source code.

The existing CLI bug is real: `cli.py:127` creates fixed `s10=0.2` Q/K `layer_options`, but `cli.py:135` passes global `options` to `LinearCalibration`. Intended behavior is fixed Q/K 0.2; actual current behavior uses global CLI options. The saved 126-op artifacts explicitly record Q/K **MSE** policies and non-0.2 scales, gate=0.1, and MSE for other linears. They did not use the intended Q/K 0.2 override; their dirty historical entrypoint is not provably the current buggy CLI. This diagnosis loads the saved scales/parameters directly and does not invoke or repair that override. Fixing the unused argument would change local Q/K output quantization only; it would not cause this calibrator to measure or propagate NPU RoPE error. Production policy and RTL remain untouched.

Model: `gemma-2b-it.Q8_0.gguf`, SHA256 `dae8a0a75dda3553c06af737adfff0c003ed1b6393932964cce72bb4f0fd41f6`. The float reference weights are reconstructed Q8_0 GGUF values cast to FP32, then read in FP64 for local GEMMs. **Original BF16/FP16 → Q8_0 source error: not measured.** No original checkpoint was found in the repository, local model directory, Hugging Face cache, or checked download/cache locations; no model was downloaded. **Recovered GGUF → our per-output-channel INT8 error** is exactly the weight-only counterfactual below. These errors are different.

OASST1 calibration: 128 examples / 34,221 valid tokens; validation: 32 / 10,141; maximum length 512, plain text, tree-held-out splits. The exact files are `calibration_outputs/data-oasst1-v1/calibration.jsonl` and `validation.jsonl`, verified against the stored manifest. The IDE-open `data-oasst1-128-32` directory has no files at those JSONL names; its SOURCE_README is not a substitute for the recorded dataset. No padding contributes to metrics. Exact commands are in `commands.jsonl`; complete dataset/model/profile metadata and script hashes are in `diagnosis_provenance.json`. Percentiles are exact over all elements, separately on both splits. Sweep thresholds are fitted only on calibration, then frozen for validation. MSE activation threshold search uses 64 deterministic calibration rows and chooses among the same six threshold candidates; its objective is input reconstruction MSE, not local output MSE.

Generic-rne is the unchanged unverified software profile. The integer backend is `fp64-exact`: all integer products and every partial sum are bounded by K×127² ≤ 264,257,536, below INT32_MAX and FP64’s consecutive-integer limit. The selected FP64 BLAS backend is `accelerate`. FP64 BLAS exactly represents these integer dot products; integer-valued results are checked and each operation/split is cross-checked on full-K sampled rows/channels against NumPy INT64. Default script mode remains pure NumPy INT64. Floating reference/counterfactual GEMMs and reductions are FP64. No reduced-precision GPU matmul is used.

## 3. Counterfactual definitions


The unchanged contract is symmetric signed INT8 [-127,127], zero point 0 for both operands; s_X = calibration input absmax / 127 per operation, and s_W[n] = max|W_float[n,:]| / 127 per output channel. ACC is an exact signed INT32-safe dot product. One shared s_10 per operation maps to signed INT10 [-512,511] with nearest-even rounding. r[n] = s_X s_W[n] / s_10 is approximated by UInt16 multiplier[n] / 2^UInt5 shift[n]. The stored minmax output candidate is max_calibration|ACC s_X s_W| / 511 (a symmetric, slightly conservative bound for the negative side), not the float-reference maximum. MSE search minimizes HW-output versus dequantized-ACC MSE on 64 calibration rows. Exact per-channel s_W and multiplier/shift arrays remain in each original module’s `scales.npz` and `qparams.npz`, located by the original `manifest.json`; their hashes and values were verified, not refitted. `quantization_parameters.csv` summarizes every operation’s s_X, s_10, s_W/multiplier/shift ranges and exact NPZ locations. The attribution CSV records the common s_X and s_10 for every operation and split.

| CSV prefix | Value | Reference / NMSE denominator |
| --- | --- | --- |
| input_only | (X_q × s_X) @ W_float.T | Y_ref / mean(Y_ref²) |
| weight_only | X_float @ (W_q × s_W).T | Y_ref / mean(Y_ref²) |
| int8_pair | ACC × s_X × s_W | Y_ref / mean(Y_ref²) |
| ideal_int10 | clip(rint(ACC × exact_ratio)) × s_10 | Y_ref / mean(Y_ref²) |
| final_local_total | actual profile output × s_10 | Y_ref / mean(Y_ref²) |
| actual_requant | actual profile output × s_10 | ideal INT10 / mean(Y_ideal²) |
| ideal_requantization_only | ideal INT10 × s_10 | INT8 pair / mean(Y_pair²) |
| requantization_only | actual profile output × s_10 | INT8 pair / mean(Y_pair²) |

Each prefix includes MSE, MAE, NMSE, maximum absolute error and reference energy. **These are separate counterfactuals, not additive contributions. No MSE subtraction is used for attribution.** Writing ΔX = X_hat − X and ΔW = W_hat − W, the pair error vector is ΔX Wᵀ + X ΔWᵀ + ΔX ΔWᵀ. Its squared norm includes cross terms; adding the independent input-only and weight-only MSEs does not reconstruct pair MSE. Requantization error can likewise correlate with the preceding error. HW-vs-ideal metrics are in real units here; historical parameter_approximation metrics used INT10 code units. NMSE percentages represent squared error relative to signal energy, not percent accuracy loss or relative amplitude error. For zero reference energy NMSE is 0 only if error is also zero, otherwise blank/null. Input clipping means |X|>127s_X; endpoint occupancy |q|=127 is reported separately from clipping. The strict |X|<s_X/2 count differs from q=0 at exact half-scale ties because nearest-even also rounds ±0.5 to zero.

## 4. Baseline attribution

Measured operations: 126/126; both-split rows: 252/252. Maximum relative reproduced-vs-stored MSE difference: 3.664e-15.

| Layer | s_X | s_10 | Input NMSE | Weight NMSE | Pair NMSE | Ideal INT10 NMSE | Final HW NMSE | Zero rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.870994 | 0.02822045 | 3.9021% | 0.0098% | 3.9100% | 3.9183% | 3.9183% | 88.2034% |
| 1 | 0.653449 | 0.009368304 | 3.9687% | 0.0175% | 3.9894% | 3.9941% | 3.9941% | 89.2054% |
| 2 | 0.8677332 | 0.007314705 | 16.5658% | 0.0134% | 16.5754% | 16.5793% | 16.5793% | 94.2588% |
| 3 | 0.6794692 | 0.005114523 | 34.6848% | 0.0136% | 34.6964% | 34.6998% | 34.6998% | 96.0109% |
| 4 | 0.549462 | 0.003328987 | 39.7588% | 0.0128% | 39.7697% | 39.7762% | 39.7763% | 96.0436% |
| 5 | 0.3957401 | 0.005974751 | 28.9799% | 0.0142% | 28.9979% | 29.0266% | 29.0266% | 93.9216% |
| 6 | 0.3423329 | 0.005328721 | 26.8242% | 0.0161% | 26.8407% | 26.8561% | 26.8561% | 92.2672% |
| 7 | 0.3126632 | 0.0087618 | 25.6581% | 0.0151% | 25.6691% | 25.9401% | 25.9400% | 90.8609% |
| 8 | 0.5708727 | 0.01383242 | 47.1881% | 0.0148% | 47.2068% | 48.1295% | 48.1294% | 97.3488% |
| 9 | 0.1579513 | 0.005881323 | 6.8128% | 0.0155% | 6.8273% | 6.8476% | 6.8476% | 69.7141% |
| 10 | 0.1985953 | 0.004704841 | 9.7600% | 0.0150% | 9.7770% | 9.7901% | 9.7901% | 76.7960% |
| 11 | 0.2596352 | 0.007084301 | 9.2729% | 0.0154% | 9.2869% | 9.3081% | 9.3082% | 80.8256% |
| 12 | 0.3489435 | 0.007809064 | 11.6986% | 0.0146% | 11.7137% | 11.7296% | 11.7295% | 88.0028% |
| 13 | 0.588631 | 0.009247759 | 15.0638% | 0.0160% | 15.0942% | 15.1083% | 15.1084% | 92.4466% |
| 14 | 0.8091608 | 0.007030724 | 10.7964% | 0.0128% | 10.8071% | 10.8110% | 10.8110% | 95.8303% |
| 15 | 0.7310743 | 0.018482 | 8.1219% | 0.0134% | 8.1354% | 8.1544% | 8.1544% | 94.1106% |
| 16 | 1.144403 | 0.01814517 | 11.6969% | 0.0140% | 11.6954% | 11.8090% | 11.8090% | 93.7367% |
| 17 | 18.48648 | 0.4180036 | 13.1922% | 0.0108% | 13.1995% | 20.0998% | 20.0998% | 99.9140% |

For the worst measured operation `model.layers.8.mlp.down_proj`, final MSE 0.006037534 divided by reference energy 0.01254438 equals NMSE 48.1294%. A small raw MSE can coexist with a large normalized error. The denominator is mean(Y_ref²), not centered variance or an average of channel NMSEs. All valid output elements are equally weighted, so longer examples contribute more tokens.


Worst 10 measured Linear operations by validation final NMSE:


| Operation | Input NMSE | Weight NMSE | Pair NMSE | Final NMSE | Final MSE |
| --- | --- | --- | --- | --- | --- |
| model.layers.8.mlp.down_proj | 47.1881% | 0.0148% | 47.2068% | 48.1294% | 0.006037534 |
| model.layers.4.mlp.down_proj | 39.7588% | 0.0128% | 39.7697% | 39.7763% | 0.01104377 |
| model.layers.3.mlp.down_proj | 34.6848% | 0.0136% | 34.6964% | 34.6998% | 0.02182938 |
| model.layers.5.mlp.down_proj | 28.9799% | 0.0142% | 28.9979% | 29.0266% | 0.005734438 |
| model.layers.6.mlp.down_proj | 26.8242% | 0.0161% | 26.8407% | 26.8561% | 0.004024074 |
| model.layers.7.mlp.down_proj | 25.6581% | 0.0151% | 25.6691% | 25.9400% | 0.003386366 |
| model.layers.17.mlp.down_proj | 13.1922% | 0.0108% | 13.1995% | 20.0998% | 0.3346652 |
| model.layers.2.mlp.down_proj | 16.5658% | 0.0134% | 16.5754% | 16.5793% | 0.02844086 |
| model.layers.13.mlp.down_proj | 15.0638% | 0.0160% | 15.0942% | 15.1084% | 0.009257316 |
| model.layers.16.mlp.down_proj | 11.6969% | 0.0140% | 11.6954% | 11.8090% | 0.03573247 |

**Important exception: layer 17 down_proj has a material INT10-stage loss.** Pair NMSE is 13.1995%, ideal INT10 NMSE is 20.0998%, and actual hardware-profile NMSE is 20.0998%. Its independently measured requantization-only MSE is 0.1017033 and NMSE is 7.4177% relative to pair energy. This is output-grid/clipping loss, not significant multiplier/shift error. Do not generalize the overall activation-dominance conclusion into “INT10 is negligible in every layer.” No MSE differences are treated as additive contributions.

## 5. Activation distributions and MLP comparison


`down_proj_activation_stats.csv` contains calibration and validation min/max/absmax, mean/std, exact |X| p50/p90/p95/p99/p99.5/p99.9/p99.95/p99.99/max, absmax/p99, absmax/p99.9, absmax/median, s_X, zero/±1/|q|≤2/endpoint occupancy, clipping, strict half-scale fraction, MAE and SQNR. Activation SQNR is 10 log10(sum(X²) / sum((X_hat−X)²)); it uses signal energy, not centered variance. `mlp_activation_comparison.csv` contains the same statistics plus input-only, weight-only, pair and final NMSE for gate/up/down in every layer. gate/up share the same input group and scale; down receives the reference GeLU(gate) × up product, before NPU quantization. High zero occupancy is a distribution diagnostic, not an error contribution or standalone proof: some reference values are already zero or carry little output energy. The separately measured input-only output error establishes the impact; zero rate and NMSE need not rank layers identically.

| Layer | Validation absmax | p99 | p99.9 | absmax/p99.9 | |X|<s_X/2 | Zero after INT8 | SQNR dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 215.8658 | 6.28987 | 13.41919 | 16.08635 | 88.2034% | 88.2034% | 14.90899 |
| 1 | 83.91769 | 4.149948 | 7.832193 | 10.71446 | 89.2054% | 89.2054% | 17.50295 |
| 2 | 105.9458 | 2.168568 | 4.958582 | 21.36616 | 94.2588% | 94.2588% | 11.66149 |
| 3 | 86.28899 | 0.990546 | 2.67754 | 32.22697 | 96.0109% | 96.0109% | 7.991349 |
| 4 | 52.00896 | 0.7056706 | 2.052566 | 25.33851 | 96.0436% | 96.0436% | 6.577106 |
| 5 | 63.12912 | 0.5231633 | 1.699527 | 37.14511 | 93.9216% | 93.9216% | 6.459531 |
| 6 | 36.32359 | 0.5056716 | 1.642788 | 22.11094 | 92.2672% | 92.2672% | 6.793653 |
| 7 | 39.70823 | 0.5352584 | 1.57261 | 25.24989 | 90.8609% | 90.8609% | 6.891937 |
| 8 | 72.50084 | 0.524115 | 1.470062 | 49.31823 | 97.3488% | 97.3488% | 4.818229 |
| 9 | 27.41071 | 0.6473226 | 1.663139 | 16.48131 | 69.7141% | 69.7141% | 11.71487 |
| 10 | 23.14849 | 0.6440876 | 1.705075 | 13.57623 | 76.7960% | 76.7960% | 10.17711 |
| 11 | 29.50646 | 0.8405846 | 2.145645 | 13.75179 | 80.8256% | 80.8256% | 10.42217 |
| 12 | 62.86323 | 0.8963935 | 2.458812 | 25.5665 | 88.0028% | 88.0028% | 9.671857 |
| 13 | 60.76543 | 1.203853 | 3.338619 | 18.20077 | 92.4466% | 92.4466% | 9.43912 |
| 14 | 97.1623 | 1.241186 | 4.4722 | 21.72584 | 95.8303% | 95.8303% | 10.76291 |
| 15 | 98.58845 | 1.648775 | 5.808441 | 16.97331 | 94.1106% | 94.1106% | 12.02198 |
| 16 | 112.1872 | 2.67064 | 7.518729 | 14.92104 | 93.7367% | 93.7367% | 11.82087 |
| 17 | 2347.783 | 3.077073 | 8.713147 | 269.4529 | 99.9140% | 99.9140% | 6.782822 |


MLP validation comparison, medians across measured layers (full per-layer table in CSV):


| Module | s_X | absmax/p99.9 | Zero rate | Input NMSE | Weight NMSE | Pair NMSE | Final NMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_proj | 0.5399041 | 4.920807 | 10.8405% | 0.1090% | 0.0022% | 0.1112% | 0.1742% |
| up_proj | 0.5399041 | 4.920807 | 10.8405% | 0.2964% | 0.0053% | 0.3013% | 0.3124% |
| down_proj | 0.5797519 | 21.546 | 93.0916% | 12.4454% | 0.0144% | 12.4566% | 13.4587% |

## 6. Diagnostic input scale sweep (production unchanged)


All alternatives remain static per-tensor signed INT8 with zero point zero. Exact calibration percentiles 99, 99.5, 99.9, 99.95 and 99.99 are divided by 127. The recorded per-channel weight scales and common s_10 are held fixed; multiplier/shift are recalculated for the new input ratio. Thus input clipping and zero-rate effects can be compared without silently refitting output scales. `down_proj_scale_sweep.csv` contains selected thresholds, scales, all requested errors and clip rates for both splits. The additional 64-row MSE policy is selected on calibration only. Validation-best below is explicitly retrospective, not a deployable selection rule.

| Layer | Baseline final NMSE | Best validation policy (retrospective) | Threshold | s_X | Input clipped | Zero rate | Final NMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.9183% | percentile_99.99 | 41.68604 | 0.3282365 | 0.0099% | 68.0600% | 3.5498% |
| 1 | 3.9941% | percentile_99.99 | 13.18572 | 0.1038245 | 0.0098% | 49.1548% | 1.8153% |
| 2 | 16.5793% | percentile_99.99 | 11.27933 | 0.08881361 | 0.0094% | 44.9304% | 4.5372% |
| 3 | 34.6998% | percentile_99.99 | 6.441558 | 0.05072093 | 0.0096% | 25.3967% | 12.9311% |
| 4 | 39.7763% | percentile_99.99 | 4.854371 | 0.0382234 | 0.0094% | 23.2091% | 6.1201% |
| 5 | 29.0266% | percentile_99.99 | 4.266167 | 0.03359187 | 0.0095% | 22.7101% | 5.5106% |
| 6 | 26.8561% | percentile_99.99 | 4.129004 | 0.03251184 | 0.0095% | 24.2816% | 3.8064% |
| 7 | 25.9400% | percentile_99.99 | 3.577202 | 0.02816695 | 0.0094% | 21.1724% | 3.3030% |
| 8 | 48.1294% | percentile_99.99 | 3.11485 | 0.02452638 | 0.0101% | 18.7408% | 6.0467% |
| 9 | 6.8476% | percentile_99.99 | 3.339219 | 0.02629306 | 0.0103% | 19.8590% | 1.6989% |
| 10 | 9.7901% | percentile_99.99 | 3.564663 | 0.02806822 | 0.0099% | 20.4059% | 1.7986% |
| 11 | 9.3082% | percentile_99.99 | 5.09224 | 0.04009638 | 0.0097% | 24.8093% | 1.9727% |
| 12 | 11.7295% | percentile_99.99 | 6.063841 | 0.04774678 | 0.0099% | 28.3755% | 5.0049% |
| 13 | 15.1084% | percentile_99.99 | 8.979519 | 0.07070487 | 0.0095% | 35.7105% | 5.0403% |
| 14 | 10.8110% | percentile_99.99 | 13.78569 | 0.1085487 | 0.0096% | 51.5806% | 10.0109% |
| 15 | 8.1544% | percentile_99.99 | 15.74381 | 0.123967 | 0.0093% | 52.7494% | 5.1009% |
| 16 | 11.8090% | percentile_99.99 | 17.09823 | 0.1346317 | 0.0095% | 46.8609% | 4.7407% |
| 17 | 20.0998% | absmax | 2347.783 | 18.48648 | 0.0000% | 99.9140% | 20.0998% |


Calibration-only selection by minimum full-calibration final NMSE among the tested policies (validation remains held out):


| Layer | Chosen policy | Baseline validation NMSE | Chosen validation NMSE | Relative NMSE reduction |
| --- | --- | --- | --- | --- |
| 0 | percentile_99.99 | 3.9183% | 3.5498% | 9.4065% |
| 1 | percentile_99.99 | 3.9941% | 1.8153% | 54.5510% |
| 2 | percentile_99.99 | 16.5793% | 4.5372% | 72.6332% |
| 3 | percentile_99.99 | 34.6998% | 12.9311% | 62.7345% |
| 4 | percentile_99.99 | 39.7763% | 6.1201% | 84.6136% |
| 5 | percentile_99.99 | 29.0266% | 5.5106% | 81.0153% |
| 6 | percentile_99.99 | 26.8561% | 3.8064% | 85.8269% |
| 7 | percentile_99.99 | 25.9400% | 3.3030% | 87.2667% |
| 8 | percentile_99.99 | 48.1294% | 6.0467% | 87.4366% |
| 9 | percentile_99.99 | 6.8476% | 1.6989% | 75.1902% |
| 10 | percentile_99.99 | 9.7901% | 1.7986% | 81.6285% |
| 11 | percentile_99.99 | 9.3082% | 1.9727% | 78.8073% |
| 12 | percentile_99.99 | 11.7295% | 5.0049% | 57.3312% |
| 13 | percentile_99.99 | 15.1084% | 5.0403% | 66.6393% |
| 14 | percentile_99.99 | 10.8110% | 10.0109% | 7.4008% |
| 15 | percentile_99.99 | 8.1544% | 5.1009% | 37.4454% |
| 16 | percentile_99.99 | 11.8090% | 4.7407% | 59.8551% |
| 17 | absmax | 20.0998% | 20.0998% | 0.0000% |

Across these 18 layers, calibration-only selection improves 17, regresses 0, and leaves 1 unchanged on validation. Median baseline validation NMSE is 13.4587%; median selected-policy validation NMSE is 4.8728%. These are medians of layer NMSEs, not a pooled output-energy-weighted model metric. The search is limited to the tested scalar thresholds, not a claim of global optimality.


Median validation final NMSE across layers for each fixed policy:


| Policy | Layers | Median final NMSE | Median input-only NMSE |
| --- | --- | --- | --- |
| absmax | 18 | 13.4587% | 12.4454% |
| percentile_99 | 18 | 28.7295% | 28.6422% |
| percentile_99.5 | 18 | 21.2486% | 21.1338% |
| percentile_99.9 | 18 | 11.9293% | 11.7627% |
| percentile_99.95 | 18 | 9.0131% | 8.9964% |
| percentile_99.99 | 18 | 4.8728% | 4.7463% |
| activation_mse_search_64_rows | 18 | 5.0226% | 4.9909% |

**Do not apply p99.99 universally:** layer 17 regresses from 20.0998% to 45.6690%. Its input-only NMSE rises from 13.1922% to 44.8033%, even though the zero rate falls from 99.9140% to 52.7085%. Rare large inputs carry important output energy. The 64-row input-reconstruction MSE search also picks this harmful threshold, while full-calibration final-output NMSE selects absmax. A lower zero rate or a better median across layers does not guarantee a better individual operation.

## 7. Output scale examined independently


With baseline X_q/W_q and scales fixed, `down_proj_output_scale_sweep.csv` reevaluates every stored MSE-search candidate plus the exact stored minmax baseline on both full splits. Each candidate uses fresh profile multiplier/shift; ideal and actual requantization-only NMSE are also reported. No validation selection changes production.

| Layer | Current clip rate | Current requant-only NMSE | Current final NMSE | Minmax s_10 | Minmax clip rate | Minmax requant-only NMSE | Minmax final NMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0007% | 0.0055% | 3.9183% | 0.0564409 | 0.0000% | 0.0123% | 3.9210% |
| 1 | 0.0008% | 0.0041% | 3.9941% | 0.01873661 | 0.0000% | 0.0080% | 3.9965% |
| 2 | 0.0002% | 0.0042% | 16.5793% | 0.01044958 | 0.0000% | 0.0070% | 16.5806% |
| 3 | 0.0000% | 0.0053% | 34.6998% | 0.007306462 | 0.0000% | 0.0104% | 34.7035% |
| 4 | 0.0004% | 0.0072% | 39.7763% | 0.006657974 | 0.0000% | 0.0192% | 39.7835% |
| 5 | 0.0004% | 0.0303% | 29.0266% | 0.0119495 | 0.0000% | 0.0741% | 29.0574% |
| 6 | 0.0006% | 0.0257% | 26.8561% | 0.01065744 | 0.0000% | 0.0681% | 26.9036% |
| 7 | 0.0005% | 0.3021% | 25.9400% | 0.0175236 | 0.0000% | 0.2064% | 25.8643% |
| 8 | 0.0005% | 1.1637% | 48.1294% | 0.02766485 | 0.0000% | 0.6376% | 47.7171% |
| 9 | 0.0000% | 0.0196% | 6.8476% | 0.00840189 | 0.0000% | 0.0389% | 6.8683% |
| 10 | 0.0001% | 0.0125% | 9.7901% | 0.006721201 | 0.0000% | 0.0239% | 9.8018% |
| 11 | 0.0001% | 0.0199% | 9.3082% | 0.0141686 | 0.0000% | 0.0629% | 9.3507% |
| 12 | 0.0000% | 0.0162% | 11.7295% | 0.01115581 | 0.0000% | 0.0328% | 11.7460% |
| 13 | 0.0001% | 0.0134% | 15.1084% | 0.01321108 | 0.0000% | 0.0251% | 15.1182% |
| 14 | 0.0001% | 0.0040% | 10.8110% | 0.01004389 | 0.0000% | 0.0080% | 10.8151% |
| 15 | 0.0000% | 0.0197% | 8.1544% | 0.0231025 | 0.0000% | 0.0307% | 8.1643% |
| 16 | 0.0023% | 0.0709% | 11.8090% | 0.03629033 | 0.0000% | 0.0408% | 11.7316% |
| 17 | 0.0011% | 7.4177% | 20.0998% | 0.8360071 | 0.0000% | 2.4465% | 15.0232% |

For layer 17, minmax s_10=0.8360071 eliminates measured clipping and lowers final NMSE to 15.0232%, with requantization-only NMSE 2.4465%. Residual output-grid loss remains; removing saturation is not equivalent to removing all INT10 quantization error. Nearby candidate scores are in the CSV, and none are applied to production.

### BOS outlier and output-search sample audit (layers 8 and 17 only)

`outlier_position_audit.json` independently reconstructs exact token positions and the historical 64-row reservoir (seed 42). The input absmax occurs at BOS in both inspected layers. All baseline INT10 clipping events in both splits occur at BOS. The historical output-scale search sampled **zero BOS rows**; all six stored search MSE scores were reproduced. Thus its preference for half the minmax scale optimizes a sample that misses these rare large outputs. This is a demonstrated sampling limitation, distinct from multiplier/shift approximation and the input absmax problem. BOS remains included in every reported metric; it has not been removed to improve results. Other layers were not audited by token position.

| Layer | Split | BOS rows | BOS input absmax | Non-BOS input absmax | INT10 clips at BOS / total |
| --- | --- | --- | --- | --- | --- |
| 8 | calibration | 128 | 72.50084 | 31.52699 | 384 / 384 |
| 8 | validation | 32 | 72.50084 | 20.01126 | 96 / 96 |
| 17 | calibration | 128 | 2347.783 | 134.2675 | 896 / 896 |
| 17 | validation | 32 | 2347.783 | 135.4066 | 224 / 224 |

## 8. Tests and limitations

`tests.log` records the full static_quant/frontend suite plus diagnostic tests. Diagnostic tests independently check all counterfactuals, Python integer RNE oracle including negative values, nonadditive MSE example, streamed-vs-dense metrics/denominators, INT64-vs-FP64 exact integer paths at K=31/257/2048/16384, overflow guards, strict zero threshold versus ties, saturation versus clipping, exact percentiles, zero-energy references, Accelerate-vs-NumPy/INT64 oracles, and fused-C-vs-NumPy metric/sweep comparisons. Hardware ABI/RTL equivalence, full-model propagated error, task accuracy and GGUF-vs-original source error remain unmeasured. A limited 128/32-example corpus cannot establish a universally safe production policy. Stored artifact checks and replay agreement validate consistency with the historical local result, not independent llama.cpp-vs-Transformers full-model equivalence.

Executed test result: **32 tests passed** (0.188 seconds).

Complete-output verification: **PASS**. All 126 operations, both splits, all 18 activation/output sweeps, frozen baseline scales, calibration-only threshold fitting, NMSE denominators, parameter widths, source hashes and replay agreement were checked; see `output_validation.json`.

Resume verification: **PASS**. Repeating the complete command exits successfully without loading the model or rewriting any of the 144 completed baseline/sweep checkpoints; see `resume_validation.json`.

## 9. Explicit answers


1. **Full quantized end-to-end simulation? No.** Original-input local Linear counterfactuals.
2. **NPU RoPE fixed-point error included? No.** Reference RoPE affects reference activations but its numerical error is not separately evaluated.
3. **Previous quantized-layer errors propagated? No.** Neither reference hooks nor this diagnostic inject quantized outputs.

4. **Dominant down_proj source: activation INT8.** Measured median input-only NMSE 12.4454%, weight-only 0.0144%, combined pair 12.4566%, final INT10 13.4587%. Median requantization-only NMSE is 0.0179% (different denominator). The ranking and closeness of independent paths identify the dominant stage; these numbers are not additive shares. Layer 17 is an important exception with substantial additional INT10 loss, as quantified above.

5. **Evidence that Q8_0 itself is inaccurate? No.** Y_ref already uses recovered GGUF weights; original-source comparison is not measured.

6. **Absmax activation bottleneck?** The input-only/pair agreement and zero occupancy support this diagnosis when activation error dominates. Median down_proj zero-after-INT8 fraction is 93.0916%; compare input-only versus weight-only/pair results above and the measured percentile sweep before choosing a policy.

7. **Clipping improvement:** absmax: median validation final NMSE 13.4587%; percentile_99: median validation final NMSE 28.7295%; percentile_99.5: median validation final NMSE 21.2486%; percentile_99.9: median validation final NMSE 11.9293%; percentile_99.95: median validation final NMSE 9.0131%; percentile_99.99: median validation final NMSE 4.8728%; activation_mse_search_64_rows: median validation final NMSE 5.0226%. This includes regression as well as improvement; validation-best choices in the layer table are retrospective.

Layer 17 is a counterexample: fixed p99.99 worsens final NMSE from 20.0998% to 45.6690%; calibration-only final-output selection retains absmax.

8. **Does down_proj require a different activation policy?** The present absmax policy is a demonstrated local-accuracy bottleneck in many layers, so activation-policy work is justified. A universal percentile replacement is contradicted by layer 17. A deployment decision needs a calibration-only selection rule validated on broader held-out data and propagated full-model evaluation. The measured counterfactuals diagnose the local bottleneck; no production policy is changed here. If all tested scalar clipping thresholds remain insufficient, per-group activation quantization is a possible future experiment, not an implemented fix.

9. **UInt16 multiplier / UInt5 shift sufficient?** The maximum validation HW-vs-ideal real-unit MSE across operations is 1.14872e-06. Maximum validation HW-vs-ideal NMSE is 0.00005744%. The original saved ratios all pass 0.001 tolerance (maximum relative error 1.50515e-5), with at most one INT10 code difference in saved validation. These parameters are sufficient for the recorded software ratios to that tolerance; this does not establish RTL correctness or guarantee future scale ranges.

10. **Before changing RTL:** include BOS/outlier rows in output-scale calibration and reevaluate the full-calibration objective; freeze a calibration-only input threshold selection method, evaluate it on a larger independent corpus and decode activations, then run quantized output injection through attention, residual, RMSNorm and GeGLU. Validate scale contracts for every consumer and run exact profile test vectors on RTL. Only an independently available original checkpoint can quantify GGUF source error. Preserve the current hardware widths until evidence identifies a limitation.
