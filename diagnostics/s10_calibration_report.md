# Common s_10 calibration with frozen factorized PoT scales

Diagnostic only. Previous PoT s_X_normal, k, s_W, W_q, calibration/validation observations are frozen in the primary pass. Existing activation caches are reused; no model forward, activation capture, download, LUT, ABI, QB, RTL or production changes. One common s_10 is used by both groups. Every candidate regenerates ONLY the normal channel multiplier/shift array; BOS applies the existing scalar 2^k before RNE and saturation.

Candidates: old s_10 * 2^(i/8), i=-16..16, exact old baseline included. Selection uses 0.5*NMSE_BOS + 0.5*NMSE_nonBOS on calibration only. The same 256 prior non-BOS row IDs and within-non-BOS weights are reused; all 128 calibration BOS rows are used independently. BOS is never weighted by corpus frequency. Balanced-score ties use rtol=1e-6, atol=1e-12, then lower worst-group NMSE, lower equal-group clipping, and closer log-distance to baseline.

Practical effective-shift support is explicitly 0..31, the current unsigned five-bit shift domain, for BOTH groups. The previous observed BOS range 11..28 was descriptive, not a hardware limit; previous normal shifts already reach 31. No negative or extended shift is accepted for the main result. Unconstrained and feasible optima, base/BOS/combined shift ranges and ratio status are recorded for every operation.

| Layer | Scheme | sX normal | k | s10 | Ratio | BOS NMSE % | Normal NMSE % | BOS clip % | Normal clip % | Effective shifts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | frozen | 0.32823654 | -2 | 0.039508632 | 1 | 0.403267 | 3.545723 | 0 | 3.864131e-05 | 17..27 |
| 0 | calibrated | 0.32823654 | -2 | 0.025618178 | 0.64842 | 0.326840 | 3.554799 | 0 | 0.001980367 | 15..26 |
| 1 | frozen | 0.10382454 | -3 | 0.013115626 | 1 | 0.655488 | 1.813293 | 0 | 3.381115e-05 | 17..28 |
| 1 | calibrated | 0.10382454 | -3 | 0.0071513459 | 0.545254 | 0.418149 | 1.837536 | 0 | 0.01657712 | 15..27 |
| 2 | frozen | 0.088813612 | -3 | 0.0073147055 | 1 | 0.324924 | 4.537046 | 0 | 0.00266142 | 14..27 |
| 2 | calibrated | 0.088813612 | -3 | 0.0067076145 | 0.917004 | 0.318053 | 4.542880 | 0 | 0.00559333 | 17..27 |
| 3 | frozen | 0.050720926 | -2 | 0.0058451697 | 1 | 0.266701 | 12.934968 | 0 | 5.31318e-05 | 16..27 |
| 3 | calibrated | 0.050720926 | -2 | 0.0041331591 | 0.707107 | 0.238712 | 12.936506 | 0 | 0.001236522 | 18..26 |
| 4 | frozen | 0.038223396 | -1 | 0.003328987 | 1 | 0.292092 | 6.123946 | 0 | 0.001226862 | 15..25 |
| 4 | calibrated | 0.038223396 | -1 | 0.0047078985 | 1.41421 | 0.299314 | 6.119757 | 0 | 6.762229e-05 | 16..26 |
| 5 | frozen | 0.033621895 | 0 | 0.0083646517 | 1 | 0.939184 | 5.509845 | 0 | 3.381115e-05 | 19..26 |
| 5 | calibrated | 0.033621895 | 0 | 0.0070338056 | 0.840896 | 0.891410 | 5.504899 | 0 | 0.0002125272 | 18..26 |
| 6 | frozen | 0.032511839 | 0 | 0.0053287211 | 1 | 0.621148 | 3.811306 | 0 | 0.0001932065 | 18..25 |
| 6 | calibrated | 0.032511839 | 0 | 0.0037679748 | 0.707107 | 0.601330 | 3.841129 | 0 | 0.001917575 | 18..25 |
| 7 | frozen | 0.028166948 | 4 | 0.0087618001 | 1 | 17.042881 | 2.292682 | 0.1464844 | 0 | 11..26 |
| 7 | calibrated | 0.028166948 | 4 | 0.0175236 | 2 | 3.948794 | 2.441909 | 0 | 0 | 12..27 |
| 8 | frozen | 0.02452638 | 5 | 0.013832423 | 1 | 17.464183 | 2.089106 | 0.1464844 | 0 | 15..27 |
| 8 | calibrated | 0.02452638 | 5 | 0.027664846 | 2 | 1.828045 | 2.490037 | 0 | 0 | 16..28 |
| 9 | frozen | 0.026544869 | 0 | 0.0042009451 | 1 | 0.615821 | 1.671769 | 0 | 0.0005941101 | 19..25 |
| 9 | calibrated | 0.026544869 | 0 | 0.0038522836 | 0.917004 | 0.611167 | 1.677090 | 0 | 0.001144749 | 14..25 |
| 10 | frozen | 0.029077519 | 0 | 0.0047048409 | 1 | 0.538641 | 1.702538 | 0.04882812 | 8.211278e-05 | 16..25 |
| 10 | calibrated | 0.029077519 | 0 | 0.0051306654 | 1.09051 | 0.412435 | 1.704347 | 0 | 3.381115e-05 | 19..25 |
| 11 | frozen | 0.041208523 | -2 | 0.0070843014 | 1 | 0.204563 | 1.880040 | 0 | 0.0001545652 | 18..27 |
| 11 | calibrated | 0.041208523 | -2 | 0.0042123508 | 0.594604 | 0.166120 | 1.925076 | 0 | 0.002400591 | 16..27 |
| 12 | frozen | 0.047746778 | -2 | 0.0055779031 | 1 | 0.137217 | 5.003376 | 0 | 0.0003139606 | 16..27 |
| 12 | calibrated | 0.047746778 | -2 | 0.0046904387 | 0.840896 | 0.127284 | 5.008593 | 0 | 0.0009080708 | 16..27 |
| 13 | frozen | 0.075991708 | -2 | 0.0092477586 | 1 | 0.149443 | 4.535483 | 0 | 0.0005651292 | 17..27 |
| 13 | calibrated | 0.075991708 | -2 | 0.007776407 | 0.840896 | 0.133262 | 4.549643 | 0 | 0.001598784 | 16..27 |
| 14 | frozen | 0.11293577 | -2 | 0.0070307245 | 1 | 0.158499 | 9.511244 | 0 | 0.0003429416 | 15..26 |
| 14 | calibrated | 0.11293577 | -2 | 0.0070307245 | 1 | 0.158499 | 9.511244 | 0 | 0.0003429416 | 15..26 |
| 15 | frozen | 0.12424406 | -1 | 0.01155125 | 1 | 0.924522 | 5.089879 | 0.04882812 | 0.0005119974 | 16..26 |
| 15 | calibrated | 0.12424406 | -1 | 0.014980117 | 1.29684 | 0.130106 | 5.092778 | 0 | 1.932065e-05 | 17..26 |
| 16 | frozen | 0.13919387 | 1 | 0.032661298 | 1 | 0.311607 | 4.337303 | 0.04882812 | 9.177311e-05 | 12..26 |
| 16 | calibrated | 0.13919387 | 1 | 0.035617398 | 1.09051 | 0.251762 | 4.342149 | 0 | 1.932065e-05 | 17..26 |
| 17 | frozen | 0.14417333 | 7 | 0.83600715 | 1 | 0.560834 | 21.035584 | 0 | 0 | 15..31 |
| 17 | calibrated | 0.14417333 | 7 | 0.54208357 | 0.64842 | 3.616711 | 12.416163 | 0.09765625 | 0 | 12..30 |
| 7 | joint | 0.033496335 | 3 | 0.016069212 | 1.83401 | 3.751494 | 1.921822 | 0 | 0 | 16..27 |
| 17 | joint | 0.17145194 | 7 | 0.54208357 | 0.64842 | 3.801739 | 11.720912 | 0.09765625 | 0 | 14..30 |
| 8 | joint | 0.029166945 | 5 | 0.027664846 | 2 | 1.938519 | 2.036026 | 0 | 0 | 16..28 |

NMSEs above use full independent validation. Candidate tables contain calibration estimates. Output-grid lower bounds use the exact FP reference rounded directly onto each candidate INT10 grid. INT8-pair errors are independent counterfactuals; no subtraction of MSEs is used as additive attribution.

frozen: BOS median/max NMSE 0.470954% / 17.464183%; non-BOS median/max 4.436393% / 21.035584%.

calibrated: BOS median/max NMSE 0.322446% / 3.948794%; non-BOS median/max 4.442515% / 12.936506%.

Optional joint pass is restricted to layers 7/8/17: 3 input-scale factors {2^-0.25,1,2^0.25}, k±1, and 3 output-scale factors {2^-0.125,1,2^0.125} around the s10-only selected point (27 candidates). Same calibration rows/objective/shift bounds; no validation selection.

GO after layers 7/8/17: validation BOS NMSE falls from 17.0429% to 3.9488% (layer 7) and 17.4642% to 1.8280% (layer 8), with non-BOS increases of 0.1492 and 0.4009 percentage points. These are material gains with modest ordinary-token costs, so expand to remaining down_proj only. Layer 17 trades BOS 0.5608%→3.6167% for non-BOS 21.0356%→12.4162%, illustrating the balanced objective, not a no-regression guarantee. All choices were fitted on calibration; validation is used only for this requested architecture-expansion decision. Also authorize the bounded optional three-layer joint diagnostic to measure residual input-scale restrictions; do not expand joint search to all 18.

## Answers A–F

A. Yes. With s_X/k frozen, doubling s_10 reduces layer-7 BOS validation NMSE 17.042881%→3.948794% and layer-8 17.464183%→1.828045%. BOS INT10 clipping drops from 0.146484375% to zero in both. This directly demonstrates that the formerly frozen output grid was a major limitation.

B (layer 7). Selected-grid lower-bound BOS NMSE is 0.034882%, compared with actual 3.948794%. The unchanged INT8-pair counterfactual is 3.928831%. The result remains far above the new grid bound; the remaining input/GEMM error is now the relevant restriction. These are separate comparisons, not additive error components.

B (layer 8). Selected-grid lower-bound BOS NMSE is 0.031024%, compared with actual 1.828045%. The unchanged INT8-pair counterfactual is 1.788323%. The result remains far above the new grid bound; the remaining input/GEMM error is now the relevant restriction. These are separate comparisons, not additive error components.

C. The balanced objective permits trade-offs. In the s10-only pass, layers 7/8 non-BOS NMSE rises 2.292682%→2.441909% and 2.089106%→2.490037% (+0.149227/+0.400931 percentage points). Layer 17 instead trades BOS 0.560834%→3.616711% for non-BOS 21.035584%→12.416163%. Across 18 operations, 14 have some non-BOS increase (often tiny), and two have BOS increases (4/17). The median non-BOS change is small, but no no-regression guarantee is claimed.

D. Selected s_10 ranges from 0.00376797483625 to 0.542083569992 (143.87× across operations). Relative to each frozen baseline the range is 0.545253866× to 2×. All 18 values and ratios are in the full CSV; a single global adjustment is not supported.

E. No selected or unconstrained-best candidate is infeasible. All 594 main-pass candidates happen to satisfy the explicit 0..31 constraint; selected combined effective shifts span 12..30. The code records both optima and tests rejection of negative/extended shifts independently. The historical observed BOS range 11..28 is not treated as an artificial constraint; the old normal shift range already included 31.

F. A small joint pass is worthwhile on the pathological layers, with modest additional gains. It was run only on 7/8/17, using 27 calibration candidates per layer. There is no validation-driven parameter selection or joint expansion to all 18.

Joint layer 7: calibration balanced NMSE improves 10.295% relatively; validation balanced NMSE improves 11.225% relatively. Validation BOS/non-BOS NMSE is 3.751494% / 1.921822%. Joint 8/17 slightly increase BOS error while improving ordinary tokens; joint 7 improves both compared with s10-only.

Joint layer 8: calibration balanced NMSE improves 7.300% relatively; validation balanced NMSE improves 7.956% relatively. Validation BOS/non-BOS NMSE is 1.938519% / 2.036026%. Joint 8/17 slightly increase BOS error while improving ordinary tokens; joint 7 improves both compared with s10-only.

Joint layer 17: calibration balanced NMSE improves 3.645% relatively; validation balanced NMSE improves 3.182% relatively. Validation BOS/non-BOS NMSE is 3.801739% / 11.720912%. Joint 8/17 slightly increase BOS error while improving ordinary tokens; joint 7 improves both compared with s10-only.

## Verification and scope

34 tests passed (6 new, 28 relevant existing). All 18 frozen baselines reproduce prior PoT MSE/MAE/NMSE/clipping; all first-pass candidates keep s_X and k fixed; both cases share one s_10; selected checkpoints precede validation. The INT8-pair metrics are identical before/after s10-only search. Prior numerical source hashes and tracked LUT/default files remain unchanged. See verification.json and commands.md in s10_calibration/. Main-pass compute: 312.72 seconds; optional three-layer joint: 80.62 seconds. Cached inputs were reused, with no new large data files. No blocker; these local metrics do not establish end-to-end quality or RTL cost.

Small validation balanced-score regressions occur in layers 4/6/9/11: +0.001516/+0.005003/+0.000334/+0.003296 percentage points. Candidate selection still uses calibration only; the held-out results do not trigger per-layer replacement. This records sampling/generalization limits rather than claiming every validation operation improves.

