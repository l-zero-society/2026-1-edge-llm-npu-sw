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
