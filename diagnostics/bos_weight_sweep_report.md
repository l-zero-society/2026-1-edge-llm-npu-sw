# BOS objective-weight sweep (candidate-table rescoring)

Selection and the 2% plateau use calibration only. Existing 594 s10 candidates, s_X_normal, k and feasibility are unchanged. No new candidate scales or calibration GEMMs. The common weight may select a different precomputed s10 per operation. Global/per-op policy comparison uses the weight-independent mean per-op worst-group NMSE, not each weight’s own objective. This is local Linear robustness, not end-to-end optimality.

Best global w_BOS = 0.25. Weights within 2% of the best calibration robust score: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]. Candidate tie handling reproduces the previous 50:50 policy exactly; robust-score ties follow the requested ordering. Dominated feasible candidates are removed before rescoring. All selected policies are Pareto-frontier members.

## Global sweep — calibration

| w BOS | BOS median | BOS max | non-BOS median | non-BOS max | Mean worst | Changed ops vs 0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.499766% | 69.846031% | 4.444466% | 13.921722% | 11.837723% | 16 |
| 0.05 | 0.348095% | 30.922628% | 4.444468% | 13.921830% | 6.158173% | 11 |
| 0.1 | 0.337764% | 21.755349% | 4.445491% | 13.921830% | 5.605269% | 10 |
| 0.15 | 0.330301% | 13.138063% | 4.445491% | 13.921830% | 5.126641% | 10 |
| 0.2 | 0.330301% | 10.573058% | 4.445491% | 13.921830% | 4.984215% | 10 |
| 0.25 | 0.330301% | 8.262692% | 4.445491% | 13.921830% | 4.938024% | 9 |
| 0.3 | 0.330301% | 6.358178% | 4.445491% | 13.921830% | 4.982867% | 9 |
| 0.35 | 0.325882% | 6.358178% | 4.447693% | 13.921830% | 4.984049% | 5 |
| 0.4 | 0.325882% | 4.870696% | 4.447693% | 13.921830% | 5.034874% | 4 |
| 0.45 | 0.322446% | 4.870696% | 4.450128% | 13.921830% | 5.035145% | 3 |
| 0.5 | 0.322446% | 3.948794% | 4.450128% | 13.921830% | 5.094056% | 0 |
| 0.55 | 0.314146% | 3.948794% | 4.450128% | 13.921830% | 5.095167% | 1 |
| 0.6 | 0.309546% | 3.948794% | 4.462614% | 13.921830% | 5.170033% | 9 |
| 0.65 | 0.309546% | 3.948794% | 4.462614% | 13.921830% | 5.172285% | 10 |
| 0.7 | 0.309546% | 3.948794% | 4.462614% | 15.115075% | 5.254077% | 11 |
| 0.75 | 0.306142% | 3.948794% | 4.472774% | 16.743959% | 5.351015% | 11 |
| 0.8 | 0.306142% | 3.948794% | 4.472774% | 16.743959% | 5.351319% | 11 |
| 0.85 | 0.302918% | 3.948794% | 4.488808% | 18.591658% | 5.458625% | 12 |
| 0.9 | 0.301166% | 3.948794% | 4.541018% | 18.591658% | 5.465907% | 12 |
| 0.95 | 0.295476% | 3.948794% | 4.588716% | 20.660091% | 5.606145% | 13 |
| 1.0 | 0.292327% | 3.939747% | 4.844803% | 20.660091% | 5.880074% | 15 |

## Per-operation oracle — calibration

| Layer | Best w BOS | s10 | BOS NMSE | non-BOS NMSE | Distinct candidates | Pareto count | Worst reduction vs global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.0332226669 | 0.372900% | 3.339436% | 7 | 10 | 0.083646% |
| 1 | 0.0 | 0.0110288831 | 0.534229% | 1.650031% | 9 | 12 | 0.558219% |
| 2 | 0.0 | 0.0086986998 | 0.359253% | 4.446411% | 9 | 13 | 0.046068% |
| 3 | 0.0 | 0.00450724197 | 0.247964% | 13.921722% | 5 | 9 | 0.000779% |
| 4 | 0.25 | 0.00470789853 | 0.299314% | 4.785533% | 3 | 3 | 0.000000% |
| 5 | 0.25 | 0.0070338056 | 0.891410% | 5.498202% | 4 | 7 | 0.000000% |
| 6 | 0.15 | 0.00448090248 | 0.622155% | 5.173951% | 4 | 5 | 0.025747% |
| 7 | 1.0 | 0.0270250859 | 3.939747% | 3.354439% | 4 | 17 | 0.229112% |
| 8 | 0.2 | 0.0253687753 | 2.113658% | 2.442232% | 5 | 27 | 3.466104% |
| 9 | 0.0 | 0.00420094507 | 0.615821% | 1.325853% | 4 | 4 | 0.011639% |
| 10 | 0.25 | 0.00513066537 | 0.412435% | 1.505578% | 2 | 4 | 0.000000% |
| 11 | 0.25 | 0.00500935756 | 0.171660% | 1.837911% | 4 | 4 | 0.000000% |
| 12 | 0.25 | 0.00557790309 | 0.137217% | 4.442521% | 5 | 12 | 0.000000% |
| 13 | 0.2 | 0.00924775857 | 0.149443% | 5.592784% | 5 | 5 | 0.029746% |
| 14 | 0.25 | 0.00766705943 | 0.160318% | 9.685647% | 6 | 9 | 0.000000% |
| 15 | 0.0 | 0.012596727 | 0.484131% | 5.517358% | 2 | 3 | 0.063939% |
| 16 | 0.0 | 0.0299505427 | 0.515402% | 4.010385% | 2 | 3 | 0.117622% |
| 17 | 0.25 | 0.418003575 | 8.262692% | 9.646772% | 13 | 17 | 0.000000% |

## Frozen-policy comparison

| Split | Policy | BOS median | BOS max | non-BOS median | non-BOS max | Mean worst | Max worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 50:50 | 0.322446% | 3.948794% | 4.450128% | 13.921830% | 5.094056% | 13.921830% |
| calibration | best_global | 0.330301% | 8.262692% | 4.445491% | 13.921830% | 4.938024% | 13.921830% |
| calibration | per_op | 0.448283% | 8.262692% | 4.444466% | 13.921722% | 4.931226% | 13.921722% |
| validation | 50:50 | 0.322446% | 3.948794% | 4.442515% | 12.936506% | 5.055746% | 12.936506% |
| validation | best_global | 0.330301% | 8.262692% | 4.439598% | 12.936506% | 4.885908% | 12.936506% |
| validation | per_op | 0.448283% | 8.262692% | 4.434124% | 12.934821% | 4.878465% | 12.934821% |

calibration: per-op versus best global mean-worst reduction = 0.006797578 percentage points (0.137658% relative). Negative means per-op is worse.

validation: per-op versus best global mean-worst reduction = 0.007442509 percentage points (0.152326% relative). Negative means per-op is worse.

Material (>5% calibration worst-group reduction) override layers: []. Strongly weight-sensitive over the full 0..1 sweep: [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17]. Sensitivity counts distinct candidates, not improvement size. Full contiguous transition intervals are in bos_weight_sweep_transitions.csv; a regime describes the 0.05 grid, not an exact continuous breakpoint.

Validation missing choices: 0. Stored validation initially covers only frozen and 50:50 selections; missing frozen choices require minimal validation-only evaluation. Existing validation is reused, one shared reference/ACC is computed per needed layer, and the 594 calibration candidates are never rerun. No inference, recapture or new model download.

## Answers A–F

A. Best single global BOS weight: **0.25** (non-BOS 0.75). Calibration mean per-operation worst-group NMSE is 4.938024%, versus 5.094056% for 50:50. This selects a different s10 for 9/18 operations.

B. There is a broad sampled plateau: **0.20–0.45**, inclusive on the tested 0.05 grid, lies within 2% of the minimum calibration robust score. This does not establish exact continuous interval boundaries.

C. Per-operation weights improve calibration mean worst-group NMSE by only **0.006798 percentage points (0.137658% relative)**. Independent validation improvement is **0.007443 percentage points (0.152326% relative)**, from 4.885908% to 4.878465%. Per-op weighting does not improve every group: validation BOS median increases from 0.330301% to 0.448283%.

D. **0/18 operations** meet the requested material benefit threshold (>5% relative reduction in calibration worst-group NMSE). The largest is layer 8 at 3.466104%; no layer overrides are justified by this criterion.

E. Strongly sensitive (at least four distinct selected candidates across the entire sweep): **0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17**. Layers 4, 10, 15, 16 are mildly sensitive; none are stable. Layer 17 has 13 distinct candidates. Sensitivity over 0–1 does not imply material benefit from per-op weights near the global optimum.

F. Recommend **one global BOS weight of 0.25**, with no overrides, for the defined local-NMSE robustness criterion. Production defaults were not changed. This is a trade-off, not a universal BOS improvement: layer 17 validation BOS NMSE rises from 3.616711% to 8.262692%, while its non-BOS NMSE falls from 12.416163% to 9.452680%. Across all layers, maximum BOS NMSE rises from 3.948794% to 8.262692%. A separate BOS-error ceiling would be a different optimization constraint and is not established by this experiment.

## Data reuse, reproducibility and checks

The 594 existing calibration candidates (18 × 33) were rescored at 21 weights. No new scales, calibration GEMMs, inference, capture, downloads, production code or configuration changes were made. Every selection, including every per-op weight and the plateau, was frozen before new validation evaluation. CSV NMSE fields are fractions; report tables express percentages.

Stored validation covered the old frozen and 50:50 policies, but not all newly selected candidates. Existing validation was reused where available; only **16 missing layer/scale combinations across 13 layers** were evaluated using cached validation activations. Each layer shared its reference/ACC across these choices. All 13 replayed 50:50 validation checks matched stored MSE, MAE, NMSE and clipping metrics (rtol 2e-12, atol 1e-15). This did not rerun the 594 candidate search. BOS calibration/validation equality can arise from the shared position-0 input; selection nevertheless accessed calibration metrics only.

Source candidate CSV and all 18 source layer JSON hashes, plus the analysis-script hash, are recorded in `bos_weight_sweep/frozen_selection.json`; each new validation record references that frozen-selection hash. Model/profile/dataset/cache identities and recovered INT8 weights were verified by the validation path. All selected policies lie on the feasible Pareto frontiers (164 frontier candidates, 430 dominated); rescoring all feasible candidates gives the same selections. The prior 50:50 policy reproduces exactly for all 18 operations. Verification results are in `bos_weight_sweep/verification.json`; exact main commands and source commit are in `bos_weight_sweep/commands.md`.

