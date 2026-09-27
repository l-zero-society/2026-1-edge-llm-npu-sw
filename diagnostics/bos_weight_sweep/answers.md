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
