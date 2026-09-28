# Row PoT likelihood diagnosis

Full held-out validation: 32 sequences, 10,141 positions, 10,109 next-token targets. Only 18 down_proj operations are quantized; all other operations remain FP. Static production/calibration artifacts are unchanged.

| mode | hidden NMSE % | normalized hidden NMSE % | logits NMSE % | KL | top1 % | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| FP | 0.000000 | 0.000000 | 0.000000 | 0.0000000 | 100.0000 | 186.637058 |
| bos_only | 14.538275 | 13.221936 | 2.710539 | 1.0798923 | 64.0864 | 220.711908 |
| general_pot | 5.953833 | 2.265110 | 1.115230 | 0.5126811 | 84.9029 | 351.777114 |
| cap0 (undefined candidates) | N/A | N/A | N/A | N/A | N/A | N/A |
| cap1 (undefined candidates) | N/A | N/A | N/A | N/A | N/A | N/A |
| cap2 (undefined candidates) | N/A | N/A | N/A | N/A | N/A | N/A |
| pair_oracle | 5.939515 | 2.243420 | 1.022508 | 0.5122681 | 85.1198 | 345.531679 |
| int10_oracle | 5.942413 | 2.247453 | 1.045180 | 0.5123469 | 84.8930 | 343.510743 |
| recentered | 5.929841 | 2.238460 | 1.073899 | 0.5128313 | 84.7944 | 353.266253 |
| arbitrary_absmax | 5.582973 | 1.796602 | 0.955596 | 0.4878345 | 86.5891 | 366.233043 |

cap0: not a complete E2E result. Sequence 0: model.layers.0.mlp.down_proj: empty strict candidate set: row(s) [23, 30, 33, 37, 42, 56, 67, 88, 95, 101, 117, 119, 121, 133, 134, 157], k0 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], cap=0, shift-k bounds=[-7,16]

cap1: not a complete E2E result. Sequence 0: model.layers.2.mlp.down_proj: empty strict candidate set: row(s) [387, 396], k0 [3, 3], cap=1, shift-k bounds=[-7,14]

cap2: not a complete E2E result. Sequence 1: model.layers.3.mlp.down_proj: empty strict candidate set: row(s) [144], k0 [4], cap=2, shift-k bounds=[-7,18]

D2/D3 compare current propagated X against X @ recovered FP W.T, never cached reference-model inputs or labels. Both oracles are diagnostic and are not implementable runtime selector proposals. Same-input k confusion is measured on each oracle trajectory; actual-trajectory agreement versus D1 is reported separately.

E uses cached calibration activations only: all 128 BOS rows and a deterministic 256-row uniform non-BOS sample, with population weights to estimate pooled output NMSE. Selection is frozen before its validation forward. This is a bounded calibration estimate, not full-token exhaustive calibration. Existing s10 is fixed. Validation does not select parameters.

F is an arbitrary absmax-row reference with exact FP64 ratio and RNE, bypassing M/S approximation. Absmax is not an optimized scalar threshold and does not guarantee the minimum possible output error; compare it as a diagnostic reference, not a proven upper bound.

Token NLL uses predictor position p and target token p+1. k features come from predictor position p, not the future target row. Tail shares use signed net excess NLL; positive-burden shares are also provided and prevent interpreting >100% net contributions incorrectly. Correlation does not establish causality.

Per-mode localization, non-BOS hidden NMSE, clipping and zero rates, histograms, and k confusion are in the CSV/JSON outputs. No large reference logits or activation caches are written; FP forward is shared within a stage when new logits comparisons require it. C0/C4/D1 reuse already measured modes.

**Baseline, scope and reproducibility.** FP/BOS-only/general PPL reproduce 186.6370579825 / 220.7119084023 / 351.7771137097. All saved baseline hidden/logit sufficient statistics match the previous experiment, not just rounded PPL. There are 10,141 logit positions and 10,109 next-token targets. Top-k/KL use all positions; NLL excludes each sequence's final position. C0, C4 and D1 reuse the reproduced modes.

Executed checkout: `49d5bd28cf1d18bbcd5e2020f7beaffe1661cd72`. The local recovered Q8_0 GGUF, frozen global w_BOS=0.25 selection and existing INT8 weight/scales/qparams were reused. Exact hashes/profile/dataset metadata: [provenance](row_pot_ppl/provenance.json); exact command and bounded calibration procedure: [commands](row_pot_ppl/commands.md); chronological execution: [log](row_pot_ppl/execution.log). No original BF16 checkpoint comparison was performed.

**A — signed token-NLL changes relative to FP (nats).**

| mode | mean | median | p90 | p95 | p99 | max | positive % | negative % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bos_only | 0.167692 | -0.029562 | 1.354351 | 2.323632 | 4.919436 | 177.946306 | 43.417 | 56.583 |
| general_pot | 0.633832 | 0.001536 | 0.879163 | 1.276572 | 2.342671 | 177.946306 | 55.020 | 44.980 |
| mode | worst tail | tokens | share of signed net excess % | share of positive-only burden % |
|---|---:|---:|---:|---:|
| bos_only | 1% | 102 | 356.360 | 63.924 |
| bos_only | 5% | 506 | 434.139 | 77.876 |
| bos_only | 10% | 1011 | 486.539 | 87.275 |
| general_pot | 1% | 102 | 90.737 | 71.054 |
| general_pot | 5% | 506 | 101.244 | 79.282 |
| general_pot | 10% | 1011 | 109.600 | 85.825 |

Shares above 100% arise because other tokens have negative delta NLL. NLL = -target_logit + logsumexp is verified per target, maximum absolute decomposition residual 3.56e-15 nats.

| mode / predictor group | targets | FP NLL | quant NLL | delta NLL | delta target logit | delta LSE |
|---|---:|---:|---:|---:|---:|---:|
| bos_only / bos_predictor | 32 | 63.653216 | 239.168434 | 175.515219 | 12.020063 | 187.535282 |
| bos_only / non_bos_predictor | 10077 | 5.043637 | 4.654505 | -0.389132 | -2.002868 | -2.392000 |
| general_pot / bos_predictor | 32 | 63.653216 | 239.168434 | 175.515219 | 12.020063 | 187.535282 |
| general_pot / non_bos_predictor | 10077 | 5.043637 | 5.122125 | 0.078488 | -0.108350 | -0.029862 |

The 32 position-0 predictions are only 0.31655% of targets but cause 87.6562% of general-PoT net excess NLL over FP. Their target logits rise by 12.0201 while LSE rises by 187.5353: the partition term outgrows the target, giving +175.5152 nats. This BOS loss is **exactly identical in BOS-only and general PoT**.

Consequently, the additional general-vs-BOS-only regression is entirely non-BOS. Its mean/median delta NLL is +0.466140/+0.054441, 58.2155% of targets worsen, and the worst 1/5/10% contribute 15.0244/53.9807/87.1055% of that added net loss. The added regression is broader than the shared BOS catastrophe. Do not attribute the 220.71 -> 351.78 PPL change to new BOS damage.

On non-BOS predictors, conditional PPL is FP 155.0329, BOS-only 105.0572, general 167.6914. These conditional metrics localize the effect; the official full PPL retains every target. BOS-only happens to improve actual-target likelihood despite worse FP fidelity: its LSE drops more than its target logits. General PoT removes much of that beneficial likelihood shift. KL to the FP distribution and true-target NLL optimize different quantities, so their ordering need not match.

FP itself has high first-position NLL (63.6532 vs 5.0436 on other positions). The existing target/prompt evaluation protocol merits a separate audit before treating this as a general quality benchmark; it was unchanged here. No tokenizer or source-model error is inferred from these numbers.

**B — predictor-row k correlation.** All features use the row producing the next-token logits, never the future target row.

| group | all targets | all mean delta NLL | non-BOS targets | non-BOS mean delta NLL |
|---|---:|---:|---:|---:|
| no_positive | 48 | 0.031727 | 48 | 0.031727 |
| positive_1_3 | 1216 | -0.040044 | 1216 | -0.040044 |
| positive_4_8 | 5756 | 1.074997 | 5724 | 0.099790 |
| positive_gt8 | 3089 | 0.086401 | 3089 | 0.086401 |
| max0 | 48 | 0.031727 | 48 | 0.031727 |
| max1 | 3235 | -0.040608 | 3235 | -0.040608 |
| max2 | 5693 | 0.126647 | 5693 | 0.126647 |
| max_ge3 | 1133 | 5.133496 | 1101 | 0.181438 |
| feature | all Pearson r | non-BOS Pearson r |
|---|---:|---:|
| positive_layers | -0.051820 | 0.048636 |
| negative_layers | 0.249615 | -0.005900 |
| sum_k | -0.096654 | 0.057347 |
| max_k | 0.423339 | 0.117731 |
| mean_k | -0.096654 | 0.057347 |

The apparent max-k association is heavily BOS-confounded (r=0.423339 overall, 0.117731 non-BOS). Positive-layer count is weakly associated with non-BOS loss (r=0.048636). These are correlations, not causal estimates.

**C — strict cap ablations could not complete.** BOS uses its original calibrated k. Non-BOS candidates are the literal intersection of {k0-1,k0,k0+1}, the requested cap, [-8,7], and S-k in [0,31]. No cap fallback was added.

| mode | full E2E metrics | first undefined case | reason |
|---|---|---|---|
| k<=0 | N/A | sequence 0, layer 0 | k0=2 gives {1,2,3}; none <=0 |
| k<=1 | N/A | sequence 0, layer 2, rows 387/396 | k0=3 gives {2,3,4}; none <=1 |
| k<=2 | N/A | sequence 1, layer 3, row 144 | k0=4 gives {3,4,5}; none <=2 |

cap2 completed only sequence 0 before failing; its partial score is not reported as full-validation PPL. All subsequent experiments continued automatically. A future cap test needs an explicitly defined boundary/fallback candidate rule; the requested strict rule does not define outputs for all rows.

**D — output-oracle comparison.** Pre-requant oracle PPL 345.5317 and final-INT10 oracle 343.5107 improve on 351.7771 by only 1.7754% and 2.3499%, respectively. Neither fixes the regression. On the D3 trajectory, activation-vs-final selector agreement is 94.78355%; activation-vs-pair is 94.82190%; pair-vs-final is 99.60885%. Across independently propagated D1/D3 paths, k agreement is 93.06774%. All 576 BOS layer/sequence k values agree with D1 in both D2 and D3. Full 16x16 confusion counts are in the summary JSON. This supports only a small selector-objective effect within the tested k0±1 set.

**E — calibration-only recentering.** All 162 candidates (18 operations x 9 quarter-octave offsets) were feasible on the calibration sample. The cached sample has 384 rows per operation: all 128 BOS plus 256 uniform non-BOS rows, seed 42, with population weights. This estimates local pooled output NMSE, not an exhaustive or propagated calibration objective. Selection was frozen before validation. Mean per-op calibration NMSE decreased 1.475613% -> 1.457461% (1.23019% relative), but E2E PPL worsened 351.7771 -> 353.2663 (+0.42332%). This tested recentering does not solve the mismatch.

Chosen j by layer 0..17: `[1, -1, 0, 0, 1, -1, 0, -2, 0, -1, 0, 0, -1, 1, 1, -2, -1, 0]`; new base = old base * 2^(j/4). Eleven bases change; layer 17 stays unchanged. [All candidates](row_pot_ppl_sx_recenter.csv), [frozen decisions and cache provenance](row_pot_ppl/recenter_selection.json).

**F — arbitrary absmax-row reference.** PPL is 366.2330, worse than PoT by 4.1094%, although hidden NMSE falls to 5.582973%, normalized hidden NMSE to 1.796602%, logits NMSE to 0.955596%, and top1 agreement rises to 86.5891%. Removing the PoT grid and M/S approximation does not resolve the PPL problem in this absmax-row experiment. It is not an exhaustive arbitrary-scale output oracle, so it does not prove an absolute limit for all continuous row-scale policies.

**Propagation localization and quantizer rates.**

| mode | first hidden >1% | first hidden >5% | worst layer | activation zero % | activation clip % | INT10 clip % | effective shift |
|---|---:|---:|---:|---:|---:|---:|---|
| bos_only | 0 | 15 | 17 | 35.311402 | 0.008142 | 0.001179 | 10..30 |
| general_pot | 16 | 17 | 17 | 41.672407 | 0.004641 | 0.001490 | 7..31 |
| pair_oracle | 16 | 17 | 17 | 41.160908 | 0.005224 | 0.001453 | 7..31 |
| int10_oracle | 16 | 17 | 17 | 41.158922 | 0.005241 | 0.001452 | 7..31 |
| recentered | 16 | 17 | 17 | 41.604476 | 0.004670 | 0.001477 | 11..31 |
| arbitrary_absmax | 17 | 17 | 17 | 41.404056 | 0.000017 | 0.001470 | ideal-ratio oracle |

Rates are pooled over the 18 down_proj invocations and all validation sequences. INT10 clipping is counted before saturation; activation clipping means |X| exceeds the representable activation threshold. F's tiny nonzero threshold-exceedance count (0.0000173%) is floating division/multiplication roundoff around the absmax endpoint, not meaningful tail clipping.

General PoT BOS hidden NMSE jumps from 0.006212% at layer 16 to 13.186506% at layer 17. D2/D3 have exactly the same BOS values; recentered and arbitrary-row final BOS NMSE remain 13.169940% and 13.176784%. Final non-BOS hidden NMSE is 2.543197% (D1), 2.522128% (D2), 2.526392% (D3), 2.515704% (E), and 2.002040% (F). All 18-layer BOS/non-BOS/MAE/cosine results and per-mode k histograms are in the layer CSV and summary JSON.

**Checks.** 52 relevant tests passed, 0 failed, including 8 new focused tests. [Test log](row_pot_ppl/tests.log), [runtime verification](row_pot_ppl/verification.json). Every completed quantized mode evaluated all 32 sequences with 18 quantized down_proj and 17 verified propagation boundaries. No FP input reinjection. All hardware paths stayed within shift 0..31. All existing tracked files and recorded source/artifact hashes are unchanged; no RTL/LUT/QB/ABI/config/default changes. New files are the driver, focused tests, diagnostic tables/report and compact diagnostic checkpoints only.

**Answers A–I.**

A. Relative to FP, the loss is tail-dominated: the worst 1% contribute 90.7368% of general-PoT net excess; 32 BOS targets alone contribute 87.6562%. Relative to BOS-only, the added loss is entirely non-BOS and broader (58.2155% of targets worsen; worst 1% contribute 15.0244%).

B. Positive k has weak non-BOS association with delta NLL (count r=0.048636; max r=0.117731). The strong all-position max-k association is largely BOS-confounded; causality is unestablished.

C. No best cap can be determined: all three requested strict candidate intersections become empty. C0/C4 remain valid reproduced baselines; partial capped runs are not full results.

D. No. D2/D3 reach PPL 345.53/343.51, still far from BOS-only 220.71 and FP 186.64.

E. Activation-MSE is not the main sufficient explanation within this candidate family. Output-aware selectors agree about 95% of the time and recover little PPL. Local output MSE also fails to optimize true-target likelihood.

F. No measured E2E benefit: calibration-only recentering changes 11 bases but PPL increases to 353.27. This conclusion is limited to the specified nine candidates and the recorded bounded calibration sample.

G. PoT granularity is not supported as the primary bottleneck: arbitrary absmax-row scaling improves fidelity yet worsens PPL to 366.23. An optimized arbitrary-threshold oracle remains unmeasured.

H. The strongest numerical localization is the layer-17 down_proj perturbation and its downstream logit-partition sensitivity, shared by every row selector. The fixed common INT10 output grid/requantization there is the next concrete suspect to isolate, not a proven sole cause: this experiment never bypasses INT10. Separately, FP fidelity is not an actual-label likelihood objective; BOS-only's favorable non-BOS NLL shift explains much of its lower PPL. Next isolate layer-17 INT10 versus pre-requant output under identical inputs/ACC and audit the unusually costly first-target protocol; do not attribute this to FP nonlinear/LUT approximations.

I. Reject adoption of the current unrestricted general-row policy for this held-out PPL criterion. Neither selector substitution nor the tested recentering is a demonstrated fix; restricted-k policy is still untested because its fallback is undefined. Keep PoT as an experimental mechanism, isolate the final down_proj output stage, and make no hardware/production change on these results.

