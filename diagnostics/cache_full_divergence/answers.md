## Diagnosis

INT8 activation threshold crossing is the mechanism most strongly supported by these controls. Small FP execution-shape differences are already present before quantization. They change a few INT8 codes, which then affect many accumulator/output channels and propagate into later hidden states and K/V. Dynamic k differences are secondary in this sample. Fixing k alone does not restore parity; supplying the FULL Xq and matching k restores essentially FP-level cache/full parity. Supplying FULL q10 gives exactly the same final metrics, so there is no evidence of an independent ACC/requant discrepancy once its integer inputs match.

This is localization on four prompts, not a new policy search or an architecture recommendation. NMSE values below are ratios unless marked %. KL is KL(FULL softmax || CACHED softmax), within the same quantization mode. Signed NLL difference is cached minus full; absolute differences are also reported.

## Scope and matched positions

Original valid prompt indices 0, 1, 2, 3 have prompt lengths 31, 599, 65, 24. There are 16 response targets per prompt: target t=0 is predicted by prompt prefill, followed by 15 actual one-token cached updates. Thus each mode has 64 comparisons, including 60 post-prefill decode positions. All 18 down_proj operations are traced: 1,152 rows per mode, 6,912 total rows; General baseline adds 1,152 new-position K/V comparisons.

Each FULL call contains exactly prompt + the first t ground-truth response tokens, not the entire response. Its final row predicts target t at absolute position prompt_length+t-1. The cached path asserts the same complete token history and position. All t=0 traces are exactly equal. No future response token or target label influences k.

Controls each retain their own evolving cached KV and residual state. They intervene only on the current matched down_proj row. Full prefix activations, ACC, hidden states and KV are never injected. Full calls temporarily produce a cache to observe the new K/V, then discard it. Existing native chat IDs/template, local GGUF and calibration-only global w_BOS=0.25 parameters are reused. All other Linear/nonlinear operations remain FP.

## First divergence

The earliest small FP difference is already in layer-0 down_proj input, before quantization. For prompt 0, t=1, its NMSE is 2.72868e-12 and maximum absolute difference is 9.53674e-5. Across all layer-0 rows, median X NMSE is 2.41229e-12 and maximum is 1.35831e-11. Across the entire FP trace, maximum X NMSE is 6.91307e-11 and maximum block-hidden NMSE is 2.14985e-11. FP logits remain near-exact and top1 agrees on all 64 targets.

The first discrete divergence in the first decoded example is:

| mode | prompt / target | layer | X NMSE before quant | X max abs difference | k FULL / cache | changed Xq | changed q10 | hidden NMSE |
|---|---|---:|---:|---:|---|---:|---:|---:|
| BOS-only | 0 / 1 | 4 | 9.95311e-11 | 2.19345e-5 | 0 / 0 | 1 / 16,384 | 79 / 2,048 | 7.00033e-6 |
| General-row | 0 / 1 | 2 | 3.14567e-11 | 3.57628e-5 | -1 / -1 | 1 / 16,384 | 48 / 2,048 | 5.22578e-6 |

In the General example, the single changed Xq code differs by one. It changes 2,012 of 2,048 accumulator channels (maximum ACC delta 106), then 48 pre-clamp integers and 48 q10 outputs, each by one code. There is no k mismatch at this trigger.

All 60 post-prefill positions show some Xq/q10 mismatch in both quantized baselines. The median first Xq/q10-mismatch layer is 2 for BOS-only and 3 for General-row; ranges are 0–11 and 1–16. First hidden NMSE >1e-6 coincides with first Xq/q10 mismatch in every baseline case. Hidden NMSE >1e-2 occurs at 21/64 BOS-only targets and 34/64 General targets; FP and both Xq/q10 controls never exceed 1e-6. Complete per-target first layers for all requested thresholds are in cache_full_divergence_first.csv.

Before the first current-row code mismatch, median X NMSE is 1.70731e-8 for BOS-only and 8.99372e-9 for General; maxima are 2.17613e-6 and 2.65020e-6. Later targets can already carry previous-token cache differences. These larger values should not be confused with the initial layer-0 FP perturbation.

## Mismatch rates

Rates use all 1,152 layer/token rows per mode, including 72 exactly equal prefill rows. Element rates average over equal-sized activation/output rows; they are not fractions of tokens.

| mode | k mismatch rows % | any Xq mismatch rows % | any q10 mismatch rows % | Xq elements mismatched % | q10 elements mismatched % |
|---|---:|---:|---:|---:|---:|
| BOS-only | 0 | 81.1632 | 81.1632 | 8.5451 | 51.7308 |
| General-row | 1.3889 | 75.9549 | 75.9549 | 8.6082 | 50.9100 |
| force-full-k | 0 | 75.9549 | 75.9549 | 7.6475 | 50.4398 |
| force-full-Xq+k | 0 | 0 | 0 | 0 | 0 |
| force-full-q10 | 0 | 14.2361 | 0 | 0.001012 | 0 |

General has 16 k-mismatched rows, spanning 14/64 target positions. None of the 60 first Xq-mismatch rows has a k mismatch: k instability appears after discrete activation-code divergence. For those 16 rows, the median activation-MSE difference (error at full-k minus error at cache-k) is -5.82782e-5 on FULL X and +2.33284e-5 on cached X. Each selected k performs better on its own activation; this does not establish random tie-breaking as the trigger. Exact per-row scores are in the trace CSV.

## Rounding-boundary evidence

Distance uses abs(abs(X/scale-floor(X/scale))-0.5). The boundary CSV stores exact median/p90/p99 for matched and mismatched elements on both FULL and cached sides of every mismatching row. The following compact summary takes the median across the 60 first-mismatch rows of each row's FULL-side quantile; these are NOT pooled element percentiles.

| mode | elements | median of row medians | median of row p90 | median of row p99 |
|---|---|---:|---:|---:|
| BOS-only | mismatched | 0.00031805 | 0.00095419 | 0.00121665 |
| BOS-only | matched | 0.27813437 | 0.46767517 | 0.49830523 |
| General-row | mismatched | 0.00013371 | 0.00045783 | 0.00048471 |
| General-row | matched | 0.27329978 | 0.46571670 | 0.49776152 |

At prompt 0 / t=1 / layer 2 in General-row, the one mismatched element is only 1.72942e-6 from a FULL rounding boundary and 9.88936e-6 from a cached boundary. Matched elements have median distance 0.26852. Combined with equal k, a one-code Xq change and the restoring Xq control, this directly supports threshold sensitivity.

## Control interpretation

- Force-full-k reduces mean KL from 0.0374340 to 0.0292734, but top1 falls from 93.75% to 90.625%; logits NMSE rises from 0.00320697 to 0.00337874. It does not restore parity. Dynamic k cannot explain the similar BOS-only mismatch rates either.
- Force-full-Xq+k restores top1 to 100%, KL to 1.79366e-11 and logits NMSE to 2.57819e-12. Mean/max absolute NLL deltas fall to 6.10773e-6 / 3.50861e-5, comparable to FP baseline. All 1,152 ACC, pre-clamp and q10 rows exactly match their FULL references. This is the decisive control for INT8 threshold sensitivity.
- Force-full-q10 produces exactly the same final per-target statistics as force-full-Xq, with no further gain. Its remaining 0.001012% Xq-element mismatches no longer propagate because the output is supplied. Both controls have maximum hidden NMSE 6.66235e-11, consistent with residual FP execution differences.
- Therefore downstream execution-shape differences provide the initial tiny perturbations, but they do not sustain the large divergence when down_proj quantized codes/outputs are held equal. An independent integer GEMM or requant mismatch is not supported by these measurements.

## K/V localization

Only the newly generated token's post-RoPE K and V are compared; no entire cache tensor is saved. Define material here as NMSE >1e-4 (0.01%), with >1e-6 also shown for sensitivity. Indices are zero-based.

| prompt | first K >1e-6 (t,L) | first V >1e-6 (t,L) | first K >1e-4 (t,L) | first V >1e-4 (t,L) |
|---|---|---|---|---|
| 0 | 1,3 | 1,3 | 1,5 | 1,4 |
| 1 | 1,6 | 1,6 | 1,8 | 1,8 |
| 2 | 1,17 | 1,17 | 2,15 | 2,13 |
| 3 | 1,8 | 1,8 | 1,9 | 1,9 |

For prompt 0 / t=1, General down_proj diverges at layer 2; K/V exceed 1e-6 in layer 3. At layer 4 V NMSE is 2.86227e-4; K first exceeds 1e-4 at layer 5 (1.10568e-4). Maximum observed K/V NMSE is 0.0132391 / 0.0357801. This follows propagation into subsequent attention layers; it is not evidence of a K/V arithmetic bug. Full per-row statistics are in cache_full_divergence_kv.csv.

## Verification and reproducibility

55 focused/relevant tests passed, zero failures: new trace/control tests 9, response-evaluation math 8, row-PoT 6, E2E Linear 4, factorized math 6 and static_quant 22. Tests cover history/absolute position equality, original-path parity, absolute-position BOS semantics, control injection scope, shared unchanged channel parameters, RNE integer path, effective shifts and real FP cache/full parity.

All observed effective shifts are 8..31, with zero violations. Post-run assertions verify all target/row counts, exact t=0 parity, full-Xq ACC/raw/q10 identity, equality of final Xq/q10-control statistics, 62 source/artifact hashes and no tracked-file modifications. Existing production/calibration/RTL/LUT/QB/ABI files and previous diagnostics are unchanged. The profile is the existing generic-rne software model; this is not an RTL bit-exactness certification.

A control-wrapper indexing error was corrected after saving all three baseline traces; eight pure tests passed before resuming. Since vectors were intentionally RAM-only, the terminated process required reconstruction of General FULL references. Completed baseline statistics were reused. Reconstructed full-logit energy fingerprints and every full-k choice exactly match the original baseline records. No repeated baseline measurements are pooled. Both run logs/provenance files are preserved.

Exact commands, local model and artifact hashes, source commit, recovery details and test output are under cache_full_divergence/{commands.md,provenance.json,baseline_provenance.json,tests.log,verification.json}. All retained numerical data are compact statistics; no activation/logit/ACC/KV tensor dumps were written.

## Answers A–I

A. Tiny differences begin in the FP down_proj input at layer 0. The first meaningful discrete divergence in prompt 0 / decoded t=1 is Xq at layer 4 for BOS-only and layer 2 for General, with unchanged k; ACC, requant and hidden differences follow.

B. Layer-0 input median/max NMSE is 2.41229e-12 / 1.35831e-11; maximum absolute difference is 2.13623e-4. The first General trigger has X NMSE 3.14567e-11 and max absolute difference 3.57628e-5. Later propagated X errors are much larger and are not the initial perturbation.

C. General k differs on 16/1,152 rows (1.3889%), across 14 targets. No first-Xq-divergence row has a k mismatch.

D. Any Xq mismatch affects 81.1632% of BOS-only rows and 75.9549% of General rows. Element mismatch rates are 8.5451% and 8.6082%; comparable rates without dynamic k rule out blaming k alone.

E. No. Force-full-k leaves substantial error, with top1 90.625% and KL 0.0292734.

F. Yes, to near-FP numerical parity: top1 100%, KL 1.79366e-11, logits NMSE 2.57819e-12; ACC/raw/q10 are exact once Xq+k match.

G. Yes. Force-full-q10 gives exactly the same final statistics as force-full-Xq, without additional benefit.

H. In the first decoded example, K/V differences exceed 1e-6 at layer 3, following the layer-2 down_proj divergence. With material defined as >1e-4, V reaches it at layer 4 and K at layer 5. Other prompts' first locations are tabulated above.

I. INT8 threshold crossing is the strongest supported main mechanism, triggered by tiny FP execution-shape differences and amplified through the network/cache. Dynamic k is secondary; an independent ACC/requant fault or large downstream-only discrepancy is not supported. No policy or architecture change was selected.
