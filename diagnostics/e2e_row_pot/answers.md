## Answers A–G

A. **Mixed result: FP fidelity improves substantially, but held-out likelihood worsens.** Block-17 hidden NMSE falls from 14.538275% to 5.953833%; logits NMSE falls from 2.710539% to 1.115230%; KL falls from 1.07989234 to 0.51268110 nats. Top-1 agreement improves from 64.0864% to 84.9029%. However, PPL increases from 220.711908 to 351.777114, versus FP 186.637058. It would be incorrect to call this an across-the-board E2E accuracy improvement.

B. **PPL degradation recovery is -384.639122%.** The numerator is -131.065205307 and the denominator is 34.074850420. This is additional degradation, not recovery: the PPL gap from FP grows from 34.074850 to 165.140056. NLL is FP 5.229165865, BOS-only 5.396858269, all-row 5.862997775 nats/target; PPL is exp(pooled NLL), not an average of per-sequence PPLs. KL uses FP predicted probabilities, whereas NLL uses actual held-out next-token targets, so their improvement directions need not agree. This distinction explains why the metrics are not logically contradictory; it does not attribute the underlying numerical cause.

C. **Not measured.** The optional arbitrary-row oracle was not run. No arbitrary/PoT accuracy-gap or claim of near-equivalence is supported by this experiment.

D. **48.296897% of non-BOS row-operation observations choose k≠0** (87,882 / 181,962). Each token contributes one observation per down_proj layer, not just one observation for the whole model.

E. **Used k range: -3..7 overall; -2..4 for non-BOS.** Overall median k=0; k=0 is 51.610076%, k≠0 is 48.389924%; 0.017531% hit the configured endpoints -8/+7 (all at +7). BOS choices equal the existing calibrated k in every layer and sequence; the observed changes are in normal-token rows.

F. **No effective-shift violations.** Selected effective shifts span 7..31. Shift feasibility rejects 0.320317% of candidate trials inside the configured k range, restricts at least one candidate for 0.949939% of rows, and changes the unconstrained activation-MSE winner for 0.006574% of rows. No row lacks a feasible candidate; no fallback or expanded search was used.

G. **Down_proj remains a significant issue for model likelihood.** Despite improved reference agreement, all-row PPL is 88.482% above FP and KL remains 0.512681 nats/token. Final conditional non-BOS hidden NMSE improves from 15.175714% to 2.543197%; BOS hidden NMSE remains 13.186506%. Post-final-RMSNorm hidden NMSE is 2.265110%. The current activation-only k rule is not sufficient evidence to adopt this as a production improvement. The source of the likelihood regression needs a separate diagnostic; this run does not isolate activation, fixed output-grid, or accumulated propagation contributions.

## Distribution and validation details

Across all 182,538 row-operation observations, activation INT8 clipping is 0.004641%, activation zero rate is 41.672407%, and INT10 output clipping is 0.001490%. Activation clipping means |X| > 127*sX(row); INT10 clipping means the rounded unsaturated integer lies outside [-512,511]. Zero rate counts q=0.

The complete 32-sequence FP/BOS-only rerun reproduces prior FP PPL 186.637057982, BOS-only PPL 220.711908402, and final hidden NMSE 14.538274593%. All per-sequence hidden/logit sufficient statistics were compared with prior results before accepting each new result (rtol 2e-12, atol 1e-12). Identical FP NLL/targets are used by both modes. All 20 final invariants and all 38 tests pass; see verification.json and tests.log. No production files or previous experiment artifacts were modified. Exact commands, source hashes and commit are recorded in commands.md and provenance.json. No blockers.

| non-BOS k | observations | fraction % |
| --- | --- | --- |
| -2 | 609 | 0.334685 |
| -1 | 16086 | 8.840307 |
| 0 | 94080 | 51.703103 |
| 1 | 58176 | 31.971511 |
| 2 | 11820 | 6.495862 |
| 3 | 1165 | 0.640244 |
| 4 | 26 | 0.014289 |

| layer | distinct k (all rows) | min k | max k | k≠0 % |
| --- | --- | --- | --- | --- |
| 0 | 6 | -2 | 3 | 67.163002 |
| 1 | 7 | -3 | 3 | 57.242875 |
| 2 | 7 | -3 | 3 | 72.142787 |
| 3 | 7 | -2 | 4 | 58.958683 |
| 4 | 6 | -2 | 3 | 48.939947 |
| 5 | 7 | -2 | 4 | 49.521743 |
| 6 | 5 | -1 | 3 | 41.277980 |
| 7 | 6 | -1 | 4 | 35.884035 |
| 8 | 6 | -1 | 5 | 36.732078 |
| 9 | 5 | -1 | 3 | 31.584656 |
| 10 | 5 | -1 | 3 | 32.294645 |
| 11 | 6 | -2 | 3 | 34.542944 |
| 12 | 7 | -2 | 4 | 47.943990 |
| 13 | 6 | -2 | 3 | 50.261315 |
| 14 | 6 | -2 | 3 | 65.989547 |
| 15 | 5 | -1 | 3 | 48.890642 |
| 16 | 5 | -1 | 3 | 43.092397 |
| 17 | 6 | -1 | 7 | 48.555369 |
