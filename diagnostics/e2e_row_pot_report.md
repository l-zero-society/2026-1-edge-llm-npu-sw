# General-row PoT: down_proj-only E2E

32 held-out sequences / 10141 valid tokens / 10109 next-token targets. FP and BOS-only are rerun on the same inputs; all stored per-sequence baseline statistics reproduce (rtol 2e-12, atol 1e-12).

Only 18 down_proj operations are quantized. All other Linear and nonlinear operations, embedding and lm_head remain FP. The existing model driver and boundary-pointer propagation checks are reused. No activation reinjection, capture, calibration, parameter search, downloads or production modifications.

Base sX, weight INT8, common s10 and per-channel M/S are frozen to global w_BOS=0.25. Each actual propagated input row selects only among raw k0±1 (k0=RNE(log2(absmax/(127*sX)))) in [-8,7], rejecting any S-k outside [0,31]. Selection uses activation reconstruction MSE only. Ties prefer k0, then k0-1, then k0+1; zero rows use k0=0. No feasible candidate causes an explicit error. One base channel multiplier is shared across rows.

| mode | final block hidden NMSE % | logits NMSE % | cosine | KL nats | top1 % | top5 overlap % | PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 0 | 0 | 1 | 0 | 100 | 100 | 186.637058 |
| bos_only | 14.538275 | 2.710539 | 0.99339565 | 1.07989234 | 64.0864 | 65.4649 | 220.711908 |
| general_pot | 5.953833 | 1.115230 | 0.99704705 | 0.51268110 | 84.9029 | 84.9246 | 351.777114 |

PPL degradation recovery: -384.639122%. Hidden is block 17 before final RMSNorm. Cosine is mean token cosine; top5 is intersection size / 5. PPL uses shifted next-token labels, excluding the last position per sequence. Reference is FP32 recovered GGUF, not original BF16.

| layer | mode | hidden NMSE % | cosine | MAE | BOS NMSE % | non-BOS NMSE % |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | bos_only | 3.136937 | 0.98498912 | 0.17993173 | 0.000183 | 4.199223 |
| 1 | bos_only | 2.972765 | 0.97822797 | 0.12573331 | 0.000184 | 5.772840 |
| 2 | bos_only | 2.791040 | 0.96883591 | 0.10666773 | 0.000195 | 8.407618 |
| 3 | bos_only | 2.867735 | 0.96053412 | 0.09825683 | 0.000201 | 11.395223 |
| 4 | bos_only | 2.585279 | 0.95756545 | 0.09041579 | 0.000209 | 11.811851 |
| 5 | bos_only | 2.509685 | 0.95224960 | 0.08726818 | 0.000232 | 13.061600 |
| 6 | bos_only | 2.414919 | 0.95030067 | 0.08649208 | 0.000249 | 12.746777 |
| 7 | bos_only | 2.351916 | 0.94924337 | 0.08785067 | 0.001585 | 12.259212 |
| 8 | bos_only | 2.310897 | 0.94829917 | 0.09003097 | 0.004568 | 12.117401 |
| 9 | bos_only | 2.310829 | 0.95241872 | 0.09318171 | 0.004514 | 10.835559 |
| 10 | bos_only | 2.451617 | 0.95280907 | 0.09979582 | 0.004481 | 10.518098 |
| 11 | bos_only | 2.546475 | 0.95332796 | 0.10574054 | 0.004484 | 10.216601 |
| 12 | bos_only | 2.782811 | 0.95065724 | 0.11453006 | 0.004468 | 10.560377 |
| 13 | bos_only | 3.315677 | 0.95364622 | 0.13576731 | 0.004461 | 9.748143 |
| 14 | bos_only | 4.930448 | 0.94173808 | 0.18330853 | 0.004445 | 11.819338 |
| 15 | bos_only | 6.638903 | 0.93332177 | 0.22909072 | 0.004592 | 13.282680 |
| 16 | bos_only | 9.040135 | 0.92800431 | 0.29862743 | 0.006212 | 14.492372 |
| 17 | bos_only | 14.538275 | 0.92668407 | 0.38323835 | 13.186506 | 15.175714 |
| 0 | general_pot | 0.737924 | 0.99557380 | 0.09885429 | 0.000183 | 0.987766 |
| 1 | general_pot | 0.544464 | 0.99525277 | 0.06381357 | 0.000184 | 1.057159 |
| 2 | general_pot | 0.498049 | 0.99364650 | 0.05092525 | 0.000195 | 1.499980 |
| 3 | general_pot | 0.436468 | 0.99317734 | 0.04308346 | 0.000201 | 1.733840 |
| 4 | general_pot | 0.351476 | 0.99361230 | 0.03738792 | 0.000209 | 1.605212 |
| 5 | general_pot | 0.312979 | 0.99343916 | 0.03443505 | 0.000232 | 1.628036 |
| 6 | general_pot | 0.280587 | 0.99382894 | 0.03262261 | 0.000249 | 1.480094 |
| 7 | general_pot | 0.265417 | 0.99402753 | 0.03273521 | 0.001585 | 1.377540 |
| 8 | general_pot | 0.259744 | 0.99401330 | 0.03347303 | 0.004568 | 1.344750 |
| 9 | general_pot | 0.252602 | 0.99469978 | 0.03364788 | 0.004514 | 1.169599 |
| 10 | general_pot | 0.264173 | 0.99483355 | 0.03549343 | 0.004481 | 1.120194 |
| 11 | general_pot | 0.274926 | 0.99486948 | 0.03710072 | 0.004484 | 1.090952 |
| 12 | general_pot | 0.306621 | 0.99451499 | 0.04045945 | 0.004468 | 1.152455 |
| 13 | general_pot | 0.382635 | 0.99456250 | 0.04896728 | 0.004461 | 1.117286 |
| 14 | general_pot | 0.592720 | 0.99306475 | 0.06609213 | 0.004445 | 1.415407 |
| 15 | general_pot | 0.837627 | 0.99174277 | 0.08436249 | 0.004592 | 1.671850 |
| 16 | general_pot | 1.141780 | 0.99099488 | 0.11068169 | 0.006212 | 1.827129 |
| 17 | general_pot | 5.953833 | 0.98699573 | 0.17098625 | 13.186506 | 2.543197 |

## Row distribution (row-operation observations)

{
  "bos": {
    "rows": 576,
    "min_k": -3,
    "max_k": 7,
    "median_k": -1.0,
    "distinct_k": 8,
    "zero_fraction": 0.2222222222222222,
    "nonzero_fraction": 0.7777777777777778,
    "at_bounds_fraction": 0.05555555555555555,
    "histogram": {
      "-8": 0,
      "-7": 0,
      "-6": 0,
      "-5": 0,
      "-4": 0,
      "-3": 64,
      "-2": 192,
      "-1": 64,
      "0": 128,
      "1": 32,
      "2": 0,
      "3": 0,
      "4": 32,
      "5": 32,
      "6": 0,
      "7": 32
    },
    "activation_clip_rate": 6.442599826388889e-05,
    "activation_zero_rate": 0.4967074924045139,
    "int10_clip_rate": 0.00021701388888888888,
    "shift_rejected_candidate_fraction": 0.0,
    "shift_restricted_row_fraction": 0.0,
    "shift_changed_choice_fraction": 0.0,
    "k_bound_rejected_candidate_fraction": 0.018518518518518517
  },
  "non_bos": {
    "rows": 181962,
    "min_k": -2,
    "max_k": 4,
    "median_k": 0.0,
    "distinct_k": 7,
    "zero_fraction": 0.517031028456491,
    "nonzero_fraction": 0.48296897154350904,
    "at_bounds_fraction": 0.0,
    "histogram": {
      "-8": 0,
      "-7": 0,
      "-6": 0,
      "-5": 0,
      "-4": 0,
      "-3": 0,
      "-2": 609,
      "-1": 16086,
      "0": 94080,
      "1": 58176,
      "2": 11820,
      "3": 1165,
      "4": 26,
      "5": 0,
      "6": 0,
      "7": 0
    },
    "activation_clip_rate": 4.6353806551127435e-05,
    "activation_zero_rate": 0.416470887049584,
    "int10_clip_rate": 1.4265083506446402e-05,
    "shift_rejected_candidate_fraction": 0.0032131250847246494,
    "shift_restricted_row_fraction": 0.009529462195403435,
    "shift_changed_choice_fraction": 6.594783526230752e-05,
    "k_bound_rejected_candidate_fraction": 0.0
  },
  "all": {
    "rows": 182538,
    "min_k": -3,
    "max_k": 7,
    "median_k": 0.0,
    "distinct_k": 10,
    "zero_fraction": 0.5161007571026307,
    "nonzero_fraction": 0.4838992428973693,
    "at_bounds_fraction": 0.00017530596368975227,
    "histogram": {
      "-8": 0,
      "-7": 0,
      "-6": 0,
      "-5": 0,
      "-4": 0,
      "-3": 64,
      "-2": 801,
      "-1": 16150,
      "0": 94208,
      "1": 58208,
      "2": 11820,
      "3": 1165,
      "4": 58,
      "5": 32,
      "6": 0,
      "7": 32
    },
    "activation_clip_rate": 4.641083348484288e-05,
    "activation_zero_rate": 0.41672407424723296,
    "int10_clip_rate": 1.4904858851307672e-05,
    "shift_rejected_candidate_fraction": 0.0032031732233711117,
    "shift_restricted_row_fraction": 0.009499391907438452,
    "shift_changed_choice_fraction": 6.57397363836571e-05,
    "k_bound_rejected_candidate_fraction": 5.843532122991742e-05
  }
}

Per-layer distinct-k counts and row/activation rates are in e2e_row_pot/row_statistics.csv. Histogram CSV has per-layer and pooled BOS/non-BOS/all groups. Rates are fractions. Shift-rejected candidate rate divides by candidate trials inside [-8,7]; restricted-row rate counts rows with any candidate rejected by shift feasibility; changed-choice rate compares to the same three-candidate activation-MSE winner without shift filtering. At-bounds means k=-8 or +7. No runtime scale is selected using Linear/reference outputs.

Observed effective shifts: 7..31; violations: 0. Arbitrary-row oracle not run: it needs distinct scalar requantization, outside this focused shift-only implementation. No claim of closeness to arbitrary scaling is made.

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

