# Factorized row requantization — frozen common s_10 experiment

Cached reference inputs reused; no model forward, recapture, download, RTL, LUT, ABI, QB or production changes. Baseline is the previously frozen stratified_64 main policy, not a validation-selected replacement. Parameters fit on calibration only; all metrics in the table use full validation. This is local Linear accuracy, not end-to-end model accuracy.

Hardware interpretation: r[m,n] = C_row[m] * r_channel[n], where r_channel[n] is the ONE stored UInt16 multiplier / 2^UInt5 shift array. B/C share that array, s_W, weights, normal scale, common s_10 and validation rows. Normal C=1. Input quantization uses s_X_base*C as well. The row factor is applied before the single RNE/saturation, never after INT10. PoT C=2^k implies effective shift s[n]-k, without clamping to UInt5 or changing packing. Arbitrary C uses exact integer arithmetic for the stored FP64 scalar binary rational. A scalar shift adjustment might eventually use row_change / row index metadata; negative or extended shifts need implementation review. RTL cost is not measured.

Non-BOS fitting: 256 representative calibration rows (early/general weighted only within non-BOS), sampled-row percentiles p99..p99.99, full non-BOS absmax, and the frozen baseline candidate. BOS fitting: all 128 calibration BOS rows, BOS-only percentile endpoints plus 16 log-spaced thresholds from p99 to absmax and C=1. PoT searches k=-8..8; these points are also included in the unrestricted benchmark without duplicate computation. BOS error is never weighted by its corpus frequency. The arbitrary result is a bounded-search software benchmark, not proof of a continuous global optimum.

| Layer | Scheme | sX normal | sX BOS | C BOS | k | BOS NMSE % | Non-BOS NMSE % | Total NMSE % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | 0.32823654 | 0.32823654 | 1 | None | 5.704308 | 3.545723 | 3.546060 |
| 0 | arbitrary | 0.32823654 | 0.083128396 | 0.25325759 | None | 0.389582 | 3.545723 | 3.545231 |
| 0 | pot | 0.32823654 | 0.082059135 | 0.25 | -2 | 0.403267 | 3.545723 | 3.545233 |
| 1 | baseline | 0.10382454 | 0.10382454 | 1 | None | 8.978443 | 1.813293 | 1.813554 |
| 1 | arbitrary | 0.10382454 | 0.016058586 | 0.15467042 | None | 0.613328 | 1.813293 | 1.813250 |
| 1 | pot | 0.10382454 | 0.012978068 | 0.125 | -3 | 0.655488 | 1.813293 | 1.813251 |
| 2 | baseline | 0.088813612 | 0.088813612 | 1 | None | 5.931592 | 4.537046 | 4.537228 |
| 2 | arbitrary | 0.088813612 | 0.01548429 | 0.17434591 | None | 0.258683 | 4.537046 | 4.536491 |
| 2 | pot | 0.088813612 | 0.011101702 | 0.125 | -3 | 0.324924 | 4.537046 | 4.536500 |
| 3 | baseline | 0.050720926 | 0.050720926 | 1 | None | 1.919551 | 12.934968 | 12.931495 |
| 3 | arbitrary | 0.050720926 | 0.013845716 | 0.27297837 | None | 0.231905 | 12.934968 | 12.930963 |
| 3 | pot | 0.050720926 | 0.012680232 | 0.25 | -2 | 0.266701 | 12.934968 | 12.930974 |
| 4 | baseline | 0.038223396 | 0.038223396 | 1 | None | 1.040390 | 6.123946 | 6.120142 |
| 4 | arbitrary | 0.038223396 | 0.016393182 | 0.42887822 | None | 0.291938 | 6.123946 | 6.119582 |
| 4 | pot | 0.038223396 | 0.019111698 | 0.5 | -1 | 0.292092 | 6.123946 | 6.119582 |
| 5 | baseline | 0.03359187 | 0.03359187 | 1 | None | 0.934307 | 5.515133 | 5.510767 |
| 5 | arbitrary | 0.033621895 | 0.024046237 | 0.71519578 | None | 0.510115 | 5.509845 | 5.505079 |
| 5 | pot | 0.033621895 | 0.033621895 | 1 | 0 | 0.939184 | 5.509845 | 5.505488 |
| 6 | baseline | 0.032511839 | 0.032511839 | 1 | None | 0.621148 | 3.811306 | 3.806352 |
| 6 | arbitrary | 0.032511839 | 0.026397275 | 0.81192808 | None | 0.446422 | 3.811306 | 3.806081 |
| 6 | pot | 0.032511839 | 0.032511839 | 1 | 0 | 0.621148 | 3.811306 | 3.806352 |
| 7 | baseline | 0.028166948 | 0.028166948 | 1 | None | 62.146786 | 2.292682 | 3.303010 |
| 7 | arbitrary | 0.028166948 | 0.31266319 | 11.100358 | None | 16.343059 | 2.292682 | 2.529850 |
| 7 | pot | 0.028166948 | 0.45067116 | 16 | 4 | 17.042881 | 2.292682 | 2.541663 |
| 8 | baseline | 0.02452638 | 0.02452638 | 1 | None | 80.088749 | 2.089106 | 6.046678 |
| 8 | arbitrary | 0.02452638 | 0.57087275 | 23.275867 | None | 17.165824 | 2.089106 | 2.854074 |
| 8 | pot | 0.02452638 | 0.78484415 | 32 | 5 | 17.464183 | 2.089106 | 2.869212 |
| 9 | baseline | 0.026293063 | 0.026293063 | 1 | None | 0.646789 | 1.697806 | 1.696419 |
| 9 | arbitrary | 0.026544869 | 0.026544869 | 1 | None | 0.615821 | 1.671769 | 1.670375 |
| 9 | pot | 0.026544869 | 0.026544869 | 1 | 0 | 0.615821 | 1.671769 | 1.670375 |
| 10 | baseline | 0.028068215 | 0.028068215 | 1 | None | 0.514550 | 1.801132 | 1.798591 |
| 10 | arbitrary | 0.029077519 | 0.021071419 | 0.72466357 | None | 0.376958 | 1.702538 | 1.699919 |
| 10 | pot | 0.029077519 | 0.029077519 | 1 | 0 | 0.538641 | 1.702538 | 1.700239 |
| 11 | baseline | 0.040096377 | 0.040096377 | 1 | None | 0.886370 | 1.973723 | 1.972652 |
| 11 | arbitrary | 0.041208523 | 0.012465878 | 0.30250727 | None | 0.156741 | 1.880040 | 1.878341 |
| 11 | pot | 0.041208523 | 0.010302131 | 0.25 | -2 | 0.204563 | 1.880040 | 1.878388 |
| 12 | baseline | 0.047746778 | 0.047746778 | 1 | None | 1.439936 | 5.003376 | 5.000831 |
| 12 | arbitrary | 0.047746778 | 0.011936695 | 0.25 | None | 0.137217 | 5.003376 | 4.999901 |
| 12 | pot | 0.047746778 | 0.011936695 | 0.25 | -2 | 0.137217 | 5.003376 | 4.999901 |
| 13 | baseline | 0.070704873 | 0.070704873 | 1 | None | 1.282941 | 5.043508 | 5.040261 |
| 13 | arbitrary | 0.075991708 | 0.015909552 | 0.20935906 | None | 0.132142 | 4.535483 | 4.531681 |
| 13 | pot | 0.075991708 | 0.018997927 | 0.25 | -2 | 0.149443 | 4.535483 | 4.531696 |
| 14 | baseline | 0.80916085 | 0.80916085 | 1 | None | 25.251046 | 10.799605 | 10.811046 |
| 14 | arbitrary | 0.11293577 | 0.028199166 | 0.24969207 | None | 0.157745 | 9.511244 | 9.503839 |
| 14 | pot | 0.11293577 | 0.028233942 | 0.25 | -2 | 0.158499 | 9.511244 | 9.503840 |
| 15 | baseline | 0.123967 | 0.123967 | 1 | None | 1.269553 | 5.107246 | 5.094559 |
| 15 | arbitrary | 0.12424406 | 0.062122031 | 0.5 | None | 0.924522 | 5.089879 | 5.076109 |
| 15 | pot | 0.12424406 | 0.062122031 | 0.5 | -1 | 0.924522 | 5.089879 | 5.076109 |
| 16 | baseline | 0.13463174 | 0.13463174 | 1 | None | 0.678278 | 4.516087 | 4.465557 |
| 16 | arbitrary | 0.13919387 | 0.22714727 | 1.6318769 | None | 0.293435 | 4.337303 | 4.284060 |
| 16 | pot | 0.13919387 | 0.27838775 | 2 | 1 | 0.311607 | 4.337303 | 4.284299 |
| 17 | baseline | 18.48648 | 18.48648 | 1 | None | 0.549480 | 77.437998 | 15.023213 |
| 17 | arbitrary | 0.14417333 | 18.48648 | 128.224 | None | 0.549480 | 21.035584 | 4.405848 |
| 17 | pot | 0.14417333 | 18.454186 | 128 | 7 | 0.560834 | 21.035584 | 4.415064 |

CSV includes full MSE/MAE/NMSE, activation zero/clipping and INT10 clipping rates for both groups; JSON checkpoints contain calibration candidate tables and frozen choices.

| Layer | PoT minus arbitrary BOS NMSE (percentage points) | PoT / arbitrary BOS MSE |
| --- | --- | --- |
| 0 | 0.013684941670162295 | 1.035127252385712 |
| 1 | 0.04216033206807203 | 1.0687402565127804 |
| 2 | 0.06624175451853206 | 1.2560733641107193 |
| 3 | 0.03479588717755417 | 1.1500435082355007 |
| 4 | 0.00015457116974123067 | 1.0005294662078763 |
| 5 | 0.42906925917790273 | 1.8411233948920973 |
| 6 | 0.17472534908283743 | 1.3913901212178685 |
| 7 | 0.6998218783821453 | 1.0428207389324413 |
| 8 | 0.2983594853975563 | 1.0173810175994944 |
| 9 | 0.0 | 1.0 |
| 10 | 0.1616831994981853 | 1.4289156567247534 |
| 11 | 0.047822591309007866 | 1.3051062472266162 |
| 12 | 0.0 | 1.0 |
| 13 | 0.017300343940471932 | 1.1309219868792666 |
| 14 | 0.0007540025823136705 | 1.0047798722260994 |
| 15 | 0.0 | 1.0 |
| 16 | 0.01817192710569515 | 1.061928240776248 |
| 17 | 0.011353369001411776 | 1.0206620124594246 |

| Scheme | Median BOS NMSE % | Max BOS NMSE % | Median non-BOS NMSE % | Max non-BOS NMSE % |
| --- | --- | --- | --- | --- |
| baseline | 1.2762473269182297 | 80.08874868521964 | 4.526566818159923 | 77.4379976064422 |
| arbitrary | 0.3832699624572665 | 17.165823789640104 | 4.436393099337672 | 21.035583596769843 |
| pot | 0.4709540330414404 | 17.464183275037662 | 4.436393099337672 | 21.035583596769843 |

## Three-layer decision

GO: layers 7/8 BOS NMSE falls from 62.1468%/80.0887% to 16.3431%/17.1658% with arbitrary correction; PoT gives 17.0429%/17.4642% (4.28%/1.74% relative MSE penalty). Layer 17 gives 0.5495% arbitrary versus 0.5608% PoT BOS NMSE, while non-BOS improves from 77.4380% to 21.0356%. B/C non-BOS outputs are identical. This preserves most BOS gain and meets the requested descriptive go/no-go test; proceed to remaining down_proj only. Remaining BOS error is not eliminated; frozen-s10 BOS saturation is 0.146484375% in layers 7/8. This is an architectural experiment, not a production acceptance rule.

## Final architectural interpretation

A. Factorization materially recovers the severe layer-7/8 BOS regressions, but does not eliminate their errors with the frozen common s_10. Exact reference-to-INT10 projection gives unavoidable BOS NMSE lower bounds of 13.239929% and 15.701946% at these output scales (layer 17: 0.0262483%). These are output-grid bounds, not a fourth scheme or an additive error decomposition; see output_grid_bound.json.

B. On layers 7/8/17, PoT adds 0.699822/0.298359/0.011353 percentage points BOS NMSE versus arbitrary, corresponding to 4.282074%/1.738102%/2.066201% relative MSE penalties. This supports a shift-only architectural follow-up. It is not universally within 20% relative MSE: layer 5 is 0.939184% versus 0.510115% (84.11% relative, 0.429069 percentage points absolute). The bounded arbitrary search includes PoT points, so a coarser arbitrary grid cannot artificially make PoT look better.

C. B/C non-BOS outputs and all recorded non-BOS metrics are exactly identical. Compared with A, nine layers improve (5,9,10,11,13,14,15,16,17), nine are unchanged, none worsen. Layer 17 still has 21.0356% non-BOS NMSE. PoT BOS is slightly worse than A on layers 5/10/17; the absolute increases are about 0.00488/0.02409/0.01135 percentage points. There is no universal token-group accuracy guarantee.

D. The positive three-layer decision was recorded before expanding; all 18 down_proj have now completed. No other projections or end-to-end inference were run. Common-s_10 optimization was deliberately omitted to isolate row factorization.

Selected PoT effective shifts lie in [11,28]. The CSV effective_shift_min/max fields represent corrected shifts for PoT and the base channel shift range for arbitrary/baseline; arbitrary correction is not a pure shift. No separate BOS channel arrays, packed parameters or dynamic scale stream were exported. Future row_change/index selection must identify logical sequence BOS, not every tile's first row or every decode row. No RTL cost or implementation claim is made.

## Verification

37 tests pass: 6 new focused mathematical tests plus 31 relevant existing static_quant/BOS-calibration tests. All 18 baseline group MSEs reproduce the previous frozen results; exact INT64 oracle checks pass, all s_10 values remain common and frozen, selection precedes validation, cache file sizes/mtimes and prior numerical source hashes are unchanged. CSVs have 9 and 54 rows. See verification.json, environment.json, commands.md and provenance.json. Total per-operation computation was 719.17 seconds; no activation capture was needed.


Common s_10 optimization was not performed. Validation was not used to select scales. See provenance.json, commands.jsonl, tests.log and per-layer checkpoints in factorized_row_requant/.
