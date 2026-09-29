# Cached-decode deployment PTQ

Only the 18 `down_proj` operations are quantized. Parameter selection used calibration chats only; validation was frozen evaluation.

Calibration: 32 chats, 11841 prompt tokens, and 1024 cached response targets (maximum 32/chat). Validation: 24 chats, 9655 prompt tokens, and 768 cached response targets sampled as the first 32 of 14630 available response targets.

## Scale decisions

| layer | old sX | new sX | old s10 | new s10 | old local NMSE % | new local NMSE % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.32823654 | 0.27601293 | 0.027936821 | 0.055873643 | 1.1042 | 0.9414 |
| 1 | 0.10382454 | 0.10382454 | 0.007798598 | 0.015597196 | 0.9898 | 0.5736 |
| 2 | 0.088813612 | 0.062800707 | 0.0073147055 | 0.0094859994 | 0.8842 | 0.8007 |
| 3 | 0.050720926 | 0.042651045 | 0.0041331591 | 0.0082663182 | 2.9796 | 2.3135 |
| 4 | 0.038223396 | 0.038223396 | 0.0047078985 | 0.006105389 | 1.0097 | 0.9905 |
| 5 | 0.033621895 | 0.028272531 | 0.0070338056 | 0.0041823258 | 0.9522 | 0.9069 |
| 6 | 0.032511839 | 0.027339089 | 0.0048864588 | 0.0037679748 | 1.0073 | 0.9923 |
| 7 | 0.028166948 | 0.033496335 | 0.0175236 | 0.0087618001 | 0.9199 | 0.7620 |
| 8 | 0.02452638 | 0.020624145 | 0.027664846 | 0.013832423 | 1.1616 | 0.7559 |
| 9 | 0.026544869 | 0.031567348 | 0.0038522836 | 0.0029705167 | 0.6696 | 0.6509 |
| 10 | 0.029077519 | 0.034579193 | 0.0051306654 | 0.0025653327 | 0.7361 | 0.7098 |
| 11 | 0.041208523 | 0.034652099 | 0.0050093576 | 0.0070843014 | 0.8271 | 0.8027 |
| 12 | 0.047746778 | 0.047746778 | 0.0055779031 | 0.0078883462 | 1.0212 | 1.0081 |
| 13 | 0.075991708 | 0.053734252 | 0.008480232 | 0.0092477586 | 1.3584 | 1.3204 |
| 14 | 0.11293577 | 0.13430402 | 0.0076670594 | 0.0064472028 | 1.8136 | 1.7256 |
| 15 | 0.12424406 | 0.12424406 | 0.014980117 | 0.013736828 | 1.1016 | 1.1014 |
| 16 | 0.13919387 | 0.11704763 | 0.035617398 | 0.029950543 | 1.0921 | 0.9874 |
| 17 | 0.14417333 | 0.10194594 | 0.41800357 | 0.20900179 | 4.9372 | 1.9400 |

Mean local NMSE changed from 1.364758% to 1.071282% (21.504% relative reduction). Fourteen sX bases and all 18 s10 values changed.

The propagated refinement capture changed mean local NMSE from 1.071282% to 1.069642%. No layer crossed the refinement trigger, so no second parameter search ran.

## Prefill/decode diagnostic

| layer | prefill/decode sX | prefill/decode s10 | prefill gain % | decode gain % | dual trigger |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.8409 | 1.5422 | 0.000 | -1.746 | no |
| 1 | 1.4142 | 1.5422 | 0.000 | 7.689 | no |
| 2 | 0.8409 | 0.8409 | 0.163 | 1.993 | no |
| 3 | 1.0000 | 1.5422 | 0.000 | 0.167 | no |
| 4 | 0.7071 | 0.9170 | 4.329 | 1.169 | no |
| 5 | 0.8409 | 0.9170 | 0.286 | 0.916 | no |
| 6 | 1.1892 | 1.0000 | 0.000 | 3.403 | no |
| 7 | 0.7071 | 1.0000 | 0.904 | 0.000 | no |
| 8 | 1.4142 | 1.0000 | 7.852 | 0.000 | no |
| 9 | 0.5946 | 1.1892 | 2.641 | 0.000 | no |
| 10 | 1.1892 | 1.0000 | 3.360 | 2.656 | no |
| 11 | 1.4142 | 1.5422 | 9.291 | 1.806 | no |
| 12 | 0.7071 | 1.8340 | 1.438 | 0.016 | no |
| 13 | 1.0000 | 1.4142 | 0.000 | 0.480 | no |
| 14 | 0.7071 | 1.0905 | -0.017 | 1.264 | no |
| 15 | 1.0000 | 1.1892 | 3.006 | 0.563 | no |
| 16 | 1.1892 | 1.0000 | 11.818 | 0.000 | no |
| 17 | 0.8409 | 1.0000 | 0.000 | 1.362 | no |

Dual-policy threshold: **not met** (0/18 qualifying layers); dual-policy E2E validation was skipped.

## Cached response validation

| mode | response NLL | PPL | KL | logits NMSE % | cosine | top1 % | FP top1 in top5 % | top5 overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.006613 | 149.397904 | 0.000000 | 0.000000 | 1.00000000 | 100.000 | 100.000 | 100.000 |
| old_general | 5.044939 | 155.234854 | 0.081807 | 0.782289 | 0.99713067 | 86.979 | 100.000 | 87.839 |
| calibrated | 5.022560 | 151.799450 | 0.074223 | 0.470313 | 0.99724546 | 88.932 | 99.740 | 88.464 |

## First assistant token

| mode | NLL | KL | logits NMSE % | cosine | top1 % | FP top1 in top5 % | top5 overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | 11.987491 | 0.000000 | 0.000000 | 1.00000000 | 100.000 | 100.000 | 100.000 |
| old_general | 11.542678 | 0.053417 | 0.794715 | 0.99843448 | 87.500 | 100.000 | 90.833 |
| calibrated | 11.862856 | 0.065945 | 0.420900 | 0.99917344 | 83.333 | 100.000 | 95.833 |

## Greedy validation

| mode | mean shared prefix | median shared prefix | aligned agreement % | exact 32-token matches |
|---|---:|---:|---:|---:|
| FP | 32.0000 | 32.0000 | 100.000 | 16/16 |
| old_general | 11.6875 | 6.5000 | 45.898 | 2/16 |
| calibrated | 12.8750 | 6.0000 | 44.727 | 4/16 |

## Answers

A. **14/18** layers changed `sX_base`.

B. **18/18** layers changed `s10`.

C. Mean local calibration output NMSE fell from **1.364758%** to **1.071282%**, a **21.504%** relative reduction.

D. No material propagated shift was found: pass-2 mean local NMSE changed by **-0.153%** and no layer crossed the absmax trigger.

E. Prefill/decode optima sometimes chose different grid points, but none of the 18 layers satisfied both the scale-ratio and >=10% local-gain conditions.

F. A separate prefill/decode scale policy is **not justified** by this diagnostic.

G. Yes. Cached response PPL improved from **155.234854** to **151.799450**; the gap above FP was reduced by **58.86%**.

H. Yes. KL fell **9.27%**, logits NMSE fell **39.88%**, and top-1 agreement rose **1.953 percentage points**.

I. Greedy preservation is mixed: mean shared prefix and exact matches improved, while median prefix and aligned token agreement declined slightly.

J. The calibrated shared policy is the better **software research baseline** because most cached deployment metrics improve; it remains a diagnostic artifact and does not replace production parameters.
