# Fusion-aware Gate s10 small-batch pilot



## 1. Scope

Only Gate s10 was searched. This is a small-batch policy pilot, not a final calibration.

## 2. Frozen baseline

All ordinary GEMM parameters, Up parameters, Gate sX_base, weights, and the absmax row selector were unchanged.

## 3. Nested calibration policy

Outer s10 selection minimized balanced GeGLU product NMSE. For every fixed s10, inner s_act selection independently minimized standalone GELU total NMSE. Each candidate s10 received newly generated M/S and a newly calibrated standalone GELU LUT.

## 4. Search grid

Nine candidates per layer: incumbent s10 × 2^(j/8), j=-4..4. No expansion was performed.

## 5. Per-layer results

| Layer | j | s10 old | s10 new | Fusion NMSE old | Fusion NMSE new | Improvement | Class | Boundary |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | -2 | 0.0289391617 | 0.0243348374 | 0.0237357363 | 0.0237197462 | 0.0674% | NEUTRAL | False |
| 1 | -1 | 0.0164190925 | 0.0150563742 | 0.00790831294 | 0.00789312008 | 0.1921% | NEUTRAL | False |
| 2 | 3 | 0.0123087022 | 0.0159624118 | 0.00702692058 | 0.00701454906 | 0.1761% | NEUTRAL | False |
| 3 | 0 | 0.0137616052 | 0.0137616052 | 0.00634780354 | 0.00634780354 | 0.0000% | NEUTRAL | False |
| 4 | -2 | 0.0100574918 | 0.00845730876 | 0.00478791611 | 0.00370969099 | 22.5197% | BENEFICIAL | False |
| 5 | -2 | 0.0132070379 | 0.0111057508 | 0.0036685421 | 0.00365995979 | 0.2339% | NEUTRAL | False |
| 6 | -3 | 0.0207656668 | 0.0160125181 | 0.0152291765 | 0.0151686019 | 0.3978% | NEUTRAL | False |
| 7 | -4 | 0.0182603583 | 0.0129120232 | 0.00474830668 | 0.00472530346 | 0.4845% | NEUTRAL | True |
| 8 | -4 | 0.0281234323 | 0.0198862697 | 0.00494435881 | 0.00487797277 | 1.3427% | BENEFICIAL | True |
| 9 | -2 | 0.0115261465 | 0.00969229524 | 0.00692685167 | 0.00622593036 | 10.1189% | BENEFICIAL | False |
| 10 | -3 | 0.0128792678 | 0.0099312731 | 0.00441080587 | 0.0030369501 | 31.1475% | BENEFICIAL | False |
| 11 | -2 | 0.0116165902 | 0.00976834905 | 0.00384443415 | 0.00373675853 | 2.8008% | BENEFICIAL | False |
| 12 | -2 | 0.0106462933 | 0.00895242985 | 0.00209415326 | 0.00208975991 | 0.2098% | NEUTRAL | False |
| 13 | -1 | 0.0139937917 | 0.0128323635 | 0.00332916669 | 0.00332547312 | 0.1109% | NEUTRAL | False |
| 14 | -3 | 0.0185164295 | 0.014278119 | 0.00647558648 | 0.00372189212 | 42.5242% | BENEFICIAL | False |
| 15 | -3 | 0.0180528483 | 0.013920649 | 0.002905901 | 0.00275739969 | 5.1103% | BENEFICIAL | False |
| 16 | -2 | 0.0245058176 | 0.0206068541 | 0.00692424865 | 0.00491811267 | 28.9726% | BENEFICIAL | False |
| 17 | -2 | 0.0598222645 | 0.0503043278 | 0.00364473143 | 0.00357015434 | 2.0462% | BENEFICIAL | False |

## 6. Aggregate local results

Changed layers: 17/18; beneficial: 9; boundary winners: 2. Mean Gate NMSE old/new: 0.000283846389/0.000319177512. Mean GELU NMSE old/new: 0.00208358012/0.0019818752. Mean fusion gate-only old/new: 0.00334285578/0.00287424819. Mean full fusion old/new: 0.00660849738/0.00613884326.

## 7. Gate-local vs fusion-product tradeoff

Gate-local NMSE was diagnostic and could worsen when full GeGLU product NMSE improved; the outer selection did not reject that intended tradeoff.

## 8. Small E2E comparison

| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| CURRENT_MIXED_BASELINE | 4.84151552 | 126.661164 | 0.12491477 | 0.00635558171 | 2.15074468 | 1.09996446 | 0.997468004 | 0.997131673 | 0.833333333 | 1 | 0.836458343 |
| FUSION_AWARE_GATE_S10 | 4.89122231 | 133.116184 | 0.122145864 | 0.00540838916 | 1.83021236 | 1.03394194 | 0.998043105 | 0.997543783 | 0.84375 | 0.984375 | 0.851041675 |

E2E classification: **IMPROVEMENT**. E2E was NOT used for parameter selection.

## 9. Boundary winners

2 layers selected j=±4; no outward search was run.

## 10. Decision for large-batch follow-up

**PROCEED_TO_LARGE_BATCH**

## 11. Limitations

GPALU product requantization was NOT modeled. GeGLU multiplication used floating-point multiplication of reconstructed branch values. Ordinary GEMM and Up parameters were unchanged; Gate sX_base was unchanged. The calibration and validation populations are deliberately small.
