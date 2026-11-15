# Full-scale GPALU Down-aware small-batch pilot



## 1. Scope

Static per-operation full-scale GPALU requantization and direct Down consumption were calibrated on the existing small batch.

## 2. Why POT-only failed

The prior POT-only pilot had negligible clipping but substantial scale-resolution loss.

## 3. Full-scale M_G/S_G hardware contract

GPALU used full signed INT16 raw multiplication without operand pre-shifts, followed by static UInt16 M_G / 2^UInt5 S_G, RNE, and signed INT8 saturation.

## 4. Direct GPALU -> Down scale contract

GPALU INT8 output fed Down directly. No second GPALU-output-to-Down-input quantization was performed.

## 5. Frozen parameters

Up, Gate sX/Wq/sW, Down Wq/sW/output s8, other GEMMs, weights, and runtime policies were fixed. Down M/S were regenerated because the physical input scale changed.

## 6. Search policy

Gate s10 used the fixed small grid. Scale probes used requested percentile anchors and log neighborhoods; 16 strongest and mandatory anchor candidates per s10 received exact Down evaluation. E2E was not used for selection.

## 7. M/S representation accuracy

Mean alpha relative error: 3.77807194e-06.

## 8. Per-layer selected parameters

| Layer | s10 | s_act | target s_h | effective s_h | M_G | S_G | Down NMSE | warning |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.0265373283 | 0.0944552634 | 0.924275264 | 0.924283971 | 37591 | 20 | 0.0238572771 | False |
| 1 | 0.0195257017 | 0.0611907816 | 0.160337449 | 0.160337473 | 55053 | 20 | 0.0113072643 | False |
| 2 | 0.020700687 | 0.0407493834 | 0.13109883 | 0.131099189 | 26217 | 19 | 0.0170123853 | False |
| 3 | 0.0106116483 | 0.0426972619 | 0.10273502 | 0.102734629 | 42495 | 20 | 0.0191022765 | False |
| 4 | 0.00922276061 | 0.0278040379 | 0.0972958818 | 0.0972951361 | 60083 | 21 | 0.0288124201 | False |
| 5 | 0.00933878605 | 0.031440355 | 0.0465937901 | 0.0465936092 | 59723 | 20 | 0.0119795749 | False |
| 6 | 0.013464869 | 0.0541775439 | 0.0754953589 | 0.0754952774 | 46341 | 20 | 0.0299356185 | False |
| 7 | 0.0129120232 | 0.0299584543 | 0.0390584402 | 0.0390586549 | 62219 | 20 | 0.0180323089 | False |
| 8 | 0.0216861308 | 0.0281530127 | 0.0334324995 | 0.0334323826 | 5187 | 16 | 0.0106618221 | False |
| 9 | 0.00815021633 | 0.0327931441 | 0.038938102 | 0.0389382006 | 24149 | 18 | 0.0116289026 | False |
| 10 | 0.0099312731 | 0.0399596883 | 0.0500991024 | 0.0500994924 | 62757 | 21 | 0.014756147 | False |
| 11 | 0.00976834905 | 0.0393041427 | 0.0745243705 | 0.0745246404 | 25891 | 19 | 0.020023957 | False |
| 12 | 0.0126606477 | 0.0339728611 | 0.0729615117 | 0.07296128 | 2505 | 17 | 0.0129604494 | False |
| 13 | 0.0166415166 | 0.0425725177 | 0.206731343 | 0.206730001 | 55017 | 22 | 0.0419047768 | False |
| 14 | 0.0130930929 | 0.0526816572 | 0.236265259 | 0.236264383 | 60097 | 22 | 0.0342868045 | False |
| 15 | 0.013920649 | 0.0560114303 | 0.222925771 | 0.22292542 | 4871 | 18 | 0.0199162568 | False |
| 16 | 0.0188965686 | 0.0760326499 | 0.220591441 | 0.220591391 | 55109 | 20 | 0.0168510921 | False |
| 17 | 0.0598222645 | 0.0819107407 | 0.327356421 | 0.327356801 | 41053 | 20 | 0.0187473505 | False |

## 9. GPALU clipping

Mean selected clip rate: 1.75899929e-05.

## 10. Local GeGLU fidelity

Mean selected ideal/HW local NMSE: 0.0198356734/0.0198357046; mean M/S representation incremental NMSE: 3.53248481e-08.

## 11. Down-projection consumer-aware fidelity

Mean selected Down NMSE: 0.0200987047.

## 12. Error-cancellation warnings

Layers: none.

## 13. Old Down scale vs new GPALU output scale

See `down_scale_alignment.csv`; equality was not forced.

## 14. Small E2E comparison

| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| FUSION_AWARE_FLOAT_PRODUCT | 4.89122231 | 133.116184 | 0.122145864 | 0.00540838916 | 1.83021236 | 1.03394194 | 0.998043105 | 0.997543783 | 0.84375 | 0.984375 | 0.851041675 |
| STATIC_POT_KG | 4.76729553 | 117.600764 | 0.345161944 | 0.0142373069 | 4.81794012 | 1.71963587 | 0.995061371 | 0.993923718 | 0.734375 | 0.973958333 | 0.732291679 |
| FULL_SCALE_LOCAL_HW | 4.83324246 | 125.617611 | 0.257355563 | 0.0103012852 | 3.48598057 | 1.46157668 | 0.996624724 | 0.995595633 | 0.78125 | 0.96875 | 0.773958344 |
| FULL_SCALE_DOWN_AWARE_HW | 4.77621061 | 118.653871 | 0.227044705 | 0.00834755595 | 2.82483373 | 1.32566289 | 0.997084738 | 0.996265133 | 0.791666667 | 0.989583333 | 0.790625011 |

### E2E deltas

| Comparison | KL rel. | NMSE rel. | PPL rel. | Top1 pp |
|---|---:|---:|---:|---:|
| static_to_local | -25.439184% | -27.645830% | 6.817003% | 4.687500 |
| local_to_downaware | -11.777814% | -18.965879% | -5.543602% | 1.041667 |
| float_to_downaware | 85.879978% | 54.344588% | -10.864429% | -5.208333 |

## 15. Arithmetic width implications

Theoretical/observed minimum signed intermediate widths: 32/31 bits.

## 16. Decision for large-batch calibration

**FULL_SCALE_PROMISING_BUT_NEEDS_REFINEMENT**

## 17. Limitations

This remained a small-batch pilot. Candidate Down evaluation used a bounded local shortlist after exhaustive scale-probe scoring. No dynamic runtime GPALU scale detector or floating-point hardware was modeled.
