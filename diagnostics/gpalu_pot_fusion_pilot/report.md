# Static GPALU POT fusion-aware small-batch pilot



## 1. Scope

A small-batch calibration of Gate s10, dependent standalone GELU LUTs, and static per-layer GPALU kG.

## 2. Hardware path modeled

Signed INT8 GELU × signed INT8 Up produced a full signed INT16 raw product. No operand was pre-shifted. Signed RNE right shift by static kG and signed INT8 saturation produced the GeGLU output.

## 3. Frozen parameters

All ordinary GEMMs, Up parameters, Gate sX, weights, and the runtime absmax row selector remained fixed.

## 4. Nested calibration policy

For each Gate s10, standalone GELU calibration selected s_act without observing product error. kG then minimized post-GPALU NMSE, and s10 was finally selected by post-GPALU fidelity. E2E was NOT used for parameter selection.

## 5. s10 search

Nine candidates centered on the previous fusion-aware s10 (j=-4..4), plus the original mixed baseline when distinct. No outward expansion was run.

## 6. kG search

Every s10 candidate evaluated static kG in {0,1,2,3,4,5,6,7}. Clipping was measured rather than automatically rejected.

## 7. Per-layer selected parameters

| Layer | j | s10 | s_act | kG | pre NMSE | post NMSE | clip |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1 | 0.0223151443 | 0.0897877064 | 5 | 0.0239196532 | 0.0350481088 | 7.62939453e-06 |
| 1 | 1 | 0.0164190925 | 0.061253002 | 4 | 0.00790831294 | 0.0116749859 | 5.34057617e-05 |
| 2 | 2 | 0.0189826137 | 0.0407891299 | 4 | 0.00701693042 | 0.0214089643 | 6.48498535e-05 |
| 3 | 0 | 0.0137616052 | 0.0553714983 | 4 | 0.00634780354 | 0.0246715153 | 3.81469727e-05 |
| 4 | -2 | 0.00711172062 | 0.0278106264 | 5 | 0.00477314527 | 0.0298215606 | 7.62939453e-06 |
| 5 | -3 | 0.00856370456 | 0.0344570296 | 4 | 0.00384788972 | 0.0130206308 | 2.28881836e-05 |
| 6 | -4 | 0.0113225601 | 0.0455577024 | 5 | 0.0107428175 | 0.0352803327 | 7.62939453e-06 |
| 7 | 0 | 0.0129120232 | 0.0299584543 | 4 | 0.00472530346 | 0.0141265721 | 2.28881836e-05 |
| 8 | -4 | 0.0140617161 | 0.0281079001 | 4 | 0.00481413914 | 0.0124440442 | 3.81469727e-06 |
| 9 | 1 | 0.0105695229 | 0.0425277653 | 3 | 0.00624355108 | 0.0134834286 | 2.67028809e-05 |
| 10 | 0 | 0.0099312731 | 0.0399596883 | 5 | 0.0030369501 | 0.0130466878 | 2.67028809e-05 |
| 11 | 0 | 0.00976834905 | 0.0393041427 | 4 | 0.00373675853 | 0.01427744 | 3.05175781e-05 |
| 12 | -2 | 0.00752806617 | 0.0302889679 | 6 | 0.002714131 | 0.0122417128 | 1.52587891e-05 |
| 13 | 2 | 0.015260338 | 0.0426439751 | 6 | 0.00332960672 | 0.0383748467 | 1.52587891e-05 |
| 14 | 0 | 0.014278119 | 0.0574497546 | 6 | 0.00372189212 | 0.0346281436 | 1.52587891e-05 |
| 15 | -1 | 0.0127652914 | 0.0513627081 | 6 | 0.00320016191 | 0.0195543256 | 1.52587891e-05 |
| 16 | 0 | 0.0206068541 | 0.0829141927 | 4 | 0.00491811267 | 0.0204799975 | 3.05175781e-05 |
| 17 | 1 | 0.0548572584 | 0.077750445 | 5 | 0.00359254092 | 0.0178539707 | 3.81469727e-06 |

## 8. GPALU clipping analysis

Mean selected clip rate: 2.2676256e-05; worst layer: 6.48498535e-05. Raw INT16 overflow count: 0.

## 9. Pre-GPALU vs post-GPALU local error

Mean current float pre-GPALU NMSE: 0.00613884326; current-s10 static-kG post NMSE: 0.0214711625; joint post NMSE: 0.0211909593.

## 10. kG distribution

Histogram: {'0': 0, '1': 0, '2': 0, '3': 1, '4': 8, '5': 5, '6': 4, '7': 0}. Mean/min/max: 4.667/3/6; moderately distributed.

## 11. Global shared-kG oracle

The global shared-kG sweep is recorded in `global_kg_oracle.csv` and was diagnostic only.

## 12. Down-proj scale alignment diagnostic

Selected GPALU output scales and ratios to fixed Down sX_base are recorded in `down_scale_alignment.csv`. Down projection was NOT recalibrated.

## 13. Small E2E comparison

| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| CURRENT_FUSION_AWARE_FLOAT_PRODUCT | 4.89122231 | 133.116184 | 0.122145864 | 0.00540838916 | 1.83021236 | 1.03394194 | 0.998043105 | 0.997543783 | 0.84375 | 0.984375 | 0.851041675 |
| CURRENT_FUSION_AWARE_STATIC_KG | 4.80818542 | 122.509113 | 0.356852975 | 0.0160817571 | 5.44210666 | 1.83915534 | 0.994762991 | 0.99360331 | 0.760416667 | 0.958333333 | 0.723958346 |
| JOINT_POST_GPALU_AWARE | 4.76729553 | 117.600764 | 0.345161944 | 0.0142373069 | 4.81794012 | 1.71963587 | 0.995061371 | 0.993923718 | 0.734375 | 0.973958333 | 0.732291679 |

### E2E deltas

| Comparison | KL rel. | NMSE rel. | PPL rel. | Top1 pp |
|---|---:|---:|---:|---:|
| float_to_current_static | 192.153138% | 197.348372% | -7.968280% | -8.333333 |
| current_static_to_joint | -3.276148% | -11.469208% | -4.006518% | -2.604167 |
| float_to_joint | 182.581770% | 163.244868% | -11.655547% | -10.937500 |

## 14. Hardware interpretation

kG was static per layer. No runtime product absmax, general GPALU M/S requantizer, or independent post-GPALU scale was modeled. Output scale followed s_h = s_act*s8_up*2^kG.

## 15. Decision for large-batch calibration

**STATIC_KG_NOT_SUPPORTED_BY_PILOT**

## 16. Limitations

This is a small-batch pilot. GPALU uses a full signed INT16 product and static UInt3-compatible kG. Down projection was not recalibrated and no extra requantization stage was added.
