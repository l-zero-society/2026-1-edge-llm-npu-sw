# Joint MLP consumer-aware small-batch pilot

## 1. Scope
Calibration-only joint optimization of the existing INT8 MLP datapath through its residual consumer.

## 2. Fixed hardware contract
No RTL datapath changes were introduced. All external VPU activation boundaries remained INT8. GPALU used signed INT8 operands, signed INT16 product, static UInt16 M_G / UInt5 S_G requantization, and signed INT8 output fed directly into Down projection.

## 3. Previous 79.17% baseline
The previous full-scale Down-aware mode had Top1 79.166667% on the same 192 targets.

## 4. What calibration restrictions were removed
s_act remained live through consumer-aware search, Down output s8 was recalibrated, GPALU search was refined, and selection extended through the post-MLP residual tensor.

## 5. Search method
The fixed small capture was used. Up to 96 deterministic GPALU candidates per layer received exact Down and residual evaluation; every exact candidate searched Down s8 and regenerated per-channel M/S. Exact counts are recorded in `verification.json`; `candidate_summary.csv` is the compact deterministic audit subset, while the full sweep remains an uncommitted work artifact. E2E metrics were never used for parameter selection.

## 6. Joint s10 / s_act results
See `layer_selection.csv` and `guarded_layer_selection.csv`.

## 7. GPALU full-scale results
Actual effective output scales derived from canonical M_G/S_G were used throughout.

## 8. Down output scale recalibration
Down Wq/sW stayed fixed; output s8 and its per-channel requantization parameters were searched jointly.

## 9. Residual-aware calibration
The current simulator reconstructs the Down result before adding the FP residual. The local objective preserves that apples-to-apples contract.

## 10. Error-cancellation analysis
Guarded warning layers: none. The guarded selection excludes candidates whose GELU, post-GPALU, or Down NMSE exceeds twice the corresponding best local value.

## 11. Local ablation
The sequential small-batch ablation is recorded in `ablation_summary.csv`.

## 12. Small E2E results
| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| FUSION_AWARE_FLOAT_PRODUCT | 4.89122231 | 133.116184 | 0.122145864 | 0.00540838916 | 1.83021236 | 1.03394194 | 0.998043105 | 0.997543783 | 0.84375 | 0.984375 | 0.851041675 |
| STATIC_POT_KG | 4.76729553 | 117.600764 | 0.345161944 | 0.0142373069 | 4.81794012 | 1.71963587 | 0.995061371 | 0.993923718 | 0.734375 | 0.973958333 | 0.732291679 |
| FULL_SCALE_DOWN_AWARE_PREVIOUS | 4.77621061 | 118.653871 | 0.227044705 | 0.00834755595 | 2.82483373 | 1.32566289 | 0.997084738 | 0.996265133 | 0.791666667 | 0.989583333 | 0.790625011 |
| FREE_SACT_ONLY | 4.81513753 | 123.363777 | 0.261372359 | 0.0122277452 | 4.13789942 | 1.60030133 | 0.996583058 | 0.995517194 | 0.776041667 | 0.984375 | 0.773958344 |
| FREE_DOWN_S8 | 4.77511985 | 118.524519 | 0.245300588 | 0.0106410571 | 3.60096024 | 1.48621156 | 0.996770297 | 0.995889959 | 0.770833333 | 0.984375 | 0.779166678 |
| FINER_GPALU_SCALE | 4.83767353 | 126.175466 | 0.209028578 | 0.0113294732 | 3.83392194 | 1.54913406 | 0.996961321 | 0.995524081 | 0.802083333 | 0.979166667 | 0.785416677 |
| JOINT_NO_RESIDUAL_OBJECTIVE | 4.84655283 | 127.300805 | 0.235112599 | 0.00984068226 | 3.33011139 | 1.42737077 | 0.996854597 | 0.995806847 | 0.78125 | 0.984375 | 0.773958345 |
| JOINT_RESIDUAL_AWARE | 4.82547198 | 124.645285 | 0.244295088 | 0.0105358316 | 3.56535165 | 1.47218489 | 0.99628414 | 0.995661414 | 0.770833333 | 0.979166667 | 0.775000012 |
| GUARDED_JOINT_RESIDUAL_AWARE | 4.82547198 | 124.645285 | 0.244295088 | 0.0105358316 | 3.56535165 | 1.47218489 | 0.99628414 | 0.995661414 | 0.770833333 | 0.979166667 | 0.775000012 |

## 13. Progress toward ~82-83% Top1 target
Guarded Top1 was 77.083333%, a -2.0833 pp change from the previous full-scale baseline. This target was interpretive only.

## 14. Implications before Normalizer/RoPE calibration
This pilot isolates MLP calibration margin. Normalizer and RoPE were not included.

## 15. Decision
**CALIBRATION_ONLY_GAIN_INSUFFICIENT**

## 16. Limitations
The calibration and validation populations remain deliberately small. Residual addition follows the current software simulator's reconstructed-Down plus FP-residual behavior. No statistical claim or large-batch conclusion is made.

s_act was consumer-aware searched rather than frozen by standalone GELU optimization. Down output s8 was allowed to recalibrate. Final local selection used post-MLP-residual fidelity. E2E metrics were never used for parameter selection.
