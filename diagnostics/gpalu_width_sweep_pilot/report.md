# CPU-only GPALU output-width sweep

## 1. Scope
This small-batch experiment isolates post-GPALU output precision: FLOAT_PRODUCT versus INT8, INT10, and INT12.

## 2. CPU execution environment
CPU only. No MPS. No ANE. No CUDA.

## 3. Frozen upstream quantization
Gate/Up/GELU parameters were identical across widths and came from the established fusion-aware baseline.

## 4. GPALU width contracts
Raw GPALU product was signed INT16. Static per-layer UInt16 M_G / UInt5 S_G produced signed INT8, INT10, or INT12 output.

## 5. Scale calibration method
Each width independently minimized balanced prefill/decode local GeGLU NMSE over percentile anchors and a ±6/16-octave neighborhood. E2E metrics were not used for calibration.

## 6. Local INT8/10/12 comparison
See `width_comparison.csv`; all three selected parameter sets are preserved under `int8/`, `int10/`, and `int12/`.

## 7. Clipping comparison
Mean selected clipping rates: INT8 1.75899929e-05, INT10 5.08626302e-06, INT12 1.48349338e-06.

## 8. M/S representation error
Mean incremental M/S losses: INT8 3.3358946e-06, INT10 1.54638534e-08, INT12 7.59808866e-08.

## 9. 192-token E2E comparison
| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| FUSION_AWARE_FLOAT_PRODUCT | 4.91378749 | 136.154121 | 0.0438021484 | 0.00174557464 | 0.590706805 | 0.595609068 | 0.99953292 | 0.999236013 | 0.916666667 | 1 | 0.907291672 |
| GPALU_INT8 | 4.82601338 | 124.712786 | 0.216476865 | 0.0084285647 | 2.85224729 | 1.33495139 | 0.996954025 | 0.99647539 | 0.770833333 | 0.963541667 | 0.781250011 |
| GPALU_INT10 | 4.84808663 | 127.496209 | 0.0857136463 | 0.00335490705 | 1.1353089 | 0.827456936 | 0.998956443 | 0.998493475 | 0.854166667 | 1 | 0.866666674 |
| GPALU_INT12 | 4.874432 | 130.899782 | 0.0664945037 | 0.00266618179 | 0.902242557 | 0.744066766 | 0.999166637 | 0.998823469 | 0.895833333 | 1 | 0.873958341 |

## 10. INT8 -> INT10 gain
Top1 delta +8.333333 pp; KL relative delta -60.405170%; NMSE relative delta -60.195986%.
Relative to the INT8-to-FLOAT_PRODUCT gap, INT10 recovered 75.728% of KL, 75.919% of logits NMSE, and 57.143% of Top1 agreement.

## 11. INT10 -> INT12 marginal gain
Top1 delta +4.166667 pp; KL relative delta -22.422500%; NMSE relative delta -20.528892%.

## 12. Implication for future RTL tradeoff
A. INT10 recovered most of the INT8 loss in KL and logits NMSE, while recovering a smaller majority of the Top1 gap.

B. INT12 provided an additional measurable gain beyond INT10 on KL, logits NMSE, and Top1; on this small sample the gain is not negligible.

C. Post-GPALU representation precision is a major error source because widening sharply reduced the gap, but it is not the only source because INT12 still trailed FLOAT_PRODUCT.

D. Widening the GPALU/VPU boundary is numerically worth considering. This experiment supplies only the accuracy evidence and does not make the RTL decision.

## 13. Limitations
The sample contains 192 aligned targets. After GPALU reconstruction, Down projection, residuals, RMSNorm, RoPE, attention, and the remaining model used FP computation. Down was not separately quantized. All widths are retained as reusable calibration baselines.
