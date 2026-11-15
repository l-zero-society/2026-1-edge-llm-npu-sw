# Mixed Linear/GELU calibration baseline



## 1. Scope

Consumer-aware calibration of 126 Linear operations followed by standalone per-layer GELUTanh LUT calibration. Status: `MIXED_LINEAR_GELU_BASELINE`; no production freeze was created.

## 2. Consumer-aware Linear quantization contract

Ordinary GEMMs were calibrated for INT8 outputs. Gate GEMMs were calibrated for INT10 outputs because they feed the activation LUT. Inputs and weights are signed INT8; accumulation is INT32; per-channel M/S and runtime absmax row exponents are used.

## 3. 126-op inventory

18 layers × seven projections: 108 ordinary INT8-output operations and 18 gate INT10-output operations.

## 4. Ordinary GEMM INT8 calibration policy

Joint `(sX_base,s8)` output-domain search with signed hardware output clamp `[-128,127]`; historical scales were evaluated but output anchors were derived from captured FP outputs.

## 5. Gate GEMM INT10 calibration policy

Joint `(sX_base,s10)` output-domain search with clamp `[-512,511]`. The old all-126 s10=0.1 values were not treated as authoritative.

## 6. Runtime row-scale policy

`compare_k_selectors_e2e.select_absmax`; operation-specific feasible k bounds enforce `0 <= S-k <= 31`.

## 7. Linear local calibration results

Selected 126 operations from bounded joint grids; 0 candidates were infeasible. Mean balanced NMSE: 0.00235812065. Details are in the three summary CSVs.

## 8. GELU INT10->INT8 LUT calibration policy

GELU LUT calibration used the newly calibrated per-op gate s10. Each table has 1024 signed INT8 entries and uses GELU tanh approximation; per-layer s_act selection compares ABSMAX with fixed tiny-tail percentiles.

## 9. GELU error decomposition

Mean input/table/total NMSE: 0.000374745859 / 0.00161138204 / 0.00208358012.

## 10. E2E evaluation

| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| MIXED_LINEAR_FP_GELU | 4.96078518 | 142.705801 | 0.107642228 | 0.00558996733 | 1.89165887 | 0.989732909 | 0.997392985 | 0.997441479 | 0.859375 | 0.989583333 | 0.845833343 |
| MIXED_LINEAR_GELU_LUT | 4.84151552 | 126.661164 | 0.12491477 | 0.00635558171 | 2.15074468 | 1.09996446 | 0.997468004 | 0.997131673 | 0.833333333 | 1 | 0.836458343 |

## 11. Historical comparison where valid

Historical down-only and prior LUT results are retained in their original diagnostics. They quantize different scopes, so no direct-equivalence claim is made.

## 12. Limitations

No GPALU product scaling was calibrated. No GeGLU fusion-aware tuning was performed. No LUT clustering was performed. GeGLU multiplication during E2E evaluation was performed in floating point on reconstructed branch values solely to isolate pre-fusion numerical error. The calibration/E2E populations are small and are not a final unseen confirmation.

## 13. Inputs for future GeGLU fusion-aware calibration

`linear_parameters.json`, gate INT10 scales, per-layer LUT tables, and `gelu_scale_pairs.*` define the pre-fusion baseline. Future work may calibrate INT8 GELU × INT8 Up product scaling without changing this experiment's records.
