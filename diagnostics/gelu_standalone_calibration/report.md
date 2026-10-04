# GeLU standalone per-op LUT calibration

## 1. Scope
Per-op fixed-INT10-input, signed-INT8-output GELUTanh LUT calibration. No fusion-aware tuning was performed. No LUT clustering was performed. `frozen_linear_quant_v1` was not modified.
## 2. GeLU numerical contract
Each layer uses its fixed gate-producer `s10`; `q=clip(RNE(x/s10),-512,511)`, address `q+512`, and `q_act=clip(RNE(GELUTanh(q*s10)/S_act),-127,127)`. Tables contain 1024 signed INT8 entries and use `torch.nn.functional.gelu(..., approximate="tanh")`.
## 3. Fixed s10 sources
All 18 scales are authoritative fixed gate-projection output scales from `calibration_outputs/gguf-all-126-oasst1/manifest.json` (SHA256 `423c628975784166aa1141ff537c46831b4a399ff032e59335d62aa358e8fdf5`); each is 0.1. No derived input scale was needed.
## 4. Calibration dataset
Reused the fingerprint-validated prior 12-chat gate capture: prefill/decode reservoirs, 64 rows per layer/group, first 16 response positions. It excludes the frozen Linear 1024-target confirmation data.
## 5. ABSMAX vs probabilistic clipping policy
Per layer, retained percentiles 99, 99.5, 99.9, 99.95, 99.99, 99.995, and 100 (ABSMAX) were evaluated against the observed code-weighted distribution. Selection minimized balanced prefill/decode total standalone activation NMSE; E2E results did not select parameters.
## 6. Per-op selected S_act summary
| Layer | Policy | Percentile | S_act | Total NMSE | ABSMAX NMSE | Saturation | LUT SHA256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | PERCENTILE_CLIPPING | 99.995 | 0.0866141732 | 0.000154241086 | 0.000235294562 | 4.95910645e-05 | `2a25cdcad43a3bad07431bea106bb1ce1f5130574fc6bb8777acc236e048037d` |
| 1 | PERCENTILE_CLIPPING | 99.95 | 0.0598425197 | 0.000205362264 | 0.000248702503 | 0.000353336334 | `eb4a0fa441458f45273a45e8749b147f1dda3923408a62184d78af5913cd8161` |
| 2 | PERCENTILE_CLIPPING | 99.99 | 0.0425196849 | 0.000593752764 | 0.000946425013 | 9.77516174e-05 | `272f1d2620338e53fd4730c2f361b1a3f38fde7e51d3aa92195e1844ef0aed2e` |
| 3 | PERCENTILE_CLIPPING | 99.995 | 0.0377952689 | 0.00165598427 | 0.00288726596 | 4.57763672e-05 | `4a9183b28db1c91e94a25a8056ceff3e0311bf9839489c9be8330f6d1d687a7b` |
| 4 | PERCENTILE_CLIPPING | 99.99 | 0.0275542032 | 0.00255729776 | 0.00675322314 | 8.63075256e-05 | `ade519f474b2c40f469d4061b8888a42db6bdc019791a33c07265092166fe174` |
| 5 | PERCENTILE_CLIPPING | 99.995 | 0.0314955099 | 0.0051230552 | 0.00917789992 | 4.33921814e-05 | `49b1876dbc661a9e1c007cecbf8a569b165a9fb0641a9039cc820d250ff962a4` |
| 6 | PERCENTILE_CLIPPING | 99.995 | 0.0322831244 | 0.00451689548 | 0.00878664214 | 4.62532043e-05 | `5f09299f9932f0a0744d72ab1e9a11db2bb1f7805721e2d1458657d1eecb2c39` |
| 7 | PERCENTILE_CLIPPING | 99.995 | 0.0322831244 | 0.00470390387 | 0.0101306634 | 4.91142273e-05 | `5f09299f9932f0a0744d72ab1e9a11db2bb1f7805721e2d1458657d1eecb2c39` |
| 8 | PERCENTILE_CLIPPING | 99.995 | 0.0275542032 | 0.00332714142 | 0.00470157779 | 4.8160553e-05 | `ade519f474b2c40f469d4061b8888a42db6bdc019791a33c07265092166fe174` |
| 9 | PERCENTILE_CLIPPING | 99.99 | 0.0267645209 | 0.00342106893 | 0.0064970891 | 8.72612e-05 | `828a3c865a4ac6b92af821cf8a836fea2e2b741987b3feb1c17f33c77e87db96` |
| 10 | PERCENTILE_CLIPPING | 99.995 | 0.0291317181 | 0.0032711791 | 0.00410011835 | 4.57763672e-05 | `cc33121e257ed41399a877ff1a51f9098e2e8eec5e3f0f0a4cef948ebdff2385` |
| 11 | PERCENTILE_CLIPPING | 99.995 | 0.0354330304 | 0.00257802296 | 0.00288565523 | 4.95910645e-05 | `b93bc64685a33717da78fa2c2a68023572d29d9e208e8835c59a162b7f69a540` |
| 12 | PERCENTILE_CLIPPING | 99.995 | 0.0346455984 | 0.00232752653 | 0.00270740768 | 4.38690186e-05 | `d96239ff50b43a58235b4dbe5dd54bd6e76f76992365c186ba2e3b23544ad8fa` |
| 13 | PERCENTILE_CLIPPING | 99.995 | 0.0393700769 | 0.00189065363 | 0.00298698748 | 4.57763672e-05 | `3b86864a7e548f92d5ca5ab0b00c0cf0c43b8d00c4c4ef9558a2879a5c1bdc79` |
| 14 | PERCENTILE_CLIPPING | 99.995 | 0.0480314961 | 0.00416946342 | 0.00685049402 | 4.8160553e-05 | `7b7f6483ed7e328e1b726fd8fd990b1fd49c92b9af3411884fbe8bf11fedab81` |
| 15 | PERCENTILE_CLIPPING | 99.995 | 0.0559055118 | 0.0032315363 | 0.00699782826 | 4.86373901e-05 | `63fd742b2e1cc86cf803ecfd92e14b4cac0cfacca9ad7fa2e674414246ae9bab` |
| 16 | PERCENTILE_CLIPPING | 99.99 | 0.0645669291 | 0.00213280636 | 0.00421073553 | 8.34465027e-05 | `eeda8969666a81bd12791abdf43e95532e1b0deec16c30b7a505fd6e138345e1` |
| 17 | PERCENTILE_CLIPPING | 99.995 | 0.0867263386 | 0.0015919961 | 0.00267388929 | 5.00679016e-05 | `8f5b520735264030a488526b077fecb4bf5e84064b133b5b01a3c657819eee58` |

## 7. Input / table / total error decomposition
Mean selected input NMSE: 0.000958927589; table NMSE: 0.00167422389; total NMSE: 0.00263621597. Metrics are computed on captured workload probability, separately by prefill/decode then balanced.
## 8. ABSMAX-vs-selected local comparison
Mean per-op ABSMAX total NMSE: 0.00465432774; selected total NMSE: 0.00263621597. Every layer retains its explicit ABSMAX candidate in `per_op_candidates.csv`.
## 9. E2E comparison
| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| FROZEN_LINEAR_FP_GELU | 4.94016528 | 139.793352 | 0.0565037455 | 0.00311548917 | 1.0542893 | 0.725382106 | 0.997556985 | 0.998526475 | 0.880208333 | 0.994791667 | 0.885416673 |
| FROZEN_LINEAR_PER_OP_ABSMAX_GELU_LUT | 4.93105741 | 138.525913 | 0.0693057478 | 0.00341697084 | 1.15631144 | 0.805871227 | 0.99773072 | 0.998389611 | 0.916666667 | 1 | 0.866666675 |
| FROZEN_LINEAR_PER_OP_SELECTED_GELU_LUT | 4.90976444 | 135.607467 | 0.0847366518 | 0.00261931863 | 0.886383949 | 0.718851087 | 0.99836302 | 0.998711925 | 0.859375 | 1 | 0.875000007 |

The primary comparisons are per-op ABSMAX versus frozen-Linear FP GeLU and per-op selected versus per-op ABSMAX. Exact deltas are in `e2e_summary.json`.
## 10. Comparison against previous global shared-LUT pilot
The same eight conversation identities and 192 targets were reused, so this comparison is direct.
| Mode | Local activation NMSE | E2E KL | E2E logits NMSE | Top1 | PPL |
|---|---:|---:|---:|---:|---:|
| Previous global shared LUT | 0.00353904424 | 0.0776858066 | 0.00362848278 | 0.901041667 | 157.169314 |
| Per-op selected LUT | 0.00263621597 | 0.0847366518 | 0.00261931863 | 0.859375 | 135.607467 |
## 11. Standalone baseline classification
Per-op ABSMAX: **BASELINE_REGRESSION**. Per-op selected: **BASELINE_REGRESSION**. Clipping comparison: **CLIPPING_TRADEOFF**.
## 12. Scale-pair artifact for future clustering
`scale_pairs.csv` and `scale_pairs.json` contain `(s10,S_act)` and log2 coordinates. No clustering was performed.
## 13. Limitations
Gate projection MAC arithmetic remains FP; this isolates the fixed INT10 activation interface and INT8 LUT output. Selection is activation-standalone and deliberately ignores the Up branch, product, and downstream error. This is a small selection evaluation, not a final confirmation or production freeze.
