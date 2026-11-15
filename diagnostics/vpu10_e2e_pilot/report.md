# CPU-only INT10 VPU path E2E precision pilot

## 1. Scope
GPALU INT10 plus hardware-like INT10 RMSNorm, Softmax, and RoPE on the established 192-token set.

## 2. RTL contract inspected
- `src/main/scala/npu/top/VPU.scala` (fc3b9c95968cd4fcd56874bda0cd21efba31595fddab1a4e6274b54f420d1de2)
- `src/main/scala/npu/core/NormUnit.scala` (f7c84b1e44a93d0c9f2ac241e2d1f42aa7a92965b549d6e877fd70b9a0018611)
- `src/main/scala/npu/core/Rope.scala` (315fc6e4e8442ae916d53e617efe178d5360555c4154c15eaab5487ac4ea6f16)
- `src/test/scala/NormUnit_Test.scala` (8eaa8b58c4f954bc129c6d3e22d00ca03e5b820adbc0e7ce9c1d3650391278fc)
- `src/test/scala/NormUnit_Distributed_Test.scala` (732022f2fef41a663902615f4ccc8040d3ab5c09a52865028989255af1c5fa02)
- `src/test/scala/Rope_Test.scala` (846ab3e86a276156bf32c69286c1eac0cbe975b107cc8674f1865c6567ec0d67)

## 3. CPU execution environment
CPU only; MPS, ANE, CUDA, Core ML, and MLX accelerators were not used.

## 4. INT10 activation path
GPALU, Normalizer inputs/outputs, and RoPE inputs/outputs use signed INT10. Static operation scales are recorded in the parameter artifacts.

## 5. Normalizer 10-bit model
The accumulator width is 35 bits. Indexing follows `normalizedIndex()` and `deltaToIndex()` with 1024 entries. Effective 10-bit coefficients remain packed in a wider software container; the 128-bit RTL write interface was not redesigned.

## 6. Norm index10 / coeff16 vs coeff10
See `norm_local_summary.csv` and the E2E table.

## 7. Softmax Exp/Scale LUT results
See `softmax_local_summary.csv`.

## 8. RoPE index10 / coeff16 results
RoPE uses a 10-bit angle index and signed 16-bit Q2.14 sine/cosine coefficients; see `rope_local_summary.csv`.

## 9. Local error decomposition
See `lut_error_decomposition.json`.

## 10. 192-token E2E results
| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| GPALU10_FP_VPU2 | 4.84808663 | 127.496209 | 0.0857136463 | 0.00335490705 | 1.1353089 | 0.827456936 | 0.998956443 | 0.998493475 | 0.854166667 | 1 | 0.866666674 |
| GPALU10_NORM10_COEFF16 | 9.76019547 | 17330.0188 | 7.63310572 | 0.202698914 | 68.5938175 | 6.6357499 | 0.960963027 | 0.941638424 | 0.0677083333 | 0.171875 | 0.0989583353 |
| GPALU10_NORM10_COEFF10 | 9.44337722 | 12624.28 | 7.3671283 | 0.21249528 | 71.908932 | 6.92632425 | 0.9628284 | 0.942757454 | 0.078125 | 0.182291667 | 0.102083335 |
| GPALU10_ROPE10 | 4.84823858 | 127.515583 | 0.0932035804 | 0.00337265046 | 1.14131332 | 0.829889444 | 0.99897415 | 0.998469845 | 0.864583333 | 1 | 0.864583341 |
| VPU10_FULL_COEFF16 | 10.3167138 | 30233.7397 | 8.06961448 | 0.21737124 | 73.5589689 | 6.91555611 | 0.959331843 | 0.94115075 | 0.0625 | 0.166666667 | 0.0968750017 |
| VPU10_FULL_COEFF10 | 9.89539259 | 19838.7543 | 7.73087438 | 0.208368817 | 70.5125265 | 6.7950496 | 0.960687304 | 0.941954283 | 0.0625 | 0.171875 | 0.096875002 |

## 11. Accuracy loss from GPALU-only INT10 baseline
{"kl_relative": 89.19420734613426, "nmse_relative": 61.10867059241387, "ppl_relative": 154.60270033382187, "top1_pp": -79.16666666666666}

## 12. Whether VPU10 remains above the desired ~82-83% region
Primary VPU10 Top1 is 6.2500%. The 82-83% region was interpretive only and was not an optimization target.

## 13. Hardware implications
The numerical result indicates whether further RTL investigation is supported; it is not an RTL implementation decision.

## 14. Limitations
This is a 192-token CPU emulation. GEMMs other than the established Gate/Up/GELU/GPALU path remained FP. Static scales came only from the established small calibration population.
