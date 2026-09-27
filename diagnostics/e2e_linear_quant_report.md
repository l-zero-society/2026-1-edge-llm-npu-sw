# End-to-end Linear-only prefill validation

32 held-out validation sequences, 10141 valid tokens, 10109 next-token targets. Selection is frozen from calibration; no recalibration.

Reference is the existing FP32 GGUF-recovered model, not original BF16. Quantized Linear outputs feed the original model driver and subsequent layers; transformer boundary pointers are asserted equal. All nonlinear math, embedding and lm_head remain the original FP implementation. No LUT approximation. INT8 GEMMs use the existing exact FP64-integer backend (K*127² < INT32 bound), checked against INT64. Output is the existing Profile RNE/saturation with BOS effective shift S-k; all effective shifts are in 0..31.

FP NLL / perplexity: 5.22916586 / 186.637058. Logits metrics include all valid positions; NLL shifts targets by one and excludes the final position. Cosine is mean per-token cosine. Top5 is set overlap / 5, not exact-set agreement.

| mode | final block hidden NMSE % | logits NMSE % | logits cosine | KL nats | top1 % | top5 overlap % | ppl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| down_only | 14.538275 | 2.710539 | 0.99339565 | 1.07989234 | 64.0864 | 65.4649 | 220.711908 |
| all_linear | 17.546855 | 3.399433 | 0.99247186 | 1.30009617 | 62.0254 | 63.9227 | 266.943033 |



| layer | mode | hidden NMSE % | cosine | BOS NMSE % | non-BOS NMSE % | FP energy | quant energy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | down_only | 3.136937 | 0.98498912 | 0.000183 | 4.199223 | 2.7013925 | 2.9712446 |
| 1 | down_only | 2.972765 | 0.97822797 | 0.000184 | 5.772840 | 1.4023366 | 1.4373755 |
| 2 | down_only | 2.791040 | 0.96883591 | 0.000195 | 8.407618 | 1.0173507 | 1.0073046 |
| 3 | down_only | 2.867735 | 0.96053412 | 0.000201 | 11.395223 | 0.90437204 | 0.87205051 |
| 4 | down_only | 2.585279 | 0.95756545 | 0.000209 | 11.811851 | 0.86556152 | 0.8378975 |
| 5 | down_only | 2.509685 | 0.95224960 | 0.000232 | 13.061600 | 0.8376427 | 0.81195251 |
| 6 | down_only | 2.414919 | 0.95030067 | 0.000249 | 12.746777 | 0.83474964 | 0.8132108 |
| 7 | down_only | 2.351916 | 0.94924337 | 0.001585 | 12.259212 | 0.85983536 | 0.84305155 |
| 8 | down_only | 2.310897 | 0.94829917 | 0.004568 | 12.117401 | 0.89567975 | 0.87967621 |
| 9 | down_only | 2.310829 | 0.95241872 | 0.004514 | 10.835559 | 0.92210417 | 0.90687111 |
| 10 | down_only | 2.451617 | 0.95280907 | 0.004481 | 10.518098 | 0.9486554 | 0.93432487 |
| 11 | down_only | 2.546475 | 0.95332796 | 0.004484 | 10.216601 | 0.97051159 | 0.95215557 |
| 12 | down_only | 2.782811 | 0.95065724 | 0.004468 | 10.560377 | 0.98993018 | 0.96548857 |
| 13 | down_only | 3.315677 | 0.95364622 | 0.004461 | 9.748143 | 1.1095583 | 1.0775592 |
| 14 | down_only | 4.930448 | 0.94173808 | 0.004445 | 11.819338 | 1.2572033 | 1.1827404 |
| 15 | down_only | 6.638903 | 0.93332177 | 0.004592 | 13.282680 | 1.4480803 | 1.3452222 |
| 16 | down_only | 9.040135 | 0.92800431 | 0.006212 | 14.492372 | 1.8474765 | 1.7347012 |
| 17 | down_only | 14.538275 | 0.92668407 | 13.186506 | 15.175714 | 2.6560576 | 2.5820329 |
| 0 | all_linear | 3.724677 | 0.98158577 | 0.001607 | 4.985523 | 2.7013925 | 2.9829825 |
| 1 | all_linear | 3.652768 | 0.97217203 | 0.002171 | 7.091513 | 1.4023366 | 1.4472796 |
| 2 | all_linear | 3.254824 | 0.96222447 | 0.002282 | 9.800567 | 1.0173507 | 1.0108375 |
| 3 | all_linear | 3.129414 | 0.95521406 | 0.002394 | 12.428561 | 0.90437204 | 0.87331578 |
| 4 | all_linear | 2.779664 | 0.95302032 | 0.002442 | 12.692060 | 0.86556152 | 0.83851362 |
| 5 | all_linear | 2.675204 | 0.94780734 | 0.002517 | 13.913497 | 0.8376427 | 0.81190778 |
| 6 | all_linear | 2.573473 | 0.94604492 | 0.002623 | 13.573594 | 0.83474964 | 0.81334324 |
| 7 | all_linear | 2.511766 | 0.94504646 | 0.004139 | 13.082107 | 0.85983536 | 0.84184943 |
| 8 | all_linear | 2.474836 | 0.94400133 | 0.007980 | 12.963895 | 0.89567975 | 0.87633422 |
| 9 | all_linear | 2.485938 | 0.94832360 | 0.007988 | 11.645072 | 0.92210417 | 0.903259 |
| 10 | all_linear | 2.644530 | 0.94867024 | 0.007975 | 11.335392 | 0.9486554 | 0.93055735 |
| 11 | all_linear | 2.761861 | 0.94892823 | 0.008050 | 11.071125 | 0.97051159 | 0.94837922 |
| 12 | all_linear | 3.020763 | 0.94596176 | 0.008081 | 11.454328 | 0.98993018 | 0.96127947 |
| 13 | all_linear | 3.601199 | 0.94928388 | 0.008243 | 10.580981 | 1.1095583 | 1.0740632 |
| 14 | all_linear | 5.322678 | 0.93661374 | 0.008559 | 12.754338 | 1.2572033 | 1.1786695 |
| 15 | all_linear | 7.236853 | 0.92679035 | 0.009218 | 14.474800 | 1.4480803 | 1.3422897 |
| 16 | all_linear | 9.892843 | 0.92092274 | 0.011963 | 15.856245 | 1.8474765 | 1.7362222 |
| 17 | all_linear | 17.546855 | 0.91946351 | 19.444427 | 16.652037 | 2.6560576 | 2.3995554 |

down_only: first hidden NMSE >1%: layer 0; >5%: layer 15; worst: layer 17 (14.538275%). Final post-RMSNorm hidden NMSE: 13.221936%.

all_linear: first hidden NMSE >1%: layer 0; >5%: layer 14; worst: layer 17 (17.546855%). Final post-RMSNorm hidden NMSE: 14.488294%.

Exact enabled module names, source/artifact hashes and frozen policy values are in e2e_linear_quant/provenance.json; per-sequence counters and metrics are resumable JSON records. Missing artifacts: []

## Answers A–G

A. **Down_proj-only is bounded but not accuracy-stable.** All 18 layers remain finite, with quant/FP hidden-energy ratios spanning 0.928969–1.099894. Final block hidden NMSE is 14.538275%; final normalized hidden NMSE is 13.221936%. Perplexity increases from 186.637058 to 220.711908 (+18.257%).

B. **All-Linear propagation is also bounded but loses more accuracy.** Energy ratios span 0.903427–1.104239. Final block hidden NMSE is 17.546855%; post-normalization NMSE is 14.488294%. Perplexity is 266.943033 (+43.028% versus FP). These energy ratios indicate no exploding activation scale; they do not establish accuracy.

C. **Material error is already present at layer 0**: total hidden NMSE 3.136937% (A), 3.724677% (B). Both first cross 1% there. Total NMSE first exceeds 5% at layer 15 (A) and layer 14 (B); the **non-BOS group crosses 5% at layer 1 in both modes**. Non-BOS NMSE reaches 13.061600% / 13.913497% at layer 5. The smaller mid-layer all-position NMSE must not be interpreted as recovery of normal-token accuracy: reference energy differs substantially between position groups. The worst layer is 17 in both modes.

D. **The early propagation problem is primarily in ordinary tokens, not BOS alone.** At layer 16, BOS NMSE is only 0.006212% / 0.011963%, while non-BOS NMSE is 14.492372% / 15.856245%. At layer 17, both groups are affected: A BOS 13.186506% versus non-BOS 15.175714%; B BOS 19.444427% versus non-BOS 16.652037%. These are independent conditional NMSEs, without BOS/non-BOS objective weighting in reporting.

E. **Logit ranking is not reliably preserved despite high cosine.** A/B mean logit cosine is 0.99339565 / 0.99247186, but top-1 agreement is only 64.0864% / 62.0254%. Top-5 overlap is 65.4649% / 63.9227%; exact top-5 set agreement is 13.3222% / 11.8825%. KL(FP||quant) is 1.07989234 / 1.30009617 nats/token.

F. **Do not treat this Linear scheme as accuracy-qualified for subsequent fixed-point nonlinear validation yet.** No acceptance threshold was specified, but the measured ranking disagreement, KL and likelihood loss are substantial. Fixed-point blocks may be implemented/tested independently; adding their errors now would obscure an already significant Linear-only loss. No loss in this experiment is attributable to a LUT or fixed-point Normalizer/RoPE/softmax.

G. **The first demonstrated bottleneck family is down_proj, starting at layer 0.** Mode A quantizes only this family and already produces 3.136937% total / 4.199223% non-BOS hidden NMSE at the first block. The large final BOS jump is localized to layer 17. Mode B introduces additional loss, but this comparison does not isolate which of q/k/v/o/gate/up contributes most; no additive MSE decomposition is implied. Next test calibration-only remedies for normal-token down_proj error and the layer-17 operating cases, keeping nonlinear math FP and confirming each frozen remedy on independent propagated validation. This task did not select any remedy or recalibrate on validation.

## Exact scope and sanity verification

- Mode A: `model.layers.{0..17}.mlp.down_proj` (18 operations).
- Mode B: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in all 18 layers (126 operations). Non-down operations use the actual historical all-126 exported scales/qparams, without correcting or recalibrating historical policy differences. Down operations use exactly the frozen global w_BOS=0.25 sX_normal, k and s10.
- RMSNorm, GeLU, RoPE, softmax, attention scaling, residual/elementwise math, embedding and lm_head remain the original FP32 model path. There is no LUT execution or KV-cache arithmetic (`use_cache=False`).
- Reference and both modes use the same 32 held-out tokenized sequences: 10,141 valid positions, 10,109 next-token targets, max length 512. The prefix-8 sanity pass verifies native FP logits equal instrumented FP logits bit-for-bit. Every full reference pass uses the unchanged original model driver.
- All 126 stored INT8 weights and per-channel scales exactly match re-quantization of the reference recovered matrices; stored non-down multiplier/shift fields reproduce exactly. Existing INT64 references agree with accelerated integer GEMMs and PoT shift application for all 126 operations. All effective shifts remain 0..31. This validates the existing software profile, not RTL bit equivalence.
- All 1,088 quantized-run transformer boundaries (2 modes × 32 sequences × 17 boundaries) passed tensor-pointer propagation assertions. Quantized wrappers accept only the actual current model input; they have no reference activation argument. Each selected operation executes once per prefill, with exactly one position-0 row and T-1 normal rows.
- 38 unit tests pass (22 existing static_quant, 16 focused); 18 final data/provenance invariants pass. Missing artifacts: **none**. Production, RTL, LUT, QB, ABI and calibration artifacts are unchanged. Source/parameter hashes and commit are in `provenance.json`; checks are in `verification.json`; exact commands and logs are in this directory.
- Decode is intentionally not run: the focused wrapper is prefill-only; decode needs cache-position-aware row handling and verification. The measured result is not end-to-end hardware accuracy and does not measure BF16-to-GGUF source error.

