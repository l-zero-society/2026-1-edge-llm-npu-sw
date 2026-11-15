# Shared GELU Activation LUT pilot



## 1. Scope

frozen_linear_quant_v1 was not modified. This is a pilot, not a final freeze.

## 2. Activation graph

gate=gate_proj(x); up=up_proj(x); act=GELUTanh(gate); h=act*up; y=down_proj(h).

## 3. Previous FP16 reference

No previous FP16 expansion implementation was located. The explicit architecture contract resolved this; no legacy unsigned LUT was used.

## 4. Integer LUT contract

Global sL; signed INT10 input [-512,511]; address=q+512; 1024 signed INT8 entries [-127,127]; global sA; RNE; no interpolation. One table shared by all layers and lanes.

Selected sL=0.01843670829728099, sA=0.07418234598354792.

## 5. Data

12 deterministic previously available calibration chats; at most 16 response positions and 64 paired rows/group/layer. E2E selection: 8 chats × 24 targets. The 1024-target confirmation identities were excluded.

## 6. Search

26 coarse percentile pairs; no refinement. Equal weight per layer and available prefill/decode group. Product NMSE ranks top three; frozen down_proj output NMSE selects among them before E2E.

## 7. Per-layer local results

See layer_summary.csv for each group and global_candidates.csv for the full coarse pool.

Diagnostic-only per-layer oracle: shared product score = 0.00470354063; mean oracle score = 0.00268097815; worst layer gap = 0.00563409204. No per-layer tables or oracle E2E were used.

## 8. Error decomposition

- Balanced input_nmse: 0.000105210103

- Balanced table_nmse: 0.00343472018

- Balanced activation_nmse: 0.00353904424

- Balanced product_nmse: 0.00470354063

- Balanced downstream_nmse: 0.0127207072

Input error compares GELU(Q10(g)) to GELU(g); table error compares reconstructed LUT output to GELU(Q10(g)); combined error compares LUT to GELU(g). Downstream reference uses the same frozen quantized down_proj on the exact-activation product.

The measured table/output quantization NMSE is about 32.6 times the input-quantization NMSE. It is the larger isolated activation error source on this capture. Product NMSE rises to 0.00470 and frozen-down_proj output NMSE to 0.01272; these different-reference metrics are not additive or a causal E2E attribution. The E2E gate fails on KL and logits NMSE despite improved Top1.

## 9. E2E comparison

| Mode | NLL | PPL | KL | NMSE | MSE | MAE | Cosine | Flat cosine | Top1 | In5 | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 4.9696909 | 143.982376 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| FROZEN_LINEAR_FP_ACT | 4.94016528 | 139.793352 | 0.0565037455 | 0.00311548917 | 1.0542893 | 0.725382106 | 0.997556985 | 0.998526475 | 0.880208333 | 0.994791667 | 0.885416673 |
| FROZEN_LINEAR_SHARED_LUT | 5.05732366 | 157.169314 | 0.0776858066 | 0.00362848278 | 1.22788761 | 0.828319292 | 0.997749682 | 0.998342911 | 0.901041667 | 0.994791667 | 0.857291675 |



Incremental LUT deltas: `{"kl": {"absolute": 0.021182061089840484, "relative": 0.3748788843453587}, "nmse": {"absolute": 0.0005129936031194808, "relative": 0.1646590871245912}, "ppl": {"absolute": 17.37596144074817, "relative": 0.12429748008302022}, "top1": {"absolute": 0.02083333333333326, "relative": 0.023668639053254354}}`.

## 10. FP16 comparison

Not applicable under the supplied direct INT10 contract.

## 11. Gate

Predeclared +5% KL/+5% NMSE/-1pp Top1 gate: **PILOT_REGRESSION**. `{"kl_pass": false, "nmse_pass": false, "top1_pass": true, "passed": false}`.

## 12. Limitations

This emulates future gate output INT10 quantization while gate/up MACs stay FP. Only down_proj is quantized. Selection data are not unseen confirmation; no final adoption is implied.

## 13. Next experiment recommendation

For the next separately authorized experiment, focus on the observed output/table quantization error and the cost of one shared output scale; preserve the frozen Linear parameters. The current shared candidate does not meet the pilot gate. No follow-up was run.
