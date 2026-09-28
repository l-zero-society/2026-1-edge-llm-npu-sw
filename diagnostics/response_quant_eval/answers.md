## Findings

General-row PoT preserves FP response behavior much better than BOS-only: response KL falls 85.91%, logit NMSE falls 76.97%, and top-1 agreement rises from 68.20% to 86.55%. Its human-reference response PPL is nevertheless higher: 59.10 versus BOS-only 36.27, and 7.32% above FP 55.07. Human-reference likelihood and preservation of the FP distribution are different objectives. The lower BOS-only PPL does not establish better preservation or better general instruction-following quality.

This supersedes raw full-sequence PPL as the main metric for this diagnostic, without modifying previous results. No semantic evaluation or end-to-end benchmark quality claim is made.

## First assistant token and response depth

Stage 1 performs a separate prompt-only forward, including the native assistant prefix and excluding all response content. Its last predictor position scores the first actual content token.

| mode | first-token NLL | KL | logit NMSE % | cosine | top1 % | FP top1 in Q top5 % | top5 overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | 11.98749 | 0 | 0 | 1 | 100 | 100 | 100 |
| BOS-only | 10.58812 | 0.44203 | 1.53673 | 0.995853 | 66.67 | 100 | 83.33 |
| General-row | 11.54268 | 0.05342 | 0.79472 | 0.998434 | 87.50 | 100 | 90.83 |

The first-response normalized hidden state is exactly equal between prompt-only and full teacher-forcing forwards for every example/mode. Tiny NLL differences between Stage 1 and Stage-2 bucket 0 arise in head evaluation at different batch sizes; the masks identify the same target.

| response index | targets | FP NLL | BOS-only NLL | General NLL | BOS-only KL | General KL |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 11.98750 | 10.58812 | 11.54268 | 0.44202 | 0.05342 |
| 1 | 24 | 8.08462 | 7.54678 | 8.21068 | 0.64625 | 0.11594 |
| 2–3 | 48 | 5.05147 | 4.59549 | 5.23453 | 0.53859 | 0.07748 |
| 4–7 | 96 | 4.52252 | 3.96169 | 4.55673 | 0.66300 | 0.07927 |
| 8–15 | 192 | 4.28513 | 3.81736 | 4.34371 | 0.69587 | 0.08088 |
| 16–31 | 384 | 4.85410 | 4.29598 | 4.90672 | 0.69535 | 0.08460 |
| >=32 | 13,862 | 3.95328 | 3.54331 | 4.02528 | 0.58121 | 0.08254 |

Distortion is already measurable at the first content token. General-row KL then settles near 0.08; there is no monotonic growth with response depth. Target difficulty and example populations differ across buckets, so this is not a causal isolation of accumulation. The first-token NLL remains high even after masking: real response starts, including unusual word pieces, can have low reference-model probability. This is not a prompt/header loss.

## Cached teacher forcing

Completed a bounded deployment-path check: first 16 valid prompts, first 32 response targets each, 512 identical targets. Each mode uses its own propagated KV cache and absolute token positions. These short-window PPLs must not be compared directly with the full-response PPLs above.

| mode | matched full NLL | cached NLL | matched full PPL | cached PPL | cached KL vs FP | cached/full top1 agreement % |
|---|---:|---:|---:|---:|---:|---:|
| FP | 5.567694 | 5.567692 | 261.82952 | 261.82915 | 0 | 100 |
| BOS-only | 5.034550 | 5.021796 | 153.63040 | 151.68341 | 0.669282 | 91.21 |
| General-row | 5.623932 | 5.612940 | 276.97645 | 273.94838 | 0.083284 | 90.82 |

FP cache parity is excellent: maximum absolute per-target NLL difference is 0.0000871. Quantized paths are not identical to full teacher forcing: maximum per-target differences are 1.47737 (BOS-only) and 2.48913 (General). Aggregate PPL differs by about 1.27% and 1.09%, respectively, while about 9% of top-1 predictions change. This execution-shape sensitivity is a remaining numerical concern. Amplification of small FP differences by quantization thresholds is a plausible explanation, not a measured attribution. The absolute-position BOS rule is tested; decoded tokens do not receive a false position-0 correction.

## Greedy continuation

Completed all 16 prompts, greedy argmax, maximum 32 tokens, existing generation EOS ID 1. Comparisons use separate, untouched prompt caches; no ground-truth response is fed during generation.

| mode | mean shared prefix | median shared prefix | aligned token agreement % | exact 32-token matches | mean generated length | EOS observed |
|---|---:|---:|---:|---:|---:|---:|
| FP | 32 | 32 | 100 | 16/16 | 32 | 0/16 |
| BOS-only | 4.5625 | 1.5 | 18.36 | 0/16 | 31.5625 | 2/16 |
| General-row | 11.6875 | 6.5 | 45.90 | 2/16 | 31.1250 | 3/16 |

General-row retains a longer exact prefix on 8 prompts, ties on 7, and is shorter on 1. First divergence is zero-indexed; for divergent sequences it equals shared-prefix length. Exact 32-token matches are censored beyond the generation limit. Aligned agreement divides by the longer generated sequence length, counting missing tokens as disagreements. Mean length differences versus FP are -0.4375 and -0.875 tokens. No FP sequence reaches EOS within 32 tokens, so all EOS-step differences are unobservable/censored, not zero. Full per-prompt results are in response_quant_eval_generation.csv.

## Response-position hidden states

| mode | first layer >1% | first layer >5% | worst layer | layer-17 hidden NMSE % |
|---|---:|---:|---:|---:|
| BOS-only | 0 | 1 | 17 | 13.88943 |
| General-row | 1 | none | 17 | 2.39725 |

Layer indices are zero-based. General-row hidden NMSE rises from 1.67141% at layer 16 to 2.39725% at layer 17. Layer 17 remains the largest cumulative response-position error, but this does not isolate its local down_proj as the sole cause. All 18 layers, cosine similarity and both first-token/response stages are in response_quant_eval_layers.csv.

Across 263,340 response predictor rows (14,630 × 18), all are non-BOS. General-row k != 0 on 46.44%; k ranges from -2 to 4, median 0. Counts: -2:864, -1:25,632, 0:141,048, 1:79,198, 2:15,069, 3:1,497, 4:32. Per-layer counts are in the summary JSON. No allowed-k-bound hit occurred.

## Scope, reproducibility and checks

- Reused local Q8_0 GGUF, existing INT8 down_proj weights and frozen calibration-only global w_BOS=0.25 selection. Only `model.layers.0..17.mlp.down_proj` is replaced. No model download, calibration activation capture or parameter search occurred.
- Recovered 24 of 32 validation anchors from the existing ready-tree OASST archive. Eight anchors are absent locally and were skipped, not synthesized. For three user anchors, a valid human assistant child was selected by dataset rank then message ID. Ancestor histories are preserved and calibration/validation trees are disjoint.
- 9,655 prompt tokens, 14,630 response content targets, 24,333 physical tokens including 48 unscored closing-template tokens. Maximum full chat length is 3,174, within model context 8,192. This differs from the old raw-message/512-token evaluation and calibration length; the numerical parameters stay frozen. Old raw PPL and new response PPL are not directly comparable populations.
- Native local GGUF `tokenizer.chat_template` is rendered/tokenized through `apply_chat_template`. Token offsets, exact prompt-prefix equality and special-token rejection verify content-only masks. The first content token is scored from the final prompt position. All modes share identical ground-truth tokens and masks.
- Each forward asserts all 17 transformer propagation boundaries. No FP reference activation is reinjected. Nonlinear operations and every other Linear, including lm_head, remain FP. The profile is existing generic-rne software semantics; no claim of verified RTL bit-exactness is made.
- 52 relevant tests passed, zero failures: response evaluation 8, row-PoT 6, E2E Linear 4, factorized math 6, static_quant 22, s10 search 6. Tests include masks/first content, score math, original quant-path equality, absolute cached BOS position, unchanged channel parameters, bucket coverage and generation censoring.
- Post-run audit verified all population counts, identical masks, complete stages, 61 immutable source/artifact/data hashes, tokenization hash and unchanged tracked files. No production, calibration, previous diagnostic, RTL/LUT/QB/ABI changes. See `response_quant_eval/verification.json` and `tests.log`.
- Commit `8770c9d533f18ee2a811fca90b3da734b4408755`; exact command, local paths, model/artifact hashes and library/profile versions are recorded in `response_quant_eval/commands.md` and `provenance.json`. Compact sufficient statistics permit report regeneration without model execution. An initial pre-forward subprocess-test startup failure was resolved by running the same tests in-process; all reported evaluation stages completed successfully.

## Answers A–H

A. General-row worsens response-only PPL versus BOS-only: 59.10 versus 36.27 (+62.95%).

B. General-row is close to FP in response PPL: 59.10 versus 55.07 (+7.32%, NLL +0.07068).

C. Yes. General-row improves KL/NMSE/top-k substantially: KL 0.08254 versus 0.58599, logit NMSE 1.131% versus 4.911%, top1 86.55% versus 68.20%.

D. Error already exists at the first token; General-row KL is 0.05342 there and approximately 0.08 later. The data do not show monotonic runaway and cannot assign a causal fraction to prefill versus response accumulation.

E. Yes, quantized cached and full teacher-forced outputs differ; FP parity is effectively exact. Matched-window cached PPL is slightly lower, but about 9% of quantized top1 predictions change. Cache/full parity needs separate numerical localization.

F. Yes on aggregate: General-row mean shared prefix is 11.69 tokens versus 4.56, with 45.90% versus 18.36% aligned agreement. It is not better on every prompt.

G. Layer 17 remains the worst cumulative response-position hidden NMSE: 2.397% General-row versus 13.889% BOS-only. This is localization, not proof of a single local mechanism.

H. General-row remains a reasonable architecture candidate for preserving FP response behavior, supported by distribution and short greedy results. Remaining cache/full sensitivity, reference-response PPL gap, limited examples and short generation prevent an RTL adoption conclusion. No final hardware decision is made.
