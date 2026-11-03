# INT10 RMSNorm output-scale probe

## 1. Exact hypothesis

A missing output gain and early phase-2 integer truncation may explain the Norm collapse. Accumulator overflow is tested independently. All scales and GPALU parameters are frozen.

## 2. Why the prior debug runner failed

The old runner rejected unrelated staged files at INITIALIZING. This probe never requires an empty index. Scientific DONE is written before isolated-worktree publication; a Git failure only creates PUSH_FAILED.

## 3. Accumulator width verification

Overflow detected: False. Maximum tested square sum 1073741824; required bits 31; configured accumulator 35 bits. Full stage capacities and observed products are in accumulator_verification.json.

## 4. RMSNorm integer scaling derivation

`q=RNE(x/s_in)`, `mean=(sum(q²)*round(2^20/N))>>20`, `epsilon_q=RNE(eps/s_in²)`. The literal rsqrt Q2.14 coefficient is exponent-corrected with truncating shifts. RTL phase 2 is arithmetic `q*scale>>14`, then INT10 saturation.
Correct model output codes need `RNE(q*rsqrt(stat)*gamma/s_out)`. The literal comparison uses `q_literal*s_out*gamma`; gamma is a software adapter outside RTL. POST_GAIN uses the already-truncated literal value, gamma, then RNE/s_out.
Fused ideal retains the exponent-normalized Q2.14 mantissa as a rational. Integer fused uses static per-channel gamma/s_out with sign, UInt16 M, UInt5 S. A wide product q*mantissa*M receives one signed RNE by the combined exponent/fraction/gain shift. This is hypothetical fused hardware; it is not the current core. Unsupported/inaccurate gain factors are recorded, never silently replaced with float.

## 5. Old emulator vs literal RTL

OLD_EMULATOR is reused unmodified. It keeps fractional normalized values, applies gamma, and quantizes with s_out; literal RTL truncates earlier. Previous combined Norm experiments also replaced Softmax, so their 6–8% Top1 is not an isolated RMSNorm precision result.

## 6. Local real-activation measurements

Only actual rows from the existing calibration population are used. Per-module prefill/decode metrics are in norm_local_comparison.csv. Below are unweighted means of per-module/group NMSE, not pooled-energy metrics.

| Mode | Mean local NMSE | Mean output RMS / FP RMS |
|---|---:|---:|
| FP_REFERENCE | 0 | 1 |
| INPUT_QUANT_ORACLE | 0.093144376 | 0.979572027 |
| OLD_EMULATOR | 0.119186381 | 0.893708507 |
| RTL_LITERAL | 0.771001211 | 0.150567502 |
| POST_GAIN_DIAGNOSTIC | 0.400380671 | 1.07027992 |
| FUSED_IDEAL_COEFFICIENT | 0.129635523 | 0.944423991 |
| FUSED_REQUANT_ORACLE | 0.129635685 | 0.944427667 |

## 7. Output-code utilization

output_code_distribution.csv records zero fraction, distinct codes, p1/p50/p99, and maximum magnitude. Literal codes are normalized core codes before the FP gamma adapter; other codes are final INT10 output codes.

## 8. Missing-gain vs early-truncation decomposition

scaling_trace.json contains pairwise measured errors: input/output quantization, LUT/fixed point, literal-to-fused, post-gain-to-literal, old-to-fused, and integer M/S-to-ideal fused. Input/output oracle includes final output quantization, so it is not a pure input-only error. Gamma and clipping are included in these paths; this is not a separate gamma-storage hardware experiment.

## 9. Small 24-token E2E results

Three fixed conversations × eight targets from the existing E2E set, disjoint from calibration by anchor/response/tree IDs. Every mode owns its cache; Softmax and RoPE stay FP. These 24-token percentages must not be compared directly with the old 192-token 85.42% Top1.

| Mode | nll | ppl | kl | nmse | mae | cosine | top1 | in5 | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_FULL | 5.47776778 | 239.311914 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| GPALU10_FP_NORM | 5.50725462 | 246.473534 | 0.0485121907 | 0.00251962348 | 0.749959426 | 0.999284224 | 0.958333333 | 1 | 0.816666678 |
| GPALU10_OLD_NORM | 8.89614154 | 7303.73789 | 6.50651996 | 0.213550134 | 7.06821372 | 0.965658001 | 0.0833333333 | 0.25 | 0.108333336 |
| GPALU10_RTL_LITERAL_NORM | 91.4399708 | 5.15080048e+39 | 89.5445502 | 6.49119391 | 45.5369976 | -0.920854083 | 0 | 0 | 0 |
| GPALU10_POST_GAIN_NORM | 12.3677734 | 235101.688 | 10.1933918 | 0.198308475 | 6.54553471 | 0.950139308 | 0 | 0.0833333333 | 0.0500000007 |
| GPALU10_FUSED_REQUANT_NORM | 9.35001911 | 11499.0432 | 6.61441192 | 0.298950386 | 8.1120663 | 0.964090098 | 0.0833333333 | 0.208333333 | 0.125000002 |

## 10. Primary diagnosis

MIXED_OR_INCONCLUSIVE
Supported descriptive cases: OUTPUT_SCALE_CONTRACT_CONFIRMED, EARLY_TRUNCATION_IS_DOMINANT, INT10_PRECISION_LIMIT_SUSPECTED.
{"input_oracle_nmse_above_5pct": true, "literal_amplitude_below_half_oracle": true, "local_fused_reduces_literal_error_50pct": true, "post_gain_error_more_than_twice_fused": true, "small_e2e_fused_improves_literal": true}

## 11. What this establishes

Q1. Overflow detected in tested boundary/actual vectors: False.
Q2. Literal output amplitude ratio is 0.150567502, compared with 0.979572027 for the proper INT10 oracle; local NMSE is 0.771001211.
Q3. Fused gain before requantization gives local NMSE 0.129635685; ideal fused 0.129635523. E2E evidence is limited to the table above.
Q4. Post-gain local NMSE 0.400380671, RMS ratio 1.07027992, versus fused NMSE 0.129635685; this quantifies what cannot be recovered after truncation.
Q5. The INT10 input/output oracle NMSE 0.093144376 is evaluated separately from the literal implementation. The measured diagnosis is MIXED_OR_INCONCLUSIVE; the former combined Norm/Softmax result alone cannot identify the cause.

## 12. What remains unverified

This is a tiny CPU numerical probe, not RTL co-simulation or synthesis. Fused coefficient banking, gamma storage, combined shifts/intermediate widths, compiler integration, and wider unseen validation remain unverified. Adding one gain is not guaranteed to fix production RTL. No E2E metrics selected coefficients or scales; no follow-up experiment is run automatically.
