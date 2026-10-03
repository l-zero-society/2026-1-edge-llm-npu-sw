# Linear Quantization Calibration Policy

## 1. Status

The final/reference Linear quant configuration is `frozen_linear_quant_v1`.

- Canonical parameters: `calibration/frozen_linear_quant_v1/parameters.json`
- Manifest: `calibration/frozen_linear_quant_v1/manifest.json`
- Current scope: only the 18 Gemma `mlp.down_proj` Linear modules

Activation LUT, RMSNorm, and RoPE fixed-point calibration are **not** included in this frozen Linear configuration yet.

The final/reference Linear quant configuration is the L13/L16/L17 cross-layer configuration. It was selected because it passes the predefined fidelity gate and provides a robust logits-NMSE reduction; it is not a strict KL/PPL optimum.

## 2. Quantization numerical contract

For (X\in\mathbb{R}^{M\times K}), (W\in\mathbb{R}^{N\times K}), and (Y=XW^T), weights use symmetric INT8 quantization with one scale per output channel:

\[
s_W[n] = \frac{\max_j |W[n,j]|}{127},\qquad
W_q[n,j] = \operatorname{clip}\left(\operatorname{RNE}\left(\frac{W[n,j]}{s_W[n]}\right),-127,127\right).
\]

Activations use symmetric INT8 quantization with a static operation scale and a runtime power-of-two row correction:

\[
s_X[m]=s_X^{base}2^{k[m]},
\]

\[
X_q[m,j] = \operatorname{clip}\left(
\operatorname{RNE}\left(\frac{X[m,j]}{s_X^{base}2^{k[m]}}\right),-127,127\right).
\]

The integer accumulator is

\[
ACC[m,n]=\sum_j X_q[m,j]W_q[n,j],
\]

with the real interpretation

\[
Y[m,n]\approx ACC[m,n]s_X^{base}2^{k[m]}s_W[n].
\]

Every Linear operation has one common static signed-INT10 output scale (s_{10}). The base per-channel requantization ratio is

\[
r_{base}[n]=\frac{s_X^{base}s_W[n]}{s_{10}}\approx\frac{M[n]}{2^{S[n]}}.
\]

The existing profile represents (M) as UInt16 and (S) as UInt5. Runtime row correction changes only the effective shift:

\[
S_{eff}[m,n]=S[n]-k[m],
\]

\[
q_{10}[m,n]\approx \operatorname{RNE}\left(ACC[m,n]\frac{M[n]}{2^{S[n]-k[m]}}\right),
\qquad -512\le q_{10}\le511,
\]

and the reconstructed Linear output is

\[
\hat{Y}[m,n]=q_{10}[m,n]s_{10}.
\]

Rounding, saturation, weight quantization, accumulator behavior, and multiplier/shift semantics are those implemented by the existing static-quantization profile.

## 3. Runtime row-scale policy

The reference runtime selector is **absmax**. For each activation row,

\[
a_m=\max_j |X[m,j]|,
\]

\[
k_0[m]=\operatorname{RNE}\left(\log_2\frac{a_m}{127s_X^{base}}\right),
\]

with (k_0=0) for an all-zero row. Hardware does not need a floating-point logarithm; threshold comparison, leading-one detection, or equivalent exponent logic can implement the same decision.

The metadata representation is signed 4-bit (k), with nominal range ([-8,+7]). That global range alone is insufficient. Each operation must also satisfy, for every output channel,

\[
0\le S[n]-k\le31.
\]

Consequently the operation-specific feasible interval is

\[
k_{min}^{op}=\max(-8,S_{max}-31),\qquad
k_{max}^{op}=\min(+7,S_{min}).
\]

The runtime selector clips (k_0) to this interval before activation quantization. The final unseen confirmation observed effective shifts from 10 through 29, within the supported ([0,31]) range.

## 4. Why MSE3 was not retained as the hardware selector

The earlier MSE3 software policy formed (k_0), evaluated (k_0-1), (k_0), and (k_0+1) when feasible, and selected the exponent with the lowest activation reconstruction MSE. It remains useful as a software reference, but its three-candidate reconstruction calculation is less attractive for direct hardware.

On identical activation rows in the committed selector diagnostic, absmax and MSE3 agreed on 94.179462% of 187,182 row decisions. The maximum observed |Δk| was 1, and the absmax effective-shift range was 10 through 29. In the subsequent fresh 384-target selector confirmation, absmax reduced logits NMSE by 11.473312% relative to MSE3, increased Top1 agreement by 0.78125 percentage points, and incurred a 5.040395% KL increase, which remained inside the predefined selector gate. These results justified absmax as the hardware-oriented row-scale policy. They do not establish that absmax is globally optimal.

## 5. Static scale calibration policy

Static (s_X^{base}) and (s_{10}) are searched jointly. The stability and cross-layer stages used

\[
s_X(j)=s_X^{old}2^{j/4},\qquad j\in\{-2,-1,0,1,2\},
\]

\[
s_{10}(i)=s_{10}^{old}2^{i/8},\qquad i\in\{-4,-3,-2,-1,0,1,2,3,4\}.
\]

For every Cartesian pair ((s_X,s_{10})), calibration performs the following exact candidate-specific procedure:

1. Approximate (s_Xs_W[n]/s_{10}) with the existing hardware profile.
2. Use that candidate's (M[n]) and (S[n]).
3. Run the absmax row selector using that candidate's (s_X) and (S).
4. Reject any candidate for which (S[n]-k[m]\notin[0,31]).
5. Quantize each row with (s_X2^{k[m]}).
6. Compute the exact integer accumulator with the frozen (W_q).
7. Apply the existing multiplier, effective shift, RNE, and signed-INT10 clamp.
8. Reconstruct \(\hat{Y}=q_{10}s_{10}\).
9. Compare it with the local floating-point output (Y_{FP}=XW^T).

Weight codes and per-channel weight scales remain frozen. Candidate-specific shifts matter because they determine both requantization and the feasible interval of the absmax exponent.

## 6. Local calibration objective

For each prefill or cached-decode group,

\[
NMSE=\frac{\|Y_Q-Y_{FP}\|_2^2}{\|Y_{FP}\|_2^2}.
\]

The balanced local score is

\[
score=\frac{NMSE_{prefill}+NMSE_{decode}}{2},
\]

or the available group's score if only one group exists. The incumbent pair is always evaluated, and selection enforces

\[
score_{selected}\le score_{old}+10^{-12}.
\]

This output-domain objective is used to generate safe local candidates. Experiments showed that local Linear NMSE alone is insufficient for final model-level selection because errors propagate and interact across layers.

## 7. Calibration history and rationale

### 7.1 MSE3-calibrated frozen baseline

The historical static (s_X/s_{10}) baseline was calibrated for the MSE3 general-row software selector. It remains the `old_absmax` incumbent when the same static parameters are executed with the absmax runtime selector.

### 7.2 HW absmax selector comparison

Applying absmax to the old static parameters produced row exponents close to MSE3 and competitive cached-decode logits fidelity. The larger selector confirmation passed its predefined gate, so absmax became the reference runtime policy.

### 7.3 Absmax-aware local joint calibration

Candidate-specific joint ((s_X,s_{10})) search replaced the structurally mismatched sequential policy that first minimized activation reconstruction error for (s_X) and then searched (s_{10}). The corrected policy evaluates both scales together using final Linear-output NMSE and exact absmax semantics.

### 7.4 Stability experiment

Using the larger calibration sample, only 6 of 18 layers exactly matched the small pilot's (s_X) choice, 8 of 18 matched its (s_{10}) choice, and 2 of 18 matched the complete pair. This data sensitivity showed that local minima should not automatically be treated as model-level optima.

### 7.5 Cross-layer E2E selection

The cross-layer experiment started from `old_absmax`. Local joint calibration generated small per-layer candidate pools. A separate E2E selection set drove an 18-layer sensitivity sweep, a six-layer shortlist, and exactly two coordinate-selection passes under KL-primary safety constraints. Seven individual local-best substitutions were E2E-beneficial on the selection set, eight were unsafe, and three were non-beneficial.

The frozen mixed configuration changes only L13, L16, and L17. L13 uses a local-second candidate, L16 uses the local-best/stability candidate, and L17 uses the pilot candidate. Thus some locally best candidates were not the best model-level choice in context. This is practical evidence that cross-layer error interaction matters, not proof of a universal cancellation law.

## 8. Final large unseen confirmation

The configuration was frozen before confirmation-data inspection. The confirmation used 32 genuinely unseen local conversations and 1,024 aligned teacher-forced cached-decode targets. Stable-ID auditing found no overlap with prior calibration, selection, or evaluation examples. Confirmation performed no search, calibration, or parameter changes.

| Mode | NLL | PPL | KL | Logits NMSE | MSE | MAE | Cosine | Flattened cosine | Top1 | FP top1 in Q top5 | Top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 5.361758998 | 213.0994585 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| old_absmax | 5.394216036 | 220.1295059 | 0.06301823473 | 0.003468309273 | 1.187737173 | 0.7474637378 | 0.9988063155 | 0.9983720284 | 0.884765625 | 0.994140625 | 0.8835937567 |
| frozen_cross_layer_absmax | 5.396941908 | 220.7303692 | 0.06391653074 | 0.002992445474 | 1.024775603 | 0.7082693005 | 0.9988679190 | 0.9986114126 | 0.8916015625 | 0.9970703125 | 0.8830078192 |

Relative to `old_absmax`, the frozen cross-layer configuration changed KL by **+1.425454%**, logits NMSE by **-13.720339%**, PPL by **+0.272959%**, and Top1 agreement by **+0.683594 percentage points**.

### 8.1 Sequence-level bootstrap

The committed bootstrap used 2,000 sequence-level resamples with seed 20261004. It required no additional inference.

| Metric | Bootstrap mean | 95% percentile interval |
|---|---:|---:|
| KL relative change | +1.428152% | [-4.714395%, +8.102109%] |
| NMSE relative change | -13.613240% | [-18.947051%, -7.567037%] |
| Top1 delta | +0.663721 pp | [-0.878906, +2.148438] pp |
| PPL relative change | +0.248973% | [-2.492702%, +3.182825%] |

The favorable replicate fractions were 0.3325 for KL improvement, 1.0 for NMSE improvement, and 0.8285 for Top1 non-degradation or improvement. The robust result is the logits-NMSE reduction. KL and PPL do not show a reliable strict improvement. Top1 trends positively, but its interval crosses zero.

## 9. Predefined E2E gate

The fixed gate is

\[
KL_{new}\le1.05KL_{old},\qquad
NMSE_{new}\le1.05NMSE_{old},\qquad
Top1_{new}\ge Top1_{old}-0.01.
\]

The L13/L16/L17 configuration passed all three conditions on the 1,024-target unseen confirmation.

## 10. Final Linear Quantization Decision

The frozen reference configuration is the L13/L16/L17 cross-layer-selected configuration.

It is frozen because:

- the predefined fidelity safety gate passes;
- the large unseen confirmation passes;
- the logits-NMSE reduction is robust under sequence-level bootstrap; and
- Top1 shows no evidence of meaningful degradation.

The 1,024-target aggregate has slightly worse KL and PPL. Therefore, the configuration is a tradeoff and is not a strict optimum across all metrics.

**Final/reference Linear quant configuration = L13/L16/L17 cross-layer configuration, selected because it passes the predefined fidelity gate and provides a robust logits-NMSE reduction; it is not a strict KL/PPL optimum.**

## 11. Scope and non-goals

The frozen scope is 18 `mlp.down_proj` modules with INT8 activations, INT8 per-channel weights, integer accumulation, runtime absmax row exponents, and signed-INT10 Linear outputs.

The following remain outside this freeze:

- activation LUT input/output scale and table construction;
- RMSNorm fixed-point/LUT scale and table;
- RoPE fixed-point sine/cosine LUT scale and table;
- remaining Linear/module quantization, as applicable; and
- full-model hardware numerical validation.

Upstream Linear scales remain frozen while the next LUT stage is calibrated. They should be reopened only if later hardware-faithful LUT, Norm, or RoPE evaluation provides evidence that the frozen Linear contract materially limits end-to-end fidelity.

## 12. File references

| Purpose | Path |
|---|---|
| Canonical frozen parameters | `calibration/frozen_linear_quant_v1/parameters.json` |
| Canonical manifest | `calibration/frozen_linear_quant_v1/manifest.json` |
| Historical old baseline | `diagnostics/final_downproj_ptq/` |
| Joint absmax pilot | `diagnostics/absmax_joint_calibration_pilot/` |
| Larger stability experiment | `diagnostics/absmax_joint_stability/` |
| Cross-layer search | `diagnostics/absmax_cross_layer_calibration/` |
| Final large confirmation | `diagnostics/frozen_cross_layer_confirmation/` |
| Joint calibration implementation | `scripts/absmax_joint_calibration.py` |
| Absmax selector implementation | `scripts/compare_k_selectors_e2e.py` |
