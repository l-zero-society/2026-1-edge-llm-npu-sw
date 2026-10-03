Fixed-Point LUT Calibration Roadmap & Activation LUT Calibration Guide
1. Purpose
현재 Linear quantization은 다음 configuration으로 freeze되어 있다.
frozen_linear_quant_v1

Canonical parameter:
calibration/frozen_linear_quant_v1/parameters.json

현재 frozen Linear configuration의 numerical contract는 다음과 같다.
\[
X_q[m,j]
=
\operatorname{RNE}
\left(
\frac{X[m,j]}
{s_X^{base}2^{k[m]}}
\right)
\]
\[
ACC[m,n]
=
\sum_j X_q[m,j]W_q[n,j]
\]
\[
q_{10}[m,n]
\approx
ACC[m,n]
\frac{M[n]}{2^{S[n]-k[m]}}
\]
\[
\hat Y[m,n]
=
q_{10}[m,n]s_{10}
\]
Runtime Row Scale은 HW-friendly absmax policy를 사용한다.
현재 단계부터는 이 Linear numerical contract를 upstream frozen baseline으로 유지하면서 Activation, RoPE, RMSNorm 순으로 실제 HW fixed-point/LUT numerical path를 추가한다.
2. Recommended Calibration Order
권장 순서는 다음과 같다.
Frozen Linear Quantization
        │
        ▼
Activation LUT Calibration
        │
        ▼
Activation LUT Freeze
        │
        ▼
RoPE LUT / Fixed-Point Calibration
        │
        ▼
RoPE Freeze
        │
        ▼
RMSNorm LUT / Fixed-Point Calibration
        │
        ▼
RMSNorm Freeze
        │
        ▼
Block-Level Numerical Validation
        │
        ▼
Full E2E Validation
        │
        ▼
Upstream Scale Reopen Decision

즉,
\[
\boxed{
Linear
\rightarrow
Activation
\rightarrow
RoPE
\rightarrow
RMSNorm
\rightarrow
E2E
}
\]
순서를 기본으로 한다.
3. Why Calibration Should Be Sequential
Activation, RoPE, RMSNorm을 한 번에 fixed-point화하면 최종 error를 다음처럼 분해하기 어려워진다.
\[
E_{\text{total}}
=
E_{\text{Linear}}
+
E_{\text{Activation}}
+
E_{\text{RoPE}}
+
E_{\text{Norm}}
+
E_{\text{interaction}}
\]
예를 들어 E2E logits NMSE가 증가했을 때 그 원인이
- Linear quantization
- Activation LUT approximation
- LUT input clipping
- RoPE sin/cos precision
- RMSNorm rsqrt approximation
- 각 unit 사이 scale mismatch
중 무엇인지 확인하기 어렵다.
따라서 각 subsystem을 순차적으로 추가하고, 각 단계에서 error contribution을 분리한다.
4. Common LUT Numerical Interface
Activation / RoPE / Norm을 calibration하기 전에 공통 LUT numerical interface를 정의하는 것이 좋다.
각 LUT profile은 최소 다음 정보를 가져야 한다.
LUTProfile
├── function
├── input_dtype
├── input_scale
├── input_zero_point
├── input_domain_min
├── input_domain_max
├── address_bits
├── address_mapping
├── table_depth
├── entry_dtype
├── output_scale
├── interpolation_mode
├── rounding_mode
└── saturation_rule

예를 들어:
Activation LUT
    function        = GeLU
    input           = INT10
    input_scale     = s_act_in
    output          = INT10
    output_scale    = s_act_out

Norm LUT
    function        = rsqrt
    input           = fixed-point variance
    output          = fixed-point reciprocal sqrt

RoPE LUT
    function        = sin / cos
    input           = phase / index
    output          = fixed-point coefficient

HW block은 가능하면 동일한 LUT infrastructure를 공유하되, calibration parameter는 각 function마다 독립적으로 가진다.
5. Activation LUT Calibration Scope
첫 번째 LUT calibration 단계에서는 Activation만 변경한다.
다음 항목은 freeze한다.
sX_base
runtime row-scale policy
runtime k
Wq
sW
M
S
s10
Linear output INT10 format

즉,
\[
\boxed{
s_X,\;RS,\;s_{10}
}
\]
은 Activation LUT calibration 중 optimization variable이 아니다.
Activation에서 새로 최적화하는 것은 다음이다.
\[
\boxed{
s_{\text{act,in}},
\quad
s_{\text{act,out}},
\quad
\text{LUT address mapping},
\quad
\text{LUT table entries}
}
\]
필요하다면 추가적으로:
- LUT depth
- LUT entry bit-width
- input clipping domain
- interpolation 여부
를 architecture candidate로 비교할 수 있다.
6. Activation Numerical Model
FP activation을
\[
y=f(x)
\]
라고 하자.
현재 frozen Linear output은 signed INT10이다.
\[
q_{10}\in[-512,511]
\]
그리고
\[
x_Q=q_{10}s_{10}
\]
으로 reconstruction된다.
Activation LUT hardware path를 일반화하면:
\[
q_{\text{in}}
=
Q_{\text{LUT,in}}
\left(
\frac{x_Q}{s_{\text{act,in}}}
\right)
\]
LUT address:
\[
a=A(q_{\text{in}})
\]
LUT table:
\[
T[a]
\]
그리고 LUT output reconstruction은
\[
\hat y_{\text{LUT}}
=
q_{\text{out}}s_{\text{act,out}}
\]
이다.
따라서 실제 HW approximation은
\[
\boxed{
f_{\text{HW}}(x_Q)
=
s_{\text{act,out}}
\cdot
T\left(
A\left(
Q_{\text{LUT,in}}
\left(
\frac{x_Q}{s_{\text{act,in}}}
\right)
\right)
\right)
}
\]
로 볼 수 있다.
7. First Question: Is a Separate LUT Input Scale Necessary?
첫 calibration에서 반드시 비교해야 할 두 architecture가 있다.
Case A — Reuse Linear Output Scale
\[
\boxed{
s_{\text{act,in}}=s_{10}
}
\]
즉 Linear INT10 output을 별도 requantization 없이 LUT에 바로 넣는다.
장점:
- extra rescale hardware 불필요
- latency 감소
- compiler contract 단순화
- activation fusion 용이
- parameter 수 감소
Data path:
ACC
 ↓
QuantAct
 ↓
INT10 @ s10
 ↓
Activation LUT

가 된다.
이 방식이 fidelity상 충분하다면 가장 선호해야 한다.
Case B — Independent LUT Input Scale
\[
s_{\text{act,in}}\neq s_{10}
\]
이라면 LUT 전에
\[
q_{\text{LUT,in}}
=
Q
\left(
q_{10}
\frac{s_{10}}{s_{\text{act,in}}}
\right)
\]
형태의 rescale이 필요하다.
장점:
- nonlinear sensitive region에 더 많은 resolution 배치 가능
단점:
- extra multiplier/shift
- extra rounding error
- hardware/control complexity 증가
- compiler optimization dimension 증가
따라서 Case A를 baseline으로 먼저 평가하고, Case B가 충분한 fidelity improvement를 제공할 때만 채택한다.
8. Activation LUT Error Decomposition
Activation LUT calibration에서는 반드시 error를 분리해서 측정한다.
FP Linear output을
\[
Y_{FP}
\]
현재 frozen quantized Linear output을
\[
Y_Q
\]
라고 한다.
8.1 Existing Linear Error After Ideal Activation
\[
E_{\text{linear→act}}
=
NMSE
\left(
f(Y_Q),
f(Y_{FP})
\right)
\]
이 metric은 LUT approximation이 전혀 없을 때도 존재한다.
즉 frozen Linear quantization error가 nonlinear function을 통과하면서 얼마나 증폭되는지를 측정한다.
8.2 Pure LUT Error
\[
E_{\text{LUT}}
=
NMSE
\left(
f_{\text{LUT}}(Y_Q),
f(Y_Q)
\right)
\]
이 metric은 같은 quantized Linear input에 대해:
Ideal FP activation
vs
Actual LUT activation

을 비교한다.
따라서 LUT 자체가 추가한 error를 직접 측정한다.
8.3 Combined Error
\[
E_{\text{combined}}
=
NMSE
\left(
f_{\text{LUT}}(Y_Q),
f(Y_{FP})
\right)
\]
이 값이 실제 HW path에서 최종적으로 발생하는 activation output error다.
따라서 calibration report에는 최소 다음 세 값이 있어야 한다.
Ideal activation on FP Linear output
Ideal activation on quantized Linear output
Actual LUT activation on quantized Linear output

9. Activation Functions to Calibrate
각 activation function마다 별도의 LUT profile을 만드는 것을 기본으로 한다.
현재 model/runtime에서 실제 사용되는 activation을 우선한다.
예:
GeLU
SiLU
exp
sigmoid

현재 NPU의 MLP activation path가 GeLU 계열이라면 GeLU부터 먼저 calibration한다.
사용하지 않는 function에 대해서 미리 LUT를 최적화할 필요는 없다.
10. LUT Table Construction
가장 단순한 LUT table construction은 다음과 같다.
LUT address에 대응하는 representative real input을
\[
x_a
\]
라고 하면,
\[
T[a]
=
\operatorname{clip}
\left(
\operatorname{RNE}
\left(
\frac{f(x_a)}
{s_{\text{act,out}}}
\right)
\right)
\]
이다.
따라서 table construction은 크게 다음 두 parameter에 의해 결정된다.
\[
s_{\text{act,in}}
\]
\[
s_{\text{act,out}}
\]
11. Input Domain Selection
LUT 전체 dynamic range를 단순히 theoretical INT10 range로 결정하지 말고 실제 activation distribution도 같이 봐야 한다.
Frozen Linear output distribution에 대해 다음을 수집한다.
min
max
absmax
mean
std
percentiles
p99
p99.9
p99.99
clipping rate candidate

특히 nonlinear function의 important region을 확인한다.
예를 들어 GeLU는 0 주변 transition region의 resolution이 중요하다.
따라서 단순 uniform real-domain mapping:
\[
[-512s_{10},511s_{10}]
\]
이 반드시 최적은 아니다.
12. LUT Address Mapping Candidates
첫 번째 implementation에서는 복잡한 mapping보다 단순한 것부터 시작한다.
Candidate 1 — Uniform Quantized-Input Mapping
INT10 또는 subset을 address로 직접 사용.
q10
 ↓
truncate / shift
 ↓
LUT address

가장 HW-friendly하다.
Candidate 2 — Uniform Real-Domain Mapping
\[
a
=
\operatorname{round}
\left(
\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
(D-1)
\right)
\]
여기서 \(D\)는 LUT depth.
구현은 조금 더 복잡하지만 calibration flexibility가 높다.
Candidate 3 — Piecewise / Nonuniform Mapping
0 근처에 더 많은 entries를 배치하는 방식.
예:
large negative region : coarse
transition region     : dense
large positive region : coarse

이 방식은 fidelity가 개선될 수 있지만 HW address-generation logic이 복잡해진다.
따라서 첫 calibration에서는:
\[
\boxed{
\text{uniform mapping first}
}
\]
를 권장한다.
Nonuniform mapping은 uniform LUT가 명확히 부족할 때만 고려한다.
13. LUT Output Scale Calibration
LUT output scale은 다음 tradeoff를 가진다.
작은 \(s_{\text{act,out}}\):
- fine resolution
- clipping risk 증가
큰 \(s_{\text{act,out}}\):
- clipping 감소
- quantization step 증가
기본 absmax-style initializer는
\[
s_{\text{out},0}
=
\frac{\max|f(x)|}
{Q_{\max}}
\]
으로 만들 수 있다.
하지만 outlier 때문에 resolution이 낭비될 수 있으므로 percentile-based candidate도 평가한다.
예:
\[
s_{\text{out}}(p)
=
\frac{P_p(|f(x)|)}
{Q_{\max}}
\]
candidate percentile:
99.0%
99.5%
99.9%
99.95%
99.99%
100%

단, final 선택은 percentile 자체가 아니라 실제 reconstruction metric으로 한다.
14. LUT Input Scale Candidate Search
Case B를 평가할 경우:
\[
s_{\text{act,in}}
=
s_{10}2^\delta
\]
같은 제한된 power-of-two search를 우선한다.
예:
\[
\delta
\in
\left\{
-\frac12,
-\frac14,
0,
+\frac14,
+\frac12
\right\}
\]
또는 HW-friendly하게 integer shift만 허용한다면:
\[
\delta\in\{-1,0,+1\}
\]
를 사용할 수 있다.
가능하면 arbitrary floating rescale보다 power-of-two relation을 선호한다.
15. LUT Calibration Objective
처음에는 local activation reconstruction을 objective로 사용한다.
FP reference:
\[
Y_{FP}^{act}
=
f(Y_{FP}^{Linear})
\]
HW reconstruction:
\[
Y_{HW}^{act}
=
f_{LUT}(Y_Q^{Linear})
\]
Primary local metric:
\[
\boxed{
NMSE_{act}
=
\frac{
\|Y_{HW}^{act}-Y_{FP}^{act}\|_2^2
}{
\|Y_{FP}^{act}\|_2^2
}
}
\]
추가 metric:
- MSE
- MAE
- cosine similarity
- clipping rate
- saturation count
- LUT input domain hit rate
- output code utilization
을 기록한다.
16. Do Not Immediately Optimize E2E
Activation LUT candidate 하나마다 full model inference를 돌리면 search cost가 지나치게 커진다.
따라서 다음 2-stage 구조를 사용한다.
Stage A — Local Candidate Generation
Activation tensor capture를 이용하여:
s_act_in
s_act_out
LUT depth
mapping
table

candidate를 빠르게 평가한다.
Local activation NMSE 기준으로 candidate를 줄인다.
예:
top 3~5 candidates

Stage B — E2E Candidate Selection
shortlist에 대해서만 실제 model-level evaluation을 수행한다.
E2E metric:
- logits KL
- logits NMSE
- Top1 agreement
보조:
- NLL
- PPL
- cosine
- top5 overlap
을 사용한다.
17. Recommended E2E Selection Policy
Linear calibration에서 사용한 경험을 그대로 적용한다.
Local optimum을 그대로 final configuration으로 사용하지 않는다.
즉:
\[
\boxed{
\text{Local LUT NMSE minimum}
\neq
\text{E2E optimum}
}
\]
일 수 있다.
E2E에서는 기존 frozen Linear + ideal activation 또는 current reference implementation을 baseline으로 잡는다.
Primary objective:
\[
\min KL
\]
Safety constraints:
\[
NMSE_{new}
\le
1.05 NMSE_{baseline}
\]
\[
Top1_{new}
\ge
Top1_{baseline}-0.01
\]
등 기존 framework를 재사용할 수 있다.
단, 정확한 gate threshold는 Activation LUT experiment 전에 고정하고 결과를 본 뒤 변경하지 않는다.
18. Activation LUT Baselines
최소 다음 세 mode를 비교하는 것을 권장한다.
Mode 0 — FP Reference
FP Linear
→ FP Activation

Mode 1 — Frozen Linear + Ideal Activation
Frozen Quant Linear
→ dequantized real value
→ FP Activation

Mode 2 — Frozen Linear + Actual LUT Activation
Frozen Quant Linear
→ INT10
→ HW LUT
→ quantized activation output

필요하면 기존 prototype:
Mode 3 — Previous FP16 Expansion LUT
Frozen Quant Linear
→ previous FP16-expansion LUT implementation

도 comparison reference로 넣을 수 있다.
이렇게 해야 새로운 LUT가 기존 FP16 expansion implementation보다 실제로 얼마나 나은지 확인할 수 있다.
19. Important Metrics for Activation LUT
각 candidate에 대해 최소 다음을 기록한다.
Metric	Meaning
Activation local NMSE	FP activation output reconstruction error
Pure LUT NMSE	LUT 자체가 추가한 error
Combined NMSE	Linear + LUT 전체 error
MAE	평균 절대 activation error
Cosine	activation vector direction preservation
Input clip rate	LUT input domain 밖으로 나가는 비율
Output clip rate	output quant saturation 비율
Code utilization	output quant code 활용 정도
Logits NMSE	model-level numerical fidelity
KL	model output distribution fidelity
Top1 agreement	FP decision preservation
NLL/PPL	language-model task metric


20. Initial Search Space Recommendation
첫 실험에서 search space를 너무 크게 만들지 않는다.
권장 initial configuration:
Input scale:
    first try s_act_in = s10

Output scale:
    percentile / scale candidates

LUT depth:
    fixed to current hardware target

Entry width:
    fixed to intended RTL width

Address mapping:
    uniform

Interpolation:
    disabled

Rounding:
    existing RNE policy

Saturation:
    target output datatype saturation

즉 처음에는 사실상
\[
\boxed{
s_{\text{act,out}}
}
\]
와 table만 주로 calibration한다.
21. Progressive Complexity
초기 LUT가 충분히 정확하지 않을 때만 search dimension을 하나씩 연다.
권장 순서:
Step 1
s_act_in = s10
uniform mapping
fixed depth
→ optimize s_act_out + table

Step 2
allow limited s_act_in search

Step 3
increase LUT depth / entry precision if needed

Step 4
evaluate interpolation

Step 5
consider nonuniform/piecewise mapping

즉 가장 단순한 HW implementation이 fidelity requirement를 만족하면 거기서 멈춘다.
22. Activation LUT Freeze Criterion
Activation LUT를 freeze하기 위해서는 최소 다음을 만족해야 한다.
Numerical
- local LUT error가 acceptable
- clipping rate sufficiently small
- output code utilization reasonable
- effective hardware arithmetic fully representable
Model-level
- predefined E2E gate pass
- larger held-out confirmation에서 catastrophic regression 없음
Engineering
- table generation deterministic
- table binary reproducible
- scale/table manifest 존재
- LUT-off path에서 frozen Linear regression test 유지
23. Activation LUT Frozen Artifact
Activation calibration이 완료되면 다음과 같은 canonical artifact를 만드는 것이 좋다.
calibration/
├── frozen_linear_quant_v1/
│   ├── parameters.json
│   └── manifest.json
│
└── frozen_activation_lut_v1/
    ├── parameters.json
    ├── manifest.json
    ├── gelu_lut.bin
    └── gelu_lut.json

예시 manifest:
{
  "name": "frozen_activation_lut_v1",
  "status": "FROZEN_REFERENCE",
  "upstream_linear": "frozen_linear_quant_v1",
  "function": "gelu",
  "input_dtype": "INT10",
  "input_scale_policy": "reuse_linear_s10",
  "output_dtype": "INT10",
  "output_scale": "...",
  "table_depth": "...",
  "entry_bits": "...",
  "address_mapping": "uniform",
  "rounding": "RNE",
  "saturation": "signed",
  "table_sha256": "..."
}

24. RoPE Calibration — Next Stage
Activation LUT freeze 이후 RoPE를 진행한다.
RoPE는:
\[
x'_0=x_0\cos\theta-x_1\sin\theta
\]
\[
x'_1=x_0\sin\theta+x_1\cos\theta
\]
이므로 주요 calibration 대상은:
angle representation
sin LUT
cos LUT
sin/cos output scale
coefficient bit-width
multiply intermediate width
RoPE output scale

이다.
이상적인 RoPE는 norm-preserving rotation이므로:
\[
\|R(\theta)x\|_2=\|x\|_2
\]
를 diagnostic으로 사용할 수 있다.
RoPE에서는 다음 error를 확인한다.
\[
NMSE(
RoPE_{HW}(X_Q),
RoPE_{FP}(X_Q)
)
\]
그리고 실제 attention Q/K downstream effect를 별도로 확인한다.
25. RMSNorm Calibration — Final LUT Stage
RMSNorm은 가장 coupling이 강하므로 마지막에 진행하는 것을 권장한다.
\[
RMS(x)
=
\sqrt{
\frac1N\sum_jx_j^2+\epsilon
}
\]
\[
y_i
=
\frac{x_i}{RMS(x)}\gamma_i
\]
필요한 numerical parameter는 예를 들면:
norm input scale
square intermediate format
sum/accumulator width
mean scaling
rsqrt LUT input scale
rsqrt LUT output scale
rsqrt table
normalization multiply format
gamma format
norm output scale

이다.
RMSNorm에서는 pointwise LUT보다 reduction error와 rsqrt approximation이 같이 들어가므로 별도의 detailed calibration이 필요하다.
26. When to Reopen Frozen Linear Scales
Activation / RoPE / Norm이 추가됐다고 해서 자동으로
\[
s_X,\quad s_{10},\quad RS
\]
를 다시 calibration하지 않는다.
Frozen Linear scale은 다음 조건에서만 reopen한다.
Condition 1
LUT/Norm/RoPE 자체의 standalone approximation error는 충분히 작지만 E2E degradation이 크다.
Condition 2
Sensitivity analysis에서 소폭의 \(s_{10}\) 변경이 명확한 E2E improvement를 만든다.
Condition 3
현재 frozen Linear output scale 때문에 LUT domain/code utilization이 구조적으로 비효율적이다.
Condition 4
HW numerical contract상 unavoidable clipping/underutilization이 발생한다.
이 경우에도 우선순위는:
\[
\boxed{
s_{10}
\rightarrow
s_X
\rightarrow
RS
}
\]
로 둔다.
전체 parameter를 한 번에 다시 열지 않는다.
27. Final Joint Numerical Validation
Activation, RoPE, RMSNorm까지 freeze된 이후 최종적으로 다음 전체 path를 validation한다.
Input activation
    ↓
Row Scale / Input Quant
    ↓
Linear / MXU
    ↓
INT10 requant
    ↓
Activation LUT
    ↓
Residual / vector operation
    ↓
RMSNorm
    ↓
RoPE
    ↓
Next Linear / Attention
    ↓
...
    ↓
Final logits

이 단계에서 처음으로 subsystem interaction을 본다.
주요 E2E metric:
\[
KL
\]
\[
Logits\ NMSE
\]
\[
Top1\ Agreement
\]
보조 metric:
\[
NLL,\quad PPL,\quad Cosine,\quad Top5
\]
을 사용한다.
28. Immediate Next Task
현재 바로 진행할 작업은 다음으로 제한한다.
1. frozen_linear_quant_v1을 upstream baseline으로 load

2. 실제 activation input/output tensor capture

3. activation distribution 분석

4. common LUT numerical interface 정의

5. 우선 s_act_in = s10으로 고정

6. LUT output scale 후보 생성

7. LUT table 생성

8. local activation NMSE 측정

9. FP activation / ideal-on-quant / actual-LUT error decomposition

10. small E2E candidate comparison

11. large held-out confirmation

12. frozen_activation_lut_v1 생성

초기 단계에서는 다음 parameter를 절대 변경하지 않는다.
sX_base
runtime RS policy
k selector
s10
M
S
Wq

29. Guiding Principle
앞으로의 calibration 원칙은 다음으로 정리한다.
\[
\boxed{
\text{Freeze upstream}
\rightarrow
\text{optimize one numerical subsystem}
\rightarrow
\text{measure its isolated error}
\rightarrow
\text{validate E2E}
\rightarrow
\text{freeze}
}
\]
그리고 upstream scale은 downstream unit 때문에 필요성이 실제로 증명된 경우에만 다시 연다.
최종 목표는 단순히 가장 낮은 quantization error가 아니라,
\[
\boxed{
\text{HW가 단순하면서도 compiler가 안정적으로 결정할 수 있는 numerical contract}
}
\]
를 만드는 것이다.