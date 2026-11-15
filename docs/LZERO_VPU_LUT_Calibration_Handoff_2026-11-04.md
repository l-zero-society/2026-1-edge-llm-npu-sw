# L-ZERO Edge LLM NPU — Activation / RMSNorm / RoPE LUT Calibration 인수인계

> **작성일:** 2026-11-04  
> **대상:** NPU System SW / Quantization·LUT Calibration 담당 팀원  
> **상태:** 완료된 소규모 실험 결과의 인수인계 및 후속 실험 명세. **LUT 최종 사양·scale은 아직 freeze하지 않음.**  
> **기준 SW:** [`l-zero-society/2026-1-edge-llm-npu-sw`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw), `lut_cali` 및 `norm10-scale-probe-results`  
> **기준 RTL:** [`l-zero-society/2026-1-edge-llm-npu-rtl`](https://github.com/l-zero-society/2026-1-edge-llm-npu-rtl), `feat/tpu-rocc-driver`  
> **모델/실행 환경:** 기존 GGUF Gemma 2B calibration 파이프라인; 이전 검증은 Mac CPU-only Python emulation 기준. RTL bit-accurate simulation을 완료했다는 뜻은 아님.

## 0. 인수인계 요약 — 먼저 읽을 것

**현재 결론:**

1. **Activation/GeGLU 경로는 작은 실험에서 가능성을 확인**했다. Gate의 INT10 출력을 받는 1,024-entry GeLU LUT(INT8 output)를 GeGLU 곱까지 고려해 calibration했을 때, 기존 mixed Linear/GeLU 대비 E2E Top1 일치율이 **83.33% → 84.38%**로 소폭 상승하고 logits NMSE는 **0.00636 → 0.00541**로 감소했다(192-token pilot). 이는 *fusion-aware Gate `s10` 재선정 실험*이며 GPALU output requantization은 **포함하지 않음**.
2. **GeGLU 이후 GPALU output width는 INT10이 INT8보다 유의미하게 좋았다.** 같은 192-token width sweep에서 **INT8 77.08%, INT10 85.42%, INT12 89.58%** Top1 일치율. 현재 INT10은 유력한 정확도/하드웨어 비용 타협안일 뿐, full-system freeze는 아님.
3. **RoPE10은 기존 pilot에서 치명적인 정확도 저하가 없었다.** GPALU10 + FP VPU 대비 RoPE10 추가 시 Top1 **85.42% → 86.46%**(192 tokens), KL **0.0857 → 0.0932**, logits NMSE **0.003355 → 0.003373**. 표본 오차를 감안하면 '개선'이 아니라 **큰 회귀가 관찰되지 않음**으로 해석. 장기 position/RTL pairing 검증은 남음.
4. **RMSNorm10은 아직 실패 상태**다. Phase 2에서 output scale/gamma를 올바르게 결합하지 않는 문제가 확인되었고, 이를 보정한 가상 Fused requantization에서도 24-target 소규모 E2E Top1은 **8.33%**에 불과했다(동일 subset의 GPALU10+FP Norm **95.83%**). 따라서 문제는 Scale LUT 한 가지로 한정되지 않는다.
5. **Accumulator overflow는 확인된 원인이 아니다.** signed INT10, N≤4096 대상 검사에서 최대 square-sum `2^30`(31-bit unsigned), 설계 accumulator 35-bit, 관찰 overflow 없음. 이전 `OverflowError`는 NLL에 `math.exp`를 적용한 **PPL 표시용 float64 overflow**였음.
6. **담당 우선순위:** (P0) RMSNorm **INT10 I/O oracle + static scale** 재검증 → (P1) 어느 Norm에서 오차가 처음 커지는지 **layer-wise propagation** → (P2) 실제 RTL과 일치하는 **gamma/output requantization 설계** → (P3) RoPE 장기 position/RTL 일치 검증 → (P4) 보존된 Activation/GeGLU baseline과 전체 통합 검증.

> **핵심 주의:** 다른 구성으로 측정한 실험의 Top1 숫자들을 하나의 순위표로 해석하지 말 것. `shared GeLU LUT`, `fusion-aware Gate s10`, `GPALU width`, `Norm/ RoPE`는 적용된 양자화 연산 범위가 서로 다르다.

---

## 1. 시스템 numerical contract와 용어

### 1.1 Activation / GeGLU / GPALU

Gemma MLP는 다음과 같이 본다.

\[
 g=XW_{gate}^{T},\quad u=XW_{up}^{T},\quad h=GELU(g)\odot u.
\]

- Gate/Up GEMM은 이전 프로젝트의 INT8 input/weight, INT32 accumulation 기반 linear quantization을 재사용한다. **Gate output은 INT10**, LUT input 주소는 `q_gate+512`(signed `[-512,511]` → `[0,1023]`).
- GeLU LUT: **1,024 entries, signed INT8 output**, 주로 `GELU(·, approximate="tanh")`; per-layer 입력 `s10`, LUT output `s_act` 및 table entry의 provenance를 함께 관리한다. 이전 global shared-LUT과 per-layer LUT는 **별도 후보**다.
- GeGLU raw product는 `INT8 × INT8 → signed INT16`. GPALU 정수 출력은 `M_G (UInt16)`, `S_G (UInt5)`의 static requantization 후 INT8/10/12 가운데 비교하였다. 입력 branch scale의 곱과 최종 output scale 간 관계가 중요하다.
- **구분:** `fusion-aware Gate s10 pilot`에서 GeGLU product는 reconstructed branch들의 FP 곱이었고, **GPALU output requantization이 아직 없었다**. 후속 `GPALU width sweep`에서 이를 별도로 모델링했다.

### 1.2 RMSNorm

Gemma RMSNorm의 실수 참조 계산은 (`gamma_i=1+weight_i`):

\[
 y_i=\frac{x_i}{\sqrt{\frac1N\sum_j x_j^2+\epsilon}}\gamma_i.
\]

`x_i≈s_in·q_i`일 때, INT10 output code의 정확한 목표는

\[
 q_{out,i}=\mathrm{sat}_{10}\!\left(\mathrm{RNE}\!\left[\frac{q_i\gamma_i}{s_{out}\sqrt{\mathrm{mean}(q^2)+\epsilon/s_{in}^2}}\right]\right).
\]

기존 `NormUnit.scala`의 Phase 2 literal은 대략 `sat_INT10((q_i * corrected_rsqrt_Q14) >> 14)`이며 **별도의 `1/s_out` requantization과 채널별 learned gamma 적용이 이 연산 코드에 없음**. LUT mantissa 10-bit index와 exponent 분리를 사용하므로 accumulator bit-width가 LUT entry 수를 직접 폭증시키는 것은 아니다. `1/s_out` 및 gamma를 얼마나 넓은 정수 intermediate에서 어떻게 합칠지는 **팀이 검증할 하드웨어/컴파일러 interface 문제**다.

### 1.3 RoPE

현재 측정한 Python 모델의 기본 후보: **signed INT10 activation**, **10-bit angle index**, **signed INT16 Q2.14 sin/cos coefficient**, 회전 후 signed INT10. Actual RTL에서 frequency 표현, angle wrapping, position index, pair indexing(모델의 half-split vs adjacent) 및 negative rounding가 일치하는지 별도 검증해야 한다.

---

## 2. 이미 완료된 관련 실험과 정확한 성능

### 2.1 Fusion을 고려한 INT10 Gate + GeLU activation LUT — 질문에 대한 직접적인 답

**소스:** [`fusion_aware_gate_s10_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/fusion_aware_gate_s10_pilot/report.md), [`e2e_summary.json`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/fusion_aware_gate_s10_pilot/e2e_summary.json).

- 비교 범위: 기존 mixed Linear/GeLU calibration의 **Gate s10만** per-layer 조정. Gate 이외 ordinary GEMM, Up, weight, row policy는 유지.
- 18개 layer 각각 9개 후보 `s10_new = s10_old × 2^(j/8)`, `j=-4…4`.
- Nested optimization: **inner**는 고정된 `s10`에서 standalone GeLU LUT `s_act`를 NMSE로 정함 → **outer**는 실제 `GELU(g) × up`의 **balanced GeGLU product NMSE** 최소화.
- 17/18개 layer의 `s10`가 바뀌었고, 9개가 실험 기준 beneficial. boundary winner 2개는 더 넓은 탐색이 필요할 수 있음.

| 동일 192-token pilot | 기존 Mixed baseline | Fusion-aware Gate s10 |
|---|---:|---:|
| E2E Top1 agreement | **83.33%** | **84.375%** |
| Logits KL(FP‖tested) | 0.124915 | **0.122146** |
| Logits NMSE | 0.00635558 | **0.00540839** |
| NLL | **4.84152** | 4.89122 |
| PPL | **126.66** | 133.12 |
| Balanced local GeGLU product NMSE | 0.00660850 | **0.00613884** |

**판정:** Local fusion error와 KL/NMSE/Top1은 개선, **NLL·PPL은 악화**. `PROCEED_TO_LARGE_BATCH`라고 기록되어 있으나 이는 **소규모 pilot 후속 진행 권고**이지 최종 채택이 아니다. GPALU product의 INT10 requantization은 아직 제외한 실험이다.

### 2.2 다른 Activation LUT 실험과 혼동하지 말 것

| 실험 | 수치/결론 | 비교 시 주의 |
|---|---|---|
| [`shared GELU activation LUT pilot`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/activation_lut_calibration_pilot/report.md) | E2E Top1 **90.10%**, KL **0.07769**, logits NMSE **0.003628**; global LUT `sL=0.0184367`, `sA=0.0741823`, local activation NMSE **0.003539** | GeGLU 전체 fused INT10 output 결과가 아님. 해당 실험의 frozen linear·양자화 범위가 별도이고 사전 정의된 KL/NMSE gate는 **REGRESSION**. |
| [`standalone per-op GeLU LUT`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/gelu_standalone_calibration/report.md) | selected LUT local activation NMSE **0.002636**; E2E Top1 **85.94%**, KL **0.08474** | GeGLU product나 Down consumer를 직접 최적화한 것은 아님; 동일 pilot에서 local NMSE 개선이 Top1 개선으로 직결되지 않았음. |
| [`mixed Linear + GeLU`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/mixed_linear_gelu_calibration/report.md) | 126 linear op(ordinary INT8 108, Gate INT10 18), GeLU LUT E2E Top1 **83.33%** | 후속 fusion-aware Gate pilot의 직접 비교 baseline. |

**따라서 “Fusion을 고려한 INT10 activation LUT가 90.1%였다”는 표현은 부정확하다.** 90.1%는 초기 **global shared GeLU LUT pilot**이고, Gate scale을 fusion product 기준으로 조정한 후속 실험은 **84.375%**이다. 실험 scope가 다른 두 값은 직접 우열 비교 대상이 아니다.

### 2.3 GPALU width sweep — 별도 결과

**소스:** [`gpalu_width_sweep_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/gpalu_width_sweep_pilot/report.md).

| 192-token 동일 실험 | INT8 output | INT10 output | INT12 output |
|---|---:|---:|---:|
| E2E Top1 | 77.08% | **85.42%** | 89.58% |
| KL | 0.21648 | **0.08571** | 0.06649 |
| Logits NMSE | 0.008429 | **0.003355** | 0.002666 |

- GeGLU `INT8×INT8` product를 INT8으로 좁힐 때 출력 range/resolution 병목이 큼. INT10은 INT8 대비 Top1 **+8.33%p**.
- INT12도 추가 개선이 있으므로 INT10은 비용/정확도 임시 타협, 절대적인 optimum 증명 아님.
- 이 sweep에서 **Down 이후와 RMSNorm/RoPE는 FP**였음. 전 NPU INT10 동작 결과로 해석하지 않는다.

### 2.4 RoPE pilot — 비교적 양호, 최종 검증은 남음

**소스:** [`vpu10_e2e_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/vpu10_e2e_pilot/report.md), [`rope_local_summary.csv`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/vpu10_e2e_pilot/rope_local_summary.csv).

| 192-token 동일 실험 | GPALU10 + FP RoPE | GPALU10 + RoPE10 |
|---|---:|---:|
| Top1 | 85.42% | 86.46% |
| KL | 0.08571 | 0.09320 |
| Logits NMSE | 0.00335491 | 0.00337265 |

RoPE local NMSE는 layer마다 대략 **5.2–5.5×10⁻⁵**, 관찰된 saturation 0. **판정:** 큰 precision 병목을 보이지 않았으나 192-token pilot이고, 큰 position에서의 phase/angle index 정확도, 모델 pair ordering, 실제 RTL bit-exact 동등성은 미확인.

### 2.5 Norm10 scale probe — 가장 중요한 미해결 이슈

**소스:** [`norm10-scale-probe-results` 브랜치 보고서](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/report.md), [`norm_local_comparison.csv`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/norm_local_comparison.csv), [`e2e_small_summary.json`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/e2e_small_summary.json), [`debugging_history.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/debugging_history.md).

**Local (FP activation capture, 5 representative modules × prefill/decode):** 아래 수치는 10개 module/group NMSE의 **산술평균**이며 energy-weighted global NMSE와 다름.

| Mode | Mean local NMSE ↓ | Reconstructed RMS / FP RMS |
|---|---:|---:|
| Input/output INT10 oracle | 0.09314 | 0.980 |
| 기존 Python emulator | 0.11919 | 0.894 |
| RTL literal | **0.77100** | **0.151** |
| Early truncation 후 post-gain | 0.40038 | 1.070 |
| Early truncation 전 fused requant oracle | **0.12964** | 0.944 |

**E2E (3 conversations × 8 targets = 24, Softmax/RoPE FP):**

| Mode | Top1 | KL | Logits NMSE |
|---|---:|---:|---:|
| GPALU10 + FP Norm | **95.83%** | 0.04851 | 0.002520 |
| GPALU10 + Old Norm | 8.33% | 6.50652 | 0.21355 |
| GPALU10 + RTL literal Norm | 0% | 89.54455 | 6.49119 |
| GPALU10 + Post-gain Norm | 0% | 10.19339 | 0.19831 |
| GPALU10 + Fused requant Norm | **8.33%** | 6.61441 | 0.29895 |

**확인된 사실:**

- RTL literal 출력 amplitude는 FP의 평균 약 **15%**, INT10 code 사용 범위가 지나치게 작다. `1/s_out`, gamma를 **최종 RNE/saturation 전** 결합하면 local NMSE는 크게 좋아진다. **하지만 E2E가 복구되지는 않음.**
- 모든 INT10 I/O oracle조차 local mean NMSE **0.093**이고, `model.layers.9.input_layernorm`에서는 prefill **0.34053**, decode **0.43770**. 이 관찰은 **현재 scale/INT10 표현 조건의 손실**을 보여주지만, INT10 bit-width 자체의 한계로 확정할 수는 없다.
- Layer 9의 oracle **output-code** zero fraction이 prefill **80.85%**, decode **83.34%**. 이는 *output code distribution*의 통계이지 input code의 zero 비율이 아니다. 또한 모델 출력의 자연스러운 희소성과 quantization에 의한 소실을 구분하지 않았으므로 **이 비율 하나만으로 scale이 나쁘다고 단정하지 말 것**. FP reference의 작은 값 분포와 input code histogram을 함께 측정해야 한다.
- 검사한 모든 실제/경계 row에서 input clipping 0%, oracle/fused output clipping도 거의 0%였음. **Clipping 부재는 좋은 resolution을 의미하지 않음.**
- UInt16-M/UInt5-S 근사는 이상적 fused coefficient와 거의 동일(`0.129635523` vs `0.129635685`); coefficient approximation이 현재 주 병목이라는 근거는 약함.
- signed INT10, 최대 N=4096의 square-sum은 최악 `2^30`; 35-bit accumulator overflow는 **미관찰**. 과거 Python `OverflowError`는 실제 accumulator가 아니라 `math.exp(nll)`의 float64 PPL 범위 초과.

**판정:** `MIXED_OR_INCONCLUSIVE`. Output-scale contract/early truncation 문제는 확실히 분석 대상이고, 기존 static I/O scale과 레이어 간 propagation 문제는 **추가 확인이 필요**. 24-target 결과는 문제를 찾기 위한 진단 자료이지 성능 최종 판정이 아니다.

---

## 3. 팀원이 인수인계받을 핵심 작업 — 우선순위 및 실험 순서

| Priority | Experiment | 독립변수/대조군 | 완료 조건 |
|---|---|---|---|
| **P0** | **Norm I/O oracle 검증** | FP Norm vs `INT10_INPUT_ONLY` vs `INT10_IO_ORACLE`; 37개 Norm을 all-enabled로도 평가 | 실제 INT10 I/O만으로 E2E 붕괴하는지 판별 |
| **P0** | **Norm scale 정책 검증** | 기존 static scale, absmax, p99·p99.9·p99.99·p99.995·p100, prefill/decode-balanced | Local clipping + zero-loss + NMSE/코사인 + E2E 후보 결과를 분리 |
| **P1** | **Layer-wise propagation** | 단일 Norm 삽입, 누적 순서 삽입, FP vs INT10 predecessor activation | 최초 급격한 악화 지점/선행 scale drift 근거 확보 |
| **P2** | **Norm bit-exact 인터페이스** | RTL literal vs model-compatible output requant vs Fused; gamma handling | 정확한 intermediate width, rounding, signedness, multiplier 및 buffer 요구사항 명세 |
| **P3** | **RoPE 정확성/long-position** | FP vs INT10; 다양한 position 및 pair-order 후보 | 매칭되는 rotation semantics, long-position drift/overflow/saturation 확인 |
| **P4** | **통합 검증** | frozen GPALU10/activation + calibrated Norm + RoPE | 같은 unseen population에서 degradation budget 충족 여부 확인 |

### 3.1 P0-A — 반드시 가장 먼저: all-Norm INT10 I/O oracle E2E

**목적:** LUT/FP-to-fixed-point 구현 오류를 제거해도 INT10 I/O representation만으로 모델이 붕괴하는지 분리한다. 기존 24-target probe는 local에서만 oracle을 실행했고, E2E 전체 37개 Norm에는 **oracle mode가 없었다**.

동일한 dataset, 독립 KV cache로 다음 mode를 고정한다.

1. `FP_FULL`: 모든 연산 FP.
2. `GPALU10_FP_NORM`: GPALU10만 적용, Norm/RoPE/Softmax는 FP (baseline).
3. `GPALU10_NORM_INPUT_ONLY_ORACLE`: Norm input을 `Q10→dequant`, 이후 RMSNorm FP; output은 FP. **입력 scale의 손실만 평가**.
4. `GPALU10_NORM_IO_ORACLE`: 입력·출력 INT10 양자화, 내부 rsqrt/gamma는 FP. **입출력 bit-width + scale 손실**.
5. `GPALU10_NORM_FUSED_REQUANT`: 현재 frozen scale + fixed-point LUT + gamma/`s_out` fused 변환.
6. (보조) `GPALU10_NORM_RTL_LITERAL`: 의도적으로 잘못된 현재 모델 contract가 어떤 결과를 내는지 regression reference로만 보관.

결정 로직: (3)부터 붕괴하면 **input scale/representation** 우선; (3)은 괜찮고 (4)부터 붕괴하면 **output scale/representation** 우선; (4)는 괜찮고 (5)가 붕괴하면 **fixed-point LUT/epsilon/rounding/gamma contract**를 우선 검토. 각 단계에서 dynamic propagated input과 FP reference의 분포를 비교한다.

### 3.2 P0-B — scale 재선정 및 검증

- 37개 Norm **각각** `s_in`, `s_out`, `epsilon_q=RNE(eps/s_in²)`, `gamma` contract를 기록한다. Per-op/per-layer static scale이 기본 후보; *runtime per-row* 또는 *per-channel input scale*은 하드웨어에서 지원되는 것으로 가정하지 말고 별도 architecture proposal로 표시.
- 각 activation에서 `absmax`, `p99`, `p99.9`, `p99.99`, `p99.995`, `p100`와 작은 log2 multiplier neighborhood를 평가. Tail을 버리는 percentile scale은 clipping률/오차가 허용될 때만 채택한다. **scale을 무조건 작게 하지 말 것.**
- Prefill/decode 분포가 다르면 각각의 NMSE를 계산하고 두 그룹을 **동일 가중 balanced objective**로 후보 선정한다. 두 그룹별 scale을 실제 실행 시 교체할 수 없다면, **단일 static scale**로 양쪽을 함께 평가해야 한다.
- `INPUT_QUANT_ONLY`, `OUTPUT_QUANT_ONLY`(동일 FP Norm output 양자화), `INT10_IO`, `IDEAL_RSQRT`, `LUT_RSQRT`, `FUSED_MS`, `RTL_LITERAL`를 분리한다. 서로 다른 reference를 가진 NMSE는 수학적으로 더해서 기여도를 만들지 않는다.
- 특히 `layer 9 input_layernorm`의 FP output distribution, `q_in=0` 비율, `q_out=0` 비율, nonzero FP value가 zero code로 소실되는 비율, 작은 성분의 에너지 점유율, outlier 영향, gamma별 scale sensitivity를 우선 조사한다.
- Input·output scale 후보는 **calibration split에서만 선택**하고 validation/E2E split에서는 후보 간 비교만 수행. 최종 확인 split에서는 재선정 금지.

### 3.3 P1 — 단일/누적 Norm 삽입 및 error propagation

37개 Norm의 실행 순서: 각 transformer layer의 `input_layernorm`, `post_attention_layernorm` 18쌍, 그리고 최종 `model.norm`.

1. `FP predecessor → single INT10 Norm → FP successor`로 모든 Norm에 대한 **local 영향** 측정.
2. `Norm[0:k] = INT10`, 나머지 FP인 **cumulative insertion**으로 0→37까지 순차 평가. 초기는 소수 토큰으로 최초 이상 구간만 탐색하고 큰 E2E 반복은 생략.
3. 매 지점에 FP predecessor와 quantized predecessor의 input tensor histogram을 캡처하여 **static scale drift** 계산: absmax ratio, RMS ratio, quantile ratio, zero-code rate, input/output clipping, local NMSE, block output NMSE, logits NMSE, KL, Top1.
4. 최초 이상 지점에는 `FP input -> INT10 Norm`, `INT10 predecessor -> FP Norm`, `INT10 predecessor -> INT10 Norm`의 2×2 조합을 비교한다. **연산 자체의 오차 vs upstream 분포 변화**를 분리한다.
5. 원인 해석 후 **propagation-aware calibration**(이전 stage의 INT10 output을 기반으로 재수집) 후보를 제한적으로 평가. E2E 데이터로 scale을 역최적화하지 않는다.

### 3.4 P2 — 실제 실행 가능한 Norm hardware/ISA/Compiler contract

- `rsqrt` LUT index10(1,024 entries) + Q2.14 coefficient(16-bit)로 시작. mantissa/exponent 분리와 odd-exponent `1/sqrt(2)` correction 재현.
- `sum(q²)` / `(sum * invN_Q20)>>20` / `epsilon_q` / `stat`의 모든 중간 width, signed/unsigned, wrap/truncation, shift, RNE를 `NormUnit.scala`와 byte-for-byte 비교한다. Python oracle에서는 임의의 integer overflow가 생기지 않게 arbitrary precision을 사용하되 *RTL 단계*의 masking/truncation은 명시적으로 적용.
- 특히 **Q2.14 corrected coefficient를 너무 일찍 작은 정수로 shift한 후 `1/s_out`을 적용하지 말 것**. Fused 구조라면 `q * rsqrt_mantissa * static_gain` 뒤 **최종** shift/RNE/saturation으로 구현하는 경로와 필요 intermediate width를 정량화.
- `gamma_i=1+weight_i`를 누가, 언제, 어느 OCM/QB/LUT bank에서 제공하는지 아키텍처/컴파일러 담당과 합의. per-channel gamma가 없다면 ideal oracle과 동일하지 않음. Extra multiplier/shift, buffer bandwidth, latency, row state, output scale ID 및 scalar/vector broadcast 비용 기록.
- Fused 재양자화의 `M/S`는 이전 실험에서 coefficient 근사 오차가 작았지만, 실제 hardware word-width와 throughput에 적합한지는 **별도 검증**.
- Scale LUT coefficient 16→10 bit 혹은 index 10→8/12 bit 축소/확장은 **오류 원인 분리 및 scale 안정화 후**만 실험한다.

### 3.5 P3 — RoPE 저위험 후속 검증

- 모델의 `rotary_dim`, sin/cos pair order(half-split vs adjacent), position indexing 및 Q0.16 frequency 표현을 FP baseline과 명시적으로 맞춘다. `Rope.scala`와 실제 model math를 vector-level golden test로 교차 검증.
- Position 집합: **0–31, 32/64/128/256/512/1024/2048/4096**, 추가로 모델이 지원하는 최대 context 부근. 지원하지 않는 position을 임의로 넣지 말 것. Overflow/wrap boundary도 검사.
- Signed INT10 input/output, angle index10, trig coeff16 Q2.14을 기본으로 두고 `FP→quant input`, `phase address discretization`, `sin/cos coefficient quantization`, `MAC shift/round`, `output clipping`을 각각 분리한다.
- RoPE는 기존 local NMSE `≈5.37×10⁻⁵`이고 E2E에 큰 회귀가 없었으므로 **Norm P0 해결 전 full width search를 할 필요는 낮음**. 다만 long-position phase drift/RTL mismatch가 나타나면 즉시 승격.

---

## 4. Large-scale calibration: 입력 규모·분할·실행 예산

아래는 **향후 팀 실험을 위한 제안 규모**이며 이전에 측정된 수치가 아니다. Mac M1 Max/CPU-only에서 여러 mode를 E2E 수행하면 상당히 비싸므로, **capture/local search는 넓게, full-model E2E는 단계적으로** 진행한다.

| Phase | 독립 conversation 수 | Position / token 규모 | 목적 |
|---|---:|---|---|
| **Smoke** | calibration 4, evaluation 3 | E2E 3×8 = **24 targets** | 기능, RNE, scale domain, 예외/검증기 확인 (이미 수행한 probe와 규모 동일) |
| **Local calibration** | **64 conversations** | prompt 길이 32/128/512/1024 등을 균형화; decode는 대화당 16–32 positions. 최소 약 **8k–16k prefill activation rows**가 나오도록 구성 | 모든 Norm/RoPE의 activation histogram/scale 후보 생성; local objective 최적화 |
| **Local held-out** | **32 conversations** | 위와 유사하나 별도 대화·topic·길이; decode 16–32 | local 후보 일반화, tail clipping·scale drift 확인 |
| **E2E screening** | **8×16 = 128 targets**, 필요 시 **16×16 = 256** | 각 mode 같은 prompt/teacher-forced target | 원인 분리 후 2–3 후보만 빠른 비교 |
| **E2E main validation** | **32×32 = 1,024 targets** | 대화·길이 strata별 결과 보고 | 후보의 Top1/KL/NMSE 및 layer/position별 회귀 평가 |
| **Final unseen confirmation** | **64×32 = 2,048 targets** (자원 허용 시 64×64=4,096) | **여기서는 후보/threshold 재선정 금지** | 최종 검증 및 freeze 제안 |

- **128개의 독립 calibration/held-out conversations(64+32+32 별도 E2E main split)** 같은 ID를 재사용하지 말 것. E2E main 32개, final 64개 역시 독립으로 관리. 총 원본 corpus에서 **각 split을 별도 배정**할 것. 표의 sample 수는 권장 출발점으로, 실제 source dataset에 독립된 대화가 부족하면 규모와 중복 제한을 명시한다.
- 동일 `anchor_id/response_id/tree_id`, 유사 prompt 변형, 동일 원본 대화의 다른 응답은 *split 간 중복*으로 취급. 고정 seed와 SHA256 manifest 필수.
- Local capture는 전체 텐서를 모델 forward마다 저장하지 말고 각 module × `prefill/decode` × sequence-length stratum당 **512–1,024개 row reservoir**와 streaming histogram/percentile sketch를 유지. FP-reference와 quantized-path input은 **동일 토큰/동일 위치**로 정렬.
- Prefill sequence 길이를 `[1–64]`, `[65–256]`, `[257–1024]`, `>1024`(모델 지원 시)로 나누고, decode는 position bucket별로 따로 보고. 네트워크가 실제로 지원하는 context를 넘기지 않는다.
- CPU 비용 때문에 E2E 1,024/2,048 target을 한 번에 모든 mode로 계산하지 않는다. **Smoke → local → screening → main → unseen**의 승인 단계로 실행. `modes×targets`에 비례한 모델 실행 시간이 필요하므로, 각 stage에서 소요 시간·메모리 실측을 기록하여 다음 stage 예산을 결정.
- 원본 activation NPZ/GGUF는 GitHub 100MB 제한과 개인 데이터 문제를 고려해 로컬/허가된 저장소에 보관; 원격에는 필요한 통계·CSV·hash manifest·중요 샘플의 비민감 summary만 publish.

> **작업량 관련:** 기존 24-target/6-mode Norm 실험도 CPU에서 약 808초가 걸렸다. 이 수치를 1,024/2,048 target에 단순 선형 적용해 완료 시간을 단정할 수는 없지만, 검증 모드 수를 줄이고 checkpoint/resume를 구현해야 한다는 근거로 충분하다.

---

## 5. 필수 Metric / error decomposition 명세

### 5.1 Local numerical metric (모듈·layer·prefill/decode·position bucket별)

| 분류 | 필수 측정값 | 목적 |
|---|---|---|
| **Precision** | `MSE`, `MAE`, `NMSE`, `cosine`, max abs error, p50/p95/p99 abs error | 출력 재구성 오차 및 tail 에러 |
| **Amplitude / distribution** | FP/output RMS ratio, mean/std, `absmax`, p50/p90/p99/p99.9/p99.99, histogram, `zero code fraction`, `unique code count` | dynamic range, scale utilization |
| **Quantization loss** | input/output clipping %, saturation count, `FP nonzero → code 0` 손실률, 작은 값 에너지 손실, code entropy | clipping 없이도 resolution 붕괴하는지 |
| **Norm statistic** | `N`, sqSum, intermediate required bits, meanSq, `epsilon_q`, mantissa index, exponent, LUT coefficient, corrected scale, relative rsqrt error | accumulator/rsqrt/epsilon 단계 증명 |
| **Requantization** | `s_in`, `s_out`, gamma min/max, M/S 정수값, coefficient relative error, RNE/shift mismatch, over/underflow flags | 실제 하드웨어 계수 구현 가능성 |
| **Propagation** | FP vs quant predecessor input NMSE/RMS/percentiles, per-block hidden NMSE, 최초 divergence layer | static scale drift vs local Norm 오차 분리 |
| **RoPE 전용** | position, rotary_dim/pair order, angle/phase-index error, sin/cos coeff error, norm preservation `||y||/||x||`, angular/cosine error | 각도 누적 오차·RTL 기능 정합성 |

- **NMSE 정의**: `Σ (y_hat − y_ref)^2 / Σ y_ref^2`. 보고서에는 `row-wise 평균`, `group-balanced 평균`, `energy-pooled global NMSE`를 **서로 구분**하고 분모가 매우 작은 row의 처리 규칙을 명시한다.
- `output zero fraction`은 원래 FP 성분이 작은 model에서도 높을 수 있으므로 그 자체만 보고 "scale 실패"라고 하지 않는다. 반드시 **FP-to-zero 에너지 비율**과 Oracle/선택 후보의 차이를 보고할 것.
- Error decomposition은 **같은 scale, 같은 입력, 같은 참조값**으로 비교한다. 서로 다른 기준의 NMSE는 가산적이지 않다.

권장 Norm path decomposition:

| Mode | 정의 | 분리되는 오차 |
|---|---|---|
| `FP_REFERENCE` | original FP RMSNorm | 기준 |
| `Q_INPUT_ONLY` | `Q10(x)·s_in` 후 FP RMSNorm | 입력 scale/width만 |
| `Q_OUTPUT_ONLY` | FP RMSNorm 후 INT10 output quant-dequant | 출력 scale/width만 |
| `IO_ORACLE` | 입력 INT10, 내부 정확한 FP RMSNorm, 출력 INT10 | I/O quant만 |
| `RSQRT_LUT_FUSED` | 같은 INT10 I/O, 정수 meanSq/LUT, gamma/`s_out` fused | LUT/statistic/fixed-point 추가 |
| `RTL_LITERAL` | 기존 RTL Phase 2 그대로 (+평가용 adapter 명시) | 실 모델 인터페이스 불일치 측정 |

### 5.2 E2E metric — 반드시 동일 대상 토큰

**Primary:** `KL(FP softmax || tested)`, `logits NMSE`, `Top1 agreement`(FP argmax와 일치율).  
**Secondary:** logits `MAE`, `cosine`, `Top5 inclusion`, `Top5 overlap`, ground-truth `NLL`, `logPPL` 및 표현 가능할 때의 `PPL`. PPL 계산에서 `exp(NLL)`이 float64 초과 시 **NLL 유지, PPL=unrepresentable**로 처리; numeric failure로 간주하지 말 것.

필수 breakdown: 전체와 layer/block별 영향, `prefill`/`decode`, short/medium/long prompt, decode position bucket, conversation별 분산·최악값, 95% confidence interval(대화 단위 bootstrap). 24/128 target pilot의 `Top1 ± n%p`는 변동이 크므로 **재현성/대조군**을 먼저 확인한다.

**권장 사전 게이트(확정 정책 아님):** 정상화된 `GPALU10_FP_NORM` 같은 동등 scope baseline과 비교해, 1,024-target E2E에서 **Top1 하락 ≤2%p, KL 증가 ≤0.02, logits NMSE 증가 ≤20%**를 1차 협의 기준으로 제안. 팀이 HW resource/cost를 고려해 **검증 전에** freeze하고 임계값을 결과에 맞춰 사후 변경하지 말 것. Baseline KL/NMSE가 거의 0인 경우 상대 증감 대신 절대값을 함께 사용.

---

## 6. Calibration search 및 재현성 규칙

1. **모드/조건부터 freeze**: 모델 GGUF hash, SW HEAD, RTL HEAD/hash, existing Gate/Up/GPALU qparam hash, tokenizer, prompts/targets, FP reference, RNG seed, rounding mode, signed saturation, `epsilon`/`gamma` contract, FP Softmax/RoPE 여부를 명시.
2. **Calibration split은 local scale 후보 생성만**: absmax + upper-tail percentiles + 제한된 log-scale neighborhood. Stage별 변수는 한 번에 하나(예: `s_in`만 → `s_out`만 → propagation-aware)씩 열어 원인을 분리한다.
3. **동일한 capture로 모든 후보 평가**: 동일 `x_FP`, same `q_in` contract, same `gamma`, same output reference. 비용 때문에 후보마다 full model 추론하지 않는다.
4. **Shortlist 2–3개만 E2E**: local balanced NMSE + worst-layer error + clipping budget + hardware feasibility로 제한한다. Accuracy-only winner를 E2E 결과에 보고 소급 선택하지 않는다.
5. **실행 재현성**: shell foreground/controlled resumable worker, `status.json`, append-only log, per-conversation checkpoint, `verification.json`, dataset hash, `commit SHA`, `report.md`. Git/네트워크 실패는 numerical experiment failure와 **분리**.
6. **원본 보호**: `lut_cali`의 기존 결과, `norm10-scale-probe-results`, `fusion_aware_gate_s10_pilot` 원본 artifact 변경 금지. 새로운 `diagnostics/norm10_large_calibration/`, `diagnostics/rope10_validation/`, `diagnostics/vpu10_integration/`를 생성하고 독립 branch/PR로 결과를 보낼 것.

추천 새 산출물:

```text
diagnostics/norm10_large_calibration/
├── data_split_manifest.json
├── source_and_rtl_hashes.json
├── capture_manifest.json
├── norm_scale_candidates.csv
├── norm_per_layer_local.csv
├── norm_code_histogram_summary.csv
├── norm_statistic_trace.json
├── norm_first_divergence.csv
├── norm_propagation_drift.csv
├── norm_selected_parameters.json
├── e2e_screening_summary.json
├── e2e_main_summary.json
├── e2e_unseen_confirmation.json
├── verification.json
└── report.md

diagnostics/rope10_validation/
├── position_coverage.json
├── rtl_pairing_and_phase_golden.csv
├── per_layer_local.csv
├── long_position_regressions.csv
├── e2e_summary.json
└── report.md
```

권장 `norm_per_layer_local.csv` 1행 단위: `(source_sha, dataset_sha, layer_name, norm_type, group, prompt_bucket, mode, s_in, s_out, row_count, element_count, nmse, cosine, rms_ratio, input_clip, output_clip, input_zero_fraction, output_zero_fraction, fp_energy_lost_to_zero, unique_output_codes, max_required_intermediate_bits, rsqrt_relative_error)`.

---

## 7. 팀원 완료 정의 / 의사결정 체크리스트

- [ ] 동일 데이터의 **all-37-Norm INT10 I/O oracle E2E**가 실행되고 결과가 저장됨.
- [ ] `Q_INPUT_ONLY` / `Q_OUTPUT_ONLY` / `IO_ORACLE` / `RSQRT_LUT_FUSED`가 서로 다른 원인을 식별함.
- [ ] **37 Norm 전체**의 FP/quant input histogram과 scale drift, worst layers, first divergence가 기록됨.
- [ ] `s_in`, `s_out` calibration policy 및 prefill/decode-balanced rationale, clipping·zero-code tradeoff가 검증됨.
- [ ] 35-bit accumulator와 모든 intermediate width, gamma delivery, output requant M/S, signed/RNE/shift가 RTL·compiler 담당과 합의됨.
- [ ] 1,024-target main E2E와 (자원 허용 시) 2,048-target unseen confirmation이 같은 baseline과 공정하게 비교됨.
- [ ] RoPE의 10-bit angle/INT16 coefficient, long position, model pairing, RTL golden vector 일치성을 검증함.
- [ ] Activation/GeGLU 결과를 다른 범위 실험과 혼동하지 않고, 최종 NPU composition에서 regression 여부를 확인함.
- [ ] Source hash/parameter hash/data split·rounding·compiler contract, 모든 표와 원인 진단을 보고서·GitHub PR에 명시함.
- [ ] **INT10이 불가능하다는 결론**은 corrected scale + oracle + propagation-aware 후보를 동일 unseen set에서 확인한 후에만 제안함. 필요하면 INT12·mixed precision은 **별도 architecture cost/benefit proposal**로 제출.

### 추가로 열려 있는 설계 결정

- Gamma per-channel를 어느 연산/메모리/컴파일러 단계가 제공할지.
- Norm의 `s_in/s_out` static granularity(레벨/operation/group/row)와 실제 ISA/QB 지원 범위.
- `s_out` requant coefficient를 LUT와 결합할지, LUT 후 multiplier/shift로 수행할지, intermediate bit width 및 timing.
- FP prefill vs propagated INT10 activation에 대한 scale 하나로 충분한지.
- Norm이 개선된 뒤 현재 RoPE/Softmax와 병합했을 때 추가 interaction error가 있는지.

---

## 8. 주요 참고 파일 모음

| 주제 | 검증된 자료 |
|---|---|
| Frozen linear/activation 정책 | [`docs/lut_calibration_guide.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/docs/lut_calibration_guide.md) |
| Mixed INT8 ordinary + INT10 Gate + GeLU | [`mixed_linear_gelu_calibration/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/mixed_linear_gelu_calibration/report.md) |
| **Fusion-aware Gate s10 + LUT** | [`fusion_aware_gate_s10_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/fusion_aware_gate_s10_pilot/report.md) · [`layer_selection.csv`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/fusion_aware_gate_s10_pilot/layer_selection.csv) |
| Early shared activation LUT (scope 다름) | [`activation_lut_calibration_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/activation_lut_calibration_pilot/report.md) |
| GPALU output width | [`gpalu_width_sweep_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/gpalu_width_sweep_pilot/report.md) |
| VPU10 / RoPE / combined Norm pilot | [`vpu10_e2e_pilot/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/lut_cali/diagnostics/vpu10_e2e_pilot/report.md) |
| RMSNorm 실제 원인 검증 | [`norm10_scale_probe/report.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/report.md) · [`e2e_small_summary.json`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/e2e_small_summary.json) |
| Runtime error 원인 | [`norm10_scale_probe/debugging_history.md`](https://github.com/l-zero-society/2026-1-edge-llm-npu-sw/blob/norm10-scale-probe-results/diagnostics/norm10_scale_probe/debugging_history.md) |
| RTL Norm 구현 | [`src/main/scala/npu/core/NormUnit.scala`](https://github.com/l-zero-society/2026-1-edge-llm-npu-rtl/blob/feat/tpu-rocc-driver/src/main/scala/npu/core/NormUnit.scala) |
| RTL RoPE 구현 | [`src/main/scala/npu/core/Rope.scala`](https://github.com/l-zero-society/2026-1-edge-llm-npu-rtl/blob/feat/tpu-rocc-driver/src/main/scala/npu/core/Rope.scala) |

**인수인계 시 한 줄 결론:** Activation/GeGLU INT10은 promising pilot 수준이고, RoPE10은 pilot상 큰 문제가 없지만 long-position·RTL 확인이 필요하며, **Normalizer는 Scale LUT만 수정해서 해결될 문제가 아닌 것이 확인되어 INT10 I/O oracle + scale 재선정 + propagation-aware calibration부터 다시 검증해야 한다.**
